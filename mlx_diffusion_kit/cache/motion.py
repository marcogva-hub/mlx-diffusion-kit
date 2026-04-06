"""WorldCache Motion-Aware Extension for TeaCache (B1).

Adds motion estimation to TeaCache's caching decision. When significant
motion is detected between frames, caching thresholds are tightened to
prevent skipping critical denoising steps.

Applicable to all 6 multi-step models (same targets as TeaCache).
All operations are vectorized — no Python loops over pixels.
"""

from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx


@dataclass
class MotionConfig:
    sensitivity: float = 2.0
    warp_cached_features: bool = False
    motion_threshold: float = 0.1
    temporal_window: int = 2
    method: str = "l1_diff"  # "l1_diff" | "gradient" | "block_matching"
    enabled: bool = True


@dataclass
class MotionState:
    prev_frames: list[mx.array] = field(default_factory=list)
    motion_magnitude: float = 0.0
    estimated_motion_vector: Optional[mx.array] = None  # [2] (dy, dx)


# Sobel kernels as constants
_SOBEL_X = mx.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=mx.float32)
_SOBEL_Y = mx.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=mx.float32)


def _apply_sobel(frame_2d: mx.array) -> tuple[mx.array, mx.array]:
    """Apply Sobel filters to a 2D frame. Returns (grad_x, grad_y).

    Args:
        frame_2d: [H, W] single-channel frame.

    Returns:
        (grad_x, grad_y) each [H-2, W-2] (valid convolution).
    """
    H, W = frame_2d.shape
    # Manual valid convolution via sliding window (vectorized)
    # Extract 3x3 patches
    patches = mx.zeros((H - 2, W - 2, 3, 3), dtype=frame_2d.dtype)
    for di in range(3):
        for dj in range(3):
            patches = patches.at[:, :, di, dj].add(
                frame_2d[di:H - 2 + di, dj:W - 2 + dj]
            )

    grad_x = mx.sum(patches * _SOBEL_X.reshape(1, 1, 3, 3), axis=(-2, -1))
    grad_y = mx.sum(patches * _SOBEL_Y.reshape(1, 1, 3, 3), axis=(-2, -1))
    return grad_x, grad_y


def _to_grayscale_2d(frame: mx.array) -> mx.array:
    """Convert frame to 2D grayscale. Handles [H,W], [H,W,C], [C,H,W]."""
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3:
        if frame.shape[-1] <= 4:  # [H, W, C]
            return mx.mean(frame, axis=-1)
        else:  # [C, H, W]
            return mx.mean(frame, axis=0)
    # Higher dims: take mean over all but last two
    while frame.ndim > 2:
        frame = mx.mean(frame, axis=0)
    return frame


def estimate_motion(
    current_frame: mx.array,
    prev_frame: mx.array,
    method: str = "l1_diff",
) -> float:
    """Estimate motion magnitude between two frames.

    Args:
        current_frame: Current frame (any spatial layout).
        prev_frame: Previous frame (same shape).
        method: "l1_diff", "gradient", or "block_matching".

    Returns:
        Motion magnitude normalized to [0, 1].
    """
    if method == "l1_diff":
        diff = mx.mean(mx.abs(current_frame - prev_frame))
        norm = mx.mean(mx.abs(prev_frame)) + 1e-6
        return min(1.0, (diff / norm).item())

    elif method == "gradient":
        return estimate_motion_gradient(current_frame, prev_frame)

    elif method == "block_matching":
        # Simplified: use L1 diff as proxy (full block matching is expensive)
        return estimate_motion(current_frame, prev_frame, method="l1_diff")

    else:
        raise ValueError(f"Unknown motion method: {method}")


def estimate_motion_gradient(current: mx.array, prev: mx.array) -> float:
    """Estimate motion via Sobel gradient magnitude change.

    Computes spatial gradients of both frames, then measures how much
    the gradient field changed — indicating edge displacement (motion).

    Args:
        current: Current frame.
        prev: Previous frame.

    Returns:
        Motion magnitude in [0, 1].
    """
    cur_2d = _to_grayscale_2d(current.astype(mx.float32))
    prev_2d = _to_grayscale_2d(prev.astype(mx.float32))

    if cur_2d.shape[0] < 3 or cur_2d.shape[1] < 3:
        # Too small for Sobel
        return estimate_motion(current, prev, method="l1_diff")

    gx_cur, gy_cur = _apply_sobel(cur_2d)
    gx_prev, gy_prev = _apply_sobel(prev_2d)

    mag_cur = mx.sqrt(gx_cur * gx_cur + gy_cur * gy_cur + 1e-8)
    mag_prev = mx.sqrt(gx_prev * gx_prev + gy_prev * gy_prev + 1e-8)

    diff = mx.mean(mx.abs(mag_cur - mag_prev))
    norm = mx.mean(mag_prev) + 1e-6
    return min(1.0, (diff / norm).item())


def motion_adjusted_threshold(
    base_threshold: float,
    motion_magnitude: float,
    config: MotionConfig,
) -> float:
    """Adjust caching threshold based on detected motion.

    High motion → lower threshold → more steps computed (conservative).
    Low motion → higher threshold → more steps skipped (aggressive).

    Formula: threshold = base_threshold / (1 + sensitivity * motion_magnitude)
    """
    if not config.enabled:
        return base_threshold
    return base_threshold / (1.0 + config.sensitivity * motion_magnitude)


def estimate_motion_vector(
    current_frame: mx.array,
    prev_frame: mx.array,
) -> mx.array:
    """Estimate a simple global motion vector (dy, dx) between frames.

    Uses center-of-mass displacement of the absolute difference map
    as a cheap proxy for optical flow direction.

    Returns:
        [2] array: (dy, dx) estimated displacement.
    """
    cur_2d = _to_grayscale_2d(current_frame.astype(mx.float32))
    prev_2d = _to_grayscale_2d(prev_frame.astype(mx.float32))

    diff = mx.abs(cur_2d - prev_2d)
    total = mx.sum(diff) + 1e-8

    H, W = diff.shape
    ys = mx.arange(H, dtype=mx.float32)
    xs = mx.arange(W, dtype=mx.float32)

    # Center of mass of difference
    com_y = mx.sum(diff * ys.reshape(-1, 1)) / total
    com_x = mx.sum(diff * xs.reshape(1, -1)) / total

    # Center of frame
    center_y = (H - 1) / 2.0
    center_x = (W - 1) / 2.0

    return mx.array([com_y.item() - center_y, com_x.item() - center_x])


def warp_features_by_motion(
    features: mx.array,
    motion_vector: mx.array,
) -> mx.array:
    """Warp cached features by estimated motion (simple translation).

    Applies integer-pixel shift to spatial features. This is a training-free
    approximation — for proper optical flow warping, a flow field would be
    needed (out of scope here).

    Args:
        features: [B, C, H, W] or [B, N, D] features.
        motion_vector: [2] (dy, dx) displacement.

    Returns:
        Shifted features (same shape, zero-padded at boundaries).
    """
    dy = int(round(motion_vector[0].item()))
    dx = int(round(motion_vector[1].item()))

    if dy == 0 and dx == 0:
        return features

    if features.ndim == 4:
        # [B, C, H, W] — shift spatially
        result = mx.zeros_like(features)
        B, C, H, W = features.shape

        src_y_start = max(0, -dy)
        src_y_end = min(H, H - dy)
        src_x_start = max(0, -dx)
        src_x_end = min(W, W - dx)

        dst_y_start = max(0, dy)
        dst_x_start = max(0, dx)

        h_len = src_y_end - src_y_start
        w_len = src_x_end - src_x_start

        if h_len > 0 and w_len > 0:
            result[:, :, dst_y_start:dst_y_start + h_len, dst_x_start:dst_x_start + w_len] = \
                features[:, :, src_y_start:src_y_end, src_x_start:src_x_end]
        return result

    # For [B, N, D] tokens — no spatial layout, return as-is
    return features


class MotionTracker:
    """Tracks inter-frame motion and adjusts caching thresholds.

    Maintains a sliding window of recent frames and provides motion-adjusted
    thresholds for TeaCache integration.
    """

    def __init__(self, config: Optional[MotionConfig] = None):
        self.config = config or MotionConfig()
        self._state = MotionState(prev_frames=[])

    def update(self, frame: mx.array) -> float:
        """Update tracker with a new frame and return motion magnitude.

        Args:
            frame: Current video frame.

        Returns:
            Estimated motion magnitude [0, 1].
        """
        if not self.config.enabled:
            self._state.motion_magnitude = 0.0
            return 0.0

        if self._state.prev_frames:
            mag = estimate_motion(
                frame, self._state.prev_frames[-1], self.config.method
            )
            # Also estimate motion vector for warping
            self._state.estimated_motion_vector = estimate_motion_vector(
                frame, self._state.prev_frames[-1]
            )
        else:
            mag = 0.0

        self._state.motion_magnitude = mag

        # Maintain sliding window
        self._state.prev_frames.append(frame)
        if len(self._state.prev_frames) > self.config.temporal_window:
            self._state.prev_frames = self._state.prev_frames[-self.config.temporal_window:]

        return mag

    def get_adjusted_threshold(self, base_threshold: float) -> float:
        """Return caching threshold adjusted by current motion."""
        return motion_adjusted_threshold(
            base_threshold, self._state.motion_magnitude, self.config
        )

    def warp_cached(self, features: mx.array) -> mx.array:
        """Warp cached features by estimated motion if enabled."""
        if (
            not self.config.warp_cached_features
            or self._state.estimated_motion_vector is None
        ):
            return features
        return warp_features_by_motion(features, self._state.estimated_motion_vector)

    @property
    def motion_magnitude(self) -> float:
        return self._state.motion_magnitude

    def reset(self) -> None:
        """Clear all motion tracking state."""
        self._state = MotionState(prev_frames=[])
