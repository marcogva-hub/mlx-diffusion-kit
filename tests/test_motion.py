"""Tests for WorldCache motion-aware extension."""

import mlx.core as mx

from mlx_diffusion_kit.cache.motion import (
    MotionConfig,
    MotionTracker,
    estimate_motion,
    estimate_motion_gradient,
    motion_adjusted_threshold,
    warp_features_by_motion,
)


def test_identical_frames_zero_motion():
    frame = mx.ones((16, 16, 3))
    mag = estimate_motion(frame, frame, method="l1_diff")
    assert mag < 0.01


def test_different_frames_high_motion():
    f1 = mx.zeros((16, 16, 3))
    f2 = mx.ones((16, 16, 3))
    mag = estimate_motion(f1, f2, method="l1_diff")
    assert mag > 0.5


def test_gradient_detects_edge_shift():
    """Sobel gradient should detect a displaced edge."""
    # Frame with vertical edge at x=4
    f1 = mx.zeros((16, 16))
    f1 = f1.at[:, 4:].add(1.0)

    # Same edge shifted to x=8
    f2 = mx.zeros((16, 16))
    f2 = f2.at[:, 8:].add(1.0)

    mag = estimate_motion_gradient(f1, f2)
    assert mag > 0.1, f"Expected significant gradient change, got {mag}"


def test_gradient_identical_zero():
    frame = mx.ones((16, 16)) * 0.5
    mag = estimate_motion_gradient(frame, frame)
    assert mag < 0.01


def test_adjusted_threshold_high_motion():
    """High motion → lower threshold."""
    cfg = MotionConfig(sensitivity=2.0)
    base = 0.3
    adjusted = motion_adjusted_threshold(base, 0.5, cfg)
    # 0.3 / (1 + 2.0 * 0.5) = 0.3 / 2.0 = 0.15
    assert abs(adjusted - 0.15) < 0.01


def test_adjusted_threshold_no_motion():
    """Zero motion → threshold unchanged."""
    cfg = MotionConfig(sensitivity=2.0)
    base = 0.3
    adjusted = motion_adjusted_threshold(base, 0.0, cfg)
    assert abs(adjusted - base) < 0.001


def test_adjusted_threshold_disabled():
    cfg = MotionConfig(enabled=False)
    assert motion_adjusted_threshold(0.3, 1.0, cfg) == 0.3


def test_warp_features_shift():
    """Warp should shift features spatially."""
    features = mx.zeros((1, 1, 8, 8))
    features = features.at[0, 0, 3, 3].add(1.0)

    mv = mx.array([0.0, 2.0])  # shift right by 2
    warped = warp_features_by_motion(features, mv)

    assert warped.shape == features.shape
    # Original value at (3,3) should now be at (3,5)
    assert warped[0, 0, 3, 5].item() > 0.5
    # Original position should be zero (shifted away)
    assert warped[0, 0, 3, 3].item() < 0.01


def test_warp_no_motion():
    features = mx.random.normal((1, 3, 8, 8))
    mv = mx.array([0.0, 0.0])
    warped = warp_features_by_motion(features, mv)
    assert mx.array_equal(warped, features)


def test_warp_3d_tokens_passthrough():
    """[B, N, D] tokens have no spatial layout → returned as-is."""
    features = mx.random.normal((2, 16, 64))
    mv = mx.array([5.0, 3.0])
    warped = warp_features_by_motion(features, mv)
    assert mx.array_equal(warped, features)


def test_tracker_history_bounded():
    cfg = MotionConfig(temporal_window=3)
    tracker = MotionTracker(cfg)

    for i in range(10):
        frame = mx.ones((8, 8)) * float(i)
        tracker.update(frame)

    assert len(tracker._state.prev_frames) <= 3


def test_tracker_reset():
    cfg = MotionConfig()
    tracker = MotionTracker(cfg)
    tracker.update(mx.ones((8, 8)))
    tracker.update(mx.ones((8, 8)) * 2)
    assert tracker.motion_magnitude > 0

    tracker.reset()
    assert tracker.motion_magnitude == 0.0
    assert len(tracker._state.prev_frames) == 0


def test_tracker_get_adjusted():
    cfg = MotionConfig(sensitivity=2.0)
    tracker = MotionTracker(cfg)
    tracker.update(mx.zeros((8, 8)))
    tracker.update(mx.ones((8, 8)))

    adjusted = tracker.get_adjusted_threshold(0.3)
    assert adjusted < 0.3  # Motion detected → tighter threshold
