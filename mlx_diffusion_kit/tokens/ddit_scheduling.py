"""B10 — DDiT Dynamic Patch Scheduling for multi-step DiT models.

Adjusts patch stride (and thus token count) per denoising step.
Early steps use large patches (coarse, fast), final steps use stride=1
(full resolution). The scheduler computes the stride; the model code
applies it by adjusting its patchify conv stride and positional embeddings.

Expected model-side API:
    stride = scheduler.get_patch_stride(step_idx)
    patches = patchify_conv(latent, stride=stride)       # fewer tokens
    pos_emb = interpolate_pos_emb(base_pos_emb, stride)  # match new grid
    output = transformer(patches + pos_emb, ...)
    decoded = unpatchify(output, stride=stride)
"""

import math
from dataclasses import dataclass


@dataclass
class DDiTScheduleConfig:
    min_patch_stride: int = 1
    max_patch_stride: int = 4
    warmup_fraction: float = 0.3
    schedule: str = "cosine"  # "linear" | "cosine" | "step"
    enabled: bool = True


def _nearest_power_of_2(x: float) -> int:
    """Round to nearest power of 2, clamped to [1, ...)."""
    if x <= 1.0:
        return 1
    log2 = math.log2(x)
    lower = 2 ** int(log2)
    upper = 2 ** (int(log2) + 1)
    return lower if (x - lower) <= (upper - x) else upper


class DDiTScheduler:
    """Per-step patch stride scheduler for dynamic-resolution DiT inference.

    Produces a monotonically non-increasing stride sequence from
    max_patch_stride down to min_patch_stride over the warmup window.
    """

    def __init__(self, total_steps: int, config: DDiTScheduleConfig | None = None):
        self.config = config or DDiTScheduleConfig()
        self.total_steps = max(total_steps, 1)
        self._warmup_steps = max(1, int(self.total_steps * self.config.warmup_fraction))

    def get_patch_stride(self, step_idx: int) -> int:
        """Return the patch stride for this step.

        Stride decreases from max_patch_stride to min_patch_stride over
        the warmup window, then stays at min_patch_stride.

        Returns:
            Power-of-2 stride value.
        """
        if not self.config.enabled:
            return self.config.min_patch_stride

        if step_idx >= self._warmup_steps:
            return self.config.min_patch_stride

        t = step_idx / self._warmup_steps  # 0.0 → 1.0

        s_max = self.config.max_patch_stride
        s_min = self.config.min_patch_stride

        if self.config.schedule == "linear":
            raw = s_max + (s_min - s_max) * t
        elif self.config.schedule == "cosine":
            # Cosine annealing: slow start, fast middle, slow end
            raw = s_min + (s_max - s_min) * 0.5 * (1.0 + math.cos(math.pi * t))
        elif self.config.schedule == "step":
            # Abrupt switch at warmup boundary
            raw = s_max if t < 1.0 else s_min
        else:
            raw = s_min

        stride = _nearest_power_of_2(raw)
        # Clamp to [min, max]
        stride = max(self.config.min_patch_stride, min(stride, self.config.max_patch_stride))
        return stride

    def get_token_reduction_factor(self, step_idx: int, spatial_dims: int = 2) -> float:
        """Return the token reduction factor for this step.

        Args:
            step_idx: Current step.
            spatial_dims: 2 for images, 3 for video (stride applies per dim).

        Returns:
            stride^spatial_dims — e.g., stride=4 in 2D → 16x fewer tokens.
        """
        stride = self.get_patch_stride(step_idx)
        return float(stride ** spatial_dims)
