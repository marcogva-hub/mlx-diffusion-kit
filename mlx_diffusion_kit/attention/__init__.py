"""Attention compression optimizations (B12)."""

from mlx_diffusion_kit.attention.ditfastattn import (
    DiTFastAttnConfig,
    DiTFastAttnManager,
    HeadStrategy,
)
from mlx_diffusion_kit.attention.residual import (
    compute_residual_scale,
    residual_gate_from_sensitivity,
    scaled_residual_add,
)

__all__ = [
    "DiTFastAttnConfig",
    "DiTFastAttnManager",
    "HeadStrategy",
    "compute_residual_scale",
    "residual_gate_from_sensitivity",
    "scaled_residual_add",
]
