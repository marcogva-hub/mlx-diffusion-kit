"""Attention compression optimizations (B12)."""

from mlx_diffusion_kit.attention.ditfastattn import (
    AttnStrategy,
    DiTFastAttnConfig,
    DiTFastAttnState,
    create_ditfastattn_state,
    ditfastattn_decide,
    ditfastattn_get_cached_attn,
    ditfastattn_get_cached_residual,
    ditfastattn_record_attn_map,
    ditfastattn_record_residual,
    ditfastattn_reset,
)
from mlx_diffusion_kit.attention.residual import (
    compute_residual_scale,
    residual_gate_from_sensitivity,
    scaled_residual_add,
)

__all__ = [
    # B12
    "AttnStrategy",
    "DiTFastAttnConfig",
    "DiTFastAttnState",
    "create_ditfastattn_state",
    "ditfastattn_decide",
    "ditfastattn_get_cached_attn",
    "ditfastattn_get_cached_residual",
    "ditfastattn_record_attn_map",
    "ditfastattn_record_residual",
    "ditfastattn_reset",
    # residual utilities
    "compute_residual_scale",
    "residual_gate_from_sensitivity",
    "scaled_residual_add",
]
