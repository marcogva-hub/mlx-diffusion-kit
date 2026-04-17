"""Token-level optimizations: merging, pruning, sparsity (B7-B10)."""

from mlx_diffusion_kit.tokens.ddit_scheduling import DDiTScheduleConfig, DDiTScheduler
from mlx_diffusion_kit.tokens.toca import (
    ToCaConfig,
    ToCaLayerState,
    ToCaState,
    create_toca_state,
    toca_compose,
    toca_get_cached,
    toca_reset,
    toca_select_tokens,
    toca_update,
)
from mlx_diffusion_kit.tokens.learned_sparsity import DiffSparseConfig, DiffSparseRouter
from mlx_diffusion_kit.tokens.pruning import (
    PruneInfo,
    ToPiConfig,
    compute_token_importance,
    topi_prune,
    topi_restore,
)
from mlx_diffusion_kit.tokens.tome import (
    MergeInfo,
    ToMeConfig,
    compute_attn_bias_for_mfa,
    compute_proportional_bias,
    compute_spatiotemporal_similarity,
    tome_merge,
    tome_unmerge,
)

__all__ = [
    # B10
    "DDiTScheduleConfig",
    "DDiTScheduler",
    # B9
    "DiffSparseConfig",
    "DiffSparseRouter",
    # B7
    "ToCaConfig",
    "ToCaLayerState",
    "ToCaState",
    "create_toca_state",
    "toca_compose",
    "toca_get_cached",
    "toca_reset",
    "toca_select_tokens",
    "toca_update",
    # B8 ToMe
    "MergeInfo",
    "ToMeConfig",
    "compute_attn_bias_for_mfa",
    "compute_proportional_bias",
    "compute_spatiotemporal_similarity",
    "tome_merge",
    "tome_unmerge",
    # B8 ToPi
    "PruneInfo",
    "ToPiConfig",
    "compute_token_importance",
    "topi_prune",
    "topi_restore",
]
