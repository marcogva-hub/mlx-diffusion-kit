"""mlx-diffusion-kit: Inference optimizations for diffusion models on MLX."""

from mlx_diffusion_kit.__version__ import __version__
from mlx_diffusion_kit.maturity import Maturity, get_maturity, list_components

# --- Attention (B12) ---
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

# --- Cache (B1-B6, B22) ---
from mlx_diffusion_kit.cache.deep_cache import (
    DeepCacheConfig,
    DeepCacheState,
    create_deepcache_state,
    deepcache_get,
    deepcache_reset,
    deepcache_should_recompute,
    deepcache_store,
)
from mlx_diffusion_kit.cache.encoder_sharing import EncoderSharingConfig
from mlx_diffusion_kit.cache.fb_cache import (
    FBCacheConfig,
    FBCacheState,
    create_fbcache_state,
    fbcache_reconstruct,
    fbcache_reset,
    fbcache_should_compute_remaining,
    fbcache_update,
)
from mlx_diffusion_kit.cache.layer_redundancy import (
    analyze_layer_redundancy,
    select_cacheable_layers,
)
from mlx_diffusion_kit.cache.multigranular import MultiGranularCache, MultiGranularConfig
from mlx_diffusion_kit.cache.smooth_cache import (
    InterpolationMode,
    SmoothCacheConfig,
    smooth_cache_interpolate,
    smooth_cache_record,
)
from mlx_diffusion_kit.cache.spectral_cache import (
    SpectralCacheConfig,
    SpectralCacheState,
    create_spectral_cache_state,
    spectral_cache_apply,
    spectral_cache_reset,
    spectral_cache_update,
)
from mlx_diffusion_kit.cache.teacache import (
    TeaCacheConfig,
    load_coefficients,
    teacache_should_compute,
    teacache_update,
)

# --- Encoder (B15) ---
from mlx_diffusion_kit.encoder.embedding_cache import TextEmbeddingCache

# --- Gating (B11) ---
from mlx_diffusion_kit.gating.tgate import TGateConfig, create_tgate_state, tgate_forward

# --- Orchestrator (B23) ---
from mlx_diffusion_kit.orchestrator import (
    BlockStrategy,
    DiffusionOptimizer,
    OrchestratorConfig,
    PISAConfig,
)

# --- Quality (B13) ---
from mlx_diffusion_kit.quality.freeu import FreeUConfig, freeu_filter

# --- Scheduler (B14) ---
from mlx_diffusion_kit.scheduler.adaptive_stepping import AdaptiveStepConfig, AdaptiveStepScheduler
from mlx_diffusion_kit.scheduler.dpm_solver_v3 import DPMSolverV3, DPMSolverV3Config, NoiseSchedule

# --- Tokens (B7-B10) ---
from mlx_diffusion_kit.tokens.ddit_scheduling import DDiTScheduleConfig, DDiTScheduler
from mlx_diffusion_kit.tokens.learned_sparsity import DiffSparseConfig, DiffSparseRouter
from mlx_diffusion_kit.tokens.pruning import ToPiConfig, topi_prune, topi_restore
from mlx_diffusion_kit.tokens.toca import (
    ToCaConfig,
    ToCaLayerState,
    ToCaState,
    create_toca_state,
    toca_compose,
    toca_reset,
    toca_select_tokens,
    toca_update,
)
from mlx_diffusion_kit.tokens.tome import (
    ToMeConfig,
    compute_attn_bias_for_mfa,
    compute_proportional_bias,
    compute_spatiotemporal_similarity,
    tome_merge,
    tome_unmerge,
)

# --- VAE (B17-B18) ---
from mlx_diffusion_kit.vae.separable_conv3d import (
    SeparableConv3D,
    build_separable_from_decomposition,
    decompose_conv3d_to_separable,
)
from mlx_diffusion_kit.vae.wavelet_cache import (
    WaveletCacheConfig,
    WaveletVAECache,
    chunked_decode_with_cache,
)

__all__ = [
    "__version__",
    # B1 TeaCache
    "TeaCacheConfig",
    "teacache_should_compute",
    "teacache_update",
    "load_coefficients",
    # B2 FBCache
    "FBCacheConfig",
    "FBCacheState",
    "create_fbcache_state",
    "fbcache_reconstruct",
    "fbcache_reset",
    "fbcache_should_compute_remaining",
    "fbcache_update",
    # B3 SpectralCache + SeaCache
    "SpectralCacheConfig",
    "SpectralCacheState",
    "create_spectral_cache_state",
    "spectral_cache_apply",
    "spectral_cache_reset",
    "spectral_cache_update",
    # B4 SmoothCache
    "InterpolationMode",
    "SmoothCacheConfig",
    "smooth_cache_interpolate",
    "smooth_cache_record",
    # B5 DeepCache
    "DeepCacheConfig",
    "DeepCacheState",
    "create_deepcache_state",
    "deepcache_get",
    "deepcache_reset",
    "deepcache_should_recompute",
    "deepcache_store",
    # MosaicDiff (layer redundancy)
    "analyze_layer_redundancy",
    "select_cacheable_layers",
    # B6 Multi-Granular Cache
    "MultiGranularCache",
    "MultiGranularConfig",
    # B7 ToCa
    "ToCaConfig",
    "ToCaLayerState",
    "ToCaState",
    "create_toca_state",
    "toca_compose",
    "toca_reset",
    "toca_select_tokens",
    "toca_update",
    # B8 ToMe
    "ToMeConfig",
    "tome_merge",
    "tome_unmerge",
    "compute_proportional_bias",
    "compute_attn_bias_for_mfa",
    "compute_spatiotemporal_similarity",
    # B8 ToPi
    "ToPiConfig",
    "topi_prune",
    "topi_restore",
    # B9 DiffSparse
    "DiffSparseConfig",
    "DiffSparseRouter",
    # B10 DDiT Scheduling
    "DDiTScheduleConfig",
    "DDiTScheduler",
    # B11 T-GATE
    "TGateConfig",
    "tgate_forward",
    "create_tgate_state",
    # B12 DiTFastAttn
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
    # B13 FreeU
    "FreeUConfig",
    "freeu_filter",
    # B14.1 DPM-Solver-v3
    "DPMSolverV3",
    "DPMSolverV3Config",
    "NoiseSchedule",
    # B14.2 Adaptive Stepping
    "AdaptiveStepConfig",
    "AdaptiveStepScheduler",
    # B15 Embedding Cache
    "TextEmbeddingCache",
    # B17 WF-VAE Cache
    "WaveletCacheConfig",
    "WaveletVAECache",
    "chunked_decode_with_cache",
    # B18 Separable Conv3D
    "SeparableConv3D",
    "build_separable_from_decomposition",
    "decompose_conv3d_to_separable",
    # B22 Encoder Sharing
    "EncoderSharingConfig",
    # B23 Orchestrator
    "BlockStrategy",
    "DiffusionOptimizer",
    "OrchestratorConfig",
    "PISAConfig",
    # Maturity tracking
    "Maturity",
    "get_maturity",
    "list_components",
]
