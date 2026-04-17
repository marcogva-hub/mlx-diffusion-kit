"""Step-level caching optimizations (B1-B6, B22)."""

from mlx_diffusion_kit.cache.motion import (
    MotionConfig,
    MotionTracker,
    estimate_motion,
    motion_adjusted_threshold,
    warp_features_by_motion,
)
from mlx_diffusion_kit.cache.deep_cache import (
    DeepCacheConfig,
    DeepCacheState,
    create_deepcache_state,
    deepcache_get,
    deepcache_reset,
    deepcache_should_recompute,
    deepcache_store,
)
from mlx_diffusion_kit.cache.layer_redundancy import (
    analyze_layer_redundancy,
    select_cacheable_layers,
)
from mlx_diffusion_kit.cache.fb_cache import (
    FBCacheConfig,
    FBCacheState,
    create_fbcache_state,
    fbcache_reconstruct,
    fbcache_reset,
    fbcache_should_compute_remaining,
    fbcache_update,
)
from mlx_diffusion_kit.cache.multigranular import (
    BWCacheConfig,
    BWCacheAllocator,
    MultiGranularCache,
    MultiGranularConfig,
    QuantCacheConfig,
    UniCPConfig,
    UniCPPolicy,
    quantcache_compress,
    quantcache_decompress,
)
from mlx_diffusion_kit.cache.encoder_sharing import (
    EncoderSharingConfig,
    EncoderSharingState,
    create_encoder_sharing_state,
    encoder_sharing_get_cached,
    encoder_sharing_should_recompute,
    encoder_sharing_update,
)
from mlx_diffusion_kit.cache.spectral_cache import (
    SpectralCacheConfig,
    SpectralCacheState,
    create_spectral_cache_state,
    spectral_cache_apply,
    spectral_cache_reset,
    spectral_cache_update,
)
from mlx_diffusion_kit.cache.smooth_cache import (
    InterpolationMode,
    SmoothCacheConfig,
    SmoothCacheState,
    create_smooth_cache_state,
    smooth_cache_interpolate,
    smooth_cache_record,
)
from mlx_diffusion_kit.cache.teacache import (
    TeaCacheConfig,
    TeaCacheState,
    create_teacache_state,
    load_coefficients,
    teacache_should_compute,
    teacache_update,
)

__all__ = [
    # B5 DeepCache
    "DeepCacheConfig",
    "DeepCacheState",
    "create_deepcache_state",
    "deepcache_get",
    "deepcache_reset",
    "deepcache_should_recompute",
    "deepcache_store",
    # MosaicDiff layer redundancy (formerly bundled with DeepCache)
    "analyze_layer_redundancy",
    "select_cacheable_layers",
    # B2 FBCache
    "FBCacheConfig",
    "FBCacheState",
    "create_fbcache_state",
    "fbcache_reconstruct",
    "fbcache_reset",
    "fbcache_should_compute_remaining",
    "fbcache_update",
    # B3 SpectralCache
    "SpectralCacheConfig",
    "SpectralCacheState",
    "create_spectral_cache_state",
    "spectral_cache_apply",
    "spectral_cache_reset",
    "spectral_cache_update",
    # B1 TeaCache
    "TeaCacheConfig",
    "TeaCacheState",
    "create_teacache_state",
    "load_coefficients",
    "teacache_should_compute",
    "teacache_update",
    # B22 Encoder Sharing
    "EncoderSharingConfig",
    "EncoderSharingState",
    "create_encoder_sharing_state",
    "encoder_sharing_get_cached",
    "encoder_sharing_should_recompute",
    "encoder_sharing_update",
    # B6 Multi-Granular Cache
    "BWCacheConfig",
    "BWCacheAllocator",
    "MultiGranularCache",
    "MultiGranularConfig",
    "QuantCacheConfig",
    "UniCPConfig",
    "UniCPPolicy",
    "quantcache_compress",
    "quantcache_decompress",
    # B4 SmoothCache
    "InterpolationMode",
    "SmoothCacheConfig",
    "SmoothCacheState",
    "create_smooth_cache_state",
    "smooth_cache_interpolate",
    "smooth_cache_record",
    # WorldCache motion
    "MotionConfig",
    "MotionTracker",
    "estimate_motion",
    "motion_adjusted_threshold",
    "warp_features_by_motion",
]
