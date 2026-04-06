"""Step-level caching optimizations (B1-B6, B22)."""

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
]
