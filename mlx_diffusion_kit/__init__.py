"""mlx-diffusion-kit: Inference optimizations for diffusion models on MLX."""

from mlx_diffusion_kit.__version__ import __version__
from mlx_diffusion_kit.cache.smooth_cache import (
    InterpolationMode,
    SmoothCacheConfig,
    smooth_cache_interpolate,
    smooth_cache_record,
)
from mlx_diffusion_kit.cache.teacache import (
    TeaCacheConfig,
    teacache_should_compute,
    teacache_update,
    load_coefficients,
)
from mlx_diffusion_kit.encoder.embedding_cache import TextEmbeddingCache
from mlx_diffusion_kit.gating.tgate import TGateConfig, tgate_forward, create_tgate_state
from mlx_diffusion_kit.quality.freeu import FreeUConfig, freeu_filter
from mlx_diffusion_kit.scheduler.adaptive_stepping import AdaptiveStepConfig, AdaptiveStepScheduler
from mlx_diffusion_kit.tokens.tome import ToMeConfig, tome_merge, tome_unmerge, compute_proportional_bias
from mlx_diffusion_kit.vae.wavelet_cache import WaveletCacheConfig, WaveletVAECache, chunked_decode_with_cache

__all__ = [
    "__version__",
    # B1 TeaCache
    "TeaCacheConfig",
    # B4 SmoothCache
    "InterpolationMode",
    "SmoothCacheConfig",
    "smooth_cache_interpolate",
    "smooth_cache_record",
    "teacache_should_compute",
    "teacache_update",
    "load_coefficients",
    # B8 ToMe
    "ToMeConfig",
    "tome_merge",
    "tome_unmerge",
    "compute_proportional_bias",
    # B11 T-GATE
    "TGateConfig",
    "tgate_forward",
    "create_tgate_state",
    # B13 FreeU
    "FreeUConfig",
    "freeu_filter",
    # B14.2 Adaptive Stepping
    "AdaptiveStepConfig",
    "AdaptiveStepScheduler",
    # B15 Embedding Cache
    "TextEmbeddingCache",
    # B17 WF-VAE Cache
    "WaveletCacheConfig",
    "WaveletVAECache",
    "chunked_decode_with_cache",
]
