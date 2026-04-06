"""mlx-diffusion-kit: Inference optimizations for diffusion models on MLX."""

from mlx_diffusion_kit.__version__ import __version__
from mlx_diffusion_kit.cache.teacache import (
    TeaCacheConfig,
    teacache_should_compute,
    teacache_update,
    load_coefficients,
)
from mlx_diffusion_kit.encoder.embedding_cache import TextEmbeddingCache
from mlx_diffusion_kit.gating.tgate import TGateConfig, tgate_forward, create_tgate_state
from mlx_diffusion_kit.quality.freeu import FreeUConfig, freeu_filter

__all__ = [
    "__version__",
    # B1 TeaCache
    "TeaCacheConfig",
    "teacache_should_compute",
    "teacache_update",
    "load_coefficients",
    # B11 T-GATE
    "TGateConfig",
    "tgate_forward",
    "create_tgate_state",
    # B13 FreeU
    "FreeUConfig",
    "freeu_filter",
    # B15 Embedding Cache
    "TextEmbeddingCache",
]
