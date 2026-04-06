"""VAE optimizations (B17-B19)."""

from mlx_diffusion_kit.vae.wavelet_cache import (
    WaveletCacheConfig,
    WaveletVAECache,
    chunked_decode_with_cache,
)

__all__ = ["WaveletCacheConfig", "WaveletVAECache", "chunked_decode_with_cache"]
