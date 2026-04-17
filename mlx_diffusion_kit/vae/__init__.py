"""VAE optimizations (B17-B19)."""

from mlx_diffusion_kit.vae.separable_conv3d import (
    SeparableConv3D,
    build_separable_from_decomposition,
    decompose_conv3d_to_separable,
)
from mlx_diffusion_kit.vae.wavelet_cache import (
    WaveletCacheConfig,
    WaveletVAECache,
    chunked_decode_with_cache,
    estimate_output_shape,
    preallocate_output_buffer,
)

__all__ = [
    # B17
    "WaveletCacheConfig",
    "WaveletVAECache",
    "chunked_decode_with_cache",
    "estimate_output_shape",
    "preallocate_output_buffer",
    # B18
    "SeparableConv3D",
    "build_separable_from_decomposition",
    "decompose_conv3d_to_separable",
]
