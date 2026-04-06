"""B13 — FreeU: Training-free UNet skip connection re-weighting.

Applicable to 5 UNet models: DAM-VSR, DiffVSR, DLoRAL, UltraVSR, VEnhancer.

Reference: Si et al., "FreeU: Free Lunch in Diffusion U-Net" (CVPR 2024).
"""

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class FreeUConfig:
    """FreeU hyperparameters.

    b1, b2: backbone scale factors for low-freq channels.
    s1, s2: skip connection high-frequency attenuation factors.
    """

    b1: float = 1.2
    b2: float = 1.4
    s1: float = 0.9
    s2: float = 0.2
    enabled: bool = True


def _spectral_attenuate(x: mx.array, scale: float) -> mx.array:
    """Attenuate high frequencies along the last spatial dimension via FFT."""
    # x shape: (..., spatial_dim)
    freq = mx.fft.rfft(x)
    n_freq = freq.shape[-1]
    # Low-freq half kept, high-freq half attenuated
    mid = n_freq // 2
    # Build scaling mask: 1.0 for low-freq, `scale` for high-freq
    mask_low = mx.ones((mid,), dtype=freq.dtype)
    mask_high = mx.full((n_freq - mid,), scale, dtype=freq.dtype)
    mask = mx.concatenate([mask_low, mask_high])
    return mx.fft.irfft(freq * mask, n=x.shape[-1])


def freeu_filter(
    h_skip: mx.array,
    h_backbone: mx.array,
    config: FreeUConfig,
) -> tuple[mx.array, mx.array]:
    """Apply FreeU re-weighting to UNet skip and backbone features.

    Args:
        h_skip: Skip connection features, shape (..., C, *spatial).
        h_backbone: Backbone features, same shape as h_skip.
        config: FreeU hyperparameters.

    Returns:
        (filtered_skip, scaled_backbone) with same shapes as inputs.
    """
    if not config.enabled:
        return h_skip, h_backbone

    # --- Backbone: scale low-frequency channels (first half) ---
    c = h_backbone.shape[-2] if h_backbone.ndim >= 2 else h_backbone.shape[-1]
    mid_c = c // 2

    # Work on channel dimension (second-to-last for spatial data)
    # For simplicity, scale the first half of channels by b1
    if h_backbone.ndim >= 2:
        slices_low = [slice(None)] * (h_backbone.ndim - 2) + [slice(0, mid_c), slice(None)]
        slices_high = [slice(None)] * (h_backbone.ndim - 2) + [slice(mid_c, None), slice(None)]
        backbone_low = h_backbone[tuple(slices_low)] * config.b1
        backbone_high = h_backbone[tuple(slices_high)] * config.b2
        h_backbone_out = mx.concatenate([backbone_low, backbone_high], axis=-2)
    else:
        h_backbone_out = h_backbone * config.b1

    # --- Skip: spectral attenuation of high frequencies ---
    h_skip_out = _spectral_attenuate(h_skip, config.s1)

    return h_skip_out, h_backbone_out
