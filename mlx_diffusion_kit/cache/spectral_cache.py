"""B3 — SpectralCache: Frequency-domain feature caching.

Algorithm:
    Features are transformed to the frequency domain (rFFT along the last
    axis), split into a low-frequency band and a high-frequency band, and
    each band is cached at its own interval:

      * Low-frequency coefficients (the first ``low_freq_ratio`` of bins)
        are recomputed every ``cache_interval_low`` steps. They encode
        bulk structure, which evolves slowly across denoising.
      * High-frequency coefficients are recomputed every
        ``cache_interval_high`` steps (default 1 = every step). They
        encode fine detail, which in VSR must stay fresh.

    The cached bands are combined (LF from cache, HF fresh or cached) and
    fed through the inverse transform to reconstruct the features.

    Crucial property: when both intervals are 1, :func:`spectral_cache_apply`
    is a bit-for-bit identity up to FFT round-trip error (~1e-6 at f32).

SeaCache extension (``spectral_velocity_aware=True``):
    Tracks a bounded history of recent transforms. When the per-band L1
    velocity between the two most recent computed transforms exceeds
    :attr:`velocity_override_thresh`, the LF cache is invalidated and
    forcibly recomputed on the next step — regardless of the interval
    counter. This catches fast scene changes that a fixed interval would
    otherwise blur across.

Notes on transforms:
    * ``"rfft"`` (default) uses :func:`mx.fft.rfft` / :func:`mx.fft.irfft`.
      MLX's implementation supports complex intermediate arrays and
      round-trips to float precision.
    * ``"dct"`` is not implemented in this pass. MLX does not expose a
      native DCT; implementing via FFT mirror-extension is possible but
      out of scope. Requesting ``"dct"`` raises ``NotImplementedError``
      with a clear message.

Applies to: 6 multi-step VSR models.

Reference:
    Baseline SpectralCache: adapted from community spectral-caching ideas
    in diffusion acceleration literature (not a single paper).
    SeaCache variant: inspired by spectral-evolution work described as
    "SeaCache" in CVPR 2026 plans; implementation-only inspiration,
    not a formal cite.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

import mlx.core as mx


@dataclass
class SpectralCacheConfig:
    """Configuration for SpectralCache.

    Attributes:
        low_freq_ratio: Fraction of rFFT bins considered "low frequency".
            0.25 is a conservative default that keeps most fine detail
            in the HF band.
        cache_interval_low: Recompute LF coefficients every N steps.
        cache_interval_high: Recompute HF every N steps. Default 1 =
            every step, which is the VSR-safe choice.
        transform: ``"rfft"`` (supported) or ``"dct"`` (raises
            NotImplementedError for now).
        spectral_velocity_aware: Enable SeaCache spectral-velocity
            invalidation of the LF cache.
        velocity_override_thresh: Per-band L1 relative velocity above
            which the LF cache is invalidated. Only used when
            ``spectral_velocity_aware=True``.
        enabled: Master switch.
    """

    low_freq_ratio: float = 0.25
    cache_interval_low: int = 4
    cache_interval_high: int = 1
    transform: Literal["rfft", "dct"] = "rfft"
    spectral_velocity_aware: bool = False
    velocity_override_thresh: float = 0.5
    enabled: bool = True


@dataclass
class SpectralCacheState:
    """Mutable state for SpectralCache.

    Attributes:
        cached_low_freq: LF bins from the most recent LF recompute.
            Complex for rFFT. None before first compute.
        cached_high_freq: HF bins from the most recent HF recompute.
            Used when ``cache_interval_high > 1``.
        prev_full_spectra: Bounded history of recent full spectra
            (tuples of ``(lf, hf)``) for SeaCache velocity calculation.
            Capped at length 2.
        last_low_recompute_step: Step index of the last LF recompute.
        last_high_recompute_step: Step index of the last HF recompute.
    """

    cached_low_freq: Optional[mx.array] = None
    cached_high_freq: Optional[mx.array] = None
    prev_full_spectra: list[tuple[mx.array, mx.array]] = field(default_factory=list)
    last_low_recompute_step: int = -1
    last_high_recompute_step: int = -1


def create_spectral_cache_state() -> SpectralCacheState:
    """Create a fresh SpectralCacheState for a new inference run."""
    return SpectralCacheState()


def _forward_transform(features: mx.array, transform: str) -> mx.array:
    """Forward transform along the last axis."""
    if transform == "rfft":
        return mx.fft.rfft(features, axis=-1)
    if transform == "dct":
        raise NotImplementedError(
            "DCT transform is not supported yet. MLX does not expose a native "
            "DCT; implementing via FFT mirror-extension is planned but out of "
            "scope for this release. Use transform='rfft' for now."
        )
    raise ValueError(f"Unknown transform: {transform}")


def _inverse_transform(freq: mx.array, n: int, transform: str) -> mx.array:
    """Inverse transform back to the original domain."""
    if transform == "rfft":
        return mx.fft.irfft(freq, n=n, axis=-1)
    if transform == "dct":
        raise NotImplementedError("DCT inverse not supported; see forward-pass note.")
    raise ValueError(f"Unknown transform: {transform}")


def _split_bands(
    spectrum: mx.array, low_freq_ratio: float
) -> tuple[mx.array, mx.array, int]:
    """Split ``spectrum`` into ``(lf, hf, split_index)`` along the last axis."""
    n_bins = spectrum.shape[-1]
    split = max(1, int(n_bins * low_freq_ratio))
    split = min(split, n_bins)  # clamp
    lf = spectrum[..., :split]
    hf = spectrum[..., split:]
    return lf, hf, split


def _combine_bands(lf: mx.array, hf: mx.array) -> mx.array:
    """Re-join ``lf`` and ``hf`` along the last axis."""
    return mx.concatenate([lf, hf], axis=-1)


def _per_band_velocity(
    spectrum_prev: mx.array,
    spectrum_curr: mx.array,
) -> float:
    """Aggregate L1-relative change between two spectra."""
    mag_prev = mx.abs(spectrum_prev)
    mag_curr = mx.abs(spectrum_curr)
    diff = mx.mean(mx.abs(mag_curr - mag_prev))
    norm = mx.mean(mag_prev) + 1e-6
    return (diff / norm).item()


def spectral_cache_apply(
    features: mx.array,
    step_idx: int,
    config: SpectralCacheConfig,
    state: SpectralCacheState,
) -> mx.array:
    """Reconstruct features through the frequency-domain cache.

    Args:
        features: Current step's features. FFT is taken along the last
            axis; any leading batch/spatial dims are preserved.
        step_idx: Current diffusion step index (for interval arithmetic).
        config: SpectralCache configuration.
        state: Mutable SpectralCache state.

    Returns:
        Reconstructed features of the same shape as the input. When
        caching is disabled or both intervals are 1, this is an identity
        up to FFT round-trip error.
    """
    if not config.enabled:
        return features

    n_last = features.shape[-1]
    spectrum = _forward_transform(features, config.transform)
    lf_curr, hf_curr, _ = _split_bands(spectrum, config.low_freq_ratio)

    # --- SeaCache: velocity-triggered LF invalidation ---
    force_low_recompute = False
    if config.spectral_velocity_aware and len(state.prev_full_spectra) >= 1:
        prev_lf, _ = state.prev_full_spectra[-1]
        vel = _per_band_velocity(prev_lf, lf_curr)
        if vel > config.velocity_override_thresh:
            force_low_recompute = True

    # --- LF recompute policy ---
    if (
        state.cached_low_freq is None
        or force_low_recompute
        or step_idx - state.last_low_recompute_step >= config.cache_interval_low
    ):
        state.cached_low_freq = lf_curr
        state.last_low_recompute_step = step_idx
    lf_used = state.cached_low_freq

    # --- HF recompute policy ---
    if (
        state.cached_high_freq is None
        or step_idx - state.last_high_recompute_step >= config.cache_interval_high
    ):
        state.cached_high_freq = hf_curr
        state.last_high_recompute_step = step_idx
    hf_used = state.cached_high_freq

    # --- SeaCache history bookkeeping ---
    if config.spectral_velocity_aware:
        state.prev_full_spectra.append((lf_curr, hf_curr))
        if len(state.prev_full_spectra) > 2:
            state.prev_full_spectra = state.prev_full_spectra[-2:]

    # --- Combine and inverse transform ---
    combined = _combine_bands(lf_used, hf_used)
    return _inverse_transform(combined, n_last, config.transform)


def spectral_cache_update(
    features: mx.array,
    step_idx: int,
    config: SpectralCacheConfig,
    state: SpectralCacheState,
) -> None:
    """Force-refresh the LF and HF caches from ``features``.

    Call after a step that the caller decided to run at full fidelity,
    to ensure the next :func:`spectral_cache_apply` starts from the
    current state. Mostly useful with step-level TeaCache in the loop.
    """
    if not config.enabled:
        return
    spectrum = _forward_transform(features, config.transform)
    lf, hf, _ = _split_bands(spectrum, config.low_freq_ratio)
    state.cached_low_freq = lf
    state.cached_high_freq = hf
    state.last_low_recompute_step = step_idx
    state.last_high_recompute_step = step_idx


def spectral_cache_reset(state: SpectralCacheState) -> None:
    """Clear all cached bands and history."""
    state.cached_low_freq = None
    state.cached_high_freq = None
    state.prev_full_spectra.clear()
    state.last_low_recompute_step = -1
    state.last_high_recompute_step = -1
