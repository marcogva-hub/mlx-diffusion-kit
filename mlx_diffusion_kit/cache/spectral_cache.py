"""B3 — SpectralCache: Frequency-domain step caching with adaptive thresholds.

Monitors high-frequency energy changes between steps via FFT. HF components
shift first when meaningful changes occur, providing earlier detection than
L1 distance. Includes sigma-adaptive thresholds and cumulative energy budgets.

Integrates TADS (Temporal Amplitude-Domain Similarity), FDC (Frequency-Domain
Change), and CEB (Cumulative Energy Budget) concepts.
"""

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx


@dataclass
class SpectralCacheConfig:
    hf_threshold: float = 0.1
    energy_budget: float = 1.0
    sigma_adaptive: bool = True
    sigma_scale: float = 2.0
    freq_split_ratio: float = 0.5
    max_consecutive_cached: int = 5
    enabled: bool = True


@dataclass
class SpectralCacheState:
    prev_spectrum: Optional[mx.array] = None
    cached_output: Optional[mx.array] = None
    accumulated_energy: float = 0.0
    consecutive_cached: int = 0


def create_spectral_cache_state() -> SpectralCacheState:
    """Create a fresh SpectralCache state."""
    return SpectralCacheState()


def _compute_spectrum(features: mx.array) -> mx.array:
    """Compute magnitude spectrum along the last dimension."""
    freq = mx.fft.rfft(features)
    return mx.abs(freq)


def spectral_cache_should_compute(
    features: mx.array,
    sigma_t: float,
    config: SpectralCacheConfig,
    state: SpectralCacheState,
) -> bool:
    """Decide whether to compute based on high-frequency energy change.

    Args:
        features: Current step's features (any shape, FFT on last dim).
        sigma_t: Current noise level (0→1, typically decreasing over steps).
        config: SpectralCache configuration.
        state: Mutable SpectralCache state.

    Returns:
        True if the step should be computed, False to reuse cache.
    """
    if not config.enabled:
        return True

    spectrum = _compute_spectrum(features)

    if state.prev_spectrum is None:
        return True

    # Split into LF and HF
    n_freq = spectrum.shape[-1]
    split = max(1, int(n_freq * config.freq_split_ratio))

    hf_cur = spectrum[..., split:]
    hf_prev = state.prev_spectrum[..., split:]

    # HF change: relative L1
    hf_diff = mx.mean(mx.abs(hf_cur - hf_prev))
    hf_norm = mx.mean(mx.abs(hf_prev)) + 1e-6
    hf_change = (hf_diff / hf_norm).item()

    # Effective threshold (sigma-adaptive)
    threshold = config.hf_threshold
    if config.sigma_adaptive:
        threshold *= config.sigma_scale * max(sigma_t, 0.01)

    # Accumulate energy
    state.accumulated_energy += hf_change

    # Decision
    if (
        hf_change < threshold
        and state.accumulated_energy < config.energy_budget
        and state.consecutive_cached < config.max_consecutive_cached
    ):
        state.consecutive_cached += 1
        return False

    # Compute — reset
    state.accumulated_energy = 0.0
    state.consecutive_cached = 0
    return True


def spectral_cache_update(
    features: mx.array,
    output: mx.array,
    state: SpectralCacheState,
) -> None:
    """Update SpectralCache state after a computed step.

    Args:
        features: Features used for spectral analysis.
        output: Full model output for this step.
        state: Mutable SpectralCache state.
    """
    state.prev_spectrum = _compute_spectrum(features)
    state.cached_output = output
