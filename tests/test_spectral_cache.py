"""Tests for B3 SpectralCache + SeaCache variant.

Contract invariants:
  - ``spectral_cache_apply`` preserves input shape.
  - When both intervals are 1, the output equals the input (identity
    up to FFT round-trip precision).
  - LF cache invalidates after ``cache_interval_low`` steps (delta-based).
  - HF cache invalidates after ``cache_interval_high`` steps.
  - When LF is cached and HF is fresh, the output's LF band equals the
    cached one and the HF band equals the current input's HF.
  - DCT raises NotImplementedError.
  - SeaCache mode: a high-velocity LF change forces a recompute even
    inside the cache_interval_low window.
  - Reset clears all state.
"""

import mlx.core as mx
import pytest

from mlx_diffusion_kit.cache.spectral_cache import (
    SpectralCacheConfig,
    SpectralCacheState,
    create_spectral_cache_state,
    spectral_cache_apply,
    spectral_cache_reset,
    spectral_cache_update,
)


def test_shape_preserved():
    cfg = SpectralCacheConfig()
    state = create_spectral_cache_state()
    x = mx.random.normal((2, 4, 8, 16))
    y = spectral_cache_apply(x, 0, cfg, state)
    assert y.shape == x.shape


def test_identity_when_both_intervals_one():
    """Both LF and HF recomputed every step → output ≈ input."""
    cfg = SpectralCacheConfig(cache_interval_low=1, cache_interval_high=1)
    state = create_spectral_cache_state()
    x = mx.random.normal((1, 2, 32))

    y = spectral_cache_apply(x, 0, cfg, state)
    # rFFT round-trip error is at the f32 epsilon scale.
    assert mx.allclose(y, x, atol=1e-4, rtol=1e-4)


def test_disabled_passthrough():
    cfg = SpectralCacheConfig(enabled=False)
    state = create_spectral_cache_state()
    x = mx.random.normal((2, 16))
    y = spectral_cache_apply(x, 5, cfg, state)
    assert mx.array_equal(y, x)


def test_lf_cache_honored():
    """Within cache_interval_low, the LF coefficients from the first step
    must dominate the output even when input changes substantially."""
    cfg = SpectralCacheConfig(
        low_freq_ratio=0.5, cache_interval_low=10, cache_interval_high=1
    )
    state = create_spectral_cache_state()

    x0 = mx.random.normal((1, 32))
    y0 = spectral_cache_apply(x0, 0, cfg, state)
    # Different input at step 1 — but LF is cached from step 0.
    x1 = mx.random.normal((1, 32)) * 10.0
    y1 = spectral_cache_apply(x1, 1, cfg, state)

    # Reconstruct what the algorithm claims to have done:
    #   y1 = irfft([cached_lf_from_x0; fresh_hf_from_x1])
    spec_x0 = mx.fft.rfft(x0, axis=-1)
    spec_x1 = mx.fft.rfft(x1, axis=-1)
    split = int(spec_x0.shape[-1] * 0.5)
    split = max(1, split)
    expected_spec = mx.concatenate([spec_x0[..., :split], spec_x1[..., split:]], axis=-1)
    expected = mx.fft.irfft(expected_spec, n=x1.shape[-1], axis=-1)

    assert mx.allclose(y1, expected, atol=1e-4, rtol=1e-4)


def test_lf_invalidates_after_interval():
    """After interval steps, LF must be recomputed from the fresh input."""
    cfg = SpectralCacheConfig(
        low_freq_ratio=0.5, cache_interval_low=3, cache_interval_high=1
    )
    state = create_spectral_cache_state()

    x0 = mx.random.normal((1, 32))
    spectral_cache_apply(x0, 0, cfg, state)
    step0_lf = state.cached_low_freq
    assert state.last_low_recompute_step == 0

    # Steps 1, 2: same LF cached.
    spectral_cache_apply(x0, 1, cfg, state)
    spectral_cache_apply(x0, 2, cfg, state)
    assert state.last_low_recompute_step == 0

    # Step 3: interval elapsed → LF recomputed.
    x3 = mx.random.normal((1, 32)) * 5.0
    spectral_cache_apply(x3, 3, cfg, state)
    assert state.last_low_recompute_step == 3
    # And the cached LF should now reflect x3, not x0.
    spec_x3 = mx.fft.rfft(x3, axis=-1)
    split = int(spec_x3.shape[-1] * 0.5)
    split = max(1, split)
    assert mx.allclose(state.cached_low_freq, spec_x3[..., :split], atol=1e-4)


def test_hf_fresh_every_step_with_interval_one():
    """Default HF interval = 1 → HF always reflects current input."""
    cfg = SpectralCacheConfig(
        low_freq_ratio=0.5, cache_interval_low=10, cache_interval_high=1
    )
    state = create_spectral_cache_state()

    x0 = mx.random.normal((1, 32))
    spectral_cache_apply(x0, 0, cfg, state)

    x1 = mx.random.normal((1, 32))
    spectral_cache_apply(x1, 1, cfg, state)
    spec_x1 = mx.fft.rfft(x1, axis=-1)
    split = int(spec_x1.shape[-1] * 0.5)
    split = max(1, split)
    assert mx.allclose(state.cached_high_freq, spec_x1[..., split:], atol=1e-4)


def test_dct_raises_not_implemented():
    cfg = SpectralCacheConfig(transform="dct")
    state = create_spectral_cache_state()
    with pytest.raises(NotImplementedError, match="DCT"):
        spectral_cache_apply(mx.random.normal((1, 16)), 0, cfg, state)


def test_seacache_forces_recompute_on_high_velocity():
    """With spectral_velocity_aware=True, a big LF change must
    invalidate the LF cache even inside the interval window."""
    cfg = SpectralCacheConfig(
        low_freq_ratio=0.5,
        cache_interval_low=100,  # effectively never expire by interval
        cache_interval_high=1,
        spectral_velocity_aware=True,
        velocity_override_thresh=0.1,  # low threshold = easy trigger
    )
    state = create_spectral_cache_state()

    # Step 0: populate cache, history gets one entry.
    x0 = mx.ones((1, 32))
    spectral_cache_apply(x0, 0, cfg, state)
    assert state.last_low_recompute_step == 0

    # Step 1: drastically different input → LF velocity explodes.
    x1 = mx.ones((1, 32)) * 100.0
    spectral_cache_apply(x1, 1, cfg, state)

    # LF should have been recomputed despite being well within the
    # cache_interval_low=100 window.
    assert state.last_low_recompute_step == 1


def test_seacache_does_not_fire_on_stable_input():
    """Without a big LF change, SeaCache should behave like baseline."""
    cfg = SpectralCacheConfig(
        low_freq_ratio=0.5,
        cache_interval_low=100,
        cache_interval_high=1,
        spectral_velocity_aware=True,
        velocity_override_thresh=0.5,
    )
    state = create_spectral_cache_state()

    x0 = mx.ones((1, 32))
    spectral_cache_apply(x0, 0, cfg, state)
    # Tiny perturbation — below the threshold.
    x1 = x0 + mx.random.normal((1, 32)) * 0.001
    spectral_cache_apply(x1, 1, cfg, state)
    assert state.last_low_recompute_step == 0  # still from step 0


def test_update_forces_refresh():
    cfg = SpectralCacheConfig(cache_interval_low=10, cache_interval_high=10)
    state = create_spectral_cache_state()

    x0 = mx.ones((1, 16))
    spectral_cache_apply(x0, 0, cfg, state)

    x5 = mx.ones((1, 16)) * 3.0
    spectral_cache_update(x5, 5, cfg, state)
    assert state.last_low_recompute_step == 5
    assert state.last_high_recompute_step == 5


def test_reset_clears_all_state():
    cfg = SpectralCacheConfig(spectral_velocity_aware=True)
    state = create_spectral_cache_state()
    spectral_cache_apply(mx.ones((1, 16)), 0, cfg, state)
    spectral_cache_apply(mx.ones((1, 16)), 1, cfg, state)
    assert state.cached_low_freq is not None
    assert len(state.prev_full_spectra) > 0

    spectral_cache_reset(state)
    assert state.cached_low_freq is None
    assert state.cached_high_freq is None
    assert state.prev_full_spectra == []
    assert state.last_low_recompute_step == -1
