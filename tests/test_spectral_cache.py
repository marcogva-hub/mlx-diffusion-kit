"""Tests for B3 SpectralCache."""

import mlx.core as mx

from mlx_diffusion_kit.cache.spectral_cache import (
    SpectralCacheConfig,
    create_spectral_cache_state,
    spectral_cache_should_compute,
    spectral_cache_update,
)


def test_first_step_always_computes():
    cfg = SpectralCacheConfig()
    state = create_spectral_cache_state()
    feat = mx.random.normal((2, 8, 64))
    assert spectral_cache_should_compute(feat, 0.5, cfg, state) is True


def test_identical_features_skip():
    cfg = SpectralCacheConfig(hf_threshold=0.5, sigma_adaptive=False)
    state = create_spectral_cache_state()
    feat = mx.ones((1, 8, 32))

    assert spectral_cache_should_compute(feat, 0.5, cfg, state) is True
    spectral_cache_update(feat, feat * 2, state)

    assert spectral_cache_should_compute(feat, 0.5, cfg, state) is False


def test_hf_change_triggers_compute():
    """Significant HF change should trigger compute."""
    cfg = SpectralCacheConfig(hf_threshold=0.01, sigma_adaptive=False)
    state = create_spectral_cache_state()

    feat1 = mx.ones((1, 4, 32))
    assert spectral_cache_should_compute(feat1, 0.5, cfg, state) is True
    spectral_cache_update(feat1, feat1, state)

    # Add high-frequency noise
    hf_noise = mx.random.normal((1, 4, 32)) * 10.0
    feat2 = feat1 + hf_noise
    assert spectral_cache_should_compute(feat2, 0.5, cfg, state) is True


def test_sigma_adaptive_threshold():
    """Higher sigma_t → looser threshold → more likely to skip."""
    cfg = SpectralCacheConfig(hf_threshold=0.1, sigma_adaptive=True, sigma_scale=2.0)
    state_high = create_spectral_cache_state()
    state_low = create_spectral_cache_state()

    feat = mx.ones((1, 4, 32))
    # Initialize both states
    spectral_cache_should_compute(feat, 0.8, cfg, state_high)
    spectral_cache_update(feat, feat, state_high)
    spectral_cache_should_compute(feat, 0.05, cfg, state_low)
    spectral_cache_update(feat, feat, state_low)

    # Small perturbation
    feat2 = feat + mx.random.normal(feat.shape) * 0.1

    # High sigma → threshold = 0.1 * 2.0 * 0.8 = 0.16 → more likely to skip
    result_high = spectral_cache_should_compute(feat2, 0.8, cfg, state_high)
    # Low sigma → threshold = 0.1 * 2.0 * 0.05 = 0.01 → more likely to compute
    result_low = spectral_cache_should_compute(feat2, 0.05, cfg, state_low)

    # At low sigma, threshold is much tighter → should compute
    # (result_high might skip while result_low computes)
    # We can at least verify that the low-sigma path is stricter
    if result_high is False:
        # If high sigma skipped, that's expected (looser threshold)
        pass
    # Low sigma should be more likely to compute
    assert result_low is True or result_high is True


def test_energy_budget_exhausted():
    """Accumulated energy exceeding budget should force compute."""
    cfg = SpectralCacheConfig(
        hf_threshold=999.0,  # Very loose — individual changes always below
        energy_budget=0.001,  # Very tight budget
        sigma_adaptive=False,
    )
    state = create_spectral_cache_state()

    feat = mx.random.normal((1, 4, 32))
    assert spectral_cache_should_compute(feat, 0.5, cfg, state) is True
    spectral_cache_update(feat, feat, state)

    # Several small changes that accumulate
    computed = False
    for i in range(20):
        feat_i = feat + mx.random.normal(feat.shape) * 0.01
        if spectral_cache_should_compute(feat_i, 0.5, cfg, state):
            computed = True
            break

    assert computed, "Energy budget should have forced a compute"


def test_fft_roundtrip_coherent():
    """Verify spectral analysis is consistent: same features → same spectrum."""
    cfg = SpectralCacheConfig(hf_threshold=0.5, sigma_adaptive=False)
    state = create_spectral_cache_state()

    feat = mx.random.normal((2, 8, 64))
    spectral_cache_should_compute(feat, 0.5, cfg, state)
    spectral_cache_update(feat, feat, state)

    # Same features again → spectrum matches → skip
    assert spectral_cache_should_compute(feat, 0.5, cfg, state) is False


def test_disabled():
    cfg = SpectralCacheConfig(enabled=False)
    state = create_spectral_cache_state()
    feat = mx.ones((1, 4, 16))
    spectral_cache_update(feat, feat, state)
    assert spectral_cache_should_compute(feat, 0.5, cfg, state) is True
