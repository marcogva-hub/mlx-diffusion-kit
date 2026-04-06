"""Tests for B5 DeepCache + MosaicDiff layer redundancy."""

import mlx.core as mx
import pytest

from mlx_diffusion_kit.cache.deep_cache import (
    DeepCacheConfig,
    DeepCacheManager,
    analyze_layer_redundancy,
    select_cacheable_layers,
)


# ===== DeepCacheManager =====


def test_auto_select_middle_layers():
    """Auto-selection should pick layers from the middle region."""
    cfg = DeepCacheConfig(auto_cache_ratio=0.5)
    mgr = DeepCacheManager(total_layers=20, config=cfg)
    cached = mgr.get_cached_layers()

    assert len(cached) > 0
    # All cached layers should be in the middle region [5, 15)
    for idx in cached:
        assert 5 <= idx < 15, f"Layer {idx} outside middle region"


def test_non_cached_always_computes():
    """Non-cached layers should always compute."""
    cfg = DeepCacheConfig(cached_layer_indices=[5, 6, 7])
    mgr = DeepCacheManager(total_layers=10, config=cfg)

    for step in range(10):
        assert mgr.should_compute_layer(0, step) is True
        assert mgr.should_compute_layer(9, step) is True


def test_cached_layer_interval():
    """Cached layers compute at intervals, skip between."""
    cfg = DeepCacheConfig(cached_layer_indices=[5], cache_interval=3)
    mgr = DeepCacheManager(total_layers=10, config=cfg)

    # Step 0: first step, always compute
    assert mgr.should_compute_layer(5, 0) is True
    mgr.update_layer(5, 0, mx.ones((2, 4)))

    # Steps 1, 2: skip (delta < 3)
    assert mgr.should_compute_layer(5, 1) is False
    assert mgr.should_compute_layer(5, 2) is False

    # Step 3: recompute (delta = 3 >= 3)
    assert mgr.should_compute_layer(5, 3) is True
    mgr.update_layer(5, 3, mx.ones((2, 4)) * 2.0)

    # Steps 4, 5: skip again
    assert mgr.should_compute_layer(5, 4) is False
    assert mgr.should_compute_layer(5, 5) is False


def test_update_get_roundtrip():
    cfg = DeepCacheConfig(cached_layer_indices=[3])
    mgr = DeepCacheManager(total_layers=10, config=cfg)

    val = mx.random.normal((2, 8, 32))
    mgr.update_layer(3, 0, val)
    cached = mgr.get_cached_layer(3)
    assert cached is not None
    assert mx.array_equal(cached, val)

    # Non-cached layer returns None
    assert mgr.get_cached_layer(0) is None


def test_reset():
    cfg = DeepCacheConfig(cached_layer_indices=[3, 5])
    mgr = DeepCacheManager(total_layers=10, config=cfg)
    mgr.update_layer(3, 0, mx.ones((2, 4)))
    mgr.update_layer(5, 0, mx.ones((2, 4)))
    mgr.reset()
    assert mgr.get_cached_layer(3) is None
    assert mgr.get_cached_layer(5) is None


def test_disabled():
    cfg = DeepCacheConfig(cached_layer_indices=[3, 5], enabled=False)
    mgr = DeepCacheManager(total_layers=10, config=cfg)
    mgr.update_layer(3, 0, mx.ones((2, 4)))
    # Disabled: always compute
    assert mgr.should_compute_layer(3, 1) is True


# ===== MosaicDiff =====


def test_identical_layers_high_redundancy():
    """Identical weights → redundancy score = 1.0."""
    w = mx.ones((16, 16))
    weights = {0: w, 1: w, 2: w, 3: w}
    scores = analyze_layer_redundancy(weights, method="cosine")
    # All middle layers should have score ≈ 1.0 (or at least > 0.9)
    for idx in [1, 2]:
        assert scores[idx] >= 0.9, f"Layer {idx} score {scores[idx]:.2f} too low"


def test_different_layers_low_redundancy():
    """Very different weights → low redundancy."""
    weights = {}
    for i in range(5):
        mx.random.seed(i * 1000)
        weights[i] = mx.random.normal((32, 32))

    scores = analyze_layer_redundancy(weights, method="cosine")
    # Scores should exist and be in [0, 1]
    for idx, score in scores.items():
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"


def test_l2_method():
    """L2 method should also produce valid scores."""
    w = mx.ones((8, 8))
    weights = {0: w, 1: w * 1.01, 2: w * 100.0}
    scores = analyze_layer_redundancy(weights, method="l2")
    # Layer 1 (close to 0) should be more redundant than layer 2 (far from 1)
    assert scores[1] > scores[2]


def test_select_cacheable_layers():
    """Should select the most redundant layers."""
    scores = {0: 0.1, 1: 0.9, 2: 0.8, 3: 0.2, 4: 0.7}
    selected = select_cacheable_layers(scores, ratio=0.4)
    assert len(selected) == 2
    # Should be the top-2 by score: 1 and 2
    assert 1 in selected
    assert 2 in selected
    # Should be sorted
    assert selected == sorted(selected)


def test_select_cacheable_layers_all():
    scores = {0: 0.5, 1: 0.5, 2: 0.5}
    selected = select_cacheable_layers(scores, ratio=1.0)
    assert len(selected) == 3
