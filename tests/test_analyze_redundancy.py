"""Tests for scripts/analyze_layer_redundancy.py with synthetic weights."""

import mlx.core as mx

from mlx_diffusion_kit.cache.deep_cache import analyze_layer_redundancy, select_cacheable_layers


def test_identical_synthetic_weights():
    """Identical weight layers should have high redundancy."""
    w = mx.ones((32, 32))
    weights = {i: w for i in range(5)}
    scores = analyze_layer_redundancy(weights, method="cosine")
    assert all(s >= 0.9 for s in scores.values())


def test_mixed_synthetic_weights():
    """Mix of similar and different weights → different scores."""
    weights = {
        0: mx.ones((16, 16)),
        1: mx.ones((16, 16)) * 1.001,  # Nearly identical to 0
        2: mx.random.normal((16, 16)),  # Very different
        3: mx.random.normal((16, 16)) * 10,  # Very different
    }
    scores = analyze_layer_redundancy(weights, method="cosine")
    # Layer 1 (similar to 0) should be more redundant than layer 2 (random)
    assert scores[1] > scores[2] or scores[1] > scores[3]


def test_select_from_scores():
    scores = {0: 0.9, 1: 0.1, 2: 0.8, 3: 0.2, 4: 0.7}
    selected = select_cacheable_layers(scores, ratio=0.4)
    assert len(selected) == 2
    # Most redundant: 0 (0.9) and 2 (0.8)
    assert 0 in selected
    assert 2 in selected
