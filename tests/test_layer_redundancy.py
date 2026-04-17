"""Tests for the MosaicDiff-style layer redundancy analyzer.

Contract invariants:
  - Redundancy score is in [0, 1].
  - Fewer than 2 layers → all scores 0.0.
  - Identical-to-neighbor layers rank above dissimilar-to-neighbor layers.
  - ``select_cacheable_layers`` returns sorted indices and respects the ratio.
"""

import mlx.core as mx

from mlx_diffusion_kit.cache.layer_redundancy import (
    analyze_layer_redundancy,
    select_cacheable_layers,
)


def test_single_layer_zero_score():
    scores = analyze_layer_redundancy({0: mx.ones((4, 4))})
    assert scores == {0: 0.0}


def test_all_identical_weights_full_redundancy():
    w = mx.ones((16, 16))
    weights = {i: w for i in range(5)}
    scores = analyze_layer_redundancy(weights)
    assert all(s == 1.0 for s in scores.values())


def test_mixed_weights_ranked_correctly():
    """A layer sandwiched between two identical neighbors should rank above
    a layer sandwiched between two very different neighbors."""
    mx.random.seed(42)
    base = mx.random.normal((8, 8))
    weights = {
        0: base,
        1: base + 1e-6,           # nearly identical to 0
        2: base + 1e-6,           # nearly identical to 1
        3: mx.random.normal((8, 8)),  # very different from neighbors
        4: mx.random.normal((8, 8)),
    }
    scores = analyze_layer_redundancy(weights)

    # Middle layer of the near-identical cluster should outscore the
    # middle layer of the random cluster.
    assert scores[1] > scores[3]


def test_scores_are_in_unit_interval():
    mx.random.seed(0)
    weights = {i: mx.random.normal((4, 4)) for i in range(6)}
    scores = analyze_layer_redundancy(weights)
    for s in scores.values():
        assert 0.0 <= s <= 1.0


def test_l2_method_runs_and_bounded():
    weights = {i: mx.random.normal((4, 4)) for i in range(4)}
    scores = analyze_layer_redundancy(weights, method="l2")
    for s in scores.values():
        assert 0.0 <= s <= 1.0


def test_select_respects_ratio():
    scores = {0: 0.1, 1: 0.9, 2: 0.4, 3: 0.8, 4: 0.2}
    selected = select_cacheable_layers(scores, ratio=0.4)
    # 40% of 5 = 2 → top two by score are indices 1 (0.9) and 3 (0.8).
    assert selected == [1, 3]


def test_select_returns_at_least_one():
    scores = {0: 0.5}
    assert select_cacheable_layers(scores, ratio=0.0) == [0]


def test_select_is_sorted():
    scores = {0: 0.9, 5: 0.8, 3: 0.7, 8: 0.95}
    selected = select_cacheable_layers(scores, ratio=0.75)
    assert selected == sorted(selected)
