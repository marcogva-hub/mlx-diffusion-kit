"""Tests for B12 DiTFastAttn."""

import mlx.core as mx

from mlx_diffusion_kit.attention.ditfastattn import (
    DiTFastAttnConfig,
    DiTFastAttnManager,
    HeadStrategy,
)


def test_before_profiling_all_full():
    """Before profiling, all heads should use FULL strategy."""
    cfg = DiTFastAttnConfig(auto_profile_steps=3)
    mgr = DiTFastAttnManager(num_layers=4, num_heads=8, config=cfg)

    for l in range(4):
        for h in range(8):
            assert mgr.get_head_strategy(l, h, 0) == HeadStrategy.FULL


def test_profiling_assigns_strategies():
    """After profiling, heads get differentiated strategies."""
    cfg = DiTFastAttnConfig(auto_profile_steps=3, sensitivity_threshold=0.1)
    mgr = DiTFastAttnManager(num_layers=2, num_heads=2, config=cfg)

    # Profile 3 steps: head (0,0) has low variance, head (0,1) has high variance
    for step in range(3):
        # Head (0,0): constant attention → low variance
        mgr.profile_step(0, 0, mx.ones((4, 4)) * 0.25, step)
        # Head (0,1): varying attention → high variance
        mgr.profile_step(0, 1, mx.random.normal((4, 4)) * (step + 1), step)
        # Head (1,0): constant
        mgr.profile_step(1, 0, mx.ones((4, 4)) * 0.5, step)
        # Head (1,1): varying
        mgr.profile_step(1, 1, mx.random.normal((4, 4)) * (step + 1) * 2, step)

    assert mgr._state.profiled
    # Low-variance heads should be CACHED
    assert mgr.get_head_strategy(0, 0, 10) == HeadStrategy.CACHED
    # High-variance heads should be FULL
    assert mgr.get_head_strategy(0, 1, 10) == HeadStrategy.FULL


def test_high_variance_head_full():
    """Explicitly high sensitivity → FULL."""
    scores = {(0, 0): 0.5, (0, 1): 0.01}
    cfg = DiTFastAttnConfig(head_sensitivity_scores=scores, sensitivity_threshold=0.1)
    mgr = DiTFastAttnManager(2, 2, cfg)
    assert mgr.get_head_strategy(0, 0, 10) == HeadStrategy.FULL
    assert mgr.get_head_strategy(0, 1, 10) == HeadStrategy.CACHED


def test_cached_only_after_start_step():
    """CACHED heads should use FULL before cache_start_step."""
    scores = {(0, 0): 0.01}
    cfg = DiTFastAttnConfig(
        head_sensitivity_scores=scores,
        sensitivity_threshold=0.1,
        cache_start_step=5,
    )
    mgr = DiTFastAttnManager(1, 1, cfg)

    assert mgr.get_head_strategy(0, 0, 3) == HeadStrategy.FULL  # Before start
    assert mgr.get_head_strategy(0, 0, 5) == HeadStrategy.CACHED  # At/after start


def test_window_mask():
    """Window mask should be diagonal band of correct width."""
    cfg = DiTFastAttnConfig(window_size=4)
    mgr = DiTFastAttnManager(1, 1, cfg)

    mask = mgr.get_window_mask(8)
    assert mask.shape == (8, 8)
    # Center should be True
    assert mask[0, 0].item()
    assert mask[4, 4].item()
    # Diagonal + 2 off-diagonals (half_w=2) should be True
    assert mask[0, 2].item()  # |0-2| = 2 <= 2
    assert not mask[0, 3].item()  # |0-3| = 3 > 2


def test_cache_get_roundtrip():
    cfg = DiTFastAttnConfig()
    mgr = DiTFastAttnManager(2, 4, cfg)

    val = mx.random.normal((8, 8))
    mgr.cache_attention(0, 1, val)
    cached = mgr.get_cached_attention(0, 1)
    assert cached is not None
    assert mx.array_equal(cached, val)

    assert mgr.get_cached_attention(0, 2) is None


def test_reset():
    scores = {(0, 0): 0.01}
    cfg = DiTFastAttnConfig(head_sensitivity_scores=scores, sensitivity_threshold=0.1)
    mgr = DiTFastAttnManager(1, 1, cfg)
    mgr.cache_attention(0, 0, mx.ones((4, 4)))

    mgr.reset()
    assert mgr.get_cached_attention(0, 0) is None
    # Strategies should be re-assigned from pre-computed scores
    assert mgr._state.profiled


def test_disabled():
    cfg = DiTFastAttnConfig(enabled=False)
    scores = {(0, 0): 0.01}
    cfg.head_sensitivity_scores = scores
    mgr = DiTFastAttnManager(1, 1, cfg)
    assert mgr.get_head_strategy(0, 0, 10) == HeadStrategy.FULL
