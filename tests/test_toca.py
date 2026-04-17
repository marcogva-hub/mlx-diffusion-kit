"""Tests for B7 ToCa — per-layer velocity-based token caching.

Contract invariants:
  - Before any cache is populated: all tokens are active (no cache to reuse).
  - ``velocity`` mode requires ≥ 2 prior updates at the same layer.
  - After two updates, the top-``recompute_ratio`` fraction of tokens by
    velocity are returned as active.
  - Active and cached index sets are disjoint, together covering [0, N).
  - ``toca_compose`` reconstructs the full token tensor so that each
    position holds exactly what was supplied in active or cached features.
  - Per-layer state: updating layer A does not affect layer B's decision.
  - ``magnitude`` mode works from step 1 onward.
"""

import mlx.core as mx

from mlx_diffusion_kit.tokens.toca import (
    ToCaConfig,
    ToCaState,
    create_toca_state,
    toca_compose,
    toca_get_cached,
    toca_reset,
    toca_select_tokens,
    toca_update,
)


def test_first_call_all_active():
    cfg = ToCaConfig(recompute_ratio=0.5)
    state = create_toca_state()
    tokens = mx.random.normal((2, 16, 8))
    active, cached = toca_select_tokens(tokens, 0, 0, cfg, state)

    assert active.shape == (2, 16)
    assert cached.shape == (2, 0)


def test_velocity_mode_needs_two_updates():
    cfg = ToCaConfig(recompute_ratio=0.5, score_mode="velocity")
    state = create_toca_state()
    tokens = mx.random.normal((1, 12, 4))

    # Update #1: only cached_tokens is populated → velocity still undefined.
    toca_update(0, tokens, state)

    active, cached = toca_select_tokens(tokens, 0, 1, cfg, state)
    # Fallback to all-active because prev_tokens is still None.
    assert active.shape == (1, 12)
    assert cached.shape == (1, 0)


def test_velocity_partition_after_two_updates():
    cfg = ToCaConfig(recompute_ratio=0.5, score_mode="velocity")
    state = create_toca_state()

    # Craft a trajectory where the first half of tokens have high velocity
    # and the second half have near-zero velocity.
    B, N, D = 1, 8, 4
    prev_tokens = mx.ones((B, N, D))
    # Half change drastically, half barely change.
    cached_tokens = mx.ones((B, N, D))
    cached_tokens = cached_tokens.at[:, :4, :].add(10.0)        # big delta
    cached_tokens = cached_tokens.at[:, 4:, :].add(1e-6)        # tiny delta

    # Simulate two updates: first prev, then cached.
    toca_update(0, prev_tokens, state)
    toca_update(0, cached_tokens, state)

    # Now a current input — the velocity scorer uses cached - prev,
    # so the current tokens' content doesn't matter for scoring.
    active, cached = toca_select_tokens(mx.ones((B, N, D)), 0, 2, cfg, state)

    assert active.shape == (B, 4)
    assert cached.shape == (B, 4)
    # High-velocity tokens (indices 0-3) should be in the active set.
    active_set = set(active[0].tolist())
    assert active_set == {0, 1, 2, 3}
    cached_set = set(cached[0].tolist())
    assert cached_set == {4, 5, 6, 7}


def test_active_cached_are_disjoint_and_cover_N():
    cfg = ToCaConfig(recompute_ratio=0.3, score_mode="velocity")
    state = create_toca_state()
    B, N, D = 2, 20, 6

    toca_update(0, mx.random.normal((B, N, D)), state)
    toca_update(0, mx.random.normal((B, N, D)), state)

    active, cached = toca_select_tokens(
        mx.random.normal((B, N, D)), 0, 2, cfg, state
    )

    for b in range(B):
        all_idx = set(active[b].tolist()) | set(cached[b].tolist())
        assert all_idx == set(range(N))
        assert len(set(active[b].tolist()) & set(cached[b].tolist())) == 0


def test_compose_reconstructs_full_tensor():
    """Given disjoint active/cached index sets, toca_compose must place
    each feature vector at the corresponding original index."""
    B, N, D = 1, 6, 3
    active_indices = mx.array([[0, 2, 4]], dtype=mx.int32)
    cached_indices = mx.array([[1, 3, 5]], dtype=mx.int32)

    active_feats = mx.array(
        [[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0], [5.0, 5.0, 5.0]]]
    )
    cached_feats = mx.array(
        [[[2.0, 2.0, 2.0], [4.0, 4.0, 4.0], [6.0, 6.0, 6.0]]]
    )

    out = toca_compose(active_feats, cached_feats, active_indices, cached_indices, N)

    assert out.shape == (B, N, D)
    # Expected: position i holds value (i+1) replicated across D.
    expected = mx.stack([mx.full((D,), float(i + 1)) for i in range(N)])
    assert mx.allclose(out[0], expected)


def test_compose_with_empty_cached():
    """When all tokens are active, cached_features is [B, 0, D] — must not break."""
    B, N, D = 1, 4, 2
    active_indices = mx.array([[0, 1, 2, 3]], dtype=mx.int32)
    cached_indices = mx.zeros((1, 0), dtype=mx.int32)
    active_feats = mx.ones((B, N, D))
    cached_feats = mx.zeros((B, 0, D))

    out = toca_compose(active_feats, cached_feats, active_indices, cached_indices, N)
    assert out.shape == (B, N, D)
    assert mx.allclose(out, mx.ones((B, N, D)))


def test_per_layer_state_independence():
    """Updating layer 0's cache must not affect layer 1's decision."""
    cfg = ToCaConfig(recompute_ratio=0.5, score_mode="velocity")
    state = create_toca_state()
    tokens = mx.random.normal((1, 8, 4))

    toca_update(0, tokens, state)
    toca_update(0, tokens, state)  # layer 0 has full history now.

    # Layer 1 has no history at all → fallback to all-active.
    active, cached = toca_select_tokens(tokens, 1, 2, cfg, state)
    assert active.shape == (1, 8)
    assert cached.shape == (1, 0)


def test_magnitude_mode_works_from_step_one():
    cfg = ToCaConfig(recompute_ratio=0.5, score_mode="magnitude")
    state = create_toca_state()

    # First update just populates cached_tokens, no velocity history.
    tokens0 = mx.random.normal((1, 8, 4))
    toca_update(0, tokens0, state)

    # Magnitude mode scores the CURRENT tokens, not the history — so
    # as soon as cached_tokens exists to prove we're past step 0,
    # magnitude can partition.
    tokens1 = mx.ones((1, 8, 4))
    # Make the first half large, second half small.
    tokens1 = tokens1.at[:, :4, :].add(100.0)
    active, cached = toca_select_tokens(tokens1, 0, 1, cfg, state)

    assert active.shape == (1, 4)
    assert cached.shape == (1, 4)
    assert set(active[0].tolist()) == {0, 1, 2, 3}


def test_disabled_all_active():
    cfg = ToCaConfig(enabled=False)
    state = create_toca_state()
    tokens = mx.random.normal((1, 8, 4))
    # Give it full history so the only reason to be all-active is `enabled=False`.
    toca_update(0, tokens, state)
    toca_update(0, tokens, state)

    active, cached = toca_select_tokens(tokens, 0, 2, cfg, state)
    assert active.shape == (1, 8)
    assert cached.shape == (1, 0)


def test_update_shifts_history():
    state = create_toca_state()
    t0 = mx.ones((1, 4, 2))
    t1 = mx.ones((1, 4, 2)) * 2.0
    t2 = mx.ones((1, 4, 2)) * 3.0

    toca_update(0, t0, state)
    ls = state.layer(0)
    assert ls.cached_tokens is not None and mx.allclose(ls.cached_tokens, t0)
    assert ls.prev_tokens is None

    toca_update(0, t1, state)
    assert mx.allclose(ls.cached_tokens, t1)
    assert mx.allclose(ls.prev_tokens, t0)

    toca_update(0, t2, state)
    assert mx.allclose(ls.cached_tokens, t2)
    assert mx.allclose(ls.prev_tokens, t1)

    assert ls.step_count == 3


def test_get_cached_returns_none_until_populated():
    state = create_toca_state()
    assert toca_get_cached(0, state) is None
    toca_update(0, mx.ones((1, 4, 2)), state)
    assert toca_get_cached(0, state) is not None


def test_reset_clears_all_layers():
    state = create_toca_state()
    toca_update(0, mx.ones((1, 4, 2)), state)
    toca_update(1, mx.ones((1, 4, 2)), state)
    assert len(state.layers) == 2

    toca_reset(state)
    assert len(state.layers) == 0
