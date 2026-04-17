"""Tests for B12 DiTFastAttn decision policy.

Contract invariants (from :func:`ditfastattn_decide` precedence):
  - Step 0 always returns FULL (no cache exists yet).
  - Disabled config always returns FULL.
  - RESIDUAL beats SHARE beats WINDOW beats FULL, when the required
    cache entry exists.
  - If a layer is in ``sharing_layers`` but no cached attn map exists,
    fall through to WINDOW / FULL (never return SHARE without a cache).
  - Same safety for RESIDUAL.
  - WINDOW only activates at ``window_start_step`` and later.
  - record/get round-trips for both caches.
  - Reset clears both cache dicts.
"""

import mlx.core as mx

from mlx_diffusion_kit.attention.ditfastattn import (
    AttnStrategy,
    DiTFastAttnConfig,
    DiTFastAttnState,
    create_ditfastattn_state,
    ditfastattn_decide,
    ditfastattn_get_cached_attn,
    ditfastattn_get_cached_residual,
    ditfastattn_record_attn_map,
    ditfastattn_record_residual,
    ditfastattn_reset,
)


def test_step_zero_is_always_full():
    cfg = DiTFastAttnConfig(
        window_start_step=0,
        sharing_layers=[0, 1],
        residual_cache_layers=[0, 1],
    )
    state = create_ditfastattn_state()
    ditfastattn_record_attn_map(0, mx.ones((4, 4)), state)
    ditfastattn_record_residual(0, mx.ones((4, 4)), state)

    for layer in (0, 1):
        assert ditfastattn_decide(layer, 0, cfg, state) == AttnStrategy.FULL


def test_disabled_is_always_full():
    cfg = DiTFastAttnConfig(enabled=False, residual_cache_layers=[0])
    state = create_ditfastattn_state()
    ditfastattn_record_residual(0, mx.ones((4, 4)), state)

    assert ditfastattn_decide(0, 5, cfg, state) == AttnStrategy.FULL


def test_residual_wins_when_configured_and_cached():
    cfg = DiTFastAttnConfig(
        window_start_step=0,
        sharing_layers=[0],
        residual_cache_layers=[0],
    )
    state = create_ditfastattn_state()
    ditfastattn_record_attn_map(0, mx.ones((4, 4)), state)
    ditfastattn_record_residual(0, mx.ones((4, 4)), state)

    assert ditfastattn_decide(0, 3, cfg, state) == AttnStrategy.RESIDUAL


def test_share_wins_over_window_when_configured_and_cached():
    cfg = DiTFastAttnConfig(
        window_start_step=0,
        sharing_layers=[0],
        residual_cache_layers=[],
    )
    state = create_ditfastattn_state()
    ditfastattn_record_attn_map(0, mx.ones((4, 4)), state)

    assert ditfastattn_decide(0, 5, cfg, state) == AttnStrategy.SHARE


def test_sharing_without_cache_falls_through():
    """A layer in sharing_layers but with no cache must NOT return SHARE."""
    cfg = DiTFastAttnConfig(window_start_step=10, sharing_layers=[0])
    state = create_ditfastattn_state()  # no cache recorded

    # Below window_start_step: should be FULL, not SHARE.
    assert ditfastattn_decide(0, 5, cfg, state) == AttnStrategy.FULL
    # At/after window_start_step: WINDOW, not SHARE.
    assert ditfastattn_decide(0, 10, cfg, state) == AttnStrategy.WINDOW


def test_residual_without_cache_falls_through():
    cfg = DiTFastAttnConfig(window_start_step=10, residual_cache_layers=[0])
    state = create_ditfastattn_state()

    assert ditfastattn_decide(0, 5, cfg, state) == AttnStrategy.FULL
    assert ditfastattn_decide(0, 10, cfg, state) == AttnStrategy.WINDOW


def test_window_activates_at_start_step():
    cfg = DiTFastAttnConfig(window_start_step=5)
    state = create_ditfastattn_state()

    # Step 4: still FULL (both before start and after step 0 for a plain layer).
    assert ditfastattn_decide(3, 4, cfg, state) == AttnStrategy.FULL
    # Step 5: boundary → WINDOW.
    assert ditfastattn_decide(3, 5, cfg, state) == AttnStrategy.WINDOW
    # Step 20: still WINDOW.
    assert ditfastattn_decide(3, 20, cfg, state) == AttnStrategy.WINDOW


def test_non_listed_layer_uses_window_not_share_or_residual():
    cfg = DiTFastAttnConfig(
        window_start_step=5,
        sharing_layers=[0],
        residual_cache_layers=[1],
    )
    state = create_ditfastattn_state()
    ditfastattn_record_attn_map(0, mx.ones((4, 4)), state)
    ditfastattn_record_residual(1, mx.ones((4, 4)), state)

    # Layer 2 is not listed anywhere → follows the window timing.
    assert ditfastattn_decide(2, 4, cfg, state) == AttnStrategy.FULL
    assert ditfastattn_decide(2, 5, cfg, state) == AttnStrategy.WINDOW


def test_attn_map_roundtrip():
    state = create_ditfastattn_state()
    val = mx.random.normal((8, 8))
    ditfastattn_record_attn_map(3, val, state)
    cached = ditfastattn_get_cached_attn(3, state)
    assert cached is not None
    assert mx.array_equal(cached, val)
    assert ditfastattn_get_cached_attn(99, state) is None


def test_residual_roundtrip():
    state = create_ditfastattn_state()
    val = mx.random.normal((4, 16))
    ditfastattn_record_residual(2, val, state)
    cached = ditfastattn_get_cached_residual(2, state)
    assert cached is not None
    assert mx.array_equal(cached, val)
    assert ditfastattn_get_cached_residual(99, state) is None


def test_reset_clears_both_caches():
    state = create_ditfastattn_state()
    ditfastattn_record_attn_map(0, mx.ones((4, 4)), state)
    ditfastattn_record_residual(1, mx.ones((4, 4)), state)
    assert state.cached_attn_maps
    assert state.cached_residuals

    ditfastattn_reset(state)
    assert state.cached_attn_maps == {}
    assert state.cached_residuals == {}


def test_four_distinct_strategies_achievable():
    """Sanity: with proper config + caches, all four enum values are reachable."""
    state = create_ditfastattn_state()
    ditfastattn_record_attn_map(1, mx.ones((4, 4)), state)
    ditfastattn_record_residual(2, mx.ones((4, 4)), state)

    cfg = DiTFastAttnConfig(
        window_start_step=5,
        sharing_layers=[1],
        residual_cache_layers=[2],
    )

    # FULL (step 0)
    assert ditfastattn_decide(0, 0, cfg, state) == AttnStrategy.FULL
    # WINDOW (step 5, plain layer)
    assert ditfastattn_decide(0, 5, cfg, state) == AttnStrategy.WINDOW
    # SHARE (layer 1 with cached map)
    assert ditfastattn_decide(1, 3, cfg, state) == AttnStrategy.SHARE
    # RESIDUAL (layer 2 with cached residual)
    assert ditfastattn_decide(2, 3, cfg, state) == AttnStrategy.RESIDUAL
