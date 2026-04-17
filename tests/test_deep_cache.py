"""Tests for B5 DeepCache.

These tests encode the algorithm's input/output contract per the DeepCache
paper (Ma et al., CVPR 2024), not an implementation sketch. The core
invariants:

  - Decision: ``should_recompute`` returns True iff the caller must run the
    deep branch this step. False means ``cached_deep_features`` MUST be used.
  - Storage: after ``deepcache_store(f, s, state)``, ``state.cached_deep_features``
    is exactly ``f`` and ``state.last_recompute_step`` is exactly ``s``.
  - Interval honoring: after storing at step s, the next recompute must be
    delayed until step ``s + cache_interval`` or later.
  - Delta-based correctness: the policy does not rely on contiguous step
    indices (TeaCache-style step skipping is safe).
"""

import mlx.core as mx
import pytest

from mlx_diffusion_kit.cache.deep_cache import (
    DeepCacheConfig,
    DeepCacheState,
    create_deepcache_state,
    deepcache_get,
    deepcache_reset,
    deepcache_should_recompute,
    deepcache_store,
)


def test_first_step_always_recomputes():
    """Before any cache is populated, the first decision must be True."""
    cfg = DeepCacheConfig(cache_interval=3)
    state = create_deepcache_state()
    assert deepcache_should_recompute(0, cfg, state) is True


def test_after_store_cache_is_reused_until_interval_elapses():
    cfg = DeepCacheConfig(cache_interval=3)
    state = create_deepcache_state()

    # Step 0: recompute, store.
    assert deepcache_should_recompute(0, cfg, state) is True
    deepcache_store(mx.ones((2, 4, 8)), 0, state)

    # Steps 1, 2: within interval → reuse.
    assert deepcache_should_recompute(1, cfg, state) is False
    assert deepcache_should_recompute(2, cfg, state) is False

    # Step 3: delta = 3 = interval → must recompute.
    assert deepcache_should_recompute(3, cfg, state) is True


def test_store_updates_state_exactly():
    """Storing must overwrite cache and last_recompute_step with given values."""
    state = create_deepcache_state()
    feats = mx.arange(24).reshape(2, 3, 4).astype(mx.float32)

    deepcache_store(feats, 7, state)
    cached = deepcache_get(state)

    assert cached is not None
    assert mx.array_equal(cached, feats)
    assert state.last_recompute_step == 7
    assert state.recompute_count == 1


def test_disabled_always_recomputes():
    """With enabled=False the policy must force recomputation every call."""
    cfg = DeepCacheConfig(cache_interval=3, enabled=False)
    state = create_deepcache_state()
    deepcache_store(mx.ones((2, 2)), 0, state)

    for s in range(1, 5):
        assert deepcache_should_recompute(s, cfg, state) is True


def test_start_step_respected():
    """Before start_step the policy must always recompute."""
    cfg = DeepCacheConfig(cache_interval=3, start_step=5)
    state = create_deepcache_state()

    # Pretend we had a cache populated — it must still not be used pre-start.
    state.cached_deep_features = mx.ones((2, 2))
    state.last_recompute_step = 0

    for s in range(0, 5):
        assert deepcache_should_recompute(s, cfg, state) is True


def test_delta_not_modulo_on_skipped_step_sequence():
    """Non-contiguous step indices (upstream skipping) must not confuse the policy.

    Scenario: TeaCache decides to run the model at steps 0, 4, 7, 12.
    DeepCache only sees those step indices. interval=3 means: run at 0,
    skip at 4 (delta=4 ≥ 3 so recompute), and importantly the decision must
    not depend on any intermediate step indices it never saw.
    """
    cfg = DeepCacheConfig(cache_interval=3)
    state = create_deepcache_state()

    assert deepcache_should_recompute(0, cfg, state) is True
    deepcache_store(mx.ones((2, 2)), 0, state)

    # Step 4 is the next step that actually runs: delta = 4 ≥ 3 → recompute.
    assert deepcache_should_recompute(4, cfg, state) is True
    deepcache_store(mx.ones((2, 2)) * 2, 4, state)

    # Step 5 (if we ran it): delta = 1 < 3 → reuse.
    assert deepcache_should_recompute(5, cfg, state) is False

    # Step 7: delta = 3 ≥ 3 → recompute.
    assert deepcache_should_recompute(7, cfg, state) is True


def test_interval_one_always_recomputes():
    """cache_interval=1 is equivalent to no caching: every step recomputes."""
    cfg = DeepCacheConfig(cache_interval=1)
    state = create_deepcache_state()
    for s in range(5):
        assert deepcache_should_recompute(s, cfg, state) is True
        deepcache_store(mx.ones((2, 2)), s, state)


def test_reset_clears_cache():
    state = create_deepcache_state()
    deepcache_store(mx.ones((4, 4)), 0, state)
    assert deepcache_get(state) is not None

    deepcache_reset(state)
    assert deepcache_get(state) is None
    assert state.last_recompute_step == -1
    assert state.recompute_count == 0


def test_get_on_fresh_state_is_none():
    state = create_deepcache_state()
    assert deepcache_get(state) is None


def test_recompute_count_tracks_actual_recomputes():
    cfg = DeepCacheConfig(cache_interval=3)
    state = create_deepcache_state()

    for step in range(10):
        if deepcache_should_recompute(step, cfg, state):
            deepcache_store(mx.ones((2, 2)), step, state)

    # Steps that recompute: 0, 3, 6, 9 → 4 recomputes.
    assert state.recompute_count == 4
