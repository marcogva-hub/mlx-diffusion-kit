"""Tests for B2 First-Block Cache.

Contract invariants:
  - First call with no prior → must compute.
  - After update, if the new first-block output is identical to the prior
    one, the decision must be skip (reuse residual).
  - After update, if the new first-block output is very different, the
    decision must be compute.
  - Reconstruction identity: for any ``fb`` and any ``residual``, after
    ``fbcache_update(fb_prev, residual, state)`` and a skip decision,
    ``fbcache_reconstruct(fb, state) == fb + residual``.
  - ``max_consecutive_cached`` is a hard ceiling.
  - ``start_step`` / ``end_step`` window is respected.
  - ``fbcache_reconstruct`` without cached residual raises.
"""

import mlx.core as mx
import pytest

from mlx_diffusion_kit.cache.fb_cache import (
    FBCacheConfig,
    FBCacheState,
    create_fbcache_state,
    fbcache_reconstruct,
    fbcache_reset,
    fbcache_should_compute_remaining,
    fbcache_update,
)


def test_first_step_always_computes():
    cfg = FBCacheConfig()
    state = create_fbcache_state()
    fb = mx.random.normal((2, 8, 16))
    assert fbcache_should_compute_remaining(fb, 0, cfg, state) is True


def test_identical_fb_triggers_skip():
    cfg = FBCacheConfig(rel_l1_thresh=0.1)
    state = create_fbcache_state()
    fb = mx.ones((1, 4, 8))

    # Compute at step 0.
    assert fbcache_should_compute_remaining(fb, 0, cfg, state) is True
    fbcache_update(fb, mx.ones((1, 4, 8)) * 0.5, state)

    # Step 1 with identical fb → change = 0 → skip.
    assert fbcache_should_compute_remaining(fb, 1, cfg, state) is False


def test_divergent_fb_triggers_compute():
    cfg = FBCacheConfig(rel_l1_thresh=0.01)
    state = create_fbcache_state()

    fb1 = mx.ones((1, 4, 8))
    assert fbcache_should_compute_remaining(fb1, 0, cfg, state) is True
    fbcache_update(fb1, mx.zeros((1, 4, 8)), state)

    fb2 = mx.ones((1, 4, 8)) * 100.0
    assert fbcache_should_compute_remaining(fb2, 1, cfg, state) is True


def test_reconstruction_identity():
    """Key algorithmic invariant: reconstruct(fb) = fb + cached_residual."""
    cfg = FBCacheConfig(rel_l1_thresh=0.1)
    state = create_fbcache_state()

    fb = mx.ones((1, 4, 8)) * 3.0
    residual = mx.ones((1, 4, 8)) * 2.0  # full_output was 5.0

    # Populate the cache.
    fbcache_should_compute_remaining(fb, 0, cfg, state)
    fbcache_update(fb, residual, state)

    # Next step: identical fb → skip. Reconstruction must equal fb + residual.
    assert fbcache_should_compute_remaining(fb, 1, cfg, state) is False
    reconstructed = fbcache_reconstruct(fb, state)
    expected = fb + residual
    assert mx.allclose(reconstructed, expected)


def test_reconstruction_with_different_fb_still_adds_cached_residual():
    """When reusing the residual, the returned value incorporates the
    *current* fb_output, not the cached one. This is the whole point of
    FBCache: the residual is stable, the first-block output is not."""
    cfg = FBCacheConfig(rel_l1_thresh=100.0)  # force skip regardless
    state = create_fbcache_state()

    fb_prev = mx.ones((1, 4)) * 1.0
    residual = mx.ones((1, 4)) * 0.3
    fbcache_should_compute_remaining(fb_prev, 0, cfg, state)
    fbcache_update(fb_prev, residual, state)

    fb_curr = mx.ones((1, 4)) * 2.0  # slightly different, but skip due to huge thresh
    assert fbcache_should_compute_remaining(fb_curr, 1, cfg, state) is False

    reconstructed = fbcache_reconstruct(fb_curr, state)
    assert mx.allclose(reconstructed, fb_curr + residual)
    # And importantly NOT equal to fb_prev + residual.
    assert not mx.allclose(reconstructed, fb_prev + residual)


def test_max_consecutive_cached_forces_recompute():
    cfg = FBCacheConfig(rel_l1_thresh=999.0, max_consecutive_cached=2)
    state = create_fbcache_state()
    fb = mx.ones((1, 4))

    # Step 0: compute.
    assert fbcache_should_compute_remaining(fb, 0, cfg, state) is True
    fbcache_update(fb, mx.zeros((1, 4)), state)

    # Steps 1, 2: skip (within ceiling).
    assert fbcache_should_compute_remaining(fb, 1, cfg, state) is False
    assert fbcache_should_compute_remaining(fb, 2, cfg, state) is False

    # Step 3: must compute (hit ceiling).
    assert fbcache_should_compute_remaining(fb, 3, cfg, state) is True


def test_start_step_window():
    cfg = FBCacheConfig(rel_l1_thresh=999.0, start_step=3)
    state = create_fbcache_state()
    fb = mx.ones((1, 4))

    # Pre-warm cache (doesn't matter what's in it; start_step blocks reuse).
    state.prev_fb_output = fb
    state.cached_residual = mx.zeros((1, 4))

    # Steps 0, 1, 2 are before start_step → always compute.
    for s in range(3):
        assert fbcache_should_compute_remaining(fb, s, cfg, state) is True

    # Step 3 and beyond: caching active, identical fb → skip.
    assert fbcache_should_compute_remaining(fb, 3, cfg, state) is False


def test_end_step_window():
    cfg = FBCacheConfig(rel_l1_thresh=999.0, end_step=5)
    state = create_fbcache_state()
    fb = mx.ones((1, 4))
    state.prev_fb_output = fb
    state.cached_residual = mx.zeros((1, 4))

    # Step 4: inside window → skip.
    assert fbcache_should_compute_remaining(fb, 4, cfg, state) is False
    # Step 5: at end_step (exclusive) → compute.
    assert fbcache_should_compute_remaining(fb, 5, cfg, state) is True


def test_disabled_always_computes():
    cfg = FBCacheConfig(enabled=False, rel_l1_thresh=999.0)
    state = create_fbcache_state()
    state.prev_fb_output = mx.ones((1, 4))
    state.cached_residual = mx.zeros((1, 4))

    for s in range(5):
        assert fbcache_should_compute_remaining(mx.ones((1, 4)), s, cfg, state) is True


def test_reconstruct_without_cache_raises():
    state = create_fbcache_state()
    with pytest.raises(RuntimeError, match="no cached residual"):
        fbcache_reconstruct(mx.ones((1, 4)), state)


def test_reset_clears_state():
    cfg = FBCacheConfig()
    state = create_fbcache_state()
    fbcache_should_compute_remaining(mx.ones((1, 4)), 0, cfg, state)
    fbcache_update(mx.ones((1, 4)), mx.zeros((1, 4)), state)
    assert state.prev_fb_output is not None
    assert state.cached_residual is not None

    fbcache_reset(state)
    assert state.prev_fb_output is None
    assert state.cached_residual is None
    assert state.step_counter == 0
    assert state.consecutive_cached == 0
