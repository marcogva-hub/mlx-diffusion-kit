"""Tests for B2 FBCache (First-Block Cache)."""

import mlx.core as mx

from mlx_diffusion_kit.cache.fbcache import (
    FBCacheConfig,
    create_fbcache_state,
    fbcache_should_compute,
    fbcache_update,
)


def test_first_step_always_computes():
    cfg = FBCacheConfig()
    state = create_fbcache_state()
    fb = mx.random.normal((2, 8, 64))
    assert fbcache_should_compute(fb, cfg, state) is True


def test_identical_inputs_skip():
    cfg = FBCacheConfig(threshold=0.1)
    state = create_fbcache_state()
    fb = mx.ones((1, 8, 32))

    assert fbcache_should_compute(fb, cfg, state) is True
    fbcache_update(fb, fb * 2, state)

    assert fbcache_should_compute(fb, cfg, state) is False


def test_different_inputs_compute():
    cfg = FBCacheConfig(threshold=0.01)
    state = create_fbcache_state()

    fb1 = mx.ones((1, 8, 32))
    assert fbcache_should_compute(fb1, cfg, state) is True
    fbcache_update(fb1, fb1, state)

    fb2 = mx.ones((1, 8, 32)) * 100.0
    assert fbcache_should_compute(fb2, cfg, state) is True


def test_max_consecutive_cached():
    cfg = FBCacheConfig(threshold=999.0, max_consecutive_cached=2)
    state = create_fbcache_state()
    fb = mx.ones((1, 4, 16))

    assert fbcache_should_compute(fb, cfg, state) is True
    fbcache_update(fb, fb, state)

    assert fbcache_should_compute(fb, cfg, state) is False  # 1
    assert fbcache_should_compute(fb, cfg, state) is False  # 2
    assert fbcache_should_compute(fb, cfg, state) is True   # forced


def test_disabled():
    cfg = FBCacheConfig(enabled=False)
    state = create_fbcache_state()
    fb = mx.ones((1, 4, 16))

    fbcache_update(fb, fb, state)
    # Even after update, disabled always computes
    for _ in range(5):
        assert fbcache_should_compute(fb, cfg, state) is True
