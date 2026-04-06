"""Tests for B22 DDT Encoder Sharing."""

import mlx.core as mx

from mlx_diffusion_kit.cache.encoder_sharing import (
    EncoderSharingConfig,
    create_encoder_sharing_state,
    encoder_sharing_get_cached,
    encoder_sharing_should_recompute,
    encoder_sharing_update,
)


def test_first_step_always_recomputes():
    """Step 0 with empty cache must recompute."""
    cfg = EncoderSharingConfig(recompute_interval=3)
    state = create_encoder_sharing_state()
    assert encoder_sharing_should_recompute(0, cfg, state) is True


def test_recompute_interval():
    """With interval=3, recompute at 0, 3, 6; cache at 1, 2, 4, 5."""
    cfg = EncoderSharingConfig(recompute_interval=3)
    state = create_encoder_sharing_state()

    # Step 0: recompute (no cache)
    assert encoder_sharing_should_recompute(0, cfg, state) is True
    encoder_sharing_update(0, mx.ones((2, 8, 64)), state)

    # Steps 1, 2: use cache
    assert encoder_sharing_should_recompute(1, cfg, state) is False
    assert encoder_sharing_should_recompute(2, cfg, state) is False

    # Step 3: recompute (interval boundary)
    assert encoder_sharing_should_recompute(3, cfg, state) is True
    encoder_sharing_update(3, mx.ones((2, 8, 64)) * 2.0, state)

    # Steps 4, 5: use cache
    assert encoder_sharing_should_recompute(4, cfg, state) is False
    assert encoder_sharing_should_recompute(5, cfg, state) is False

    # Step 6: recompute
    assert encoder_sharing_should_recompute(6, cfg, state) is True


def test_update_then_get_cached():
    """After update, get_cached should return the same tensor."""
    state = create_encoder_sharing_state()
    val = mx.random.normal((2, 8, 64))
    encoder_sharing_update(0, val, state)

    cached = encoder_sharing_get_cached(state)
    assert cached is not None
    assert mx.array_equal(cached, val)


def test_update_overwrites_previous():
    """A new update should replace the old cached value."""
    state = create_encoder_sharing_state()
    encoder_sharing_update(0, mx.ones((2, 4)), state)
    encoder_sharing_update(3, mx.ones((2, 4)) * 5.0, state)

    cached = encoder_sharing_get_cached(state)
    assert mx.allclose(cached, mx.ones((2, 4)) * 5.0)
    assert state.last_computed_step == 3


def test_disabled_always_recomputes():
    """Disabled config → always recompute."""
    cfg = EncoderSharingConfig(recompute_interval=3, enabled=False)
    state = create_encoder_sharing_state()

    encoder_sharing_update(0, mx.ones((2, 4)), state)

    for step in range(10):
        assert encoder_sharing_should_recompute(step, cfg, state) is True


def test_get_cached_empty():
    """Get cached on empty state returns None."""
    state = create_encoder_sharing_state()
    assert encoder_sharing_get_cached(state) is None


def test_interval_1_always_recomputes():
    """Interval=1 means recompute every step (delta >= 1 always true for next step)."""
    cfg = EncoderSharingConfig(recompute_interval=1)
    state = create_encoder_sharing_state()

    for step in range(5):
        assert encoder_sharing_should_recompute(step, cfg, state) is True
        encoder_sharing_update(step, mx.ones((2, 4)), state)


def test_non_sequential_steps_with_skips():
    """Delta-based recompute handles TeaCache-skipped steps correctly."""
    cfg = EncoderSharingConfig(recompute_interval=3)
    state = create_encoder_sharing_state()

    # Step 0: recompute (no cache)
    assert encoder_sharing_should_recompute(0, cfg, state) is True
    encoder_sharing_update(0, mx.ones((2, 4)), state)

    # Steps 1, 2 skipped by TeaCache, step 4 is next computed
    # delta = 4 - 0 = 4 >= 3 → recompute
    assert encoder_sharing_should_recompute(4, cfg, state) is True
    encoder_sharing_update(4, mx.ones((2, 4)) * 2.0, state)

    # Step 5: delta = 5 - 4 = 1 < 3 → cache
    assert encoder_sharing_should_recompute(5, cfg, state) is False

    # Step 7: delta = 7 - 4 = 3 >= 3 → recompute
    assert encoder_sharing_should_recompute(7, cfg, state) is True
