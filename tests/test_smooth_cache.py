"""Tests for B4 SmoothCache + Taylor interpolation."""

import mlx.core as mx
import pytest

from mlx_diffusion_kit.cache.smooth_cache import (
    InterpolationMode,
    SmoothCacheConfig,
    create_smooth_cache_state,
    smooth_cache_interpolate,
    smooth_cache_record,
)


def test_linear_between_two_points():
    """LINEAR: midpoint between step 0 (val=0) and step 10 (val=10) → val=5."""
    state = create_smooth_cache_state()
    cfg = SmoothCacheConfig(mode=InterpolationMode.LINEAR)

    smooth_cache_record(0, mx.array([0.0, 0.0]), state)
    smooth_cache_record(10, mx.array([10.0, 10.0]), state)

    result = smooth_cache_interpolate(5, state, cfg)
    assert mx.allclose(result, mx.array([5.0, 5.0]), atol=1e-5)


def test_linear_quarter_point():
    """LINEAR: 25% between step 0 and step 4 → step 1."""
    state = create_smooth_cache_state()
    cfg = SmoothCacheConfig(mode=InterpolationMode.LINEAR)

    smooth_cache_record(0, mx.array([0.0]), state)
    smooth_cache_record(4, mx.array([8.0]), state)

    result = smooth_cache_interpolate(1, state, cfg)
    assert mx.allclose(result, mx.array([2.0]), atol=1e-5)


def test_taylor_1_extrapolation_direction():
    """TAYLOR_1: increasing features → extrapolation should continue increasing."""
    state = create_smooth_cache_state()
    cfg = SmoothCacheConfig(mode=InterpolationMode.TAYLOR_1)

    smooth_cache_record(0, mx.array([1.0, 2.0]), state)
    smooth_cache_record(5, mx.array([6.0, 7.0]), state)

    # Extrapolate to step 10: d1 = (6-1)/5 = 1.0 per step, dt = 5
    # Expected: 6 + 1*5 = 11, 7 + 1*5 = 12
    result = smooth_cache_interpolate(10, state, cfg)
    assert mx.allclose(result, mx.array([11.0, 12.0]), atol=1e-5)


def test_taylor_1_exact_derivative():
    """TAYLOR_1: exact derivative computation."""
    state = create_smooth_cache_state()
    cfg = SmoothCacheConfig(mode=InterpolationMode.TAYLOR_1)

    # Linear function: f(s) = 2*s + 3
    smooth_cache_record(2, mx.array([7.0]), state)   # 2*2+3 = 7
    smooth_cache_record(5, mx.array([13.0]), state)  # 2*5+3 = 13

    # d1 = (13-7)/(5-2) = 2.0, dt = 3, expected: 13 + 2*3 = 19
    result = smooth_cache_interpolate(8, state, cfg)
    assert mx.allclose(result, mx.array([19.0]), atol=1e-5)


def test_taylor_2_parabola():
    """TAYLOR_2: 3 points on f(s) = s^2 → exact quadratic interpolation."""
    state = create_smooth_cache_state()
    cfg = SmoothCacheConfig(mode=InterpolationMode.TAYLOR_2)

    # f(s) = s^2: points at s=0,2,4
    smooth_cache_record(0, mx.array([0.0]), state)   # 0^2 = 0
    smooth_cache_record(2, mx.array([4.0]), state)   # 2^2 = 4
    smooth_cache_record(4, mx.array([16.0]), state)  # 4^2 = 16

    # Predict at s=6: f(6)=36
    # d1 = (16-4)/(4-2) = 6.0
    # d1_prev = (4-0)/(2-0) = 2.0
    # d2 = (6-2)/((4-0)/2) = 4/2 = 2.0
    # dt = 6-4 = 2
    # result = 16 + 6*2 + 0.5*2*4 = 16+12+4 = 32
    # Note: not exact 36 since Taylor-2 from these 3 points gives 32
    # (Taylor expansion from last point, not global fit)
    result = smooth_cache_interpolate(6, state, cfg)
    expected = 16.0 + 6.0 * 2 + 0.5 * 2.0 * 4  # = 32
    assert mx.allclose(result, mx.array([expected]), atol=1e-4)


def test_taylor_2_fallback_to_taylor_1():
    """TAYLOR_2 with only 2 entries should fall back to TAYLOR_1."""
    state = create_smooth_cache_state()
    cfg_t2 = SmoothCacheConfig(mode=InterpolationMode.TAYLOR_2)
    cfg_t1 = SmoothCacheConfig(mode=InterpolationMode.TAYLOR_1)

    smooth_cache_record(0, mx.array([1.0, 2.0]), state)
    smooth_cache_record(5, mx.array([6.0, 7.0]), state)

    result_t2 = smooth_cache_interpolate(10, state, cfg_t2)
    result_t1 = smooth_cache_interpolate(10, state, cfg_t1)
    assert mx.allclose(result_t2, result_t1, atol=1e-6)


def test_history_pruning():
    """History should be capped at max_history."""
    state = create_smooth_cache_state(max_history=3)

    for i in range(10):
        smooth_cache_record(i, mx.array([float(i)]), state)

    assert len(state.history) == 3
    # Should keep the last 3
    assert state.history[0][0] == 7
    assert state.history[1][0] == 8
    assert state.history[2][0] == 9


def test_disabled_returns_last():
    """Disabled SmoothCache returns the last recorded features."""
    state = create_smooth_cache_state()
    cfg = SmoothCacheConfig(enabled=False)

    smooth_cache_record(0, mx.array([1.0]), state)
    smooth_cache_record(5, mx.array([10.0]), state)

    # Midpoint would be 5.5 for linear, but disabled → return last
    result = smooth_cache_interpolate(3, state, cfg)
    assert mx.allclose(result, mx.array([10.0]))


def test_single_entry_returns_that_entry():
    """With only one history entry, return it regardless of mode."""
    state = create_smooth_cache_state()
    cfg = SmoothCacheConfig(mode=InterpolationMode.TAYLOR_1)

    smooth_cache_record(5, mx.array([42.0, 7.0]), state)
    result = smooth_cache_interpolate(10, state, cfg)
    assert mx.allclose(result, mx.array([42.0, 7.0]))


def test_empty_history_raises():
    """Interpolation with no history should raise ValueError."""
    state = create_smooth_cache_state()
    cfg = SmoothCacheConfig()

    with pytest.raises(ValueError, match="no history"):
        smooth_cache_interpolate(5, state, cfg)


def test_outputs_finite():
    """All interpolation modes should produce finite outputs."""
    for mode in InterpolationMode:
        state = create_smooth_cache_state()
        cfg = SmoothCacheConfig(mode=mode)

        smooth_cache_record(0, mx.random.normal((4, 16)), state)
        smooth_cache_record(3, mx.random.normal((4, 16)), state)
        smooth_cache_record(6, mx.random.normal((4, 16)), state)

        result = smooth_cache_interpolate(8, state, cfg)
        assert mx.all(mx.isfinite(result)).item(), f"Non-finite output for {mode}"


def test_shapes_preserved():
    """Output shape should match input feature shape."""
    for shape in [(4, 16), (2, 8, 32), (1, 4, 8, 64)]:
        state = create_smooth_cache_state()
        cfg = SmoothCacheConfig(mode=InterpolationMode.LINEAR)

        smooth_cache_record(0, mx.random.normal(shape), state)
        smooth_cache_record(5, mx.random.normal(shape), state)

        result = smooth_cache_interpolate(3, state, cfg)
        assert result.shape == shape, f"Shape mismatch for {shape}"
