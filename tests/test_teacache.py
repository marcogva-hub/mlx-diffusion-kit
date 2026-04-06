"""Tests for B1 TeaCache step caching."""

import mlx.core as mx
import pytest

from mlx_diffusion_kit.cache.teacache import (
    TeaCacheConfig,
    create_teacache_state,
    load_coefficients,
    teacache_should_compute,
    teacache_update,
)


def test_first_step_always_computes():
    cfg = TeaCacheConfig()
    state = create_teacache_state()
    x = mx.random.normal((4, 8))
    assert teacache_should_compute(x, step_idx=0, config=cfg, state=state) is True


def test_identical_inputs_skip():
    cfg = TeaCacheConfig(rel_l1_thresh=0.5)
    state = create_teacache_state()
    x = mx.ones((4, 8))

    # First step — compute
    assert teacache_should_compute(x, step_idx=0, config=cfg, state=state) is True
    teacache_update(x, x * 2, state)

    # Second step with identical input — should skip (distance=0)
    assert teacache_should_compute(x, step_idx=1, config=cfg, state=state) is False


def test_very_different_inputs_compute():
    cfg = TeaCacheConfig(rel_l1_thresh=0.01)
    state = create_teacache_state()

    x1 = mx.ones((4, 8))
    assert teacache_should_compute(x1, step_idx=0, config=cfg, state=state) is True
    teacache_update(x1, x1 * 2, state)

    x2 = mx.ones((4, 8)) * 100.0  # Very different
    assert teacache_should_compute(x2, step_idx=1, config=cfg, state=state) is True


def test_max_consecutive_cached():
    cfg = TeaCacheConfig(rel_l1_thresh=999.0, max_consecutive_cached=2)
    state = create_teacache_state()
    x = mx.ones((4, 8))

    # Step 0 — compute (first step)
    assert teacache_should_compute(x, step_idx=0, config=cfg, state=state) is True
    teacache_update(x, x, state)

    # Steps 1,2 — skip (consecutive_cached = 1, 2)
    assert teacache_should_compute(x, step_idx=1, config=cfg, state=state) is False
    assert teacache_should_compute(x, step_idx=2, config=cfg, state=state) is False

    # Step 3 — forced compute (consecutive_cached hit max)
    assert teacache_should_compute(x, step_idx=3, config=cfg, state=state) is True


def test_poly_coeffs_scaling():
    """Polynomial rescaling changes the effective distance."""
    # Coefficients that amplify the distance: 0 + 10*x
    cfg = TeaCacheConfig(
        rel_l1_thresh=0.5,
        poly_coeffs=[0.0, 10.0],
    )
    state = create_teacache_state()
    x1 = mx.ones((4, 8))

    assert teacache_should_compute(x1, step_idx=0, config=cfg, state=state) is True
    teacache_update(x1, x1, state)

    # Small change — without poly would be small, with 10x amplification → compute
    x2 = mx.ones((4, 8)) * 1.1
    assert teacache_should_compute(x2, step_idx=1, config=cfg, state=state) is True


def test_load_coefficients_cogvideox():
    cfg = load_coefficients("cogvideox")
    assert cfg.rel_l1_thresh == 0.3
    assert cfg.poly_coeffs is not None
    assert len(cfg.poly_coeffs) == 5


def test_load_coefficients_case_insensitive():
    cfg = load_coefficients("CogVideoX")
    assert cfg.poly_coeffs is not None


def test_load_coefficients_missing():
    with pytest.raises(FileNotFoundError):
        load_coefficients("nonexistent_model")


def test_start_step_window():
    cfg = TeaCacheConfig(start_step=3)
    state = create_teacache_state()
    x = mx.ones((4, 8))

    # Steps before start_step — always compute
    for step in range(3):
        assert teacache_should_compute(x, step_idx=step, config=cfg, state=state) is True


def test_end_step_window():
    cfg = TeaCacheConfig(end_step=5)
    state = create_teacache_state()
    x = mx.ones((4, 8))

    # Step at end_step — always compute
    assert teacache_should_compute(x, step_idx=5, config=cfg, state=state) is True
    assert teacache_should_compute(x, step_idx=10, config=cfg, state=state) is True


def test_disabled_always_computes():
    cfg = TeaCacheConfig(enabled=False)
    state = create_teacache_state()
    x = mx.ones((4, 8))

    for step in range(10):
        assert teacache_should_compute(x, step_idx=step, config=cfg, state=state) is True
