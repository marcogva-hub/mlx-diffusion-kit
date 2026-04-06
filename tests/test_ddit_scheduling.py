"""Tests for B10 DDiT Dynamic Patch Scheduling."""

import math

from mlx_diffusion_kit.tokens.ddit_scheduling import (
    DDiTScheduleConfig,
    DDiTScheduler,
)


def test_linear_schedule_monotonic_decrease():
    """Linear schedule: stride should decrease monotonically."""
    cfg = DDiTScheduleConfig(max_patch_stride=4, min_patch_stride=1, schedule="linear", warmup_fraction=0.5)
    sched = DDiTScheduler(total_steps=20, config=cfg)

    strides = [sched.get_patch_stride(i) for i in range(20)]
    # During warmup (steps 0-9): non-increasing
    for i in range(1, 10):
        assert strides[i] <= strides[i - 1], f"Step {i}: {strides[i]} > {strides[i-1]}"
    # After warmup: min stride
    for i in range(10, 20):
        assert strides[i] == 1


def test_cosine_schedule_shape():
    """Cosine schedule: slower at start, faster in middle."""
    cfg = DDiTScheduleConfig(max_patch_stride=4, min_patch_stride=1, schedule="cosine", warmup_fraction=1.0)
    sched = DDiTScheduler(total_steps=100, config=cfg)

    strides = [sched.get_patch_stride(i) for i in range(100)]
    # First stride should be max
    assert strides[0] == 4
    # Last stride should be min
    assert strides[-1] == 1
    # Non-increasing overall
    for i in range(1, 100):
        assert strides[i] <= strides[i - 1] or strides[i] == strides[i - 1], \
            f"Step {i}: {strides[i]} > {strides[i-1]}"


def test_step_schedule_abrupt():
    """Step schedule: max stride during warmup, min after."""
    cfg = DDiTScheduleConfig(max_patch_stride=4, min_patch_stride=1, schedule="step", warmup_fraction=0.5)
    sched = DDiTScheduler(total_steps=20, config=cfg)

    # During warmup (steps 0-9): max stride
    for i in range(10):
        assert sched.get_patch_stride(i) == 4, f"Step {i} should be 4"
    # After warmup: min stride
    for i in range(10, 20):
        assert sched.get_patch_stride(i) == 1, f"Step {i} should be 1"


def test_strides_are_powers_of_2():
    """All returned strides must be powers of 2."""
    for schedule in ["linear", "cosine", "step"]:
        cfg = DDiTScheduleConfig(max_patch_stride=8, min_patch_stride=1, schedule=schedule, warmup_fraction=0.6)
        sched = DDiTScheduler(total_steps=50, config=cfg)

        for i in range(50):
            stride = sched.get_patch_stride(i)
            assert stride > 0
            assert (stride & (stride - 1)) == 0, f"Stride {stride} at step {i} is not power of 2"


def test_token_reduction_factor_2d():
    """Reduction factor should be stride^2 for 2D."""
    cfg = DDiTScheduleConfig(max_patch_stride=4, min_patch_stride=1, schedule="step", warmup_fraction=0.5)
    sched = DDiTScheduler(total_steps=10, config=cfg)

    # During warmup: stride=4 → factor=16
    assert sched.get_token_reduction_factor(0, spatial_dims=2) == 16.0
    # After warmup: stride=1 → factor=1
    assert sched.get_token_reduction_factor(9, spatial_dims=2) == 1.0


def test_token_reduction_factor_3d():
    """Reduction factor should be stride^3 for 3D video."""
    cfg = DDiTScheduleConfig(max_patch_stride=4, min_patch_stride=1, schedule="step", warmup_fraction=0.5)
    sched = DDiTScheduler(total_steps=10, config=cfg)

    assert sched.get_token_reduction_factor(0, spatial_dims=3) == 64.0
    assert sched.get_token_reduction_factor(9, spatial_dims=3) == 1.0


def test_disabled_returns_min_stride():
    """Disabled scheduler always returns min_patch_stride."""
    cfg = DDiTScheduleConfig(max_patch_stride=4, min_patch_stride=1, enabled=False)
    sched = DDiTScheduler(total_steps=20, config=cfg)

    for i in range(20):
        assert sched.get_patch_stride(i) == 1


def test_warmup_fraction_zero():
    """warmup_fraction=0 → always min stride (immediate full res)."""
    cfg = DDiTScheduleConfig(max_patch_stride=4, min_patch_stride=1, warmup_fraction=0.0, schedule="linear")
    sched = DDiTScheduler(total_steps=20, config=cfg)

    # warmup_steps = max(1, int(20*0)) = 1, so step 0 is warmup, rest is min
    for i in range(1, 20):
        assert sched.get_patch_stride(i) == 1
