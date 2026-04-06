"""Tests for B14.2 Adaptive Stepping scheduler."""

import mlx.core as mx

from mlx_diffusion_kit.scheduler.adaptive_stepping import (
    AdaptiveStepConfig,
    AdaptiveStepScheduler,
)


def test_identical_outputs_skip():
    """Identical consecutive outputs should trigger step skipping."""
    cfg = AdaptiveStepConfig(min_steps=2, tolerance=0.001)
    timesteps = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    sched = AdaptiveStepScheduler(timesteps, cfg)

    output = mx.ones((1, 4, 16))

    # First min_steps shouldn't skip
    for i in range(cfg.min_steps):
        skip = sched.should_skip_step(i, output, output)
        assert not skip, f"Step {i} should not skip (below min_steps)"

    # After min_steps, identical outputs → skip
    skip = sched.should_skip_step(cfg.min_steps, output, output)
    assert skip, "Identical outputs after min_steps should trigger skip"


def test_different_outputs_no_skip():
    """Very different outputs should not skip."""
    cfg = AdaptiveStepConfig(min_steps=1, tolerance=0.001)
    timesteps = [1.0, 0.5, 0.0]
    sched = AdaptiveStepScheduler(timesteps, cfg)

    out1 = mx.zeros((1, 4, 16))
    out2 = mx.ones((1, 4, 16)) * 10.0

    # Burn min_steps
    sched.should_skip_step(0, out1, out2)
    # Very different outputs → no skip
    skip = sched.should_skip_step(1, out1, out2)
    assert not skip


def test_min_steps_respected():
    """Should never skip before min_steps even with identical outputs."""
    cfg = AdaptiveStepConfig(min_steps=5, tolerance=0.0001)
    timesteps = list(range(10))
    sched = AdaptiveStepScheduler(timesteps, cfg)

    output = mx.ones((2, 8))
    for i in range(5):
        skip = sched.should_skip_step(i, output, output)
        assert not skip, f"Step {i} should not skip (below min_steps=5)"


def test_reset():
    cfg = AdaptiveStepConfig(min_steps=1, tolerance=0.001)
    timesteps = [1.0, 0.5, 0.0]
    sched = AdaptiveStepScheduler(timesteps, cfg)

    output = mx.ones((1, 4))
    sched.should_skip_step(0, output, output)
    sched.should_skip_step(1, output, output)

    sched.reset()
    assert sched.num_skipped == 0
    assert sched.get_effective_timesteps() == timesteps


def test_get_effective_timesteps():
    cfg = AdaptiveStepConfig(min_steps=1, tolerance=0.001)
    timesteps = [1.0, 0.8, 0.6, 0.4, 0.2]
    sched = AdaptiveStepScheduler(timesteps, cfg)

    output = mx.ones((1, 4))
    # Burn min_steps
    sched.should_skip_step(0, output, output)
    # This should trigger a skip of step 2
    sched.should_skip_step(1, output, output)

    effective = sched.get_effective_timesteps()
    assert len(effective) < len(timesteps)


def test_disabled():
    cfg = AdaptiveStepConfig(enabled=False)
    timesteps = [1.0, 0.5, 0.0]
    sched = AdaptiveStepScheduler(timesteps, cfg)

    output = mx.ones((1, 4))
    for i in range(3):
        assert not sched.should_skip_step(i, output, output)
    assert sched.num_skipped == 0
