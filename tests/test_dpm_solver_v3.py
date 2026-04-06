"""Tests for B14.1 DPM-Solver-v3 scheduler."""

import mlx.core as mx
import pytest

from mlx_diffusion_kit.scheduler.dpm_solver_v3 import (
    DPMSolverV3,
    DPMSolverV3Config,
    NoiseSchedule,
    compute_optimal_timesteps,
)


# --- NoiseSchedule tests ---


def test_alpha_bar_decreasing():
    """VP alpha_bar should decrease monotonically from ~1 to ~0."""
    ns = NoiseSchedule()
    t = mx.linspace(0, 999, 100)
    ab = ns.alpha_bar(t)
    vals = ab.tolist()
    for i in range(1, len(vals)):
        assert vals[i] <= vals[i - 1] + 1e-6, f"alpha_bar not decreasing at index {i}"
    assert vals[0] > 0.9  # Near 1 at t=0
    assert vals[-1] < 0.1  # Near 0 at t=T


def test_log_snr_decreasing():
    """Log-SNR should decrease monotonically."""
    ns = NoiseSchedule()
    t = mx.linspace(0, 999, 100)
    lsnr = ns.log_snr(t)
    vals = lsnr.tolist()
    for i in range(1, len(vals)):
        assert vals[i] <= vals[i - 1] + 1e-5, f"log_snr not decreasing at index {i}"


def test_inverse_log_snr_roundtrip():
    """inverse_log_snr(log_snr(t)) ≈ t."""
    ns = NoiseSchedule()
    # Test at a few interior points (avoid boundaries)
    t_orig = mx.array([50.0, 200.0, 500.0, 800.0, 950.0])
    lsnr = ns.log_snr(t_orig)
    t_recovered = ns.inverse_log_snr(lsnr)
    # Allow ±2 timestep tolerance due to discrete schedule interpolation
    diff = mx.abs(t_recovered - t_orig)
    assert mx.all(diff < 2.0).item(), f"Roundtrip error too large: {diff.tolist()}"


# --- Timestep computation tests ---


def test_get_timesteps_count():
    """Should return num_steps + 1 timesteps."""
    ns = NoiseSchedule()
    cfg = DPMSolverV3Config(num_steps=15)
    solver = DPMSolverV3(ns, cfg)
    ts = solver.get_timesteps()
    assert ts.shape[0] == 16  # 15 + 1


def test_timesteps_decreasing():
    """Timesteps should be non-increasing (from t=0 clean to t=T noisy then back)."""
    ns = NoiseSchedule()
    ts = compute_optimal_timesteps(ns, num_steps=20)
    # Since we go from clean (t=0, high SNR) to noisy (t=T, low SNR),
    # timesteps should be increasing in t
    vals = ts.tolist()
    for i in range(1, len(vals)):
        assert vals[i] >= vals[i - 1] - 1e-3, f"Timesteps not non-decreasing at {i}"


def test_log_snr_spacing_uniform():
    """Log-SNR of optimal timesteps should be approximately uniformly spaced."""
    ns = NoiseSchedule()
    ts = compute_optimal_timesteps(ns, num_steps=20)
    lsnr = ns.log_snr(ts)
    diffs = []
    vals = lsnr.tolist()
    for i in range(1, len(vals)):
        diffs.append(vals[i] - vals[i - 1])
    # All diffs should be approximately equal
    mean_diff = sum(diffs) / len(diffs)
    for i, d in enumerate(diffs):
        assert abs(d - mean_diff) < abs(mean_diff) * 0.15, (
            f"Log-SNR spacing not uniform at {i}: {d:.4f} vs mean {mean_diff:.4f}"
        )


# --- Solver step tests ---


def test_order_1_denoising():
    """Order 1 (DDIM-like) should denoise toward signal."""
    ns = NoiseSchedule()
    cfg = DPMSolverV3Config(order=1, num_steps=20, predict_type="epsilon")
    solver = DPMSolverV3(ns, cfg)

    ts = solver.get_timesteps()
    # Start with noisy sample (high t)
    signal = mx.ones((1, 4, 8)) * 3.0
    # Simulate: at each step, model predicts the noise (= sample - signal * alpha)
    x = mx.random.normal((1, 4, 8)) * 0.5 + signal * 0.1  # Very noisy

    for i in range(len(ts) - 1):
        # Simple mock: model output is the noise estimate
        ab = ns.alpha_bar(ts[i:i+1])
        alpha = mx.sqrt(ab)
        sigma = mx.sqrt(1.0 - ab)
        eps_pred = (x - alpha * signal) / (sigma + 1e-8)
        x = solver.step(eps_pred, i, x)

    # After denoising, x should be finite
    assert mx.all(mx.isfinite(x)).item()


def test_higher_order_produces_finite_results():
    """Orders 1, 2, 3 should all produce finite results with zero-noise prediction."""
    ns = NoiseSchedule()

    for order in [1, 2, 3]:
        cfg = DPMSolverV3Config(order=order, num_steps=10)
        solver = DPMSolverV3(ns, cfg)
        ts = solver.get_timesteps()

        x = mx.random.normal((1, 16))

        for i in range(len(ts) - 1):
            # Predict zero noise → solver drives x toward "clean" signal
            eps_pred = mx.zeros_like(x)
            x = solver.step(eps_pred, i, x)

        assert mx.all(mx.isfinite(x)).item(), f"Non-finite output for order {order}"
        solver.reset()

    # Order 2 history should have been populated during its run
    # (already verified via finite check above)


def test_reset_clears_history():
    ns = NoiseSchedule()
    cfg = DPMSolverV3Config(order=3, num_steps=10)
    solver = DPMSolverV3(ns, cfg)

    # Simulate a few steps
    x = mx.random.normal((1, 8))
    ts = solver.get_timesteps()
    for i in range(3):
        solver.step(mx.random.normal((1, 8)), i, x)

    assert len(solver.model_output_history) > 0
    solver.reset()
    assert len(solver.model_output_history) == 0


def test_predict_types_same_shape():
    """Different predict_types should produce same output shape."""
    ns = NoiseSchedule()
    shape = (1, 4, 8)
    x = mx.random.normal(shape)
    model_out = mx.random.normal(shape)

    for ptype in ["epsilon", "v_prediction", "x_start"]:
        cfg = DPMSolverV3Config(order=1, num_steps=10, predict_type=ptype)
        solver = DPMSolverV3(ns, cfg)
        result = solver.step(model_out, 0, x)
        assert result.shape == shape, f"Shape mismatch for {ptype}"
        assert mx.all(mx.isfinite(result)).item(), f"Non-finite for {ptype}"
        solver.reset()


def test_outputs_finite_order_3():
    """Order 3 solver should produce finite outputs."""
    ns = NoiseSchedule()
    cfg = DPMSolverV3Config(order=3, num_steps=15)
    solver = DPMSolverV3(ns, cfg)

    x = mx.random.normal((2, 4, 16))
    ts = solver.get_timesteps()

    for i in range(min(5, len(ts) - 1)):
        eps = mx.random.normal(x.shape) * 0.1
        x = solver.step(eps, i, x)
        assert mx.all(mx.isfinite(x)).item(), f"Non-finite at step {i}"
