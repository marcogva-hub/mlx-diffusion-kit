"""Tests for attention/residual.py utilities."""

import math

import mlx.core as mx

from mlx_diffusion_kit.attention.residual import (
    compute_residual_scale,
    residual_gate_from_sensitivity,
    scaled_residual_add,
)


def test_scaled_residual_add_no_gate():
    x = mx.ones((2, 4)) * 3.0
    res = mx.ones((2, 4)) * 2.0
    result = scaled_residual_add(x, res, scale=1.0)
    assert mx.allclose(result, mx.ones((2, 4)) * 5.0)


def test_scaled_residual_add_with_scale():
    x = mx.ones((2, 4)) * 3.0
    res = mx.ones((2, 4)) * 2.0
    result = scaled_residual_add(x, res, scale=0.5)
    assert mx.allclose(result, mx.ones((2, 4)) * 4.0)  # 3 + 0.5*2 = 4


def test_scaled_residual_add_with_gate():
    x = mx.ones((2, 4)) * 3.0
    res = mx.ones((2, 4)) * 2.0
    gate = mx.array([[0.5, 0.5, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0]])
    result = scaled_residual_add(x, res, scale=1.0, gate=gate)
    # Row 0: 3 + 0.5*1*2 = 4, Row 1: 3 + 1*1*2 = 5
    assert mx.allclose(result[0], mx.ones((4,)) * 4.0)
    assert mx.allclose(result[1], mx.ones((4,)) * 5.0)


def test_compute_residual_scale_inverse_sqrt():
    s0 = compute_residual_scale(0, 24, "inverse_sqrt")
    s1 = compute_residual_scale(1, 24, "inverse_sqrt")
    s10 = compute_residual_scale(10, 24, "inverse_sqrt")

    assert s0 == 1.0  # 1/sqrt(1)
    assert abs(s1 - 1.0 / math.sqrt(2)) < 1e-6
    assert s0 > s1 > s10  # Decreasing


def test_compute_residual_scale_linear():
    s0 = compute_residual_scale(0, 10, "linear")
    s5 = compute_residual_scale(5, 10, "linear")
    s9 = compute_residual_scale(9, 10, "linear")

    assert s0 == 1.0
    assert abs(s5 - 0.5) < 1e-6
    assert abs(s9 - 0.1) < 1e-6


def test_compute_residual_scale_constant():
    for i in range(10):
        assert compute_residual_scale(i, 10, "constant") == 1.0


def test_residual_gate_from_sensitivity():
    scores = {0: 0.9, 1: 0.1, 2: 0.5}
    assert residual_gate_from_sensitivity(scores, 0) == 0.9
    assert residual_gate_from_sensitivity(scores, 1) == 0.1
    assert residual_gate_from_sensitivity(scores, 2) == 0.5


def test_residual_gate_default():
    assert residual_gate_from_sensitivity({}, 0, default=0.7) == 0.7


def test_residual_gate_clamped():
    scores = {0: 1.5, 1: -0.3}
    assert residual_gate_from_sensitivity(scores, 0) == 1.0
    assert residual_gate_from_sensitivity(scores, 1) == 0.0
