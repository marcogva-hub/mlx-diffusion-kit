"""Tests for B11 T-GATE cross-attention gating."""

import mlx.core as mx

from mlx_diffusion_kit.gating.tgate import (
    TGateConfig,
    create_tgate_state,
    tgate_forward,
)


def _make_fns(self_calls: list, cross_calls: list):
    """Create mock self-attn and cross-attn functions with call counters."""

    def self_attn(x):
        self_calls.append(1)
        return x

    def cross_attn(x, ctx):
        cross_calls.append(1)
        return mx.ones_like(x) * 0.5

    return self_attn, cross_attn


def test_cross_attn_called_before_gate_step():
    cfg = TGateConfig(gate_step=3)
    state = create_tgate_state()
    x = mx.zeros((2, 4))
    ctx = mx.ones((2, 4))
    self_c, cross_c = [], []
    sa, ca = _make_fns(self_c, cross_c)

    for step in range(3):
        tgate_forward(0, step, cfg, state, sa, ca, x, ctx)

    assert len(cross_c) == 3  # Called every step before gate


def test_cross_attn_not_called_after_gate_step():
    cfg = TGateConfig(gate_step=2)
    state = create_tgate_state()
    x = mx.zeros((2, 4))
    ctx = mx.ones((2, 4))
    self_c, cross_c = [], []
    sa, ca = _make_fns(self_c, cross_c)

    # Steps 0,1 — compute cross-attn
    for step in range(2):
        tgate_forward(0, step, cfg, state, sa, ca, x, ctx)
    assert len(cross_c) == 2

    # Steps 2,3,4 — should use cache
    for step in range(2, 5):
        tgate_forward(0, step, cfg, state, sa, ca, x, ctx)
    assert len(cross_c) == 2  # Still 2 — not called again


def test_cached_value_reused():
    cfg = TGateConfig(gate_step=1)
    state = create_tgate_state()
    x = mx.zeros((2, 4))
    ctx = mx.ones((2, 4))
    sa = lambda x: x
    ca = lambda x, ctx: mx.ones_like(x) * 0.42

    # Step 0 — compute and cache
    out0 = tgate_forward(0, 0, cfg, state, sa, ca, x, ctx)

    # Step 1 — use cached value (cross_attn NOT called)
    ca_never = lambda x, ctx: mx.ones_like(x) * 999.0  # Would give different result
    out1 = tgate_forward(0, 1, cfg, state, sa, ca_never, x, ctx)

    # Both should use the cached 0.42 value
    assert mx.allclose(out0, out1)


def test_disabled_always_calls_cross_attn():
    cfg = TGateConfig(gate_step=1, enabled=False)
    state = create_tgate_state()
    x = mx.zeros((2, 4))
    ctx = mx.ones((2, 4))
    self_c, cross_c = [], []
    sa, ca = _make_fns(self_c, cross_c)

    for step in range(5):
        tgate_forward(0, step, cfg, state, sa, ca, x, ctx)

    assert len(cross_c) == 5  # Called every step


def test_multi_layer():
    cfg = TGateConfig(gate_step=1)
    state = create_tgate_state()
    x = mx.zeros((2, 4))
    ctx = mx.ones((2, 4))

    # Each layer gets its own cached value
    vals = {}
    for layer_idx in range(3):
        val = float(layer_idx + 1)
        ca = lambda x, ctx, v=val: mx.ones_like(x) * v
        tgate_forward(layer_idx, 0, cfg, state, lambda x: x, ca, x, ctx)
        vals[layer_idx] = val

    # Verify each layer cached independently
    for layer_idx in range(3):
        cached = state.cached_cross_attn[layer_idx]
        expected = mx.ones((2, 4)) * vals[layer_idx]
        assert mx.allclose(cached, expected)
