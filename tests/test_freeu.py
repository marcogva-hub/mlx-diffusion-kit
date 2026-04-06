"""Tests for B13 FreeU filter."""

import mlx.core as mx
import pytest

from mlx_diffusion_kit.quality.freeu import FreeUConfig, freeu_filter


def test_disabled_passthrough():
    """When disabled, output == input exactly."""
    cfg = FreeUConfig(enabled=False)
    h_skip = mx.random.normal((2, 8, 16))
    h_back = mx.random.normal((2, 8, 16))
    out_skip, out_back = freeu_filter(h_skip, h_back, cfg)
    assert mx.array_equal(out_skip, h_skip)
    assert mx.array_equal(out_back, h_back)


def test_output_shapes_match_input():
    """Shapes must be preserved."""
    cfg = FreeUConfig()
    for shape in [(4, 8, 16), (2, 4, 8, 16), (1, 2, 4, 8, 16)]:
        h_skip = mx.random.normal(shape)
        h_back = mx.random.normal(shape)
        out_s, out_b = freeu_filter(h_skip, h_back, cfg)
        assert out_s.shape == h_skip.shape, f"Skip shape mismatch for {shape}"
        assert out_b.shape == h_back.shape, f"Backbone shape mismatch for {shape}"


def test_backbone_is_modified():
    """Backbone output should differ from input when enabled."""
    cfg = FreeUConfig()
    h_skip = mx.random.normal((2, 8, 16))
    h_back = mx.random.normal((2, 8, 16))
    _, out_back = freeu_filter(h_skip, h_back, cfg)
    assert not mx.array_equal(out_back, h_back)


def test_skip_is_modified():
    """Skip output should differ from input when enabled."""
    cfg = FreeUConfig()
    h_skip = mx.random.normal((2, 8, 32))
    h_back = mx.random.normal((2, 8, 32))
    out_skip, _ = freeu_filter(h_skip, h_back, cfg)
    assert not mx.array_equal(out_skip, h_skip)


def test_outputs_are_finite():
    """No NaN or Inf in outputs."""
    cfg = FreeUConfig()
    h_skip = mx.random.normal((4, 16, 64))
    h_back = mx.random.normal((4, 16, 64))
    out_s, out_b = freeu_filter(h_skip, h_back, cfg)
    assert mx.all(mx.isfinite(out_s)).item()
    assert mx.all(mx.isfinite(out_b)).item()
