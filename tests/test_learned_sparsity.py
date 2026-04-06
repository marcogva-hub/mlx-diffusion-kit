"""Tests for B9 DiffSparse learned sparsity (stub)."""

import mlx.core as mx
import pytest

from mlx_diffusion_kit.tokens.learned_sparsity import (
    DiffSparseConfig,
    DiffSparseRouter,
)


def test_stub_returns_correct_token_count():
    """Stub should return budget * N tokens."""
    cfg = DiffSparseConfig(budget=0.5)
    router = DiffSparseRouter(input_dim=32, config=cfg)
    tokens = mx.random.normal((2, 16, 32))
    selected, scores = router(tokens)
    assert selected.shape == (2, 8, 32)
    assert scores.shape == (2, 16)


def test_stub_budget_quarter():
    cfg = DiffSparseConfig(budget=0.25)
    router = DiffSparseRouter(input_dim=64, config=cfg)
    tokens = mx.random.normal((1, 20, 64))
    selected, _ = router(tokens)
    assert selected.shape == (1, 5, 64)


def test_from_pretrained_raises():
    with pytest.raises(NotImplementedError, match="router weights required"):
        DiffSparseRouter.from_pretrained("/fake/path")


def test_disabled_passthrough():
    """Disabled router should return all tokens."""
    cfg = DiffSparseConfig(enabled=False)
    router = DiffSparseRouter(input_dim=32, config=cfg)
    tokens = mx.random.normal((2, 16, 32))
    selected, scores = router(tokens)
    assert selected.shape == tokens.shape
    assert mx.array_equal(selected, tokens)


def test_stub_outputs_finite():
    cfg = DiffSparseConfig(budget=0.5)
    router = DiffSparseRouter(input_dim=32, config=cfg)
    tokens = mx.random.normal((2, 16, 32))
    selected, scores = router(tokens)
    assert mx.all(mx.isfinite(selected)).item()
    assert mx.all(mx.isfinite(scores)).item()


def test_minimum_one_token():
    """Even with very low budget, keep at least 1 token."""
    cfg = DiffSparseConfig(budget=0.01)
    router = DiffSparseRouter(input_dim=16, config=cfg)
    tokens = mx.random.normal((1, 4, 16))
    selected, _ = router(tokens)
    assert selected.shape[1] >= 1


def test_strict_mode_raises():
    """strict=True without pretrained weights should raise RuntimeError."""
    cfg = DiffSparseConfig(strict=True)
    router = DiffSparseRouter(input_dim=32, config=cfg)
    tokens = mx.random.normal((1, 8, 32))
    with pytest.raises(RuntimeError, match="no pretrained weights"):
        router(tokens)


def test_non_strict_warns(caplog):
    """strict=False should log a warning on first call."""
    import logging

    cfg = DiffSparseConfig(strict=False)
    router = DiffSparseRouter(input_dim=32, config=cfg)
    tokens = mx.random.normal((1, 8, 32))

    with caplog.at_level(logging.WARNING):
        router(tokens)
    assert "no pretrained weights" in caplog.text

    # Second call should NOT warn again
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        router(tokens)
    assert "no pretrained weights" not in caplog.text
