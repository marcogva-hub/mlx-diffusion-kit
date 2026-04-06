"""Tests for B8 ToPi token pruning."""

import mlx.core as mx
import pytest

from mlx_diffusion_kit.tokens.pruning import (
    PruneInfo,
    ToPiConfig,
    compute_token_importance,
    topi_prune,
    topi_restore,
)


def test_prune_restore_shape():
    """Prune then restore should give original shape."""
    cfg = ToPiConfig(prune_ratio=0.3, importance="norm", restore_mode="copy")
    tokens = mx.random.normal((2, 20, 32))
    pruned, info = topi_prune(tokens, cfg)
    assert pruned.shape == (2, 14, 32)  # 20 * 0.7 = 14
    restored = topi_restore(pruned, info, cfg)
    assert restored.shape == (2, 20, 32)


def test_prune_ratio_zero_passthrough():
    """prune_ratio=0 should be a no-op."""
    cfg = ToPiConfig(prune_ratio=0.0)
    tokens = mx.random.normal((1, 16, 32))
    pruned, info = topi_prune(tokens, cfg)
    assert pruned.shape == tokens.shape
    assert mx.array_equal(pruned, tokens)


def test_importance_norm_keeps_high_norm():
    """Norm importance should keep tokens with highest L2 norms."""
    cfg = ToPiConfig(prune_ratio=0.5, importance="norm")
    # Create tokens where first half has high norm, second half low
    high = mx.ones((1, 5, 8)) * 10.0
    low = mx.ones((1, 5, 8)) * 0.01
    tokens = mx.concatenate([high, low], axis=1)  # [1, 10, 8]

    pruned, info = topi_prune(tokens, cfg)
    assert pruned.shape == (1, 5, 8)

    # Kept tokens should be mostly from the high-norm set
    avg_norm = mx.mean(mx.linalg.norm(pruned, axis=-1)).item()
    assert avg_norm > 5.0, f"Expected high-norm tokens kept, got avg_norm={avg_norm}"


def test_importance_random_correct_count():
    """Random importance should produce correct number of kept tokens."""
    cfg = ToPiConfig(prune_ratio=0.4, importance="random")
    tokens = mx.random.normal((2, 20, 16))
    pruned, info = topi_prune(tokens, cfg)
    assert pruned.shape == (2, 12, 16)  # 20 * 0.6 = 12


def test_importance_attention():
    """Attention importance uses column-sum of attention weights."""
    cfg = ToPiConfig(prune_ratio=0.5, importance="attention")
    tokens = mx.random.normal((1, 8, 16))
    # Attention weights: token 0 gets most attention (column 0 has highest sum)
    attn = mx.zeros((1, 8, 8))
    attn = attn.at[:, :, 0].add(1.0)  # All rows attend to token 0

    scores = compute_token_importance(tokens, "attention", attn)
    assert scores.shape == (1, 8)
    # Token 0 should have highest importance
    assert mx.argmax(scores[0]).item() == 0


def test_restore_copy():
    """Copy restore: pruned tokens == their nearest kept token."""
    cfg = ToPiConfig(prune_ratio=0.5, importance="norm", restore_mode="copy")
    # Distinct tokens with clear norms
    tokens = mx.random.normal((1, 10, 8))
    pruned, info = topi_prune(tokens, cfg)
    restored = topi_restore(pruned, info, cfg)

    # Check restored shape
    assert restored.shape == (1, 10, 8)
    # Pruned positions should have non-zero values (copied from kept)
    for b in range(1):
        for i in range(info.pruned_indices.shape[1]):
            pos = info.pruned_indices[b, i].item()
            val = restored[b, pos]
            assert mx.any(val != 0.0).item(), f"Pruned token at {pos} is zero"


def test_restore_zero():
    """Zero restore: pruned tokens should be zero."""
    cfg = ToPiConfig(prune_ratio=0.5, importance="norm", restore_mode="zero")
    tokens = mx.random.normal((1, 10, 8))
    pruned, info = topi_prune(tokens, cfg)
    restored = topi_restore(pruned, info, cfg)

    assert restored.shape == (1, 10, 8)
    for b in range(1):
        for i in range(info.pruned_indices.shape[1]):
            pos = info.pruned_indices[b, i].item()
            assert mx.allclose(restored[b, pos], mx.zeros((8,))), f"Token at {pos} not zero"


def test_disabled():
    cfg = ToPiConfig(enabled=False)
    tokens = mx.random.normal((2, 16, 32))
    pruned, info = topi_prune(tokens, cfg)
    assert mx.array_equal(pruned, tokens)


def test_outputs_finite():
    for mode in ["copy", "zero", "lerp"]:
        cfg = ToPiConfig(prune_ratio=0.3, importance="norm", restore_mode=mode)
        tokens = mx.random.normal((2, 16, 32))
        pruned, info = topi_prune(tokens, cfg)
        restored = topi_restore(pruned, info, cfg)
        assert mx.all(mx.isfinite(pruned)).item(), f"Non-finite pruned for {mode}"
        assert mx.all(mx.isfinite(restored)).item(), f"Non-finite restored for {mode}"
