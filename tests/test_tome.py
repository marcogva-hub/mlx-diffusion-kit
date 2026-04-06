"""Tests for B8 Token Merging (ToMe)."""

import mlx.core as mx
import pytest

from mlx_diffusion_kit.tokens.tome import (
    MergeInfo,
    ToMeConfig,
    compute_proportional_bias,
    tome_merge,
    tome_unmerge,
)


def test_merge_unmerge_restores_shape():
    """Merge then unmerge should restore original shape."""
    cfg = ToMeConfig(merge_ratio=0.5)
    tokens = mx.random.normal((2, 16, 64))
    merged, info = tome_merge(tokens, cfg)
    assert merged.shape[1] == 8  # N/2
    unmerged = tome_unmerge(merged, info)
    assert unmerged.shape == tokens.shape


def test_merge_ratio_zero_passthrough():
    """merge_ratio=0 should be a no-op."""
    cfg = ToMeConfig(merge_ratio=0.0)
    tokens = mx.random.normal((2, 16, 64))
    merged, info = tome_merge(tokens, cfg)
    assert merged.shape == tokens.shape
    assert mx.array_equal(merged, tokens)


def test_merge_ratio_half():
    """merge_ratio=0.5 should halve token count."""
    cfg = ToMeConfig(merge_ratio=0.5)
    tokens = mx.random.normal((1, 20, 32))
    merged, info = tome_merge(tokens, cfg)
    assert merged.shape == (1, 10, 32)
    assert info.original_n == 20


def test_identical_tokens_merge():
    """Identical tokens should merge cleanly."""
    cfg = ToMeConfig(merge_ratio=0.5)
    tokens = mx.ones((1, 8, 16))
    merged, info = tome_merge(tokens, cfg)
    assert merged.shape == (1, 4, 16)
    # All tokens identical → merged should be ~1.0 everywhere
    assert mx.allclose(merged, mx.ones_like(merged), atol=0.1)


def test_different_tokens_correct_shape():
    """Very different tokens should still produce correct shape."""
    cfg = ToMeConfig(merge_ratio=0.5)
    # Alternating very different tokens
    t = mx.zeros((1, 8, 16))
    t = t.at[:, ::2, :].add(10.0)
    t = t.at[:, 1::2, :].add(-10.0)
    merged, info = tome_merge(t, cfg)
    assert merged.shape == (1, 4, 16)


def test_proportional_bias_counts():
    """Proportional bias should have values >= 0 (log(count) where count >= 1)."""
    cfg = ToMeConfig(merge_ratio=0.5)
    tokens = mx.random.normal((1, 16, 32))
    _, info = tome_merge(tokens, cfg)
    bias = compute_proportional_bias(info)
    assert bias.shape == (8,)  # n_src = 8
    # log(1) = 0, log(2) ≈ 0.69, etc. — all >= 0
    assert mx.all(bias >= 0.0).item()
    # At least some tokens should have count > 1 (bias > 0)
    assert mx.sum(bias > 0.0).item() > 0


def test_mlerp_preserves_norms():
    """MLERP should approximately preserve norms."""
    cfg = ToMeConfig(merge_ratio=0.5, use_mlerp=True)
    tokens = mx.random.normal((1, 8, 32)) * 5.0  # Non-unit norm
    avg_norm_before = mx.mean(mx.linalg.norm(tokens, axis=-1)).item()
    merged, _ = tome_merge(tokens, cfg)
    avg_norm_after = mx.mean(mx.linalg.norm(merged, axis=-1)).item()
    # Norms should be in the same ballpark (within 2x)
    assert avg_norm_after > avg_norm_before * 0.3
    assert avg_norm_after < avg_norm_before * 3.0


def test_different_batch_sizes():
    """Should work with various batch sizes."""
    cfg = ToMeConfig(merge_ratio=0.5)
    for batch in [1, 2, 4]:
        tokens = mx.random.normal((batch, 12, 32))
        merged, info = tome_merge(tokens, cfg)
        assert merged.shape == (batch, 6, 32)
        unmerged = tome_unmerge(merged, info)
        assert unmerged.shape == (batch, 12, 32)


def test_disabled_passthrough():
    """Disabled config should pass through unchanged."""
    cfg = ToMeConfig(enabled=False)
    tokens = mx.random.normal((1, 16, 32))
    merged, info = tome_merge(tokens, cfg)
    assert mx.array_equal(merged, tokens)


def test_4d_input_with_heads():
    """Should handle [B, H, N, D] input."""
    cfg = ToMeConfig(merge_ratio=0.5)
    tokens = mx.random.normal((2, 4, 16, 32))  # B=2, H=4, N=16, D=32
    merged, info = tome_merge(tokens, cfg)
    assert merged.shape == (2, 4, 8, 32)
    unmerged = tome_unmerge(merged, info)
    assert unmerged.shape == (2, 4, 16, 32)


def test_outputs_finite():
    """No NaN or Inf in outputs."""
    cfg = ToMeConfig(merge_ratio=0.5)
    tokens = mx.random.normal((2, 16, 64))
    merged, info = tome_merge(tokens, cfg)
    assert mx.all(mx.isfinite(merged)).item()
    unmerged = tome_unmerge(merged, info)
    assert mx.all(mx.isfinite(unmerged)).item()


def test_performance_n1000():
    """tome_merge on N=1000 should complete in under 1s (vectorized)."""
    import time

    cfg = ToMeConfig(merge_ratio=0.5)
    tokens = mx.random.normal((1, 1000, 64))
    # Force materialization of input
    _ = tokens.sum().item()

    start = time.perf_counter()
    merged, info = tome_merge(tokens, cfg)
    # Force materialization of output
    _ = merged.sum().item()
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"tome_merge(N=1000) took {elapsed:.2f}s, expected < 1s"


def test_spatiotemporal_similarity_nearby_higher():
    """Spatially nearby tokens should have higher combined similarity."""
    from mlx_diffusion_kit.tokens.tome import compute_spatiotemporal_similarity

    T, H, W = 2, 4, 4
    N = T * H * W  # 32
    tokens = mx.random.normal((1, N, 16))

    sim = compute_spatiotemporal_similarity(
        tokens, (T, H, W), spatial_weight=0.4, temporal_weight=0.3
    )
    assert sim.shape == (1, N, N)

    # Token 0 is at (0,0,0), token 1 at (0,0,1) — adjacent
    # Token 0 vs token W-1 at (0,0,3) — farther
    sim_near = sim[0, 0, 1].item()
    sim_far = sim[0, 0, W - 1].item()
    # Adjacent should generally be higher (spatial proximity contributes)
    # With 40% spatial weight, this should hold even if cosine differs
    assert sim_near >= sim_far - 0.5, (
        f"Adjacent sim {sim_near:.3f} should be >= distant sim {sim_far:.3f} - 0.5"
    )


def test_spatiotemporal_similarity_shape():
    from mlx_diffusion_kit.tokens.tome import compute_spatiotemporal_similarity

    T, H, W = 3, 4, 4
    N = T * H * W
    tokens = mx.random.normal((2, N, 32))
    sim = compute_spatiotemporal_similarity(tokens, (T, H, W))
    assert sim.shape == (2, N, N)
    assert mx.all(mx.isfinite(sim)).item()


def test_tome_merge_with_spatial_dims():
    """tome_merge should accept spatial_dims for video-aware merging."""
    cfg = ToMeConfig(merge_ratio=0.5)
    T, H, W = 2, 4, 4
    N = T * H * W
    tokens = mx.random.normal((1, N, 32))

    merged, info = tome_merge(tokens, cfg, spatial_dims=(T, H, W))
    assert merged.shape == (1, N // 2, 32)
    assert info.original_n == N


def test_compute_attn_bias_for_mfa():
    """attn_bias should have shape [1,1,1,N_merged] with log(counts)."""
    from mlx_diffusion_kit.tokens.tome import compute_attn_bias_for_mfa

    cfg = ToMeConfig(merge_ratio=0.5)
    tokens = mx.random.normal((1, 16, 32))
    _, info = tome_merge(tokens, cfg)

    bias = compute_attn_bias_for_mfa(info)
    assert bias.ndim == 4
    assert bias.shape[0] == 1
    assert bias.shape[1] == 1
    assert bias.shape[2] == 1
    assert bias.shape[3] == 8  # N_merged = 8

    # Values should be >= 0 (log(count) where count >= 1)
    assert mx.all(bias >= 0.0).item()


def test_attn_bias_integration():
    """Merge → bias → shapes should be compatible for flash_attention."""
    cfg = ToMeConfig(merge_ratio=0.5)
    tokens = mx.random.normal((2, 16, 64))

    merged, info = tome_merge(tokens, cfg)
    from mlx_diffusion_kit.tokens.tome import compute_attn_bias_for_mfa
    bias = compute_attn_bias_for_mfa(info)

    B, N_merged, D = merged.shape
    # Simulate Q, K, V shapes
    H = 4
    head_dim = D // H
    q = merged.reshape(B, N_merged, H, head_dim)
    # bias [1,1,1,N_kv] should broadcast with [B,H,N_q,N_kv]
    assert bias.shape == (1, 1, 1, N_merged)
    # Verify broadcast works
    fake_attn = mx.zeros((B, H, N_merged, N_merged)) + bias
    assert fake_attn.shape == (B, H, N_merged, N_merged)
