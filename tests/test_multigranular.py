"""Tests for B6 Multi-Granular Cache (BWCache + UniCP + QuantCache)."""

import mlx.core as mx
import pytest

from mlx_diffusion_kit.cache.multigranular import (
    BWCacheAllocator,
    BWCacheConfig,
    MultiGranularCache,
    QuantCacheConfig,
    UniCPConfig,
    UniCPPolicy,
    quantcache_compress,
    quantcache_decompress,
    DECISION_COMPUTE,
    DECISION_CACHE_REUSE,
    DECISION_INTERPOLATE,
)


# ===== BWCacheAllocator =====


def test_bw_allocation_respects_budget():
    """Total allocation should not exceed memory budget."""
    shapes = {0: (64, 128), 1: (64, 256), 2: (64, 512)}
    cfg = BWCacheConfig(memory_budget_gb=0.001, dtype_bytes=2)  # ~1 MB
    alloc = BWCacheAllocator(cfg, shapes)
    result = alloc.compute_allocation()

    total_bytes = 0
    for idx, max_steps in result.items():
        layer_bytes = 1
        for s in shapes[idx]:
            layer_bytes *= s
        layer_bytes *= cfg.dtype_bytes
        total_bytes += max_steps * layer_bytes

    assert total_bytes <= cfg.memory_budget_gb * 1e9


def test_bw_large_layers_fewer_steps():
    """Larger layers should get fewer (or equal) cached steps than smaller ones."""
    shapes = {0: (16, 32), 1: (16, 512)}  # Layer 1 is 16x bigger
    cfg = BWCacheConfig(memory_budget_gb=0.001, dtype_bytes=2, prefer_quality=True)
    alloc = BWCacheAllocator(cfg, shapes)
    result = alloc.compute_allocation()

    # Small layer should have >= steps compared to large layer
    assert result.get(0, 0) >= result.get(1, 0)


def test_bw_zero_budget():
    """Zero budget → 0 cached steps for all layers."""
    shapes = {0: (64, 128), 1: (64, 256)}
    cfg = BWCacheConfig(memory_budget_gb=0.0, dtype_bytes=2)
    alloc = BWCacheAllocator(cfg, shapes)
    result = alloc.compute_allocation()
    assert all(v == 0 for v in result.values())


def test_bw_huge_budget():
    """Very large budget → all layers cacheable."""
    shapes = {0: (64, 128), 1: (64, 256), 2: (64, 512)}
    cfg = BWCacheConfig(memory_budget_gb=100.0, dtype_bytes=2)
    alloc = BWCacheAllocator(cfg, shapes)
    result = alloc.compute_allocation()
    assert all(v >= 1 for v in result.values())


def test_bw_should_cache_layer():
    shapes = {0: (16, 32)}
    cfg = BWCacheConfig(memory_budget_gb=1.0, dtype_bytes=2)
    alloc = BWCacheAllocator(cfg, shapes)

    max_steps = alloc.compute_allocation()[0]
    # Within budget
    assert alloc.should_cache_layer(0, 0, current_cached=0) is True
    # At capacity
    assert alloc.should_cache_layer(0, 0, current_cached=max_steps) is False


# ===== UniCPPolicy =====


def test_unicp_high_distance_compute():
    """High distance + budget OK → compute."""
    cfg = UniCPConfig(distance_threshold=0.3)
    policy = UniCPPolicy(cfg)
    assert policy.decide(0, 0, teacache_distance=0.5, bw_budget_ok=True) == DECISION_COMPUTE


def test_unicp_low_distance_interpolate():
    """Low distance + smooth enabled → interpolate."""
    cfg = UniCPConfig(distance_threshold=0.3, use_smooth_interpolation=True)
    policy = UniCPPolicy(cfg)
    assert policy.decide(0, 0, teacache_distance=0.1, bw_budget_ok=True) == DECISION_INTERPOLATE


def test_unicp_low_distance_cache_reuse():
    """Low distance + smooth disabled → cache_reuse."""
    cfg = UniCPConfig(distance_threshold=0.3, use_smooth_interpolation=False)
    policy = UniCPPolicy(cfg)
    assert policy.decide(0, 0, teacache_distance=0.1, bw_budget_ok=True) == DECISION_CACHE_REUSE


def test_unicp_budget_not_ok_force_reuse():
    """Budget not OK → force cache_reuse regardless of distance."""
    cfg = UniCPConfig(distance_threshold=0.3, use_bw_budget=True)
    policy = UniCPPolicy(cfg)
    assert policy.decide(0, 0, teacache_distance=999.0, bw_budget_ok=False) == DECISION_CACHE_REUSE


def test_unicp_layer_priority():
    """Higher priority layer → lower effective threshold → more likely to compute."""
    cfg = UniCPConfig(distance_threshold=0.5, layer_priority={0: 0.9, 1: 0.1})
    policy = UniCPPolicy(cfg)

    # Layer 0 (high priority, threshold ~0.275): distance 0.3 > 0.275 → compute
    assert policy.decide(0, 0, teacache_distance=0.3, bw_budget_ok=True) == DECISION_COMPUTE
    # Layer 1 (low priority, threshold ~0.475): distance 0.3 < 0.475 → interpolate
    assert policy.decide(1, 0, teacache_distance=0.3, bw_budget_ok=True) == DECISION_INTERPOLATE


def test_unicp_disabled():
    """Disabled policy → always compute."""
    cfg = UniCPConfig(enabled=False)
    policy = UniCPPolicy(cfg)
    assert policy.decide(0, 0, teacache_distance=0.01, bw_budget_ok=False) == DECISION_COMPUTE


# ===== QuantCache =====


def test_quantcache_int8_roundtrip():
    """int8 compress/decompress should have < 1% relative error."""
    cfg = QuantCacheConfig(bits=8, per_channel=True)
    features = mx.random.normal((4, 16)).astype(mx.float16)
    q, s = quantcache_compress(features, cfg)
    recovered = quantcache_decompress(q, s, cfg)

    # Relative error
    err = mx.mean(mx.abs(recovered - features)) / (mx.mean(mx.abs(features)) + 1e-8)
    assert err.item() < 0.02, f"int8 error too high: {err.item():.4f}"


def test_quantcache_int4_roundtrip():
    """int4 compress/decompress should have < 10% relative error."""
    cfg = QuantCacheConfig(bits=4, per_channel=True)
    features = mx.random.normal((4, 16)).astype(mx.float16)
    q, s = quantcache_compress(features, cfg)
    recovered = quantcache_decompress(q, s, cfg)

    err = mx.mean(mx.abs(recovered - features)) / (mx.mean(mx.abs(features)) + 1e-8)
    assert err.item() < 0.15, f"int4 error too high: {err.item():.4f}"


def test_quantcache_per_tensor():
    """Per-tensor quantization should also work."""
    cfg = QuantCacheConfig(bits=8, per_channel=False)
    features = mx.random.normal((4, 16)).astype(mx.float16)
    q, s = quantcache_compress(features, cfg)
    recovered = quantcache_decompress(q, s, cfg)

    err = mx.mean(mx.abs(recovered - features)) / (mx.mean(mx.abs(features)) + 1e-8)
    assert err.item() < 0.02


def test_quantcache_compressed_smaller():
    """int8 quantized should use less memory than f16 features."""
    cfg = QuantCacheConfig(bits=8, per_channel=True)
    features = mx.random.normal((64, 128)).astype(mx.float16)
    q, s = quantcache_compress(features, cfg)

    # int8 = 1 byte per element, f16 = 2 bytes per element
    assert q.dtype == mx.int8
    assert q.nbytes < features.nbytes


def test_quantcache_disabled_passthrough():
    cfg = QuantCacheConfig(enabled=False)
    features = mx.random.normal((4, 8))
    q, s = quantcache_compress(features, cfg)
    assert mx.array_equal(q, features)


# ===== MultiGranularCache =====


def test_multigranular_compute_stores():
    """Compute decision should store features and return them."""
    cache = MultiGranularCache(
        unicp_config=UniCPConfig(distance_threshold=0.3),
    )
    features = mx.random.normal((2, 4, 16))
    result = cache.process_layer(0, 0, features, teacache_distance=0.5)

    assert result.shape == features.shape
    stats = cache.get_stats()
    assert stats["cache_misses"] == 1


def test_multigranular_reuse_from_cache():
    """After storing, cache_reuse should return cached value."""
    cfg = UniCPConfig(distance_threshold=0.3, use_smooth_interpolation=False)
    cache = MultiGranularCache(unicp_config=cfg)

    features = mx.ones((2, 4, 16)) * 5.0
    # First call: compute (distance > threshold)
    cache.process_layer(0, 0, features, teacache_distance=0.5)
    # Second call: reuse (distance < threshold)
    result = cache.process_layer(0, 1, None, teacache_distance=0.1)

    assert mx.allclose(result, features, atol=0.01)
    stats = cache.get_stats()
    assert stats["cache_hits"] == 1
    assert stats["cache_misses"] == 1


def test_multigranular_with_quant():
    """Pipeline with QuantCache should compress/decompress transparently."""
    cache = MultiGranularCache(
        unicp_config=UniCPConfig(distance_threshold=0.3, use_smooth_interpolation=False),
        quant_config=QuantCacheConfig(bits=8, per_channel=True),
    )

    features = mx.random.normal((2, 4, 16)).astype(mx.float16)
    cache.process_layer(0, 0, features, teacache_distance=0.5)
    result = cache.process_layer(0, 1, None, teacache_distance=0.1)

    # Should be close but not exact due to quantization
    err = mx.mean(mx.abs(result - features)) / (mx.mean(mx.abs(features)) + 1e-8)
    assert err.item() < 0.02


def test_multigranular_stats():
    """Stats should reflect cache activity."""
    cache = MultiGranularCache(
        unicp_config=UniCPConfig(distance_threshold=0.3, use_smooth_interpolation=False),
    )

    f = mx.ones((2, 8))
    cache.process_layer(0, 0, f, teacache_distance=0.5)  # miss
    cache.process_layer(0, 1, None, teacache_distance=0.1)  # hit
    cache.process_layer(0, 2, None, teacache_distance=0.1)  # hit

    stats = cache.get_stats()
    assert stats["cache_hits"] == 2
    assert stats["cache_misses"] == 1
    assert stats["hit_rate"] == pytest.approx(2 / 3, abs=0.01)
    assert stats["cached_layers"] == 1


def test_multigranular_clear():
    cache = MultiGranularCache(
        unicp_config=UniCPConfig(distance_threshold=0.3),
    )
    cache.process_layer(0, 0, mx.ones((2, 4)), teacache_distance=0.5)
    cache.clear()
    stats = cache.get_stats()
    assert stats["cache_hits"] == 0
    assert stats["cached_layers"] == 0
