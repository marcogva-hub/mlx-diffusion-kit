"""Tests for B17 WF-VAE Causal Cache."""

import mlx.core as mx
import pytest

from mlx_diffusion_kit.vae.wavelet_cache import (
    WaveletCacheConfig,
    WaveletVAECache,
    chunked_decode_with_cache,
)


def test_get_state_uncached_returns_none():
    cache = WaveletVAECache()
    assert cache.get_state(0) is None
    assert cache.get_state(99) is None


def test_set_then_get_state():
    cache = WaveletVAECache()
    state = mx.ones((4, 8))
    cache.set_state(0, state)
    retrieved = cache.get_state(0)
    assert retrieved is not None
    assert mx.array_equal(retrieved, state)


def test_clear():
    cache = WaveletVAECache()
    cache.set_state(0, mx.ones((2,)))
    cache.set_state(1, mx.ones((3,)))
    assert cache.num_cached() == 2
    cache.clear()
    assert cache.num_cached() == 0
    assert cache.get_state(0) is None


def test_max_cached_layers():
    cfg = WaveletCacheConfig(max_cached_layers=2)
    cache = WaveletVAECache(cfg)
    cache.set_state(0, mx.ones((2,)))
    cache.set_state(1, mx.ones((2,)))
    cache.set_state(2, mx.ones((2,)))  # Should be ignored
    assert cache.num_cached() == 2
    assert cache.get_state(2) is None


def test_max_cached_layers_allows_update():
    """Updating an existing layer should work even at capacity."""
    cfg = WaveletCacheConfig(max_cached_layers=1)
    cache = WaveletVAECache(cfg)
    cache.set_state(0, mx.ones((2,)))
    cache.set_state(0, mx.ones((2,)) * 5.0)  # Update existing
    assert cache.num_cached() == 1
    assert mx.allclose(cache.get_state(0), mx.ones((2,)) * 5.0)


def test_disabled_cache():
    cfg = WaveletCacheConfig(enabled=False)
    cache = WaveletVAECache(cfg)
    cache.set_state(0, mx.ones((2,)))
    assert cache.get_state(0) is None
    assert cache.num_cached() == 0


def test_chunked_decode_propagates_states():
    """States from chunk N should be visible to decode_fn for chunk N+1."""
    cache = WaveletVAECache()
    call_log = []

    def mock_decode(chunk, c):
        # Record what states were available
        available = {i: c.get_state(i) is not None for i in range(3)}
        call_log.append(available)
        # Each chunk "produces" states for layers 0,1,2
        new_states = {i: chunk * (i + 1) for i in range(3)}
        # Decoded output: [B, C, T, H, W] — T=1 per chunk
        decoded = mx.expand_dims(chunk, axis=2)
        return decoded, new_states

    # 3 chunks, each [1, 4, 8, 8]
    chunks = [mx.ones((1, 4, 8, 8)) * (i + 1) for i in range(3)]
    result = chunked_decode_with_cache(mock_decode, chunks, cache)

    # Chunk 0: no states available
    assert not any(call_log[0].values())
    # Chunk 1: all states from chunk 0 available
    assert all(call_log[1].values())
    # Chunk 2: all states from chunk 1 available
    assert all(call_log[2].values())

    # Output concatenated along dim=2 (temporal)
    assert result.shape == (1, 4, 3, 8, 8)


def test_chunked_decode_empty_raises():
    cache = WaveletVAECache()
    with pytest.raises(ValueError):
        chunked_decode_with_cache(lambda c, ca: (c, {}), [], cache)


def test_chunked_decode_single_chunk():
    cache = WaveletVAECache()

    def mock_decode(chunk, c):
        return mx.expand_dims(chunk, axis=2), {}

    chunks = [mx.ones((1, 4, 8, 8))]
    result = chunked_decode_with_cache(mock_decode, chunks, cache)
    assert result.shape == (1, 4, 1, 8, 8)
