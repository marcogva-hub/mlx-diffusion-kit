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


# --- Streaming / Buffer tests ---

def test_output_buffer_mode():
    """Buffer mode should produce identical result to concat mode."""
    from mlx_diffusion_kit.vae.wavelet_cache import estimate_output_shape, preallocate_output_buffer

    cache1 = WaveletVAECache()
    cache2 = WaveletVAECache()

    def mock_decode(chunk, c):
        decoded = mx.expand_dims(chunk, axis=2)  # [B,C,1,H,W]
        return decoded, {}

    # 3 chunks: [1, 4, 8, 8] → decoded [1, 4, 1, 8, 8] each
    chunks = [mx.ones((1, 4, 8, 8)) * (i + 1) for i in range(3)]

    # Concat mode
    result_concat = chunked_decode_with_cache(mock_decode, chunks, cache1)

    # Buffer mode — need 5D chunks for estimate_output_shape
    chunks_5d = [mx.ones((1, 4, 1, 8, 8)) * (i + 1) for i in range(3)]
    shape = estimate_output_shape(chunks_5d)
    buf = preallocate_output_buffer(shape)
    result_buffer = chunked_decode_with_cache(mock_decode, chunks, cache2, output_buffer=buf)

    assert result_buffer.shape == result_concat.shape
    assert mx.allclose(result_buffer, result_concat)


def test_callback_mode():
    """Callback should be called once per chunk."""
    cache = WaveletVAECache()
    callback_log = []

    def mock_decode(chunk, c):
        return mx.expand_dims(chunk, axis=2), {}

    def my_callback(idx, decoded):
        callback_log.append((idx, decoded.shape))

    chunks = [mx.ones((1, 4, 8, 8)) for _ in range(4)]
    result = chunked_decode_with_cache(mock_decode, chunks, cache, callback=my_callback)

    assert len(callback_log) == 4
    assert callback_log[0][0] == 0
    assert callback_log[3][0] == 3


def test_estimate_output_shape():
    from mlx_diffusion_kit.vae.wavelet_cache import estimate_output_shape

    chunks = [mx.zeros((2, 3, 4, 16, 16)) for _ in range(3)]
    shape = estimate_output_shape(chunks, spatial_upsample=2, temporal_upsample=1)
    assert shape == (2, 3, 12, 32, 32)  # T=4*3=12, H=16*2=32, W=16*2=32


def test_estimate_output_shape_temporal_upsample():
    from mlx_diffusion_kit.vae.wavelet_cache import estimate_output_shape

    chunks = [mx.zeros((1, 4, 2, 8, 8)) for _ in range(2)]
    shape = estimate_output_shape(chunks, temporal_upsample=3)
    assert shape == (1, 4, 12, 8, 8)  # T=2*2*3=12
