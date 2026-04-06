"""Tests for B15 TextEmbeddingCache."""

import mlx.core as mx
import pytest

from mlx_diffusion_kit.encoder.embedding_cache import TextEmbeddingCache


@pytest.fixture
def cache(tmp_path):
    return TextEmbeddingCache(cache_dir=tmp_path / "emb_cache")


def _make_encoder(call_count: list):
    """Mock encoder that tracks call count."""

    def encoder_fn(prompt, **kwargs):
        call_count.append(1)
        return mx.ones((1, 768))

    return encoder_fn


def test_cache_miss_then_hit(cache):
    calls = []
    enc = _make_encoder(calls)

    out1 = cache.get_or_compute("upscale 4x", enc)
    assert len(calls) == 1
    assert out1.shape == (1, 768)

    out2 = cache.get_or_compute("upscale 4x", enc)
    assert len(calls) == 1  # encoder NOT called again
    assert mx.allclose(out1, out2)


def test_different_prompts_different_keys(cache):
    calls = []
    enc = _make_encoder(calls)

    cache.get_or_compute("prompt A", enc)
    cache.get_or_compute("prompt B", enc)
    assert len(calls) == 2
    assert cache.cache_size() == 2


def test_clear(cache):
    calls = []
    enc = _make_encoder(calls)

    cache.get_or_compute("test", enc)
    assert cache.cache_size() == 1
    cache.clear()
    assert cache.cache_size() == 0

    # After clear, next call should recompute
    cache.get_or_compute("test", enc)
    assert len(calls) == 2


def test_cache_size_empty(cache):
    assert cache.cache_size() == 0


def test_different_encoder_id_different_key(cache):
    """Same prompt with different encoder_id should produce separate cache entries."""
    calls = []
    enc = _make_encoder(calls)

    cache.get_or_compute("upscale 4x", enc, encoder_id="t5-xxl")
    cache.get_or_compute("upscale 4x", enc, encoder_id="clip-l")
    assert len(calls) == 2
    assert cache.cache_size() == 2


def test_same_encoder_id_same_key(cache):
    """Same prompt + same encoder_id should cache hit."""
    calls = []
    enc = _make_encoder(calls)

    cache.get_or_compute("upscale 4x", enc, encoder_id="t5-xxl")
    cache.get_or_compute("upscale 4x", enc, encoder_id="t5-xxl")
    assert len(calls) == 1
    assert cache.cache_size() == 1
