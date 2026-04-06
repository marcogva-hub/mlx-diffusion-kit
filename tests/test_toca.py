"""Tests for B7 ToCa (Token Caching)."""

import mlx.core as mx

from mlx_diffusion_kit.tokens.toca import ToCaConfig, TokenCacheManager


def test_first_step_all_dynamic():
    """No cache → all tokens are dynamic."""
    cfg = ToCaConfig(cache_ratio=0.5)
    mgr = TokenCacheManager(cfg)
    tokens = mx.random.normal((2, 16, 32))
    mask = mgr.identify_stable_tokens(tokens, step_idx=0)
    assert mask.shape == (2, 16)
    assert not mx.any(mask).item()


def test_identical_tokens_stable():
    """Identical tokens between steps → stable (cacheable)."""
    cfg = ToCaConfig(cache_ratio=0.5, similarity_threshold=0.9, use_attention_scores=False)
    mgr = TokenCacheManager(cfg)
    tokens = mx.ones((1, 10, 32))

    # First step: all dynamic, update cache
    mask0 = mgr.identify_stable_tokens(tokens, step_idx=0)
    assert not mx.any(mask0).item()
    mgr.update_cache(tokens, mask0, step_idx=0)

    # Second step with same tokens: some should be stable
    mask1 = mgr.identify_stable_tokens(tokens, step_idx=1)
    assert mx.any(mask1).item(), "Identical tokens should be marked stable"


def test_very_different_tokens_all_dynamic():
    """Very different tokens between steps → all dynamic."""
    cfg = ToCaConfig(cache_ratio=0.5, similarity_threshold=0.99, use_attention_scores=False)
    mgr = TokenCacheManager(cfg)
    t1 = mx.ones((1, 10, 32))
    mask0 = mgr.identify_stable_tokens(t1, step_idx=0)
    mgr.update_cache(t1, mask0, step_idx=0)

    t2 = -mx.ones((1, 10, 32))  # Opposite direction → cos_sim ≈ -1
    mask1 = mgr.identify_stable_tokens(t2, step_idx=1)
    assert not mx.any(mask1).item()


def test_cache_ratio_respected():
    """At most cache_ratio fraction of tokens should be stable."""
    cfg = ToCaConfig(cache_ratio=0.3, similarity_threshold=0.0, use_attention_scores=False)
    mgr = TokenCacheManager(cfg)
    tokens = mx.ones((1, 20, 16))

    mgr.identify_stable_tokens(tokens, step_idx=0)
    mgr.update_cache(tokens, mx.zeros((1, 20), dtype=mx.bool_), step_idx=0)

    mask = mgr.identify_stable_tokens(tokens, step_idx=1)
    n_stable = mx.sum(mask.astype(mx.int32)).item()
    max_allowed = int(20 * 0.3)
    assert n_stable <= max_allowed, f"Got {n_stable} stable, max allowed {max_allowed}"


def test_recompute_interval_forces_dynamic():
    """After recompute_interval steps, all tokens should be dynamic."""
    cfg = ToCaConfig(cache_ratio=0.5, recompute_interval=3, use_attention_scores=False)
    mgr = TokenCacheManager(cfg)
    tokens = mx.ones((1, 10, 16))

    # Step 0: full compute
    mask0 = mgr.identify_stable_tokens(tokens, step_idx=0)
    mgr.update_cache(tokens, mask0, step_idx=0)

    # Step 1: some stable
    mask1 = mgr.identify_stable_tokens(tokens, step_idx=1)

    # Step 3: forced recompute (delta = 3 >= interval=3)
    mask3 = mgr.identify_stable_tokens(tokens, step_idx=3)
    assert not mx.any(mask3).item(), "Recompute interval should force all dynamic"


def test_apply_cache_reconstructs():
    """apply_cache should combine cached and computed tokens correctly."""
    cfg = ToCaConfig()
    mgr = TokenCacheManager(cfg)

    cached = mx.ones((1, 4, 8)) * 10.0
    computed = mx.ones((1, 4, 8)) * 20.0
    mgr._state.cached_tokens = cached

    mask = mx.array([[True, False, True, False]])
    result = mgr.apply_cache(computed, mask)

    assert result.shape == (1, 4, 8)
    # Stable tokens (0, 2) should have cached value (10)
    assert mx.allclose(result[0, 0], mx.ones((8,)) * 10.0)
    assert mx.allclose(result[0, 2], mx.ones((8,)) * 10.0)
    # Dynamic tokens (1, 3) should have computed value (20)
    assert mx.allclose(result[0, 1], mx.ones((8,)) * 20.0)
    assert mx.allclose(result[0, 3], mx.ones((8,)) * 20.0)


def test_get_dynamic_indices():
    cfg = ToCaConfig()
    mgr = TokenCacheManager(cfg)
    mask = mx.array([[True, False, True, False, False]])  # 3 dynamic
    indices = mgr.get_dynamic_indices(mask)
    assert indices.shape[1] == 3  # 3 dynamic tokens


def test_reset():
    cfg = ToCaConfig()
    mgr = TokenCacheManager(cfg)
    mgr._state.cached_tokens = mx.ones((1, 4, 8))
    mgr._state.last_full_step = 5
    mgr.reset()
    assert mgr._state.cached_tokens is None
    assert mgr._state.last_full_step == -1


def test_disabled():
    cfg = ToCaConfig(enabled=False)
    mgr = TokenCacheManager(cfg)
    tokens = mx.ones((1, 8, 16))
    mgr._state.cached_tokens = tokens
    mask = mgr.identify_stable_tokens(tokens, step_idx=1)
    assert not mx.any(mask).item()
