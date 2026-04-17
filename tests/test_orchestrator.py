"""Tests for B23 Orchestrator + PISA block approximation."""

import mlx.core as mx

from mlx_diffusion_kit.cache.teacache import TeaCacheConfig
from mlx_diffusion_kit.gating.tgate import TGateConfig
from mlx_diffusion_kit.orchestrator import (
    BlockStrategy,
    DiffusionOptimizer,
    OrchestratorConfig,
    PISAConfig,
)
from mlx_diffusion_kit.tokens.tome import ToMeConfig


def test_single_step_no_teacache_no_tgate():
    """Single-step models should never use TeaCache or T-GATE."""
    cfg = OrchestratorConfig(
        teacache=TeaCacheConfig(),
        tgate=TGateConfig(gate_step=3),
        is_single_step=True,
    )
    opt = DiffusionOptimizer(cfg)

    x = mx.ones((1, 16, 64))

    # TeaCache: always compute for single-step
    assert opt.should_compute_step(0, x) is True
    assert opt.should_compute_step(10, x) is True

    # T-GATE: always compute cross-attn for single-step
    assert opt.should_compute_cross_attn(0, 0) is True
    assert opt.should_compute_cross_attn(0, 100) is True

    # No internal states allocated
    assert opt.teacache_state is None
    assert opt.tgate_state is None


def test_multi_step_teacache_active():
    """Multi-step models should use TeaCache."""
    cfg = OrchestratorConfig(
        teacache=TeaCacheConfig(rel_l1_thresh=0.5),
        is_single_step=False,
    )
    opt = DiffusionOptimizer(cfg)

    x = mx.ones((1, 16, 64))

    # First step: always compute
    assert opt.should_compute_step(0, x) is True
    opt.update_step_cache(x, x * 2)

    # Same input: should skip
    assert opt.should_compute_step(1, x) is False

    # Cached output available
    cached = opt.get_cached_output()
    assert cached is not None


def test_multi_step_tgate():
    """T-GATE should gate cross-attention after gate_step."""
    cfg = OrchestratorConfig(
        tgate=TGateConfig(gate_step=3),
        is_single_step=False,
    )
    opt = DiffusionOptimizer(cfg)

    # Before gate_step: compute cross-attn
    assert opt.should_compute_cross_attn(0, 0) is True
    assert opt.should_compute_cross_attn(0, 2) is True

    # At/after gate_step: use cache
    assert opt.should_compute_cross_attn(0, 3) is False
    assert opt.should_compute_cross_attn(0, 10) is False


def test_tgate_cache_store_retrieve():
    """T-GATE should store and retrieve cross-attention outputs."""
    cfg = OrchestratorConfig(
        tgate=TGateConfig(gate_step=2),
        is_single_step=False,
    )
    opt = DiffusionOptimizer(cfg)

    val = mx.ones((2, 4)) * 0.42
    opt.cache_cross_attn(0, val)
    retrieved = opt.get_cached_cross_attn(0)
    assert retrieved is not None
    assert mx.allclose(retrieved, val)

    # Layer not cached
    assert opt.get_cached_cross_attn(99) is None


def test_pisa_block_strategy():
    """PISA should mark low-sensitivity blocks as APPROXIMATE."""
    # 5 blocks, approx_ratio=0.6 → 3 blocks approximated
    # Sorted scores: 0.02, 0.05, 0.1, 0.8, 0.9
    # Bottom 3: blocks 4(0.02), 2(0.05), 0(0.1)
    scores = {0: 0.1, 1: 0.9, 2: 0.05, 3: 0.8, 4: 0.02}
    cfg = OrchestratorConfig(
        pisa=PISAConfig(approx_ratio=0.6, sensitivity_scores=scores),
        num_blocks=5,
    )
    opt = DiffusionOptimizer(cfg)

    # Low sensitivity blocks → APPROXIMATE
    assert opt.get_block_strategy(4, 0) == BlockStrategy.APPROXIMATE  # 0.02
    assert opt.get_block_strategy(2, 0) == BlockStrategy.APPROXIMATE  # 0.05
    assert opt.get_block_strategy(0, 0) == BlockStrategy.APPROXIMATE  # 0.1

    # High sensitivity blocks → COMPUTE
    assert opt.get_block_strategy(1, 0) == BlockStrategy.COMPUTE  # 0.9
    assert opt.get_block_strategy(3, 0) == BlockStrategy.COMPUTE  # 0.8


def test_pisa_disabled():
    """PISA disabled → all blocks COMPUTE."""
    scores = {0: 0.01, 1: 0.01, 2: 0.01}
    cfg = OrchestratorConfig(
        pisa=PISAConfig(approx_ratio=0.5, sensitivity_scores=scores, enabled=False),
    )
    opt = DiffusionOptimizer(cfg)

    for i in range(3):
        assert opt.get_block_strategy(i, 0) == BlockStrategy.COMPUTE


def test_pisa_no_scores():
    """PISA without sensitivity scores → all blocks COMPUTE."""
    cfg = OrchestratorConfig(pisa=PISAConfig(approx_ratio=0.5))
    opt = DiffusionOptimizer(cfg)
    assert opt.get_block_strategy(0, 0) == BlockStrategy.COMPUTE


def test_tome_merge_unmerge():
    """Orchestrator should merge and unmerge tokens via ToMe."""
    cfg = OrchestratorConfig(tome=ToMeConfig(merge_ratio=0.5))
    opt = DiffusionOptimizer(cfg)

    tokens = mx.random.normal((2, 16, 32))
    merged = opt.merge_tokens(tokens)
    assert merged.shape == (2, 8, 32)

    unmerged = opt.unmerge_tokens(merged)
    assert unmerged.shape == (2, 16, 32)


def test_tome_not_configured():
    """No ToMe config → passthrough."""
    cfg = OrchestratorConfig(tome=None)
    opt = DiffusionOptimizer(cfg)

    tokens = mx.random.normal((2, 16, 32))
    result = opt.merge_tokens(tokens)
    assert mx.array_equal(result, tokens)

    result2 = opt.unmerge_tokens(result)
    assert mx.array_equal(result2, tokens)


def test_composition_multi_step():
    """Test TeaCache + T-GATE + ToMe together."""
    cfg = OrchestratorConfig(
        teacache=TeaCacheConfig(rel_l1_thresh=0.5),
        tgate=TGateConfig(gate_step=3),
        tome=ToMeConfig(merge_ratio=0.5),
        is_single_step=False,
    )
    opt = DiffusionOptimizer(cfg)

    x = mx.ones((1, 16, 64))

    # Step 0: compute
    assert opt.should_compute_step(0, x) is True
    merged = opt.merge_tokens(x)
    assert merged.shape[1] == 8
    assert opt.should_compute_cross_attn(0, 0) is True
    unmerged = opt.unmerge_tokens(merged)
    assert unmerged.shape == x.shape
    opt.update_step_cache(x, unmerged)

    # Step 1: TeaCache skip (identical input)
    assert opt.should_compute_step(1, x) is False

    # Step 5: T-GATE gate
    assert opt.should_compute_cross_attn(0, 5) is False


def test_skip_strategy_when_block_cached():
    """get_block_strategy should return SKIP when block is cached and TeaCache has residual."""
    cfg = OrchestratorConfig(
        teacache=TeaCacheConfig(),
        is_single_step=False,
    )
    opt = DiffusionOptimizer(cfg)

    # Manually populate block cache and TeaCache residual
    opt._block_cache[0] = mx.ones((2, 4))
    opt._teacache_state.cached_residual = mx.ones((2, 4))

    assert opt.get_block_strategy(0, 5) == BlockStrategy.SKIP
    # Block not in cache → COMPUTE
    assert opt.get_block_strategy(1, 5) == BlockStrategy.COMPUTE


def test_skip_not_returned_single_step():
    """Single-step models should never get SKIP strategy."""
    cfg = OrchestratorConfig(
        teacache=TeaCacheConfig(),
        is_single_step=True,
    )
    opt = DiffusionOptimizer(cfg)

    # Even if we manually put something in block_cache, single-step → no SKIP
    opt._block_cache[0] = mx.ones((2, 4))
    assert opt.get_block_strategy(0, 5) == BlockStrategy.COMPUTE


def test_reset():
    """Reset should clear all internal state."""
    cfg = OrchestratorConfig(
        teacache=TeaCacheConfig(),
        tgate=TGateConfig(),
        tome=ToMeConfig(),
        is_single_step=False,
    )
    opt = DiffusionOptimizer(cfg)

    x = mx.ones((1, 8, 32))
    opt.should_compute_step(0, x)
    opt.update_step_cache(x, x)
    opt.merge_tokens(x)
    opt.cache_cross_attn(0, x)

    opt.reset()

    assert opt.teacache_state is not None
    assert opt.teacache_state.prev_modulated_input is None
    assert opt.tgate_state is not None
    assert len(opt.tgate_state.cached_cross_attn) == 0


def test_spectral_cache_state_not_corrupted_by_update_step_cache():
    """Regression test for P8 Fix 1: update_step_cache must NOT force-refresh
    SpectralCache state.

    The caller's SpectralCache operates on an arbitrary intermediate feature
    stream chosen by the caller. The orchestrator's modulated_input is a
    different tensor (shape and semantics). If update_step_cache were to
    write modulated_input into the SpectralCache state, the next
    apply_spectral_cache call with real features would either crash on
    shape mismatch or silently produce corrupted output.

    This test uses deliberately different shapes for modulated_input and
    the features passed to apply_spectral_cache to make any coupling
    between them visible.
    """
    from mlx_diffusion_kit.cache.spectral_cache import SpectralCacheConfig

    cfg = OrchestratorConfig(
        spectral_cache=SpectralCacheConfig(
            low_freq_ratio=0.25,
            cache_interval_low=4,
            cache_interval_high=1,
        ),
        is_single_step=False,
    )
    opt = DiffusionOptimizer(cfg)

    # Use one shape for the caller's features, and a DIFFERENT shape for
    # the model's modulated_input. If update_step_cache touched SpectralCache,
    # it would store state with the modulated_input shape and the next
    # apply_spectral_cache call would fail or corrupt.
    features_shape = (1, 8, 64)
    modulated_shape = (1, 16, 32)

    features_0 = mx.random.normal(features_shape)
    modulated_0 = mx.random.normal(modulated_shape)
    output_0 = mx.random.normal(modulated_shape)

    # Step 0: caller applies spectral cache on features, then
    # update_step_cache is called on model input/output.
    y0 = opt.apply_spectral_cache(features_0, step_idx=0)
    assert y0.shape == features_shape
    opt.update_step_cache(modulated_0, output_0, step_idx=0)

    # Step 1: caller applies spectral cache again with the SAME shape as
    # before. If SpectralCache state had been force-refreshed with
    # modulated_input at step 0, this call would either crash on shape
    # mismatch (concatenation across bands of incompatible shapes) or
    # reconstruct from nonsense bands. Must succeed and match shape.
    features_1 = mx.random.normal(features_shape)
    y1 = opt.apply_spectral_cache(features_1, step_idx=1)
    assert y1.shape == features_shape
    assert mx.all(mx.isfinite(y1)).item()
