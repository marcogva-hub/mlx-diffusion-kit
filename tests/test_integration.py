"""Integration tests: composition of Phase 1+2+3 components together.

Scenario 1: Multi-step DiT (simulates SparkVSR)
Scenario 2: Single-step DiT (simulates SeedVR2)
Scenario 3: VAE chunked decode
"""

import mlx.core as mx

from mlx_diffusion_kit.cache.smooth_cache import InterpolationMode, SmoothCacheConfig
from mlx_diffusion_kit.cache.teacache import TeaCacheConfig, load_coefficients
from mlx_diffusion_kit.encoder.embedding_cache import TextEmbeddingCache
from mlx_diffusion_kit.gating.tgate import TGateConfig
from mlx_diffusion_kit.orchestrator import (
    BlockStrategy,
    DiffusionOptimizer,
    OrchestratorConfig,
    PISAConfig,
)
from mlx_diffusion_kit.tokens.tome import ToMeConfig
from mlx_diffusion_kit.vae.wavelet_cache import WaveletVAECache, chunked_decode_with_cache


# ---------------------------------------------------------------------------
# Scenario 1 — Multi-step DiT (SparkVSR-like: 20 steps, CogVideoX backbone)
# ---------------------------------------------------------------------------


class TestMultiStepDiT:
    """Simulate a 20-step SparkVSR run with TeaCache + T-GATE + ToMe."""

    def setup_method(self):
        self.n_steps = 20
        self.n_tokens = 16
        self.dim = 64
        self.n_layers = 4
        self.gate_step = 5

        self.cfg = OrchestratorConfig(
            teacache=TeaCacheConfig(rel_l1_thresh=0.5),
            tgate=TGateConfig(gate_step=self.gate_step),
            tome=ToMeConfig(merge_ratio=0.5),
            is_single_step=False,
            num_blocks=self.n_layers,
        )
        self.opt = DiffusionOptimizer(self.cfg)

    def test_teacache_skips_steps(self):
        """TeaCache should skip some steps when inputs are similar."""
        computed_steps = 0
        x = mx.ones((1, self.n_tokens, self.dim))

        for step in range(self.n_steps):
            # Slowly diverging input (simulates denoising convergence)
            noise = mx.random.normal(x.shape) * (0.01 * step)
            modulated = x + noise

            if self.opt.should_compute_step(step, modulated):
                computed_steps += 1
                # Simulate forward: identity + small perturbation
                output = modulated + mx.random.normal(modulated.shape) * 0.001
                self.opt.update_step_cache(modulated, output)

        # With similar inputs, should skip at least some steps
        assert computed_steps < self.n_steps, (
            f"Expected some skipped steps, but computed all {self.n_steps}"
        )

    def test_tgate_gates_cross_attn(self):
        """Cross-attention should not be computed after gate_step."""
        cross_attn_calls = 0

        for step in range(self.n_steps):
            for layer in range(self.n_layers):
                if self.opt.should_compute_cross_attn(layer, step):
                    cross_attn_calls += 1
                    # Cache the value for post-gate usage
                    val = mx.ones((1, 8, self.dim)) * step
                    self.opt.cache_cross_attn(layer, val)

        # Cross-attn only called for steps < gate_step across all layers
        expected = self.gate_step * self.n_layers
        assert cross_attn_calls == expected, (
            f"Expected {expected} cross-attn calls, got {cross_attn_calls}"
        )

    def test_tome_halves_tokens(self):
        """ToMe should reduce token count by merge_ratio."""
        x = mx.random.normal((1, self.n_tokens, self.dim))
        merged = self.opt.merge_tokens(x)
        assert merged.shape == (1, self.n_tokens // 2, self.dim)

        unmerged = self.opt.unmerge_tokens(merged)
        assert unmerged.shape == x.shape

    def test_embedding_cache_reuse(self, tmp_path):
        """TextEmbeddingCache should avoid recomputing for same prompt."""
        cache = TextEmbeddingCache(cache_dir=tmp_path / "emb")
        call_count = []

        def mock_t5(prompt, **kw):
            call_count.append(1)
            return mx.random.normal((1, 128, 1024))

        prompt = "enhance video quality 4x"
        emb1 = cache.get_or_compute(prompt, mock_t5)
        emb2 = cache.get_or_compute(prompt, mock_t5)

        assert len(call_count) == 1
        assert mx.allclose(emb1, emb2)

    def test_full_pipeline_composition(self, tmp_path):
        """Full pipeline: embedding + TeaCache + ToMe + T-GATE together."""
        # 1. Text embedding (cached)
        emb_cache = TextEmbeddingCache(cache_dir=tmp_path / "emb")
        calls = []
        emb = emb_cache.get_or_compute(
            "upscale", lambda p, **kw: (calls.append(1), mx.ones((1, 8, 64)))[1]
        )
        assert len(calls) == 1

        # 2. Run steps
        computed = 0
        cross_attn_after_gate = 0
        x = mx.ones((1, self.n_tokens, self.dim))

        for step in range(10):
            modulated = x + mx.random.normal(x.shape) * 0.005

            if self.opt.should_compute_step(step, modulated):
                computed += 1
                merged = self.opt.merge_tokens(modulated)
                assert merged.shape[1] == self.n_tokens // 2

                for layer in range(self.n_layers):
                    if not self.opt.should_compute_cross_attn(layer, step):
                        cross_attn_after_gate += 1

                unmerged = self.opt.unmerge_tokens(merged)
                self.opt.update_step_cache(modulated, unmerged)

        # Sanity: not all steps computed, gating happened
        assert computed < 10
        assert cross_attn_after_gate > 0


# ---------------------------------------------------------------------------
# Scenario 2 — Single-step DiT (SeedVR2-like)
# ---------------------------------------------------------------------------


class TestSingleStepDiT:
    """Simulate a single-step SeedVR2 run with ToMe + PISA."""

    def setup_method(self):
        scores = {i: 0.1 * i for i in range(10)}  # 0: 0.0, 9: 0.9
        self.cfg = OrchestratorConfig(
            tome=ToMeConfig(merge_ratio=0.5),
            pisa=PISAConfig(approx_ratio=0.3, sensitivity_scores=scores),
            is_single_step=True,
            num_blocks=10,
        )
        self.opt = DiffusionOptimizer(self.cfg)

    def test_no_teacache_no_tgate(self):
        """Single-step: TeaCache and T-GATE should be inactive."""
        x = mx.ones((1, 16, 64))
        assert self.opt.should_compute_step(0, x) is True
        assert self.opt.teacache_state is None
        assert self.opt.tgate_state is None
        assert self.opt.should_compute_cross_attn(0, 0) is True

    def test_tome_active(self):
        x = mx.random.normal((1, 20, 64))
        merged = self.opt.merge_tokens(x)
        assert merged.shape == (1, 10, 64)

    def test_pisa_approximates_low_sensitivity(self):
        """Bottom 30% sensitivity blocks should be APPROXIMATE."""
        # scores: 0→0.0, 1→0.1, 2→0.2, ... 9→0.9
        # Bottom 30% = 3 blocks: 0(0.0), 1(0.1), 2(0.2)
        approx_count = 0
        compute_count = 0
        for i in range(10):
            strategy = self.opt.get_block_strategy(i, 0)
            if strategy == BlockStrategy.APPROXIMATE:
                approx_count += 1
            else:
                compute_count += 1

        assert approx_count == 3
        assert compute_count == 7

    def test_tome_plus_pisa(self):
        """ToMe + PISA should compose: tokens merged, blocks stratified."""
        x = mx.random.normal((1, 16, 32))
        merged = self.opt.merge_tokens(x)
        assert merged.shape[1] == 8

        # Simulate per-block execution
        outputs = []
        for block_idx in range(10):
            strategy = self.opt.get_block_strategy(block_idx, 0)
            if strategy == BlockStrategy.APPROXIMATE:
                outputs.append(merged * 0.99)  # Identity + small scale
            else:
                outputs.append(merged + mx.random.normal(merged.shape) * 0.01)

        # Unmerge final output
        final = self.opt.unmerge_tokens(outputs[-1])
        assert final.shape == x.shape


# ---------------------------------------------------------------------------
# Scenario 3 — VAE Chunked Decode
# ---------------------------------------------------------------------------


class TestVAEChunkedDecode:
    """Test WaveletVAECache with chunked temporal decoding."""

    def test_states_propagate_between_chunks(self):
        """Conv states from chunk N should be available in chunk N+1."""
        cache = WaveletVAECache()
        n_layers = 5
        chunk_states_seen = []

        def mock_decoder(latent_chunk, c):
            # Record which layers have cached states
            seen = {i: c.get_state(i) is not None for i in range(n_layers)}
            chunk_states_seen.append(seen)

            # Produce new states
            new_states = {}
            for i in range(n_layers):
                new_states[i] = latent_chunk[:, :, :1, :, :]  # Halo slice
            decoded = latent_chunk  # Identity decode
            return decoded, new_states

        # 4 chunks: [B=1, C=4, T=2, H=8, W=8]
        chunks = [mx.random.normal((1, 4, 2, 8, 8)) for _ in range(4)]
        result = chunked_decode_with_cache(mock_decoder, chunks, cache)

        # Output concatenated on temporal dim
        assert result.shape == (1, 4, 8, 8, 8)  # T = 2*4 = 8

        # Chunk 0: no states
        assert not any(chunk_states_seen[0].values())
        # Chunk 1+: all states available
        for i in range(1, 4):
            assert all(chunk_states_seen[i].values()), f"Chunk {i} missing states"

    def test_cache_accumulates_states(self):
        cache = WaveletVAECache()

        def mock_decoder(chunk, c):
            new_states = {0: chunk, 1: chunk * 2}
            return mx.expand_dims(chunk, axis=2), new_states

        chunks = [mx.ones((1, 4, 8, 8)), mx.ones((1, 4, 8, 8)) * 2]
        chunked_decode_with_cache(mock_decoder, chunks, cache)

        # After 2 chunks, cache should have states from the last chunk
        s0 = cache.get_state(0)
        s1 = cache.get_state(1)
        assert s0 is not None
        assert s1 is not None
        # States should reflect the second chunk's values
        assert mx.allclose(s0, mx.ones((1, 4, 8, 8)) * 2)
        assert mx.allclose(s1, mx.ones((1, 4, 8, 8)) * 4)

    def test_empty_cache_at_start(self):
        cache = WaveletVAECache()
        assert cache.num_cached() == 0
        for i in range(10):
            assert cache.get_state(i) is None


# ---------------------------------------------------------------------------
# Scenario 4 — Multi-step DiT with SmoothCache (TeaCache + interpolation)
# ---------------------------------------------------------------------------


class TestSmoothCacheIntegration:
    """Verify SmoothCache interpolates skipped steps instead of reusing stale cache."""

    def test_interpolated_output_differs_from_raw_cache(self):
        """When TeaCache skips, SmoothCache should produce interpolated output
        that differs from the raw cached_residual."""
        # Setup: TeaCache (aggressive threshold) + SmoothCache LINEAR
        cfg_with_smooth = OrchestratorConfig(
            teacache=TeaCacheConfig(rel_l1_thresh=999.0, max_consecutive_cached=10),
            smooth_cache=SmoothCacheConfig(mode=InterpolationMode.LINEAR),
            is_single_step=False,
        )
        opt = DiffusionOptimizer(cfg_with_smooth)

        dim = 32
        n_tokens = 8

        # Step 0: compute → features = 1.0
        x0 = mx.ones((1, n_tokens, dim))
        assert opt.should_compute_step(0, x0) is True
        out0 = mx.ones((1, n_tokens, dim)) * 1.0
        opt.update_step_cache(x0, out0, step_idx=0)

        # Step 1: skip (threshold=999 → always skip after first)
        x1 = mx.ones((1, n_tokens, dim)) * 1.001
        assert opt.should_compute_step(1, x1) is False

        # With only 1 history entry, smooth cache returns that entry
        cached_1 = opt.get_cached_output(step_idx=1)
        assert cached_1 is not None

        # Force-compute step 5 to give SmoothCache a second data point
        opt._teacache_state.accumulated_distance = 999.0  # Force compute
        opt._teacache_state.consecutive_cached = 0
        x5 = mx.ones((1, n_tokens, dim)) * 1.001
        assert opt.should_compute_step(5, x5) is True
        out5 = mx.ones((1, n_tokens, dim)) * 5.0
        opt.update_step_cache(x5, out5, step_idx=5)

        # Now skip step 7 — SmoothCache should interpolate between (0, 1.0) and (5, 5.0)
        x7 = mx.ones((1, n_tokens, dim)) * 1.001
        # Force the TeaCache state so it will skip
        opt._teacache_state.accumulated_distance = 0.0
        opt._teacache_state.consecutive_cached = 0
        assert opt.should_compute_step(7, x7) is False

        interpolated = opt.get_cached_output(step_idx=7)
        raw_cached = opt.teacache_state.cached_residual

        assert interpolated is not None
        assert raw_cached is not None
        # Interpolated should differ from raw cached residual
        assert not mx.allclose(interpolated, raw_cached, atol=0.01), (
            "SmoothCache interpolation should differ from raw TeaCache cache"
        )

    def test_smooth_cache_taylor_1_integration(self):
        """Taylor-1 extrapolation should produce outputs that follow the trend."""
        cfg = OrchestratorConfig(
            teacache=TeaCacheConfig(rel_l1_thresh=999.0, max_consecutive_cached=10),
            smooth_cache=SmoothCacheConfig(mode=InterpolationMode.TAYLOR_1),
            is_single_step=False,
        )
        opt = DiffusionOptimizer(cfg)

        # Record two computed steps with increasing features
        x = mx.ones((1, 4, 16))
        opt.should_compute_step(0, x)
        opt.update_step_cache(x, mx.ones((1, 4, 16)) * 2.0, step_idx=0)

        # Force compute for step 5
        opt._teacache_state.accumulated_distance = 999.0
        opt._teacache_state.consecutive_cached = 0
        opt.should_compute_step(5, x)
        opt.update_step_cache(x, mx.ones((1, 4, 16)) * 7.0, step_idx=5)

        # Extrapolate to step 10: d1 = (7-2)/5 = 1.0/step, dt=5 → 7+5=12
        output_10 = opt.get_cached_output(step_idx=10)
        expected = mx.ones((1, 4, 16)) * 12.0
        assert mx.allclose(output_10, expected, atol=1e-4)

    def test_smooth_cache_reset(self):
        """Reset should clear SmoothCache history."""
        cfg = OrchestratorConfig(
            teacache=TeaCacheConfig(),
            smooth_cache=SmoothCacheConfig(),
            is_single_step=False,
        )
        opt = DiffusionOptimizer(cfg)

        x = mx.ones((1, 4, 8))
        opt.should_compute_step(0, x)
        opt.update_step_cache(x, x, step_idx=0)
        assert len(opt.smooth_cache_state.history) == 1

        opt.reset()
        assert len(opt.smooth_cache_state.history) == 0


# ---------------------------------------------------------------------------
# Scenario 5 — End-to-end compose of 4 rebuilt components (P7 + P8)
# ---------------------------------------------------------------------------


def test_rebuilt_components_compose_without_desync():
    """P9.1 regression: simulate a 10-step multi-step DiT pass with TeaCache,
    FBCache, SpectralCache, and DeepCache all active through
    DiffusionOptimizer. Verify no cross-component state corruption, shapes
    remain coherent, and reset() cleans everything.

    This is the scenario that the P7 SpectralCache wiring bug (fixed in P8
    Fix 1) would have broken. It stays green as long as the orchestrator
    wires the rebuilt components correctly.
    """
    from mlx_diffusion_kit.cache.teacache import TeaCacheConfig
    from mlx_diffusion_kit.cache.fb_cache import FBCacheConfig
    from mlx_diffusion_kit.cache.spectral_cache import SpectralCacheConfig
    from mlx_diffusion_kit.cache.deep_cache import DeepCacheConfig

    opt_cfg = OrchestratorConfig(
        teacache=TeaCacheConfig(rel_l1_thresh=0.1),
        fbcache=FBCacheConfig(rel_l1_thresh=0.1),
        spectral_cache=SpectralCacheConfig(
            low_freq_ratio=0.25,
            cache_interval_low=3,
            cache_interval_high=1,
        ),
        deep_cache=DeepCacheConfig(cache_interval=2),
        is_single_step=False,
        num_blocks=8,
        total_steps=10,
    )
    opt = DiffusionOptimizer(opt_cfg)

    # Shapes mimic a small DiT pass — kept deliberately small so the test
    # runs in the millisecond range.
    modulated_shape = (1, 256, 64)   # latent input to the model
    feature_shape = (1, 128, 128)    # intermediate features for SpectralCache
    fb_shape = (1, 64, 128)          # first-block output
    deep_shape = (1, 32, 128)        # deep-branch output

    for step in range(10):
        modulated = mx.random.normal(modulated_shape)
        features = mx.random.normal(feature_shape)
        fb_out = mx.random.normal(fb_shape)
        full_out = mx.random.normal(fb_shape)
        deep_out = mx.random.normal(deep_shape)

        # ---- Step-level (TeaCache) ----
        compute = opt.should_compute_step(step_idx=step, modulated_input=modulated)
        assert isinstance(compute, bool)

        # ---- Block-level (FBCache) ----
        compute_remaining = opt.should_compute_remaining_blocks(fb_out, step_idx=step)
        assert isinstance(compute_remaining, bool)
        if not compute_remaining:
            reconstructed = opt.fbcache_reconstruct_output(fb_out)
            assert reconstructed.shape == fb_shape
        else:
            # Record what the "remaining blocks" would have produced so
            # future skips have a residual to reuse.
            residual = full_out - fb_out
            opt.fbcache_update_residual(fb_out, residual)

        # ---- Deep-branch (DeepCache) ----
        if opt.should_recompute_deep(step_idx=step):
            opt.store_deep_features(deep_out, step_idx=step)
        else:
            cached_deep = opt.get_cached_deep_features()
            assert cached_deep is not None, f"DeepCache cache missing at step {step}"
            assert cached_deep.shape == deep_shape

        # ---- Feature-level freq caching (SpectralCache) ----
        # This is the path that was broken pre-P8-Fix-1. The caller's
        # feature stream (feature_shape) is independent from modulated_input
        # (modulated_shape) — they must stay independent through all steps.
        cached_features = opt.apply_spectral_cache(features, step_idx=step)
        assert cached_features.shape == feature_shape, (
            f"SpectralCache desync at step {step}: expected {feature_shape}, "
            f"got {cached_features.shape}"
        )
        assert mx.all(mx.isfinite(cached_features)).item(), (
            f"SpectralCache produced non-finite values at step {step}"
        )

        # ---- Orchestrator step-cache update ----
        # Per P8 Fix 1, this must NOT touch SpectralCache state.
        if compute:
            opt.update_step_cache(
                modulated_input=modulated,
                output=mx.random.normal(modulated_shape),
                step_idx=step,
            )

    # ---- Reset cleans everything ----
    opt.reset()
    assert opt._teacache_state.prev_modulated_input is None
    assert opt._fbcache_state.cached_residual is None
    assert opt._spectral_cache_state.cached_low_freq is None
    assert opt._deep_cache_state.cached_deep_features is None
