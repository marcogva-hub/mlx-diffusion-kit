"""Integration tests: composition of Phase 1+2+3 components together.

Scenario 1: Multi-step DiT (simulates SparkVSR)
Scenario 2: Single-step DiT (simulates SeedVR2)
Scenario 3: VAE chunked decode
"""

import mlx.core as mx

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
