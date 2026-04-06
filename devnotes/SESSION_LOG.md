---
## [2026-04-06 11:00] Phase P0: Repo scaffolding

### Plan
- **Objective:** Create mlx-diffusion-kit repo structure with all packages, stubs, and config
- **Files to modify:** All new files (pyproject.toml, all __init__.py, orchestrator.py, stubs, tests)
- **Dependencies impacted:** None (greenfield)

### Changes made
- `pyproject.toml` — package config, deps: mlx>=0.25.0, optional mfa + dev [HIGH]
- `mlx_diffusion_kit/__init__.py` — version + __all__ [HIGH]
- `mlx_diffusion_kit/orchestrator.py` — DiffusionOptimizer stub [HIGH]
- All subpackage `__init__.py` — docstrings only [HIGH]
- `tests/test_smoke.py` — import + version + submodule tests [HIGH]
- `scripts/` — two stubs [HIGH]

### Dependency & regression check
- No existing code — greenfield setup

### Tech cost assessment
- N/A — scaffolding only

### Confidence
- Overall: [HIGH]
- Risks: none

---
## [2026-04-06 11:05] Phase P1.1: B15 TextEmbeddingCache

### Plan
- **Objective:** Disk-backed text embedding cache keyed by sha256(prompt)
- **Files to modify:** encoder/embedding_cache.py (new), encoder/__init__.py, __init__.py
- **Dependencies impacted:** None (new module)

### Changes made
- `encoder/embedding_cache.py` — TextEmbeddingCache class with get_or_compute, clear, cache_size [HIGH]
- `tests/test_embedding_cache.py` — 4 tests: miss→hit, different prompts, clear, empty [HIGH]

### Tech cost assessment
- I/O: One mx.load (mmap) per cache hit. mx.savez on miss. Negligible vs encoder runtime.
- Memory: No in-memory cache — always disk-backed.

### Confidence
- Overall: [HIGH]

---
## [2026-04-06 11:10] Phase P1.2: B13 FreeU filter

### Plan
- **Objective:** FFT-based skip connection re-weighting for UNet models
- **Files to modify:** quality/freeu.py (new), quality/__init__.py
- **Dependencies impacted:** None (new module)

### Changes made
- `quality/freeu.py` — FreeUConfig + freeu_filter using mx.fft.rfft/irfft [HIGH]
- `tests/test_freeu.py` — 5 tests: disabled passthrough, shapes, modification, finiteness [HIGH]

### Tech cost assessment
- Compute: 1 FFT + 1 iFFT per skip tensor. O(n log n) per spatial dim. Negligible vs conv layers.
- Memory: No extra buffers (in-place scaling + temporary freq tensor).

### Confidence
- Overall: [HIGH]

---
## [2026-04-06 11:15] Phase P1.3: B11 T-GATE

### Plan
- **Objective:** Cross-attention gating after convergence step
- **Files to modify:** gating/tgate.py (new), gating/__init__.py
- **Dependencies impacted:** None (new module)

### Changes made
- `gating/tgate.py` — TGateConfig, TGateState, tgate_forward, create_tgate_state [HIGH]
- `tests/test_tgate.py` — 5 tests: before/after gate, cached reuse, disabled, multi-layer [HIGH]

### Tech cost assessment
- Memory: One cached mx.array per layer (same size as cross-attn output). For 30 layers × (B,N,D) = modest.
- Compute: Zero after gate_step (dict lookup only).

### Confidence
- Overall: [HIGH]

---
## [2026-04-06 11:20] Phase P1.4: B1 TeaCache

### Plan
- **Objective:** Step-level caching with L1 distance + polynomial rescaling
- **Files to modify:** cache/teacache.py (new), cache/coefficients/cogvideox.json (new), cache/__init__.py
- **Dependencies impacted:** None (new module)

### Changes made
- `cache/teacache.py` — TeaCacheConfig, TeaCacheState, teacache_should_compute, teacache_update, load_coefficients [HIGH]
- `cache/coefficients/cogvideox.json` — Published CogVideoX-5B polynomial coefficients [HIGH]
- `tests/test_teacache.py` — 11 tests: first step, identical skip, different compute, max consecutive, poly scaling, load coeffs, windows, disabled [HIGH]

### Tech cost assessment
- Compute: One mean(abs(diff)) per step = O(n). Negligible vs DiT forward.
- Memory: Stores prev_modulated_input + cached_residual = 2× input size.

### Confidence
- Overall: [HIGH]
- Risks: Polynomial coefficients are model-specific. CogVideoX verified from paper. Other models need calibration.

---
## [2026-04-06 12:00] Phase P2.1: B8 Token Merging (ToMe)

### Plan
- **Objective:** Bipartite cosine matching → merge similar tokens (N→N/2)
- **Files to modify:** tokens/tome.py (new), tokens/__init__.py

### Changes made
- `tokens/tome.py` — ToMeConfig, MergeInfo, tome_merge, tome_unmerge, compute_proportional_bias [HIGH]
  - Bipartite partition (first n_src = src, rest = dst), cosine sim matching
  - MLERP norm-preserving merge, 3D and 4D (B,H,N,D) support
- `tests/test_tome.py` — 11 tests: merge/unmerge shape, ratio 0/0.5, identical/different tokens, proportional bias, MLERP norms, batch sizes, disabled, 4D, finite [HIGH]

### Tech cost assessment
- Compute: O(n_dst × n_src) similarity + O(n_dst) argmax per merge. For N=4096, ratio=0.5 → 2048×2048 sim matrix.
- Memory: similarity matrix n_dst × n_src. Temporary, freed after merge.
- Note: scatter-add loop in merge is O(n_dst) Python — acceptable for now, future optimization with mx.scatter_add if profiled as bottleneck.

### Confidence
- Overall: [HIGH]
- Risks: Python loop in merge step. Profile-driven optimization path identified.

---
## [2026-04-06 12:10] Phase P2.2: B17 WF-VAE Causal Cache

### Plan
- **Objective:** Conv state propagation for chunked temporal VAE decoding
- **Files to modify:** vae/wavelet_cache.py (new), vae/__init__.py

### Changes made
- `vae/wavelet_cache.py` — WaveletCacheConfig, WaveletVAECache, chunked_decode_with_cache [HIGH]
- `tests/test_wavelet_cache.py` — 9 tests: get/set, clear, max layers, disabled, state propagation, empty/single chunk [HIGH]

### Tech cost assessment
- Memory: One mx.array per cached conv layer. For 30 conv layers with typical hidden dims → modest.
- Compute: Dict lookup per layer per chunk. O(1).

### Confidence
- Overall: [HIGH]

---
## [2026-04-06 12:20] Phase P2.3: B14.2 Adaptive Stepping

### Plan
- **Objective:** Skip redundant diffusion steps based on output convergence
- **Files to modify:** scheduler/adaptive_stepping.py (new), scheduler/__init__.py
- **Files to modify:** scheduler/adaptive_stepping.py (new), scheduler/__init__.py

### Changes made
- `scheduler/adaptive_stepping.py` — AdaptiveStepConfig, AdaptiveStepScheduler [HIGH]
- `tests/test_adaptive_stepping.py` — 6 tests: identical skip, different no-skip, min_steps, reset, effective timesteps, disabled [HIGH]

### Tech cost assessment
- Compute: One MSE per step = O(n). Negligible vs model forward.
- Memory: Set of skipped indices. O(num_steps).

### Confidence
- Overall: [HIGH]

---
## [2026-04-06 13:00] Phase P3.1: B23 Orchestrator + PISA

### Plan
- **Objective:** Central orchestrator composing TeaCache, ToMe, T-GATE, PISA with BlockStrategy enum
- **Files to modify:** orchestrator.py (rewrite from stub), tests/test_smoke.py (fix API change)

### Changes made
- `orchestrator.py` — Full rewrite: BlockStrategy, PISAConfig, OrchestratorConfig, DiffusionOptimizer [HIGH]
- `tests/test_orchestrator.py` — 11 tests [HIGH]
- `tests/test_smoke.py:L22` — Fixed for new API [HIGH]

### Dependency & regression check
- Imports from cache, gating, quality, tokens — all verified compatible

### Confidence
- Overall: [HIGH]

---
## [2026-04-06 13:10] Phase P3.2: B9 DiffSparse (stub)

### Changes made
- `tokens/learned_sparsity.py` — DiffSparseConfig, DiffSparseRouter(nn.Module) stub [HIGH]
- `tests/test_learned_sparsity.py` — 6 tests [HIGH]

### Confidence
- Overall: [HIGH]

---
## [2026-04-06 13:20] Phase P_int: Integration tests

### Changes made
- `tests/test_integration.py` — 12 tests across 3 scenarios (multi-step, single-step, VAE) [HIGH]

### Confidence
- Overall: [HIGH]

---
## [2026-04-06 14:00] Phase P4.1: B4 SmoothCache + Taylor interpolation

### Plan
- **Objective:** Interpolation for skipped steps (Linear, Taylor-1, Taylor-2)
- **Files to modify:** cache/smooth_cache.py (new), orchestrator.py, cache/__init__.py, __init__.py, test_integration.py

### Changes made
- `cache/smooth_cache.py` — InterpolationMode enum, SmoothCacheConfig, SmoothCacheState, smooth_cache_record, smooth_cache_interpolate with 3 modes [HIGH]
- `orchestrator.py` — Added smooth_cache to OrchestratorConfig, SmoothCacheState init, record in update_step_cache, interpolate in get_cached_output [HIGH]
- `tests/test_smooth_cache.py` — 12 tests: linear exact, Taylor-1 exact, Taylor-2 parabola, fallback, pruning, disabled, empty, finite, shapes [HIGH]
- `tests/test_integration.py` — 3 new SmoothCache scenarios (interpolated differs from raw, Taylor-1 extrapolation, reset) [HIGH]

### Dependency & regression check
- orchestrator.py: `update_step_cache` and `get_cached_output` got `step_idx` parameter with default=0 — backward compatible [HIGH]
- All 23 existing orchestrator/integration tests pass without modification

### Tech cost assessment
- Compute: Linear interpolation O(n) elementwise. Taylor-1/2 same + 1-2 divisions.
- Memory: History buffer capped at 3 entries = 3× one feature tensor. Constant regardless of step count.

### Confidence
- Overall: [HIGH]

---
## [2026-04-06 15:00] Phase P4.2: B10 DDiT Scheduling + B22 Encoder Sharing

### Plan
- **Objective:** Dynamic patch scheduling per step + encoder block sharing across steps
- **Files to modify:** tokens/ddit_scheduling.py (new), cache/encoder_sharing.py (new), orchestrator.py, cache/__init__.py, tokens/__init__.py

### Changes made
- `tokens/ddit_scheduling.py` — DDiTScheduleConfig, DDiTScheduler with linear/cosine/step schedules, power-of-2 stride quantization [HIGH]
- `cache/encoder_sharing.py` — EncoderSharingConfig/State, should_recompute, get_cached, update [HIGH]
- `orchestrator.py` — Added ddit_schedule + encoder_sharing configs, DDiTScheduler + EncoderSharingState init, 4 new methods (should_recompute_encoder, get_cached_encoder_output, update_encoder_cache, get_patch_stride) [HIGH]
- `tests/test_ddit_scheduling.py` — 8 tests: linear/cosine/step schedules, powers of 2, reduction factor 2D/3D, disabled, warmup=0 [HIGH]
- `tests/test_encoder_sharing.py` — 7 tests: first step, interval=3, update/get, overwrite, disabled, empty, interval=1 [HIGH]

### Dependency & regression check
- orchestrator.py: OrchestratorConfig got `ddit_schedule`, `encoder_sharing`, `total_steps` fields with None/50 defaults — backward compatible
- All existing 97 tests pass without modification (new fields are Optional with defaults)

### Tech cost assessment
- DDiT: O(1) per step (math.cos + nearest_power_of_2)
- Encoder sharing: O(1) decision per step, one mx.array cached (size = encoder output)

### Confidence
- Overall: [HIGH]

---
## [2026-04-06 16:00] Phase P4.3: B14.1 DPM-Solver-v3

### Plan
- **Objective:** High-order ODE solver (orders 1-3) with log-SNR uniform timestep spacing
- **Files to modify:** scheduler/dpm_solver_v3.py (new), scheduler/__init__.py, __init__.py

### Changes made
- `scheduler/dpm_solver_v3.py` — NoiseSchedule (VP, alpha_bar, log_snr, inverse_log_snr), DPMSolverV3Config, DPMSolverV3 (step, 1st/2nd/3rd order), compute_optimal_timesteps, predict_type conversion (epsilon/v/x_start) [HIGH]
- `tests/test_dpm_solver_v3.py` — 11 tests: alpha_bar/log_snr monotonic, inverse roundtrip, timestep count, log-SNR uniform spacing, orders 1-3 finite, reset, predict_types [HIGH]

### Tech cost assessment
- NoiseSchedule init: O(T) cumprod (1000 betas). One-time.
- inverse_log_snr: O(T log T) binary search per target value. Called once at init.
- step: O(n) elementwise math per step. Negligible vs model forward.
- Memory: model_output_history capped at 3 entries (order 3).

### Confidence
- Overall: [HIGH]
- Risks: EMS coefficients (paper-specific) not yet implemented — current impl uses standard DPM-Solver multistep. EMS can be added as pre-calibrated per-model coefficients similar to TeaCache.
