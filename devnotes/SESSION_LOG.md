> **Historical note:** This session log records the development of mlx-diffusion-kit v0.1.0,
> built over a single session on 2026-04-06. All phases (P0 through P6.2) are documented below.

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

---
## [2026-04-06 17:00] Phase P4.4: B6 Multi-Granular Cache

### Plan
- **Objective:** BWCache (bandwidth-aware allocation) + UniCP (unified policy) + QuantCache (int8/int4 compression)
- **Files to modify:** cache/multigranular.py (new), orchestrator.py, cache/__init__.py

### Changes made
- `cache/multigranular.py` — 3 components + unified MultiGranularCache class [HIGH]
  - BWCacheAllocator: budget-aware per-layer step allocation, prefer_quality mode
  - UniCPPolicy: 3-signal decision (TeaCache distance, SmoothCache interpolation, BW budget)
  - QuantCache: int8/int4 per-channel/per-tensor compress/decompress
  - MultiGranularCache: pipeline combining all 3 + stats tracking
- `orchestrator.py` — Added multigranular config + MultiGranularCache lifecycle + property + reset [HIGH]
- `tests/test_multigranular.py` — 21 tests across all components [HIGH]

### Dependency & regression check
- Imports SmoothCache concepts (interpolation) by reference in UniCP, no direct code dependency
- orchestrator.py: MultiGranularConfig added to OrchestratorConfig with None default — backward compatible
- All 123 existing tests pass

### Tech cost assessment
- BWCache allocation: O(n_layers log n_layers) sort. One-time at init.
- UniCP decide: O(1) per (layer, step) pair.
- QuantCache compress: O(n) per tensor + one max reduction. Decompress: O(n) multiply.
- Memory: int8 = 50% of f16, int4 = 25% of f16 (packed as int8 in current impl, true int4 packing future).

### Confidence
- Overall: [HIGH]
- Note: int4 uses int8 dtype with reduced range (clamp to [-7,7]). True 4-bit packing would halve memory further but requires bitwise ops.

---
## [2026-04-06 18:00] Phase P6.1: ToPi + ToMe video scoring + attn_bias

### Plan
- **Objective:** Complete B8 token optimization module with pruning, video-aware similarity, and mlx-mfa bridge
- **Files to modify:** tokens/pruning.py (new), tokens/tome.py (add functions + modify signature), tokens/__init__.py, __init__.py

### Changes made
- `tokens/pruning.py` — ToPiConfig, PruneInfo, compute_token_importance (attention/norm/random), topi_prune, topi_restore (copy/zero/lerp). Fully vectorized, no .item() calls. [HIGH]
- `tokens/tome.py` — Added compute_spatiotemporal_similarity (cosine + spatial + temporal proximity), compute_attn_bias_for_mfa (log-count in [1,1,1,N_merged] shape), tome_merge now accepts spatial_dims parameter [HIGH]
- `tests/test_pruning.py` — 10 tests: shape restore, passthrough, norm importance, random count, attention importance, copy/zero restore, disabled, finite [HIGH]
- `tests/test_tome.py` — 5 new tests: spatiotemporal nearby higher, shape, merge with spatial_dims, attn_bias shape/values, integration [HIGH]

### Dependency & regression check
- tome_merge signature extended with optional spatial_dims/weights — backward compatible (defaults to None)
- All 150 existing tests pass without modification

### Tech cost assessment
- ToPi prune: O(N log N) argsort + O(N_pruned × N_kept) nearest neighbor. No .item() calls.
- Spatiotemporal similarity: O(N²) pairwise distance matrices. Same as cosine_similarity.
- attn_bias: O(N_src) one-liner wrapping compute_proportional_bias.

### Confidence
- Overall: [HIGH]

---
## [2026-04-06 19:00] Phase P6.2: DiffSparse guard + maturity + docs + LCSA

### Plan
- **Objective:** Library-quality improvements: production guards, maturity tracking, docs, LCSA interaction handling

### Changes made
- `tokens/learned_sparsity.py` — Added strict mode (RuntimeError) + one-time warning for non-pretrained fallback [HIGH]
- `maturity.py` — New module: Maturity enum, COMPONENT_MATURITY registry, get_maturity(), list_components() [HIGH]
- `cache/coefficients/README.md` — TeaCache coefficient format documentation [HIGH]
- `tokens/tome.py` — lcsa_compatible flag in ToMeConfig, caps merge_ratio at 0.3 with warning, LCSA docstring note [HIGH]
- `tests/test_learned_sparsity.py` — +2 tests: strict raises, non-strict warns once [HIGH]
- `tests/test_maturity.py` — 5 tests: known/unknown components, all/filtered listing [HIGH]
- `tests/test_tome.py` — +2 tests: LCSA cap, LCSA passthrough below cap [HIGH]

### Confidence
- Overall: [HIGH]

---
## [2026-04-06 20:00] Phase P5.1: B2 FBCache + B3 SpectralCache

### Changes made
- `cache/fbcache.py` — FBCacheConfig/State, fbcache_should_compute (rel_l1 on first block output), fbcache_update [HIGH]
- `cache/spectral_cache.py` — SpectralCacheConfig/State, spectral_cache_should_compute (FFT-based HF change + sigma-adaptive thresholds + energy budget), spectral_cache_update [HIGH]
- `orchestrator.py` — Added fbcache/spectral_cache configs + states, cascade priority in should_compute_step (TeaCache → SpectralCache → FBCache), update_step_cache updates all caches [HIGH]
- `tests/test_fbcache.py` — 5 tests [HIGH]
- `tests/test_spectral_cache.py` — 7 tests [HIGH]

### Tech cost assessment
- FBCache: One rel_l1 (O(n)) on first-block output only. Negligible.
- SpectralCache: One rfft + magnitude + HF comparison per step. O(n log n) on last dim.

### Confidence
- Overall: [HIGH]

---
## [2026-04-06 20:30] Phase P5.2: B5 DeepCache + MosaicDiff

### Changes made
- `cache/deep_cache.py` — DeepCacheConfig/State, DeepCacheManager (auto-select middle layers, interval-based recompute), analyze_layer_redundancy (cosine/l2), select_cacheable_layers [HIGH]
- `orchestrator.py` — Added deep_cache config + DeepCacheManager lifecycle + 3 new methods (should_compute_layer_deep, get_deep_cached_layer, update_deep_cache_layer) [HIGH]
- `scripts/analyze_layer_redundancy.py` — Full CLI tool: load weights, analyze, recommend, save JSON [HIGH]
- `tests/test_deep_cache.py` — 11 tests: auto-select, compute/skip intervals, roundtrip, reset, disabled, redundancy analysis, selection [HIGH]

### Tech cost assessment
- DeepCacheManager: O(1) per layer per step (set lookup + delta comparison)
- analyze_layer_redundancy: O(n_layers × weight_size) cosine similarity. One-time analysis.

### Confidence
- Overall: [HIGH]

---
## [2026-04-06 21:00] Phase P5.3: B7 ToCa Token Caching

### Changes made
- `tokens/toca.py` — ToCaConfig, ToCaState, TokenCacheManager with identify_stable_tokens (cosine sim + attention priority + cache_ratio cap), apply_cache (mx.where), get_dynamic_indices, reset. Fully vectorized. [HIGH]
- `orchestrator.py` — Added toca config + TokenCacheManager lifecycle + property + reset [HIGH]
- `tests/test_toca.py` — 9 tests [HIGH]

### Confidence
- Overall: [HIGH]

---
## [2026-04-06 21:30] Phase P5.4: B12 DiTFastAttn + residual

### Changes made
- `attention/ditfastattn.py` — HeadStrategy enum, DiTFastAttnConfig/State, DiTFastAttnManager with auto-profiling (variance-based sensitivity), per-head strategy assignment (FULL/WINDOW/CACHED), window mask generation, cache/get roundtrip [HIGH]
- `attention/residual.py` — scaled_residual_add (with optional gate), compute_residual_scale (inverse_sqrt/linear/constant), residual_gate_from_sensitivity [HIGH]
- `orchestrator.py` — Added ditfastattn config + DiTFastAttnManager lifecycle + property + reset [HIGH]
- `tests/test_ditfastattn.py` — 8 tests: pre-profile all FULL, profiling assigns strategies, high/low variance, cache_start_step, window mask, cache roundtrip, reset, disabled [HIGH]
- `tests/test_residual.py` — 8 tests: add without gate, with scale, with gate, inverse_sqrt/linear/constant scale, sensitivity gate, default, clamping [HIGH]

### Confidence
- Overall: [HIGH]

---
## [2026-04-06 22:00] Phase P5.5: WorldCache Motion-Aware Extension

### Changes made
- `cache/motion.py` — MotionConfig/State, estimate_motion (l1_diff + gradient + block_matching), Sobel filter, motion_adjusted_threshold, estimate_motion_vector (CoM), warp_features_by_motion (integer shift), MotionTracker class [HIGH]
- `cache/teacache.py` — Added optional `motion: MotionConfig` field to TeaCacheConfig [HIGH]
- `orchestrator.py` — MotionTracker lifecycle, motion-adjusted threshold in should_compute_step, `frame` parameter, property + reset [HIGH]
- `tests/test_motion.py` — 13 tests: zero/high motion, gradient edge detection, threshold adjustment, warp shift, tracker history/reset [HIGH]

### Tech cost assessment
- estimate_motion: O(H×W) for l1_diff, O(H×W) for gradient (Sobel is 3×3 conv → O(H×W))
- motion_adjusted_threshold: O(1)
- warp_features_by_motion: O(H×W) array slicing

### Confidence
- Overall: [HIGH]

---
## [2026-04-06 22:30] Phase P5.6: Functional Scripts + VAE Streaming

### Changes made
- `scripts/calibrate_teacache.py` — Full offline calibration: load features, compute L1 distances, polynomial fit via numpy, threshold selection by target skip ratio [HIGH]
- `vae/wavelet_cache.py` — Added output_buffer mode (in-place write), callback mode (per-chunk notification), estimate_output_shape, preallocate_output_buffer [HIGH]
- `tests/test_calibrate_teacache.py` — 3 tests: synthetic calibration, JSON output, too-few-files error [HIGH]
- `tests/test_analyze_redundancy.py` — 3 tests: identical/mixed weights, selection [HIGH]
- `tests/test_wavelet_cache.py` — +4 tests: buffer mode, callback mode, shape estimation, temporal upsample [HIGH]

### Tech cost assessment
- Calibration: O(S × N) where S=steps, N=feature size. One-time offline operation.
- Buffer mode: zero extra memory vs concat mode (no intermediate list).

### Confidence
- Overall: [HIGH]

---
## [2026-04-07 08:00] Phase P7.0: Honest rebuild — baseline reset

### Plan
- **Objective:** Reset the maturity of B2/B3/B5/B7/B12 to STUB because the current implementations are algorithmically incorrect when audited against the original prompt specification. Re-implement correctly across P7.1–P7.6.
- **Files to modify:** `mlx_diffusion_kit/maturity.py` (downgrade 5 entries), `devnotes/SESSION_LOG.md` (this entry).
- **Dependencies impacted:** None yet — no code deleted or rewritten in this phase. Per-phase cleanup happens in P7.1–P7.5.

### Audit findings that motivated the reset
- **B5 DeepCache** was a generic per-layer cache (`dict[int, mx.array]`) with auto-selection of "middle" indices. It had no notion of UNet deep-branch semantics. Correct algorithm caches the bottleneck output as a single tensor and recomputes only the shallow encoder/decoder layers each step.
- **B2 FBCache** used the first-block output as a skip signal but cached and returned the **entire model output**. Correct algorithm caches the **residual** (full_output − fb_output) and reconstructs as `fb_output + cached_residual`, skipping only the remaining blocks. Also missing `start_step`/`end_step`.
- **B7 ToCa** used single-step cosine similarity with a single global cache. Correct algorithm needs per-layer state and **L1-velocity scoring** across ≥2 previous steps. Also contained a Python `for b_idx in range(B)` loop — the same anti-pattern fixed earlier in ToMe.
- **B3 SpectralCache** was a step-level skip decision using HF magnitude change as signal. Correct algorithm caches **LF Fourier coefficients** across steps, recomputes HF each step, and **reconstructs via inverse transform**. None of the frequency-domain caching existed.
- **B12 DiTFastAttn** had 3 strategies (FULL/WINDOW/CACHED) with variance-based auto-profiling. Correct design per prompt has 4 strategies (FULL/WINDOW/SHARE/RESIDUAL) with explicit per-layer config lists.

### Rebuild decisions approved by user (R1 path)
- Delete the 5 modules and their tests. New tests derived from algorithm input/output contract, not from the candidate code.
- **B19** Flash-VAED / Neodragon decoder distill: skip (separate project).
- **B3** SpectralCache: implement SeaCache variant (`spectral_velocity_aware` flag) in the same pass.
- **B12** DiTFastAttn: follow prompt spec exactly — 4 strategies, `sharing_layers`/`residual_cache_layers` config lists, per-layer (not per-head) decisions.
- **B18** Separable Conv3D: implementer's judgment. Plan: Mode A = `nn.Module` R(2+1)D block for new models; Mode B = SVD decomposition of pretrained `conv3d` weights into (spatial_2D, temporal_1D) factors with reported reconstruction error.
- **MosaicDiff** (redundancy analyzer, previously bundled with DeepCache): move to a separate module so the new DeepCache file stays surgical. Preserves `scripts/analyze_layer_redundancy.py` functionality.

### Changes made
- `maturity.py:L21-L39` — B2/B3/B5/B7/B12 downgraded to STUB with inline rationale [HIGH]
- `devnotes/SESSION_LOG.md` — this entry [HIGH]

### Dependency & regression check
- No code deleted in this phase. Existing 245 tests still pass (tested before this commit).
- Per-phase integration-point rewiring in the orchestrator happens inside each P7.N commit.

### Tech cost assessment
- N/A (metadata + docs only)

### Confidence
- Overall: [HIGH]
- Risks: none for this setup commit. Per-phase risks logged in each P7.N entry.

---
## [2026-04-07 08:30] Phase P7.1: B5 DeepCache rebuild

### Plan
- **Objective:** Replace the generic per-layer cache masquerading as DeepCache with a paper-faithful deep-branch caching policy. Move the MosaicDiff redundancy analyzer (previously bundled) to its own module.
- **Files to modify:** `cache/deep_cache.py` (rewrite), `cache/layer_redundancy.py` (new), `cache/__init__.py`, `orchestrator.py`, `scripts/analyze_layer_redundancy.py`; `tests/test_deep_cache.py` (rewrite), `tests/test_layer_redundancy.py` (new), `tests/test_analyze_redundancy.py` (delete — superseded).
- **Dependencies impacted:** orchestrator (rewired), analyze script (import path change).

### Changes made
- `cache/deep_cache.py` — rewrite. New API: `DeepCacheConfig{cache_interval, start_step, enabled}`, `DeepCacheState{cached_deep_features, last_recompute_step, recompute_count}`, plus `create_deepcache_state`, `deepcache_should_recompute`, `deepcache_store`, `deepcache_get`, `deepcache_reset`. Decision is delta-based, not modulo, so TeaCache-skipped step sequences remain correct. [HIGH]
- `cache/layer_redundancy.py` — new. `analyze_layer_redundancy` and `select_cacheable_layers` moved here, unchanged in behavior. Docstring clarifies this is a tooling utility, not a runtime cache. [HIGH]
- `cache/__init__.py` — expose new DeepCache API + MotionConfig/MotionTracker (previously forgotten from __all__). [HIGH]
- `orchestrator.py` — DeepCacheManager-based methods removed. New: `should_recompute_deep`, `get_cached_deep_features`, `store_deep_features`, plus `deep_cache_state` property. Integration contract documented: the caller's model wrapper splits at the deep/shallow boundary and invokes these. [HIGH]
- `scripts/analyze_layer_redundancy.py:L17` — import path updated to `cache.layer_redundancy`. [HIGH]
- `tests/test_deep_cache.py` — 10 new tests derived from the algorithm contract: first-step recompute, store-then-reuse-until-interval, exact state mutation, disabled passthrough, start_step respected, delta-based correctness under non-contiguous step indices, interval=1 edge case, reset, get-on-fresh, recompute-count telemetry. [HIGH]
- `tests/test_layer_redundancy.py` — 8 new tests: single-layer zero score, all-identical full redundancy, mixed-weight ranking, score bounds, l2 method, ratio selection, min selection size, sorted output. [HIGH]
- `tests/test_analyze_redundancy.py` — deleted (superseded by test_layer_redundancy). [HIGH]

### Dependency & regression check
- grep for `DeepCacheManager`, `should_compute_layer_deep`, `get_deep_cached_layer`, `update_deep_cache_layer`, `deep_cache_manager`, `analyze_layer_redundancy`, `select_cacheable_layers` in tests/, mlx_diffusion_kit/, scripts/ → only expected references remain.
- Full test suite: 249 pass (previously 245 — net +4 because redundancy now has 8 tests vs the prior 3). No regressions.

### Tech cost assessment
- Compute: decision is O(1) per step. Store is O(1) (reference assignment). No FFT, no matrix ops.
- Memory: one cached tensor (size model-dependent). No per-layer dict.

### Confidence
- Overall: [HIGH]
- Risks: model wrappers that previously called `should_compute_layer_deep(layer_idx, step_idx)` are incompatible. Since this API was undocumented/internal and not exported at top-level, no external breakage. Internal orchestrator tests did not use it.

---
## [2026-04-07 09:00] Phase P7.2: B2 FBCache rebuild

### Plan
- **Objective:** Replace the TeaCache-lite skip with block-level residual caching per the First-Block Cache algorithm. Rename `fbcache.py` → `fb_cache.py` to match the prompt's target filename.
- **Files to modify:** `cache/fb_cache.py` (new), `cache/fbcache.py` (delete), `cache/__init__.py`, `orchestrator.py`; `tests/test_fb_cache.py` (new), `tests/test_fbcache.py` (delete).
- **Dependencies impacted:** orchestrator cascade layout changes. FBCache is no longer in the `should_compute_step` cascade — it operates at a different granularity.

### Changes made
- `cache/fb_cache.py` — new file. Config: `rel_l1_thresh=0.1`, `start_step=0`, `end_step=None`, `max_consecutive_cached=5`, `enabled=True`. State: `prev_fb_output`, `cached_residual` (= `full_output - fb_output`), `step_counter`, `consecutive_cached`. Functions: `create_fbcache_state`, `fbcache_should_compute_remaining` (decision), `fbcache_update` (stores residual), `fbcache_reconstruct` (returns `fb + cached_residual`), `fbcache_reset`. [HIGH]
- `cache/fbcache.py` — deleted. [HIGH]
- `cache/__init__.py` — import path updated; `fbcache_should_compute` removed from __all__, replaced with `fbcache_should_compute_remaining`, `fbcache_reconstruct`, `fbcache_reset`. [HIGH]
- `orchestrator.py` — FBCache removed from the `should_compute_step` step-level cascade (it operates at block boundary, not step boundary). New methods: `should_compute_remaining_blocks`, `fbcache_update_residual`, `fbcache_reconstruct_output`. `update_step_cache` no longer takes `first_block_output`, no longer updates FBCache state. `get_cached_output` no longer returns FBCache data (would be incorrect: FBCache stores a residual, not a full output). [HIGH]
- `tests/test_fb_cache.py` — 11 new tests: first-step compute, identical-skip, divergent-compute, **reconstruction identity** (`reconstruct(fb) = fb + cached_residual`), **fb dependence** (reconstruction uses *current* fb, not cached), max_consecutive ceiling, start_step window, end_step window, disabled passthrough, reconstruct-without-cache raises, reset. [HIGH]
- `tests/test_fbcache.py` — deleted (superseded). [HIGH]

### Dependency & regression check
- grep for `fbcache\|FBCache\|first_block_output` in tests/test_orchestrator.py, tests/test_integration.py → no matches. No callers to update.
- Full test suite: 255 pass (previously 249 — net +6 because the new test file has 11 tests vs the prior 5). No regressions.

### Tech cost assessment
- Compute: one rel_l1 per decision (O(n)). One subtract for residual store. One add for reconstruction. Negligible vs model forward.
- Memory: `prev_fb_output` (first-block shape) + `cached_residual` (full-output shape) = modest. Not 2x full-output as the prior broken impl would suggest.

### Confidence
- Overall: [HIGH]
- Risks: model wrappers that previously received "skip everything" from FBCache now get "skip remaining blocks" semantics. If any user relied on the prior (incorrect) behavior, their code breaks. Acceptable because the prior behavior was wrong per the paper.

---
## [2026-04-07 09:30] Phase P7.3: B7 ToCa rebuild

### Plan
- **Objective:** Replace single-step global cosine caching with per-layer velocity-based token caching per Zou et al.
- **Files to modify:** `tokens/toca.py` (rewrite), `tokens/__init__.py`, `orchestrator.py`; `tests/test_toca.py` (rewrite).

### Changes made
- `tokens/toca.py` — rewrite. New API: `ToCaConfig{recompute_ratio, score_mode, enabled}`, per-layer `ToCaLayerState{cached_tokens, prev_tokens, step_count}`, top-level `ToCaState{layers: dict[int, ...]}`. Functions: `create_toca_state`, `toca_select_tokens` (returns `(active_indices, cached_indices)`, sorted by position), `toca_compose` (reassembles `[B, N, D]` from disjoint active+cached pieces), `toca_update` (shifts `prev ← cached`, `cached ← tokens`), `toca_get_cached`, `toca_reset`. Score modes: `"velocity"` needs 2-step history, `"magnitude"` works from step 1. Fallback to all-active when history is insufficient. [HIGH]
- `tokens/__init__.py` — export new API, drop `TokenCacheManager`. [HIGH]
- `orchestrator.py` — `TokenCacheManager` references replaced with function-based API: `toca_select`, `toca_record`, `toca_compose_tokens`, `toca_state` property. State storage is `_toca_state: Optional[ToCaState]`. [HIGH]
- `tests/test_toca.py` — 13 new tests: first-call all-active, velocity mode needs 2 updates, velocity partitioning with crafted trajectory, disjoint union covers N, compose reconstructs correctly, compose with empty cached, per-layer independence, magnitude mode from step 1, disabled passthrough, update shifts history, get_cached Noneness, reset clears. [HIGH]

### Dependency & regression check
- grep for `TokenCacheManager`, `self._toca[^_]` → no matches outside orchestrator internals (now updated).
- Full test suite: 258 pass (previously 255 — net +3; test count jumped from 9 to 13 but some integration tests cover overlapping ground). No regressions.

### Tech cost assessment
- Compute: per decision, one mean-abs per token (O(N·D)) + one argsort (O(N log N)). No Python batch loops in selection. Compose has a small batch loop for scatter-add (B typically 1-8).
- Memory: per layer, 2 × `[B, N, D]` tensors (cached + prev). For 24-layer DiT with B=1, N=4096, D=1024: ~200 MB. Documented as the tradeoff.

### Confidence
- Overall: [HIGH]
- Risks: the compose function uses a per-batch Python loop for scatter-add. Acceptable because B is small; matches existing ToPi and ToMe patterns. If profiled as a bottleneck, a vectorized `at[]` path via flat batch offsets is possible.

---
## [2026-04-07 10:00] Phase P7.4: B3 SpectralCache + SeaCache rebuild

### Plan
- **Objective:** Replace the step-level skip decision with actual frequency-domain LF/HF caching and inverse-transform reconstruction, per the prompt. Add the SeaCache variant (`spectral_velocity_aware`) that invalidates LF cache on high per-band velocity.
- **Files to modify:** `cache/spectral_cache.py` (rewrite), `cache/__init__.py`, `orchestrator.py`; `tests/test_spectral_cache.py` (rewrite).

### Changes made
- `cache/spectral_cache.py` — rewrite. Config: `low_freq_ratio=0.25`, `cache_interval_low=4`, `cache_interval_high=1`, `transform="rfft"|"dct"`, `spectral_velocity_aware=False`, `velocity_override_thresh=0.5`. State: `cached_low_freq`, `cached_high_freq`, `prev_full_spectra` (bounded at 2 for SeaCache), `last_low_recompute_step`, `last_high_recompute_step`. Core function `spectral_cache_apply(features, step_idx, config, state) -> features` does the round trip: rFFT → split LF/HF → apply policy → combine → irFFT. Delta-based interval arithmetic so TeaCache-skipped steps don't desync. DCT raises NotImplementedError with a clear message (MLX has no native DCT). [HIGH]
- `cache/__init__.py` — export new API, drop `spectral_cache_should_compute`. [HIGH]
- `orchestrator.py` — SpectralCache removed from `should_compute_step` cascade (it's not a skip gate anymore). New method `apply_spectral_cache(features, step_idx)` for caller-driven application. `should_compute_step` signature simplified (removed `sigma_t` parameter since nothing else used it). `update_step_cache` updates spectral state via the new `spectral_cache_update(features, step_idx, config, state)` signature. `get_cached_output` docstring now explains that SpectralCache has its own reconstruction path. [HIGH]
- `tests/test_spectral_cache.py` — 11 new tests: shape preservation, **identity when both intervals = 1** (the canonical reconstruction test), disabled passthrough, **LF cache honored** (verifies the combined spectrum explicitly), LF invalidates after interval, HF always fresh at interval=1, DCT raises, **SeaCache forces recompute on high velocity**, SeaCache stable on small change, update forces refresh, reset clears all. [HIGH]

### Dependency & regression check
- `should_compute_step` signature changed (`sigma_t` removed). Before change I grep'd for callers — no tests use that kwarg. Integration tests use keyword arguments that no longer include `sigma_t` in their call sites; no breakage.
- Full test suite: 262 pass (previously 258 — net +4). No regressions.

### Tech cost assessment
- Compute: forward + inverse rFFT per call. MLX complex64 round-trips at ~1e-6 error. For N=4096 tokens along the last axis, O(N log N).
- Memory: two band caches (LF + HF). Sum is ~one spectrum-sized complex array. For SeaCache: up to 2 additional full spectra in history.

### Confidence
- Overall: [HIGH]
- Risks: FFT is taken along the last axis only. For feature maps where the last axis is the channel/embedding dim (standard DiT layout), this is correct. If a user passes features with a different axis convention, they must reshape first. Documented implicitly via the `axis=-1` contract in the docstring.

---
## [2026-04-07 10:30] Phase P7.5: B12 DiTFastAttn rebuild

### Plan
- **Objective:** Replace the 3-strategy per-head variance-profiled design with the paper's 4-strategy per-layer policy (FULL / WINDOW / SHARE / RESIDUAL).
- **Files to modify:** `attention/ditfastattn.py` (rewrite), `attention/__init__.py`, `orchestrator.py`; `tests/test_ditfastattn.py` (rewrite).

### Changes made
- `attention/ditfastattn.py` — rewrite. Four-strategy `AttnStrategy` enum. Config: `window_start_step`, `window_size`, `sharing_layers: list[int]`, `residual_cache_layers: list[int]`. State: two caches (`cached_attn_maps`, `cached_residuals`) keyed by layer_idx. Functions: `create_ditfastattn_state`, `ditfastattn_decide`, `ditfastattn_record_attn_map`, `ditfastattn_get_cached_attn`, `ditfastattn_record_residual`, `ditfastattn_get_cached_residual`, `ditfastattn_reset`. Decision precedence documented in docstring: step==0 → FULL; RESIDUAL wins if configured+cached; SHARE wins over WINDOW; WINDOW activates at window_start_step. Safety fallback: missing cache → drop to next priority tier. [HIGH]
- `attention/__init__.py` — expose new API. `DiTFastAttnManager` and `HeadStrategy` removed. [HIGH]
- `orchestrator.py` — `DiTFastAttnManager` references replaced with function-based API. State storage is `_ditfastattn_state: Optional[DiTFastAttnState]`. New methods: `get_attn_strategy`, `record_attn_map`, `get_cached_attn_map`, `record_attn_residual`, `get_cached_attn_residual`. [HIGH]
- `tests/test_ditfastattn.py` — 12 new tests: step 0 is FULL, disabled is FULL, RESIDUAL beats SHARE beats WINDOW, missing-cache safety fallback for both SHARE and RESIDUAL, WINDOW boundary, non-listed layer behavior, roundtrip for each cache, reset clears both, and a comprehensive **"four distinct strategies achievable"** test. [HIGH]

### Dependency & regression check
- grep for `DiTFastAttnManager`, `HeadStrategy`, `self._ditfastattn[^_]` → no matches outside the rewritten orchestrator internals.
- Full test suite: 266 pass (previously 262 — net +4). No regressions.

### Tech cost assessment
- Compute: all decision functions are O(1) (list/dict membership + integer compares).
- Memory: two dicts keyed by layer_idx. Per entry: one mx.array reference, no copy. Caller owns the underlying tensors.

### Confidence
- Overall: [HIGH]
- Risks: the safety fallback (missing-cache → drop to next strategy) means users who configure `sharing_layers=[0]` but forget to call `record_attn_map` will silently get WINDOW/FULL instead of SHARE. Documented in the docstring. An alternative would be to raise, but that's noisier and less forgiving.

---
## [2026-04-07 11:00] Phase P7.6: B18 Separable Conv3D (new)

### Plan
- **Objective:** Implement R(2+1)D separable 3D convolution in both modes: new-module (Mode A) and SVD decomposition of a pretrained dense Conv3d kernel (Mode B).
- **Files to modify:** `vae/separable_conv3d.py` (new), `vae/__init__.py`; `tests/test_separable_conv3d.py` (new).

### Changes made
- `vae/separable_conv3d.py` — new file. [HIGH]
  - Mode A: `SeparableConv3D(nn.Module)`. Uses `nn.Conv2d` for spatial (kH, kW) followed by `nn.Conv1d` for temporal (kT). Forward reshapes via transpose to (N*T, H, W, C) for spatial, then (N*H'*W', T, mid) for temporal. Configurable `mid_channels` (rank).
  - Mode B: `decompose_conv3d_to_separable(W, rank=None) -> (spatial, temporal, error)`. Reshapes `W[out, kT, kH, kW, in]` to `(kT*out, kH*kW*in)` via transpose, runs SVD on CPU stream (MLX SVD is CPU-only at current MLX version on M1 Max), distributes sqrt(Σ) evenly between U and V^T, reshapes factors to `(mid, kH, kW, in)` and `(out, kT, mid)`. Returns the relative Frobenius reconstruction error for the caller to judge acceptability.
  - Bridge helper: `build_separable_from_decomposition` wraps Mode B factors into a Mode A module.
- `vae/__init__.py` — export all three symbols. [HIGH]
- `tests/test_separable_conv3d.py` — 10 tests. Mode A: forward shape (5 → 3 temporal kernel reduces T by 2), forward finiteness, parameter-count formula validation, input-validation errors. Mode B: **full-rank lossless reconstruction** (error < 1e-4), reduced-rank error is positive and bounded, **monotone error decrease with increasing rank**, bridge produces a working module. [HIGH]

### Dependency & regression check
- No orchestrator integration per prompt spec (B18 is a VAE building block, not a step-level decision).
- Full test suite: 276 pass (previously 266 — net +10, matches the 10 new tests). No regressions.

### Tech cost assessment
- Mode A: parameter count follows `kH·kW·in·mid + kT·mid·out + out_bias`. For a 3³ kernel, `in=out=mid=64`, separable = 28_864 vs dense = 110_592, a 3.8× reduction.
- Mode B: SVD on `(kT·out, kH·kW·in)` matrix. For `out=64, kT=3, kH=kW=3, in=64` → SVD on `(192, 576)`, CPU-bound. One-time cost at model conversion.

### Confidence
- Overall: [HIGH]
- Risks: MLX `mx.linalg.svd` requires `stream=mx.cpu` at this MLX version. Documented and verified. If a future MLX ships GPU SVD, the decomposition should automatically get faster; the API is stable either way.
- Risks: Mode B at reduced rank is lossy by definition. The returned `reconstruction_error` lets the user decide; no silent accuracy degradation.

---
## [2026-04-07 11:30] Phase P7 wrap-up — docs, exports, final verification

### Changes made
- `mlx_diffusion_kit/__init__.py` — re-exported the rebuilt APIs. Export count grew from 46 to 89. All 89 pass `hasattr(mod, name)` smoke check. [HIGH]
- `README.md` — table updated: B5 maturity note (UNet-generalized scope, 5 models not 3), B7 target narrowed to multi-step DiT, B12 moved to Beta, B18 added row. Status line: 21→23 components, 245+→276+ tests, maturity counts corrected. [HIGH]
- `CLAUDE.md:L174-176` — implementation-status block updated (89 exports, 23 components, 276+ tests across 27 test files). [HIGH]
- `CHANGELOG.md` — new "Unreleased — P7 rebuild branch" section documenting the audit + rebuild of B2/B3/B5/B7/B12 and the new B18, with rationale for each. Also covers the orchestrator API reshuffles (cascade layout change, method renames). [HIGH]
- `docs/API_MANUAL.md` — appended "Rebuilt / Added in P7 (2026-04-07)" section with full per-component signatures. Version banner bumped from 46 to 89 exports. [HIGH]

### Dependency & regression check
- Import smoke: 89/89 exports importable via `mlx_diffusion_kit.<name>`.
- Full test suite: **276 pass, 0 failures**. Up from 245 baseline on `main` (net +31 across the 6 rebuilt/new components; deltas recorded per-phase in the P7.1–P7.6 entries).
- Orchestrator API changes not visible at top-level — all were internal method renames (e.g., `should_compute_layer_deep` → `should_recompute_deep`). External users consuming only the documented public surface are unaffected where the algorithms were paper-faithful; where they depended on the previous incorrect semantics (notably FBCache caching full output), they break by design.

### Final component maturity snapshot
- STABLE: 9 (B1, B4, B8 ToMe, B11, B13, B14.1, B15, B17, B23)
- BETA: 11 (B2, B3, B5, B6, B7, B8 ToPi, B10, B12, B14.2, B18, B22)
- EXPERIMENTAL: 1 (WorldCache motion)
- STUB: 2 (B9 DiffSparse, B19 Flash-VAED/Neodragon — separate project)

### Confidence
- Overall: [HIGH]
- Next: ready for user review on the branch. Per the prompt, no push until review. Once reviewed, merge `feat/readme-backlog-p7` into `main` and tag as needed (not `v0.1.0` since that tag is already on the original release — next tag should be `v0.2.0` or a pre-release like `v0.2.0-rc.1` because the external API changed in non-trivial ways).

---
## [2026-04-07 12:30] Phase P8.0: Post-review fixes from mlx-code-review P7 audit

### Plan
- **Objective:** Apply the four fixes surfaced by the `mlx-code-review` skill's audit of branch `feat/readme-backlog-p7`:
  (1) remove SpectralCache force-refresh from `update_step_cache` (silent correctness bug — orchestrator wiring introduced in P7.4);
  (2) add a B18 forward-equivalence test (contract guarantee previously untested);
  (3) add a `bias` parameter to `build_separable_from_decomposition` (previously silently dropped bias);
  (4) add shape validation to `fbcache_reconstruct` (previously produced cryptic MLX errors on shape mismatch).
- **Files to modify:** `mlx_diffusion_kit/orchestrator.py`, `mlx_diffusion_kit/vae/separable_conv3d.py`, `mlx_diffusion_kit/cache/fb_cache.py`, `tests/test_orchestrator.py`, `tests/test_separable_conv3d.py`, `tests/test_fb_cache.py`.
- **Dependencies impacted:**
  - `orchestrator.py` drops the `spectral_cache_update` import — external top-level API unchanged.
  - `build_separable_from_decomposition` gains an optional `bias` kwarg (backward compatible).
  - No other user-facing API changes.

### Changes made
- `orchestrator.py:58-62` — removed `spectral_cache_update` from the import group [HIGH]
- `orchestrator.py:387-395` — removed the SpectralCache block from `update_step_cache` and replaced with a comment explaining why the orchestrator must not touch SpectralCache state here [HIGH]
- `vae/separable_conv3d.py:238-292` — added optional `bias: Optional[mx.array] = None` parameter to `build_separable_from_decomposition`; when supplied, installed on `mod.temporal.bias` [HIGH]
- `cache/fb_cache.py:168-195` — added `ValueError` in `fbcache_reconstruct` when `fb_output.shape != state.cached_residual.shape`, with a message pointing the caller to `fbcache_reset` [HIGH]
- `tests/test_orchestrator.py` — 1 new test: `test_spectral_cache_state_not_corrupted_by_update_step_cache`. Uses deliberately different shapes for `modulated_input` and `apply_spectral_cache` features to make any coupling between them visible. Passes after the fix; would have crashed on shape mismatch before [HIGH]
- `tests/test_separable_conv3d.py` — 3 new tests: `test_mode_b_forward_matches_dense_conv3d_at_full_rank` (contract guarantee — max-abs-diff ~1.4e-5 on `(4,3,3,3,3)` kernel), `test_build_separable_preserves_bias` (per-channel mean of `with-bias − without-bias` output equals the bias element), `test_build_separable_no_bias_by_default` (no implicit zero bias is created when `bias=None`) [HIGH]
- `tests/test_fb_cache.py` — 2 new tests: `test_fbcache_reconstruct_raises_on_shape_mismatch` (positive case for new ValueError; also re-checks the RuntimeError path still fires post-reset) and `test_fbcache_reconstruct_same_shape_still_works` (negative control: same-shape path unchanged) [HIGH]

### Dependency & regression check
- `grep -n spectral_cache_update mlx_diffusion_kit/orchestrator.py` → empty ✓
- `grep -n "bias=False" mlx_diffusion_kit/vae/separable_conv3d.py` inspected: only remaining occurrences are in `SeparableConv3D.__init__` (line 114, the R(2+1)D spatial stage which correctly carries no bias — bias belongs on the temporal stage mathematically) and in docstring lines — the bridge function no longer contains a hardcoded `bias=False` ✓
- `python -c "from mlx_diffusion_kit import *"` → no ImportError ✓
- Full test suite: **282 pass, 0 failures** (up from 276 baseline on `main`; net +6 from this phase: 4 explicitly required by the prompt + 2 positive-control tests for robustness) ✓
- No existing tests were modified — only new tests added.

### Tech cost assessment
- Fix 1: removes one unconditional function call (`spectral_cache_update`) per `update_step_cache` invocation. Net compute: -1 rFFT + -1 magnitude per call. Negligible.
- Fix 2: adds one forward-pass comparison per test run. Negligible.
- Fix 3: adds one optional parameter. No compute change when `bias=None` (default). When `bias` is supplied, one additional bias add per forward pass — but that was presumably the intent of the pretrained bias being kept in the first place.
- Fix 4: adds one shape comparison per `fbcache_reconstruct` call. Tuple equality — O(1). Negligible.

### Deferred (acknowledged, not implemented in P8)
- **Integration tests covering the rebuilt components end-to-end** through `DiffusionOptimizer` (review finding #3). Slated for a future P9 phase. Lower priority now that Fix 1 closes the most likely integration-level bug.
- **`analyze_layer_redundancy` `.item()`-in-loop** (review finding #6). Acceptable for an offline analysis tool; not worth vectorizing unless it becomes a bottleneck in practice.

### Confidence
- Overall: [HIGH]
- Risks: Fix 1 is the only behavior change with user-visible consequences. Callers who relied on the incorrect force-refresh semantics for SpectralCache must now explicitly call `apply_spectral_cache` with the feature stream they want cached. This is documented in the in-code comment and in the CHANGELOG already covers the orchestrator API moves. Pre-1.0 library; acceptable.

### Commit sequence on this branch
```
e7fe0b4 fix(orchestrator): remove SpectralCache force-refresh from update_step_cache
f27e2b4 test(B18): add forward-equivalence test for Mode B at full rank
c2a0a73 feat(B18): add bias parameter to build_separable_from_decomposition
11298b0 feat(fbcache): validate shapes in fbcache_reconstruct
```

### Status
Ready for merge review on `feat/readme-backlog-p7`. No push yet per prompt directive.

---
## [2026-04-18 10:00] Phase P9.0-P9.2: v0.2.0 review cleanup

### Plan
- **Objective:** Apply all 5 findings from the mlx-code-review of main@v0.2.0.
  P9.0 exception-safety fix + codebase sweep for the same anti-pattern;
  P9.1 integration test for rebuilt components (carried from P8);
  P9.2 minor cleanup (DiffSparse TODO, unused imports, dead telemetry).
- **Branch:** `chore/p9-cleanup` from `main@v0.2.0` (commit 18e8273).
- **Target tag after merge:** `v0.2.1`.
- **Files touched:** orchestrator.py, cache/fb_cache.py, cache/teacache.py,
  cache/multigranular.py, scheduler/dpm_solver_v3.py, scheduler/adaptive_stepping.py,
  vae/wavelet_cache.py, tokens/learned_sparsity.py,
  tests/test_orchestrator.py, tests/test_integration.py, tests/test_fb_cache.py.
- **Dependencies impacted:** None at public-API level. FBCacheState loses the
  internal `step_counter` field (never documented as public).

### Changes made — P9.0 (exception-safety fix)
- `orchestrator.py:10` — added `replace` to existing `from dataclasses import ...` line [HIGH]
- `orchestrator.py:301-313` — replaced save-mutate-restore of `cfg.rel_l1_thresh`
  with `dataclasses.replace(cfg, rel_l1_thresh=adjusted)`. TeaCacheConfig is a
  @dataclass of scalars + Optional[list[float]] so `replace` is a cheap shallow
  copy. Exception-safe by construction. [HIGH]
- `tests/test_orchestrator.py` — new regression test
  `test_motion_adjusted_threshold_does_not_mutate_user_config_on_exception`.
  Primes TeaCacheState via should_compute_step + update_step_cache, corrupts
  `prev_modulated_input` to a string, forces teacache_should_compute to raise,
  then asserts `tc_cfg.rel_l1_thresh` is unchanged. **Verified to fail against
  the pre-fix code** via `git stash push -- orchestrator.py` round-trip, confirming
  the test is not tautological. [HIGH]

### P9.0.2 — anti-pattern sweep results
Grep patterns searched:
- `original = ` → 0 matches in mlx_diffusion_kit/
- `= original$` (line-end restore) → 0 matches
- `saved = `, `prev_value = `, `old_value = ` → 0 matches
- `finally:` (where restore could hide) → 0 matches
- `setattr(` (invisible mutation) → 0 matches
- `^\s+(config|cfg|self\.config|self\.cfg)\.[a-z_]+ = ` (direct attribute writes
  on dataclass instances) → 0 matches
- `config\.|cfg\.` with ` = ` on the right side, filtered to exclude reads
  (`threshold = self.config.X`, etc.) → 0 write-sites

**Conclusion:** the motion branch in orchestrator.py was the only occurrence of
save-mutate-restore in the codebase. No other refactor needed.

### Changes made — P9.1 (integration test)
- `tests/test_integration.py` — new test
  `test_rebuilt_components_compose_without_desync` (Scenario 5). Simulates a
  10-step multi-step DiT pass with TeaCache, FBCache, SpectralCache, and
  DeepCache all active through DiffusionOptimizer. Uses deliberately distinct
  shapes for the four components' tensors (modulated_input, features, fb_out,
  deep_out) so any cross-cache shape corruption is visible. Asserts reset()
  clears all four caches. [HIGH]

### Changes made — P9.2 (minor cleanup)
- `tokens/learned_sparsity.py:59` — TODO(future) comment on `self._pretrained
  = False` explaining the intentionally-unreachable learned branch [HIGH]
- `orchestrator.py:37,43` — remove `MotionConfig` and `fbcache_reset` from
  import lists (neither referenced in the file after P7) [HIGH]
- `cache/teacache.py:12`, `cache/multigranular.py:10`,
  `scheduler/dpm_solver_v3.py:14`, `scheduler/adaptive_stepping.py:7`,
  `orchestrator.py:10`, `vae/wavelet_cache.py:11` — remove unused `field`
  from `from dataclasses import ...` in 6 files (swept via ast+grep heuristic) [HIGH]
- `cache/fb_cache.py` — remove `step_counter: int = 0` from FBCacheState,
  remove the `state.step_counter += 1` line from fbcache_should_compute_remaining,
  remove the `state.step_counter = 0` line from fbcache_reset, update docstring [HIGH]
- `tests/test_fb_cache.py:177` — remove the `assert state.step_counter == 0`
  line from the reset test [HIGH]

### Dependency & regression check
- `grep -n "cfg.rel_l1_thresh = " mlx_diffusion_kit/orchestrator.py` → empty ✓
- `grep -rn "step_counter" mlx_diffusion_kit/cache/fb_cache.py tests/test_fb_cache.py` → empty ✓
- `grep -n "MotionConfig\|fbcache_reset" mlx_diffusion_kit/orchestrator.py` → empty ✓
- `grep -n "TODO(future)" mlx_diffusion_kit/tokens/learned_sparsity.py` → 1 match at L59 ✓
- Import smoke (`import mlx_diffusion_kit`, all submodules) → OK, 89 top-level exports ✓
- Full test suite: **284 pass** (previously 282 on v0.2.0 — net +2: P9.0 regression + P9.1 integration)
- No existing tests modified other than the test_fb_cache.py reset test (which had to drop the `step_counter` assertion)

### Tech cost assessment
- P9.0: `dataclasses.replace` on a scalar dataclass is a single `__init__` call.
  Zero MLX array allocation. Negligible runtime cost vs the previous save-mutate-restore.
- P9.1: one integration test, 10 iterations of small mx.arrays. Runs in ~60 ms.
- P9.2: pure deletion — net reduction of ~15 lines of source, one int field per
  FBCacheState instance (~8 bytes saved per cache).

### Commit sequence on this branch
```
1beb966 chore: P9.2 minor cleanup (unused imports, dead telemetry, future-TODO)
2f5461a test(integration): end-to-end scenario with 4 rebuilt components active
29c5d8a fix(orchestrator): make motion-adjusted threshold exception-safe
```

### Status
Ready for merge review on `chore/p9-cleanup`. No push per prompt directive.
After merge to main, suggested tag `v0.2.1` (patch bump — bug fix +
integration tests + cleanup, no new features, no breaking API changes).

### Confidence
- Overall: [HIGH]
- Risks: The P9.0.2 sweep used 6 grep patterns to look for variants of
  save-mutate-restore. All returned empty. If there's a variant the patterns
  didn't anticipate (e.g., mutation via a wrapper function that itself
  writes-through to a config, or mutation via a method that takes cfg by
  reference and mutates it), it wouldn't show. Given the small codebase size
  (~5600 LOC) and the narrow set of dataclass configs, this is very unlikely —
  but noted here for completeness.
