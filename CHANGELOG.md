# Changelog

All notable changes to mlx-diffusion-kit are documented here.

## [0.2.1] — 2026-04-18

P9 review cleanup on top of the P7 rebuild + P8 fixes merged in v0.2.0.

### Fixed
- **Orchestrator exception-safety** — the motion-adjusted TeaCache
  threshold path no longer mutates the user-supplied `TeaCacheConfig`
  in place. Uses `dataclasses.replace` so the original config is
  untouched even if `teacache_should_compute` raises mid-call. Bug
  introduced in P5.5 (WorldCache motion), surfaced by P9 review.
- **FBCache** — `fbcache_reconstruct` now validates shape against the
  cached residual and raises a clear `ValueError` on mismatch (vs
  cryptic MLX broadcast error).

### Added (tests)
- Regression test for the exception-safety fix — stash-pop validated.
- Four-component integration test exercising TeaCache + FBCache +
  SpectralCache + DeepCache together through `DiffusionOptimizer`.
- B18 forward-equivalence test: composed `SeparableConv3D` matches
  dense `nn.Conv3d(W)(x)` at full rank (atol 1e-3).
- B18 `build_separable_from_decomposition(bias=...)` gains + test.

### Removed
- Dead `step_counter` field on `FBCacheState` (was write-only).
- Unused imports cleaned across 6 files (`field`, `MotionConfig`,
  `fbcache_reset`).

### Docs
- CLAUDE.md: codified three test-methodology principles validated
  across the P7/P8/P9 cycles (contract-derived tests, stash-pop
  regression validation, multi-grep anti-pattern sweeps).
- `TODO(future)` on DiffSparseRouter's `_pretrained` flag explaining
  the intentionally-unreachable learned branch.

### [abandoned] P10 SkipSR
After implementation, the `feat/skipsr` branch was abandoned without
merge. SkipSR (arXiv 2510.08799) requires a trainable mask predictor,
image-space scoring, and mask-aware rotary positional encodings — none
of which can be faithfully reproduced as a training-free orchestration
layer. Aligns with the prior exclusion of SLA/LLSA/SALAD.

## [0.2.0] — 2026-04-07

Post-release audit of the initial v0.1.0 caught that five components
(B2, B3, B5, B7, B12) had been merged with shallow or semantically
incorrect implementations that passed tests only because the tests
had been written against the implementation, not the reference
algorithm. All five were rebuilt on the `feat/readme-backlog-p7`
branch; B18 was newly implemented.

### Rebuilt
- **B5 DeepCache** — now caches the UNet deep-branch output as a
  single tensor and skips recomputation for `cache_interval` steps,
  per Ma et al. (CVPR 2024). Previous impl was a generic per-layer cache.
- **B2 FBCache** — now skips only blocks 2..N and reconstructs via
  `output = fb_output + cached_residual`. Previous impl cached the full
  output (wrong semantics). File renamed `fbcache.py` → `fb_cache.py`.
  Config gains `start_step` / `end_step`.
- **B7 ToCa** — now per-layer, velocity-based scoring from 2-step
  history, returning disjoint (active, cached) index arrays that cover
  [0, N). Previous impl was single-step cosine over a global cache.
- **B3 SpectralCache** — now performs real frequency-domain round-trip
  (rFFT → split LF/HF → apply per-band caching → combine → irFFT).
  Identity when both intervals = 1. **SeaCache variant** added as a
  `spectral_velocity_aware` flag that invalidates the LF cache on
  high per-band velocity. Previous impl was a step-level skip decision
  with no actual frequency-domain caching.
- **B12 DiTFastAttn** — now exposes the paper's 4-strategy enum
  `AttnStrategy { FULL, WINDOW, SHARE, RESIDUAL }` with explicit
  per-layer config lists `sharing_layers` and `residual_cache_layers`.
  Safety fallback: missing-cache → drop to next priority tier. Previous
  impl had only 3 strategies and conflated SHARE with RESIDUAL.

### Added
- **B18 Separable Conv3D (R(2+1)D)** — `SeparableConv3D` `nn.Module`
  for new models (Mode A) and `decompose_conv3d_to_separable` SVD
  utility for pretrained Conv3d kernels (Mode B), with
  reconstruction-error reporting for rank/accuracy tradeoffs.
- **MosaicDiff** layer-redundancy analyzer moved to its own module
  (`cache/layer_redundancy.py`) so DeepCache stays surgical.

### Internal
- Orchestrator rewired to function-based APIs for all rebuilt components
  (`DeepCacheManager`, `DiTFastAttnManager`, `TokenCacheManager` classes
  removed). FBCache and SpectralCache removed from the `should_compute_step`
  cascade — they operate at different granularities and got their own
  methods (`should_compute_remaining_blocks`, `apply_spectral_cache`).
- `should_compute_step` signature simplified (dropped `sigma_t`,
  `first_block_output`).
- Top-level `__init__.py` exports grew from 46 to 89.
- Test count rose from 245 to 276 (new acceptance tests derived from
  algorithm contracts, not candidate implementations).

## [0.1.0] — 2026-04-06

Initial release. 21 optimization components for diffusion/VSR inference on MLX.

### Step-Level Caching (B1-B6)
- **B1 TeaCache** — timestep-aware step caching with polynomial rescaling.
  CogVideoX-5B coefficients included. CVPR 2025 Highlight.
- **B1 WorldCache** — motion-aware extension for TeaCache. Sobel gradient
  and L1 motion estimation, sigma-adaptive threshold adjustment, feature
  warping via integer shift.
- **B2 FBCache** — zero-calibration first-block cache fallback.
- **B3 SpectralCache** — FFT-based high-frequency change detection with
  cumulative energy budget and sigma-adaptive thresholds.
- **B4 SmoothCache** — Taylor interpolation (order 1/2) for skipped steps.
  Eliminates "stutter" artifacts from consecutive cache reuse.
- **B5 DeepCache + MosaicDiff** — UNet layer-level caching with weight
  redundancy analysis for principled layer selection.
- **B6 Multi-Granular Cache** — BWCache (bandwidth-aware allocation) +
  UniCP (unified caching policy) + QuantCache (int8/int4 compression).

### Token-Level Optimizations (B7-B10)
- **B7 ToCa** — per-token caching between steps via cosine similarity
  with attention-score prioritization.
- **B8 ToMe** — token merging via bipartite cosine matching with MLERP,
  spatiotemporal video-aware scoring, `attn_bias` bridge for mlx-mfa.
  Fully vectorized (no Python loops).
- **B8 ToPi** — token pruning by importance (attention/norm/random) with
  copy/zero/lerp restore modes.
- **B9 DiffSparse** — learned router interface (stub). Strict mode guard
  prevents silent fallback in production.
- **B10 DDiT Scheduling** — dynamic patch stride per step (linear/cosine/step
  schedules) with power-of-2 quantization.

### Gating & Attention (B11-B12)
- **B11 T-GATE** — cross-attention gating after convergence step.
- **B12 DiTFastAttn** — per-head attention strategy (FULL/WINDOW/CACHED)
  with auto-profiling via variance analysis.
- **Residual utilities** — scaled residual add with gating, per-layer
  scaling (inverse_sqrt/linear/constant), sensitivity-based gates.

### Quality & Scheduling (B13-B15)
- **B13 FreeU** — FFT-based skip connection re-weighting for UNet.
- **B14.1 DPM-Solver-v3** — high-order ODE solver (orders 1-3) with
  log-SNR uniform spacing. Epsilon/v-prediction/x-start support.
- **B14.2 Adaptive Stepping** — MSE-based convergence detection with
  min_steps floor.
- **B15 TextEmbeddingCache** — disk-backed embedding cache with encoder_id
  keys and atomic writes.

### VAE Optimization (B17)
- **B17 WF-VAE Causal Cache** — conv state propagation for chunked
  temporal decoding. Pre-allocated buffer mode and streaming callbacks.

### Infrastructure (B22-B23)
- **B22 Encoder Sharing** — delta-based encoder block reuse across steps.
- **B23 Orchestrator + PISA** — central composition of all components.
  BlockStrategy enum (COMPUTE/SKIP/APPROXIMATE), cascade step-cache
  priority (TeaCache > SpectralCache > FBCache), sensitivity-based
  block approximation.
- **Maturity system** — per-component maturity tracking (STABLE/BETA/
  EXPERIMENTAL/STUB).

### Scripts
- `scripts/calibrate_teacache.py` — offline TeaCache polynomial calibration.
- `scripts/analyze_layer_redundancy.py` — MosaicDiff weight analysis CLI.

### Testing
- 245+ tests across 25 test files, 0 failures.
- Integration tests covering 4 scenarios: multi-step DiT, single-step DiT,
  VAE chunked decode, SmoothCache interpolation.
