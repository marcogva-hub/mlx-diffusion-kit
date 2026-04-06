# Changelog

All notable changes to mlx-diffusion-kit are documented here.

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
