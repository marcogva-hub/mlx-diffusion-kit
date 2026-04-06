# mlx-diffusion-kit Architecture

Version: **0.1.0**

## 1) System Overview

`mlx-diffusion-kit` is a pure-Python optimization layer for diffusion model
inference on MLX / Apple Silicon. It sits above model code and below the
user pipeline:

```
User Pipeline (inference script)
  -> DiffusionOptimizer (orchestrator.py)
    -> Step-level caching (TeaCache, FBCache, SpectralCache, SmoothCache, DeepCache)
    -> Token-level optimization (ToMe, ToPi, ToCa, DDiT, DiffSparse)
    -> Attention compression (DiTFastAttn, T-GATE)
    -> Quality enhancement (FreeU)
    -> Scheduler optimization (DPM-Solver-v3, Adaptive Stepping)
    -> VAE optimization (WaveletVAECache)
    -> Text encoder caching (TextEmbeddingCache)
  -> mlx-mfa (optional: Flash Attention kernels via attn_bias bridge)
```

## 2) Module Structure

```
mlx_diffusion_kit/
├── __init__.py              # 46 public exports
├── __version__.py           # "0.1.0"
├── maturity.py              # Component maturity registry
├── orchestrator.py          # B23 — DiffusionOptimizer + PISA
│
├── cache/                   # Step-level caching (B1-B6, B22)
│   ├── teacache.py          # B1 — TeaCache + polynomial rescaling
│   ├── motion.py            # B1 ext — WorldCache motion estimation
│   ├── fbcache.py           # B2 — First-Block Cache (zero-calibration)
│   ├── spectral_cache.py    # B3 — FFT-based HF change detection
│   ├── smooth_cache.py      # B4 — Taylor interpolation for skipped steps
│   ├── deep_cache.py        # B5 — UNet layer caching + MosaicDiff
│   ├── multigranular.py     # B6 — BWCache + UniCP + QuantCache
│   ├── encoder_sharing.py   # B22 — DiT encoder block sharing
│   └── coefficients/        # Pre-calibrated TeaCache coefficients
│       └── cogvideox.json
│
├── tokens/                  # Token-level optimizations (B7-B10)
│   ├── toca.py              # B7 — Per-token caching between steps
│   ├── tome.py              # B8 — Token Merging (bipartite matching)
│   ├── pruning.py           # B8 — Token Pruning (ToPi)
│   ├── learned_sparsity.py  # B9 — DiffSparse router (stub)
│   └── ddit_scheduling.py   # B10 — Dynamic patch stride scheduling
│
├── gating/                  # Conditional gating (B11)
│   └── tgate.py             # B11 — Cross-attention gating
│
├── attention/               # Attention compression (B12)
│   ├── ditfastattn.py       # B12 — Per-head strategy (FULL/WINDOW/CACHED)
│   └── residual.py          # Residual connection utilities
│
├── quality/                 # Quality enhancement (B13)
│   └── freeu.py             # B13 — UNet skip connection re-weighting
│
├── scheduler/               # Scheduler optimizations (B14)
│   ├── dpm_solver_v3.py     # B14.1 — High-order ODE solver
│   └── adaptive_stepping.py # B14.2 — Convergence-based step pruning
│
├── encoder/                 # Text encoder (B15)
│   └── embedding_cache.py   # B15 — Disk-backed embedding cache
│
└── vae/                     # VAE optimization (B17)
    └── wavelet_cache.py     # B17 — Causal conv state cache + streaming
```

## 3) Design Principles

### 3.1 Model-Type Aware
All components respect the `is_single_step` flag. Multi-step-only features
(TeaCache, T-GATE, ToCa, encoder sharing) are automatically disabled for
single-step models (SeedVR2, DOVE, FlashVSR, DLoRAL, UltraVSR).

### 3.2 Static Shapes (Shiva-DiT)
MLX recompiles computation graphs when tensor shapes change. ToMe and ToPi
produce fixed output sizes (N/2, N×0.7) to avoid recompilation overhead.

### 3.3 Training-Free First
All Phase 1-2 components are entirely training-free. Phase 3 introduces
optional learned components (DiffSparse) that require fine-tuning.

### 3.4 Composable
Each component works standalone. The orchestrator (B23) composes them but
does not require all to be active. Users can enable any subset.

### 3.5 Profile-Driven
Performance estimates come from MLX production logs on Apple M1 Max,
not paper benchmarks on NVIDIA hardware.

### 3.6 Vectorized
All critical paths use MLX array operations. No Python `.item()` loops in
hot paths. The ToMe scatter-add uses `mx.array.at[].add()` for GPU-native
accumulation.

## 4) Orchestrator Architecture

The `DiffusionOptimizer` is the central composition point:

```
                        ┌─────────────────────┐
                        │  DiffusionOptimizer  │
                        └────────┬────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
    Step-level decision    Token-level          Block-level decision
          │                      │                      │
    ┌─────┴──────┐         ┌─────┴─────┐          ┌─────┴──────┐
    │ TeaCache   │         │ ToMe      │          │ PISA       │
    │ SpectralC  │         │ ToPi      │          │ DeepCache  │
    │ FBCache    │         │ ToCa      │          │ DiTFastAttn│
    │ SmoothC    │         │ DDiT      │          │ T-GATE     │
    └────────────┘         └───────────┘          └────────────┘
```

### Step-Cache Cascade Priority
When `should_compute_step()` is called, the first configured cache wins:
1. **TeaCache** — polynomial-calibrated L1 distance (best quality)
2. **SpectralCache** — FFT-based HF energy detection
3. **FBCache** — first-block output distance (zero calibration)

### Block Strategy
Each transformer block gets one of:
- **COMPUTE** — full forward pass
- **SKIP** — reuse cached output (TeaCache decided to skip)
- **APPROXIMATE** — PISA identity + scale (low-sensitivity blocks)

## 5) mlx-mfa Integration

`mlx-diffusion-kit` connects to `mlx-mfa` via the `attn_bias` bridge:

```python
merged_tokens, info = tome_merge(tokens, config)
bias = compute_attn_bias_for_mfa(info)  # [1, 1, 1, N_merged]
# Pass to mlx-mfa:
output = flash_attention(q, k, v, attn_bias=bias)
```

The bias vector `log(count)` ensures merged tokens receive proportionally
more attention, correcting for the reduced token count.

## 6) Maturity Levels

Components are tracked via `mlx_diffusion_kit.maturity`:

| Level | Meaning | Count |
|-------|---------|-------|
| STABLE | Tested, production-ready, API frozen | 9 |
| BETA | Functional and tested, API may change | 9 |
| EXPERIMENTAL | Functional, limited testing | 2 |
| STUB | Interface only, no implementation | 3 |
