# CLAUDE.md

## MANDATORY — Self-update
After every significant change (new module, API change, bug fix, design decision), update this CLAUDE.md with the relevant information. This file is the single source of truth for Claude Code sessions.

## MANDATORY — Git safety
NEVER delete, move, or `rm -rf` the repository directory.
ALWAYS `git push origin master` before starting any destructive operation.
NEVER run `rm -rf` on any path containing the repo root.

## Python environment
Always use the shared venv: `~/code/venv_mlx_vae_gemini`

This venv contains all required dependencies: mlx, numpy, pytest, mlx-mfa (mlx-flashattention-steel).

Install in dev mode:
```bash
cd ~/code/mlx-diffusion-kit
~/code/venv_mlx_vae_gemini/bin/python -m pip install -e ".[dev]"
```

Run tests:
```bash
~/code/venv_mlx_vae_gemini/bin/python -m pytest tests/ -q
```

## What is this project?

`mlx-diffusion-kit` is a Python library of inference optimizations for diffusion and video super-resolution (VSR) models on MLX / Apple Silicon. It complements `mlx-mfa` (mlx-flashattention-steel) which handles Metal Flash Attention kernels.

**Scope**: everything OUTSIDE the attention kernel — step caching, token merging/pruning, cross-attention gating, VAE optimization, scheduling, orchestration.

**NOT in scope**: attention kernels themselves (those live in mlx-mfa), model weights, training.

## Relationship to mlx-mfa

```
mlx-mfa (mlx-flashattention-steel)     mlx-diffusion-kit
┌──────────────────────────────┐      ┌──────────────────────────────┐
│ Flash Attention kernels      │      │ Step-level caching           │
│ Sparse / GNA / Paged         │◄─────│ Token merging / pruning      │
│ KV cache management          │      │ Cross-attention gating       │
│ SVDQuantLinear               │      │ VAE optimization             │
│ TurboQuant KV                │      │ Scheduling                   │
│ attn_bias (Plan A — A1)      │      │ Orchestrator (PISA)          │
└──────────────────────────────┘      └──────────────────────────────┘
         Kernel layer                      Optimization layer
```

mlx-diffusion-kit is an **optional** dependency of mlx-mfa. Some features (Token Merging proportional attention) use `attn_bias` from mlx-mfa when available, but all components work standalone.

## Target models — 11 MLX VSR ports

### Single-step (no inter-step caching)
| Model | Backbone | VAE | Key trait |
|---|---|---|---|
| **SeedVR2** | DiT 48b, adaptive window | Custom causal 3D | Production ref. DiT=22%, VAE=77% |
| **DOVE** | DiT CogVideoX1.5-5B | CogVideoX | Single-step DiT |
| **FlashVSR** | DiT Wan2.1, LCSA | WanVAE + TC Decoder | LCSA sparse attn |
| **DLoRAL** | UNet SD, Dual-LoRA | SD VAE | ~1B params |
| **UltraVSR** | UNet SD + RTS | SD VAE | ~1B params |

### Multi-step (step caching applicable)
| Model | Backbone | VAE | Steps |
|---|---|---|---|
| **SparkVSR** | DiT CogVideoX1.5-5B-I2V | CogVideoX | ~20–30 |
| **STAR** | DiT CogVideoX-5B | CogVideoX | Multi |
| **Vivid-VR** | DiT CogVideoX1.5-5B + CN | CogVideoX | Multi |
| **DAM-VSR** | SVD UNet + CN, dual UNet | SD VAE | ~30 |
| **DiffVSR** | UNet SD | SD VAE | 20–50 |
| **VEnhancer** | ControlNet + ModelScope UNet | SD VAE | 15–50 |

### Performance profile (SeedVR2, verified production)
| Phase | % du total |
|---|---|
| VAE encode | 22.5% |
| DiT | 21.8% |
| VAE decode | 54.4% |
| Post-process | 1.3% |

**Key insight**: SeedVR2 strategy is VAE-first. DiT optimizations (ToMe, PISA) give ~8–13% e2e. VAE optimizations (B17+B18+B19) give ~35–50% e2e.

## Architecture

```
mlx_diffusion_kit/
├── cache/                    # SECTION I — Step-Level (6 multi-step models)
│   ├── teacache.py           # B1 — TeaCache + motion-aware (WorldCache)
│   ├── fbcache.py            # B2 — First-Block Cache
│   ├── spectral_cache.py     # B3 — SpectralCache + ERTACache
│   ├── smooth_cache.py       # B4 — SmoothCache + Taylor
│   ├── deep_cache.py         # B5 — DeepCache + MosaicDiff (UNet only)
│   ├── multigranular.py      # B6 — BWCache, UniCP, QuantCache
│   ├── motion.py             # WorldCache motion estimation
│   └── coefficients/         # Pre-calibrated coefficients (JSON)
├── tokens/                   # SECTION II — Token-Level (ALL models)
│   ├── tome.py               # B8 — Token Merging (ToMe/ToFu)
│   ├── toca.py               # B7 — TokenCache / ToCa (multi-step)
│   ├── pruning.py            # B8 — ToPi pruning
│   ├── learned_sparsity.py   # B9 — DiffSparse / E-DiT
│   └── ddit_scheduling.py    # B10 — Dynamic patch scheduling
├── gating/                   # SECTION III — Conditional Gating (multi-step)
│   └── tgate.py              # B11 — T-GATE cross-attention gating
├── attention/                # SECTION IV — Attention Compression
│   ├── ditfastattn.py        # B12 — DiTFastAttn orchestration
│   └── residual.py
├── quality/                  # SECTION V — Quality (training-free)
│   └── freeu.py              # B13 — FreeU (UNet re-weighting)
├── scheduler/                # SECTION VI — Scheduler (multi-step)
│   ├── dpm_solver_v3.py      # B14.1
│   └── adaptive_stepping.py  # B14.2
├── encoder/                  # SECTION VII — Text Encoder (ALL models)
│   └── embedding_cache.py    # B15 — Pre-computed text embeddings
├── vae/                      # SECTION VIII — VAE Optimization
│   ├── wavelet_cache.py      # B17 — WF-VAE causal cache
│   ├── separable_conv3d.py   # B18 — (2+1)D depthwise separable
│   └── decoder_distill.py    # B19 — Neodragon / Flash-VAED
└── orchestrator.py           # SECTION IX — B23 Orchestrator + PISA
```

## Design principles

1. **Model-type aware**: `is_single_step` gates all inter-step logic.
2. **Static shapes (Shiva-DiT)**: Post-merge token count = fixed multiple (N/2, N/4).
3. **VAE-ecosystem multiplier**: One VAE optimization benefits all models sharing it.
4. **Training-free first**: Phases 1–2 entirely training-free. Phase 3 introduces optional learned components.
5. **Composable**: Each component works standalone. B23 orchestrates but doesn't require all.
6. **Profile-driven**: Gains estimated from MLX production logs, not paper profiles.

## Component IDs

| ID | Component | Applies to | Phase |
|---|---|---|---|
| B1 | TeaCache + motion | 6 multi-step | 1 |
| B2 | First-Block Cache | 6 multi-step | 2 |
| B3 | SpectralCache | 6 multi-step | 2 |
| B4 | SmoothCache | 6 multi-step | 4 |
| B5 | DeepCache | 3 UNet multi-step | 2 |
| B6 | Multi-granular cache | 6 multi-step | 4 |
| B7 | ToCa | 6 multi-step | 3 |
| B8 | Token Merging | ALL 11 | 2 |
| B9 | DiffSparse | DiT models | 3 |
| B10 | DDiT scheduling | multi-step DiT | 4 |
| B11 | T-GATE | 6 multi-step | 1 |
| B12 | DiTFastAttn | multi-step DiT | 3 |
| B13 | FreeU | 5 UNet | 1 |
| B14 | DPM-Solver-v3 / adaptive | 6 multi-step | 2–4 |
| B15 | Text embedding cache | ALL 11 | 1 |
| B17 | WF-VAE causal cache | SeedVR2 + CogVideoX | 2 |
| B18 | Separable Conv3D | SeedVR2 | 3 |
| B19 | Neodragon decoder distill | SeedVR2 | 3 |
| B23 | Orchestrator + PISA | ALL 11 | 3 |

## Testing conventions

- Every component gets its own `tests/test_{component}.py`
- Test with synthetic data (random mx.arrays), no model weights required
- Test round-trip: merge/unmerge, cache/retrieve, gate/ungate
- Test shape preservation and numerical sanity (finite, reasonable range)
- Test config edge cases (disabled, 0 merge ratio, etc.)

## Key technical decisions

- **Vectorized scatter-add** — ToMe uses `mx.array.at[].add()` for GPU-native accumulation. No Python `.item()` loops in hot paths.
- **Delta-based recompute** — Encoder sharing uses `step_idx - last_computed >= interval` instead of modulo, robust to TeaCache-skipped steps.
- **Atomic cache writes** — TextEmbeddingCache writes via tmp+rename to prevent corruption on crash.
- **Cascade step-cache priority** — Orchestrator: TeaCache > SpectralCache > FBCache (first configured wins).
- **LCSA compatibility** — ToMe caps merge_ratio at 0.3 when `lcsa_compatible=True` for FlashVSR.
- **DiffSparse strict guard** — `strict=True` raises RuntimeError without pretrained weights to prevent silent fallback.
- **Motion-adjusted threshold** — WorldCache formula: `threshold / (1 + sensitivity * motion_magnitude)`.

## Current implementation status

- **89 public exports** in `__init__.py`
- **23 components** (9 STABLE, 11 BETA, 1 EXPERIMENTAL, 2 STUB)
- **276+ tests** across 27 test files
- **2 functional scripts** (calibrate_teacache.py, analyze_layer_redundancy.py)

## Output constraint — MANDATORY
NEVER produce a monolithic response exceeding 20000 tokens.
### Reading large files
NEVER open an entire file without checking its size first. Before reading any source file:
1. Run `wc -l <file>` to check line count
2. If > 500 lines, NEVER read the whole file. Instead:
   - Use `grep -n` to locate relevant sections
   - Use `head -n` / `tail -n` to read specific portions
   - Use `sed -n 'START,ENDp'` to extract targeted line ranges
   - Read the file in chunks using view with line ranges
3. If you need to understand a file's structure, use `grep -n "function\|class\|struct\|def \|void \|enum" <file>` first
### Writing output
For long tasks, systematically break down the work:
1. Make ONE change (one fix, one file)
2. Commit
3. Test
4. Briefly summarize what was done (~200 words max)
5. Move to the next change
NEVER write long recap reports at the end of a session.
Summarize in 500 words maximum, using a table format when relevant.
