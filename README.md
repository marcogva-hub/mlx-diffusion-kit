# mlx-diffusion-kit

Inference optimizations for diffusion and video super-resolution models on
MLX / Apple Silicon. Training-free techniques that reduce compute by 2-5x
without quality loss.

Current version: **0.2.1** — 23 optimization components, 284+ tests.

## Foreword

This library was born from the same frustration that drove
[mlx-mfa](https://github.com/marcogva-hub/mlx-flashattention-steel): VSR
inference on Apple Silicon is painfully slow. `mlx-mfa` tackles the attention
kernel; `mlx-diffusion-kit` tackles everything else — step caching, token
merging, cross-attention gating, VAE optimization, scheduling, and
orchestration.

The two libraries are complementary:

```
mlx-mfa                              mlx-diffusion-kit
┌──────────────────────────┐        ┌──────────────────────────┐
│ Flash Attention kernels  │        │ Step-level caching       │
│ Sparse / GNA / Paged     │◄───────│ Token merging / pruning  │
│ KV cache management      │        │ Cross-attention gating   │
│ attn_bias (native Metal) │        │ VAE optimization         │
│ TurboQuant KV            │        │ Scheduling               │
│ SVDQuantLinear            │        │ Orchestrator (PISA)      │
└──────────────────────────┘        └──────────────────────────┘
        Kernel layer                     Optimization layer
```

## Installation

```bash
pip install -e ".[dev]"
```

With mlx-mfa integration (for proportional attention via `attn_bias`):

```bash
pip install -e ".[mfa]"
```

Requirements: Python >= 3.10, MLX >= 0.25.0, Apple Silicon Mac.

## Current Status

- **9 STABLE** components — tested, integrated, production-ready API.
- **11 BETA** components — functional and tested, API may evolve.
- **1 EXPERIMENTAL** — functional, use with caution.
- **2 STUB** — interface defined, implementation pending.
- **276+ tests** pass, 0 failures.
- Primary validation hardware: Apple M1 Max.

## Component Overview

| ID | Component | Section | Maturity | Applies to |
|----|-----------|---------|----------|------------|
| B1 | TeaCache + WorldCache | Step Cache | Stable | 6 multi-step |
| B2 | First-Block Cache | Step Cache | Beta | 6 multi-step |
| B3 | SpectralCache | Step Cache | Beta | 6 multi-step |
| B4 | SmoothCache + Taylor | Step Cache | Stable | 6 multi-step |
| B5 | DeepCache (+ MosaicDiff layer-redundancy tool) | Step Cache | Beta | 5 UNet multi-step |
| B6 | Multi-Granular Cache | Step Cache | Beta | 6 multi-step |
| B7 | ToCa (Token Cache) | Tokens | Beta | multi-step DiT |
| B8 | ToMe + ToPi | Tokens | Stable / Beta | ALL 11 |
| B9 | DiffSparse | Tokens | Stub | DiT models |
| B10 | DDiT Scheduling | Tokens | Beta | multi-step DiT |
| B11 | T-GATE | Gating | Stable | 6 multi-step |
| B12 | DiTFastAttn (4 strategies) | Attention | Beta | multi-step DiT |
| B13 | FreeU | Quality | Stable | 5 UNet |
| B14 | DPM-Solver-v3 / Adaptive | Scheduler | Stable / Beta | 6 multi-step |
| B15 | Text Embedding Cache | Encoder | Stable | ALL 11 |
| B17 | WF-VAE Causal Cache | VAE | Stable | SeedVR2 + CogVideoX |
| B18 | Separable Conv3D + SVD utility | VAE | Beta | SeedVR2 VAE |
| B22 | Encoder Sharing | Cache | Beta | multi-step DiT |
| B23 | Orchestrator + PISA | Orchestrator | Stable | ALL 11 |

## Quick Start

```python
import mlx_diffusion_kit as mdk

# 1. Cache text embeddings (all models)
emb_cache = mdk.TextEmbeddingCache()
embedding = emb_cache.get_or_compute("enhance 4x", my_t5_encoder, encoder_id="t5-xxl")

# 2. Step caching for multi-step models
from mlx_diffusion_kit.cache import TeaCacheConfig, load_coefficients
config = load_coefficients("cogvideox")  # Pre-calibrated

# 3. Token merging (all models)
merged, info = mdk.tome_merge(tokens, mdk.ToMeConfig(merge_ratio=0.5))
# ... run attention on merged tokens ...
output = mdk.tome_unmerge(merged_output, info)

# 4. Full orchestration
from mlx_diffusion_kit.orchestrator import DiffusionOptimizer, OrchestratorConfig
opt = DiffusionOptimizer(OrchestratorConfig(
    teacache=config,
    tome=mdk.ToMeConfig(merge_ratio=0.5),
    tgate=mdk.TGateConfig(gate_step=5),
    is_single_step=False,
))
```

See `docs/API_MANUAL.md` for complete API reference.

## Target Models

### Single-step (no inter-step caching)
| Model | Backbone | Key trait |
|-------|----------|-----------|
| SeedVR2 | DiT 48b | Production ref. DiT=22%, VAE=77% |
| DOVE | DiT CogVideoX1.5-5B | Single-step DiT |
| FlashVSR | DiT Wan2.1, LCSA | Sparse attention |
| DLoRAL | UNet SD, Dual-LoRA | ~1B params |
| UltraVSR | UNet SD + RTS | ~1B params |

### Multi-step (step caching applicable)
| Model | Backbone | Steps |
|-------|----------|-------|
| SparkVSR | DiT CogVideoX1.5-5B-I2V | ~20-30 |
| STAR | DiT CogVideoX-5B | Multi |
| Vivid-VR | DiT CogVideoX1.5-5B + CN | Multi |
| DAM-VSR | SVD UNet + CN | ~30 |
| DiffVSR | UNet SD | 20-50 |
| VEnhancer | ControlNet + ModelScope UNet | 15-50 |

## Scripts

```bash
# Calibrate TeaCache coefficients for a new model
python scripts/calibrate_teacache.py --features-dir ./features/ --output coefficients.json

# Analyze layer redundancy for DeepCache
python scripts/analyze_layer_redundancy.py --weights model.npz --output scores.json
```

## Documentation

- `docs/API_MANUAL.md` — Complete API reference for all 89 exports
- `docs/ARCHITECTURE.md` — Module structure and design principles
- `CHANGELOG.md` — Version history

## License

MIT — see `LICENSE` for details.
