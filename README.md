# mlx-diffusion-kit

Inference optimizations for diffusion and video super-resolution models on MLX / Apple Silicon.

Complements [mlx-mfa](https://github.com/your-org/mlx-mfa) (Flash Attention kernels) with everything outside the attention kernel: step caching, token merging/pruning, cross-attention gating, VAE optimization, scheduling, and orchestration.

## Installation

```bash
pip install -e ".[dev]"
```

With mlx-flashattention-steel integration:
```bash
pip install -e ".[mfa]"
```

## Components

| ID | Component | Section | Applies to |
|----|-----------|---------|------------|
| B1 | TeaCache + motion | Step Cache | 6 multi-step |
| B2 | First-Block Cache | Step Cache | 6 multi-step |
| B3 | SpectralCache | Step Cache | 6 multi-step |
| B4 | SmoothCache | Step Cache | 6 multi-step |
| B5 | DeepCache | Step Cache | 3 UNet multi-step |
| B6 | Multi-granular cache | Step Cache | 6 multi-step |
| B7 | ToCa | Tokens | 6 multi-step |
| B8 | Token Merging (ToMe) | Tokens | ALL 11 |
| B9 | DiffSparse | Tokens | DiT models |
| B10 | DDiT scheduling | Tokens | multi-step DiT |
| B11 | T-GATE | Gating | 6 multi-step |
| B12 | DiTFastAttn | Attention | multi-step DiT |
| B13 | FreeU | Quality | 5 UNet |
| B14 | DPM-Solver-v3 | Scheduler | 6 multi-step |
| B15 | Text Embedding Cache | Encoder | ALL 11 |
| B17 | WF-VAE causal cache | VAE | SeedVR2 + CogVideoX |
| B18 | Separable Conv3D | VAE | SeedVR2 |
| B19 | Neodragon decoder | VAE | SeedVR2 |
| B23 | Orchestrator + PISA | Orchestrator | ALL 11 |

## License

MIT
