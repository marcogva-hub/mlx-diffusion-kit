# mlx-diffusion-kit API Manual

Version: **0.1.0** — 46 public exports.

All exports are available from the top-level package:
```python
import mlx_diffusion_kit as mdk
mdk.__version__  # "0.1.0"
```

**`__version__`** — Package version string.

---

## Step-Level Caching

### TeaCache (B1)

**`TeaCacheConfig`** — Configuration for timestep-aware step caching.
- `rel_l1_thresh: float = 0.3` — L1 distance threshold for skipping.
- `poly_coeffs: list[float] | None` — Polynomial rescaling coefficients.
- `start_step: int = 0` — First step where caching is active.
- `end_step: int | None` — Last step (None = until end).
- `max_consecutive_cached: int = 5` — Max consecutive skips.
- `motion: MotionConfig | None` — WorldCache motion extension.
- `enabled: bool = True`

**`teacache_should_compute(modulated_input, step_idx, config, state) -> bool`**
Decide whether to compute or reuse cache for this step.

**`teacache_update(modulated_input, output, state) -> None`**
Update cache state after a computed step.

**`load_coefficients(model_name: str) -> TeaCacheConfig`**
Load pre-calibrated coefficients from `cache/coefficients/`. Available: `"cogvideox"`.

### SmoothCache (B4)

**`SmoothCacheConfig`** — Interpolation for skipped steps.
- `mode: InterpolationMode = LINEAR` — LINEAR, TAYLOR_1, or TAYLOR_2.
- `enabled: bool = True`

**`InterpolationMode`** — Enum: `LINEAR`, `TAYLOR_1`, `TAYLOR_2`.

**`smooth_cache_record(step_idx, features, state) -> None`**
Record features from a computed step.

**`smooth_cache_interpolate(target_step, state, config) -> mx.array`**
Interpolate features for a skipped step.

---

## Token-Level Optimizations

### ToMe (B8)

**`ToMeConfig`** — Token merging configuration.
- `merge_ratio: float = 0.5` — Fraction of tokens to merge.
- `similarity: str = "cosine"` — `"cosine"` or `"l2"`.
- `use_mlerp: bool = True` — Norm-preserving merge.
- `lcsa_compatible: bool = False` — Cap ratio at 0.3 for FlashVSR.
- `enabled: bool = True`

**`tome_merge(tokens, config, spatial_dims=None, spatial_weight=0.3, temporal_weight=0.5) -> (merged, MergeInfo)`**
Merge similar tokens via bipartite matching. Supports [B,N,D] and [B,H,N,D].
Optional `spatial_dims=(T,H,W)` enables video-aware spatiotemporal scoring.

**`tome_unmerge(merged, info) -> mx.array`**
Reconstruct full token sequence from merged tokens.

**`compute_proportional_bias(info) -> mx.array`**
Token counts as `log(count)` vector [N_merged] for attention weighting.

**`compute_attn_bias_for_mfa(info) -> mx.array`**
Proportional bias in [1,1,1,N_merged] shape for `mlx-mfa` `flash_attention(attn_bias=...)`.

**`compute_spatiotemporal_similarity(tokens, spatial_dims, spatial_weight, temporal_weight, config) -> mx.array`**
Combined cosine + spatial + temporal proximity similarity matrix [B,N,N].

### ToPi (B8)

**`ToPiConfig`** — Token pruning configuration.
- `prune_ratio: float = 0.3` — Fraction to prune.
- `importance: str = "norm"` — `"attention"`, `"norm"`, or `"random"`.
- `restore_mode: str = "copy"` — `"copy"`, `"zero"`, or `"lerp"`.
- `enabled: bool = True`

**`topi_prune(tokens, config, attention_weights=None) -> (pruned, PruneInfo)`**
Prune low-importance tokens. Returns [B, N_kept, D].

**`topi_restore(pruned_output, info, config) -> mx.array`**
Restore pruned tokens to original length [B, N, D].

### DiffSparse (B9) — Stub

**`DiffSparseConfig`** — Learned router configuration.
- `budget: float = 0.5` — Fraction of tokens to keep.
- `router_hidden_dim: int = 64`
- `strict: bool = False` — Raise RuntimeError without pretrained weights.
- `enabled: bool = True`

**`DiffSparseRouter(input_dim, config)`** — `nn.Module` router MLP.
Without pretrained weights: deterministic fallback (first N tokens).
With `strict=True`: raises RuntimeError. Use `from_pretrained(path)` to load weights.

---

## Gating

### T-GATE (B11)

**`TGateConfig`** — Cross-attention gating.
- `gate_step: int = 5` — Step after which cross-attention is cached.
- `enabled: bool = True`

**`tgate_forward(layer_idx, step_idx, config, state, self_attn_fn, cross_attn_fn, x, context) -> mx.array`**
Execute transformer block with T-GATE gating.

**`create_tgate_state() -> TGateState`**
Create fresh T-GATE state.

---

## Quality

### FreeU (B13)

**`FreeUConfig`** — UNet skip connection re-weighting.
- `b1: float = 1.2` / `b2: float = 1.4` — Backbone scale factors.
- `s1: float = 0.9` / `s2: float = 0.2` — Skip attenuation factors.
- `enabled: bool = True`

**`freeu_filter(h_skip, h_backbone, config) -> (filtered_skip, scaled_backbone)`**
Apply FreeU FFT-based re-weighting. Training-free.

---

## Schedulers

### DPM-Solver-v3 (B14.1)

**`DPMSolverV3Config`** — High-order ODE solver.
- `order: int = 3` — 1 (DDIM), 2, or 3.
- `num_steps: int = 15`
- `predict_type: str = "epsilon"` — `"epsilon"`, `"v_prediction"`, `"x_start"`.
- `enabled: bool = True`

**`DPMSolverV3(noise_schedule, config)`** — Solver instance.
- `.get_timesteps() -> mx.array` — Log-SNR uniform timesteps.
- `.step(model_output, timestep_idx, sample) -> mx.array` — One solver step.
- `.reset()` — Clear history for new run.

**`NoiseSchedule(schedule_type, beta_start, beta_end, num_train_timesteps)`**
VP noise schedule with `alpha_bar(t)`, `log_snr(t)`, `inverse_log_snr(lsnr)`.

### Adaptive Stepping (B14.2)

**`AdaptiveStepConfig`** — Convergence-based step pruning.
- `min_steps: int = 10` — Hard floor.
- `tolerance: float = 0.01` — MSE convergence threshold.
- `enabled: bool = True`

**`AdaptiveStepScheduler(base_timesteps, config)`**
- `.should_skip_step(step_idx, prev_output, curr_output) -> bool`
- `.get_effective_timesteps() -> list[float]`
- `.reset()`

---

## Encoder

### TextEmbeddingCache (B15)

**`TextEmbeddingCache(cache_dir)`** — Disk-backed text encoder cache.
- `.get_or_compute(prompt, encoder_fn, encoder_id="default", **kwargs) -> mx.array`
  Cache key includes encoder_id + prompt + kwargs. Atomic write via tmp+rename.
- `.clear()` — Remove all cached embeddings.
- `.cache_size() -> int`

---

## VAE

### WaveletVAECache (B17)

**`WaveletCacheConfig`** — Causal conv state cache.
- `enabled: bool = True`
- `max_cached_layers: int | None`

**`WaveletVAECache(config)`** — Per-layer conv state storage.
- `.get_state(layer_idx) -> mx.array | None`
- `.set_state(layer_idx, state)`
- `.clear()` / `.num_cached() -> int`

**`chunked_decode_with_cache(decode_fn, latent_chunks, cache, output_buffer=None, callback=None) -> mx.array`**
Decode chunks sequentially, propagating conv states. Three modes:
buffer (in-place write), callback (per-chunk), or list+concat.

---

## Orchestrator

### DiffusionOptimizer (B23)

**`OrchestratorConfig`** — Central configuration bundle.
All fields are `Optional` with `None` default (disabled when absent):
`teacache`, `fbcache`, `spectral_cache`, `smooth_cache`, `tome`, `tgate`,
`toca`, `ditfastattn`, `deep_cache`, `freeu`, `pisa`, `multigranular`,
`ddit_schedule`, `encoder_sharing`. Plus `is_single_step: bool`, `num_blocks: int`,
`total_steps: int`.

**`DiffusionOptimizer(config)`** — Central optimizer.
- `.should_compute_step(step_idx, modulated_input, ...) -> bool` — Step-cache cascade.
- `.update_step_cache(modulated_input, output, step_idx, ...) -> None`
- `.get_cached_output(step_idx) -> mx.array | None` — With SmoothCache interpolation.
- `.merge_tokens(tokens) -> mx.array` / `.unmerge_tokens(tokens) -> mx.array`
- `.should_compute_cross_attn(layer_idx, step_idx) -> bool` — T-GATE.
- `.get_block_strategy(block_idx, step_idx) -> BlockStrategy` — COMPUTE/SKIP/APPROXIMATE.
- `.should_compute_layer_deep(layer_idx, step_idx) -> bool` — DeepCache.
- `.should_recompute_encoder(step_idx) -> bool` — Encoder sharing.
- `.get_patch_stride(step_idx) -> int` — DDiT scheduling.
- `.reset()` — Clear all state.

**`BlockStrategy`** — Enum: `COMPUTE`, `SKIP`, `APPROXIMATE`.

**`PISAConfig`** — Profile-Informed Selective Activation.
- `approx_ratio: float = 0.3` — Fraction of blocks to approximate.
- `sensitivity_scores: dict[int, float] | None` — Block sensitivities.
- `enabled: bool = True`

---

## Supporting Types

**`DDiTScheduleConfig`** / **`DDiTScheduler`** — Dynamic patch stride scheduling.

**`EncoderSharingConfig`** — Delta-based encoder block reuse.

**`MultiGranularCache`** / **`MultiGranularConfig`** — BWCache + UniCP + QuantCache bundle.

**`Maturity`** — Enum: `STABLE`, `BETA`, `EXPERIMENTAL`, `STUB`.

**`get_maturity(component_id) -> Maturity`** — Query component maturity.

**`list_components(maturity=None) -> dict`** — List components by maturity level.
