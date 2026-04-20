# mlx-diffusion-kit API Manual

Version: **0.2.1** — 89 public exports.

All exports are available from the top-level package:
```python
import mlx_diffusion_kit as mdk
mdk.__version__  # "0.2.1"
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

---

## Rebuilt / Added in P7 (2026-04-07)

These components were audited against their reference papers, found to be
shallow or semantically incorrect in the initial release, and rebuilt.

### FBCache (B2)

**`FBCacheConfig`** — Configuration for First-Block Cache.
- `rel_l1_thresh: float = 0.1` — relative L1 threshold on first-block output.
- `start_step: int = 0` / `end_step: int | None` — caching window.
- `max_consecutive_cached: int = 5` — safety ceiling.
- `enabled: bool = True`

**`fbcache_should_compute_remaining(fb_output, step_idx, config, state) -> bool`**
True if the caller must run blocks 2..N; False to reuse via reconstruct.

**`fbcache_update(fb_output, residual, state) -> None`**
Record `(fb, residual = full_output - fb_output)` after a compute step.

**`fbcache_reconstruct(fb_output, state) -> mx.array`**
Return `fb_output + cached_residual`. Raises if no cache populated.

**`fbcache_reset(state) -> None`** — Clear state.

**`FBCacheState`** — Mutable per-inference state. Attributes:
- `prev_fb_output: mx.array | None`
- `cached_residual: mx.array | None`
- `consecutive_cached: int`

**`create_fbcache_state() -> FBCacheState`** — Fresh state factory.

### SpectralCache (B3)

**`SpectralCacheConfig`** — Frequency-domain feature caching.
- `low_freq_ratio: float = 0.25`
- `cache_interval_low: int = 4` / `cache_interval_high: int = 1`
- `transform: "rfft" | "dct"` — DCT raises NotImplementedError.
- `spectral_velocity_aware: bool = False` — SeaCache variant.
- `velocity_override_thresh: float = 0.5`
- `enabled: bool = True`

**`spectral_cache_apply(features, step_idx, config, state) -> mx.array`**
Round-trip via rFFT: split LF/HF, apply per-band caching policy, combine,
inverse transform. Identity (up to f32 precision) when both intervals = 1.

**`spectral_cache_update(features, step_idx, config, state) -> None`**
Force-refresh both bands from features.

**`spectral_cache_reset(state) -> None`**

**`SpectralCacheState`** — Attributes:
- `cached_low_freq: mx.array | None`
- `cached_high_freq: mx.array | None`
- `last_low_recompute_step: int` / `last_high_recompute_step: int`
- `prev_full_spectra: list[tuple[mx.array, mx.array]]` — SeaCache history.

**`create_spectral_cache_state() -> SpectralCacheState`** — Fresh state factory.

### DeepCache (B5)

**`DeepCacheConfig`** — UNet deep-branch caching.
- `cache_interval: int = 3` / `start_step: int = 0` / `enabled: bool = True`

**`deepcache_should_recompute(step_idx, config, state) -> bool`**
Delta-based (not modulo), so TeaCache-skipped step sequences stay correct.

**`deepcache_store(features, step_idx, state)`** / **`deepcache_get(state)`** / **`deepcache_reset(state)`**.

**`DeepCacheState`** — Attributes:
- `cached_deep_features: mx.array | None`
- `last_recompute_step: int` (−1 before first store)
- `recompute_count: int` — total recomputes across the run.

**`create_deepcache_state() -> DeepCacheState`** — Fresh state factory.

### MosaicDiff layer redundancy (moved out of DeepCache)

**`analyze_layer_redundancy(layer_weights: dict, method="cosine"|"l2") -> dict[int, float]`**
**`select_cacheable_layers(scores, ratio=0.5) -> list[int]`**

### ToCa (B7)

**`ToCaConfig`** — Per-layer velocity-based token caching.
- `recompute_ratio: float = 0.5`
- `score_mode: "velocity" | "magnitude"`
- `enabled: bool = True`

**`toca_select_tokens(tokens, layer_idx, step_idx, config, state) -> (active, cached)`**
Returns two sorted-by-position index arrays partitioning `[0, N)`.
Fallback to all-active when history is insufficient.

**`toca_compose(active_feats, cached_feats, active_idx, cached_idx, total_n) -> mx.array`**
Reassemble full `[B, N, D]` tensor from disjoint pieces.

**`toca_update(layer_idx, tokens, state)`** — shifts `prev ← cached`, `cached ← tokens`.

**`toca_get_cached(layer_idx, state)`** / **`toca_reset(state)`**.

**`ToCaLayerState`** — Per-layer history slot. Attributes:
- `cached_tokens: mx.array | None` — most recent tokens.
- `prev_tokens: mx.array | None` — one step older; enables velocity scoring.
- `step_count: int`

**`ToCaState`** — Container over all layers. Use `state.layer(idx)` to
get-or-create the `ToCaLayerState` for a given layer.

**`create_toca_state() -> ToCaState`** — Fresh state factory.

### DiTFastAttn (B12)

**`AttnStrategy`** — Enum: `FULL`, `WINDOW`, `SHARE`, `RESIDUAL`.

**`DiTFastAttnConfig`** — Per-layer policy.
- `window_start_step: int = 10`
- `window_size: int = 64`
- `sharing_layers: list[int]` — layers eligible for SHARE.
- `residual_cache_layers: list[int]` — layers eligible for RESIDUAL.

**`ditfastattn_decide(layer_idx, step_idx, config, state) -> AttnStrategy`**
Precedence: step 0 → FULL; RESIDUAL > SHARE > WINDOW > FULL, with
safety fallback when a required cache is missing.

**`ditfastattn_record_attn_map`** / **`ditfastattn_get_cached_attn`**
**`ditfastattn_record_residual`** / **`ditfastattn_get_cached_residual`**
**`ditfastattn_reset`**

**`DiTFastAttnState`** — Attributes:
- `cached_attn_maps: dict[int, mx.array]` — for SHARE strategy.
- `cached_residuals: dict[int, mx.array]` — for RESIDUAL strategy.

**`create_ditfastattn_state() -> DiTFastAttnState`** — Fresh state factory.

### Separable Conv3D (B18)

**`SeparableConv3D(in_channels, out_channels, kernel_size=(kT,kH,kW), mid_channels=None, ...)`**
R(2+1)D `nn.Module`: spatial `Conv2d((kH,kW))` then temporal `Conv1d(kT)`.
Input `(N, T, H, W, in)` → output `(N, T', H', W', out)`.

**`decompose_conv3d_to_separable(conv3d_weight, rank=None) -> (spatial, temporal, error)`**
SVD-based factorization of a pretrained `Conv3d.weight[out, kT, kH, kW, in]`.
At full rank, reconstruction error < 1e-4 (float32). At reduced rank, the
decomposition is lossy and the returned error lets the caller judge.

**`build_separable_from_decomposition(spatial, temporal, in_ch, out_ch, kernel_size, ...) -> SeparableConv3D`**
Convenience bridge from Mode B output to a Mode A module.
