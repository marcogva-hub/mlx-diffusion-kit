"""B23 — Diffusion Optimization Orchestrator + PISA.

Composes all optimization components (TeaCache, ToMe, T-GATE, FreeU, PISA)
into a unified decision layer. Three modes per transformer block per step:
  COMPUTE — full forward pass
  SKIP — reuse cached output from previous step (multi-step only)
  APPROXIMATE — PISA lightweight approximation (identity + scale)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import mlx.core as mx

from mlx_diffusion_kit.attention.ditfastattn import (
    DiTFastAttnConfig,
    DiTFastAttnManager,
    HeadStrategy,
)
from mlx_diffusion_kit.cache.deep_cache import (
    DeepCacheConfig,
    DeepCacheState,
    create_deepcache_state,
    deepcache_get,
    deepcache_reset,
    deepcache_should_recompute,
    deepcache_store,
)
from mlx_diffusion_kit.cache.motion import MotionConfig, MotionTracker
from mlx_diffusion_kit.cache.fb_cache import (
    FBCacheConfig,
    FBCacheState,
    create_fbcache_state,
    fbcache_reconstruct,
    fbcache_reset,
    fbcache_should_compute_remaining,
    fbcache_update,
)
from mlx_diffusion_kit.cache.multigranular import MultiGranularCache, MultiGranularConfig
from mlx_diffusion_kit.cache.encoder_sharing import (
    EncoderSharingConfig,
    EncoderSharingState,
    create_encoder_sharing_state,
    encoder_sharing_get_cached,
    encoder_sharing_should_recompute,
    encoder_sharing_update,
)
from mlx_diffusion_kit.cache.spectral_cache import (
    SpectralCacheConfig,
    SpectralCacheState,
    create_spectral_cache_state,
    spectral_cache_should_compute,
    spectral_cache_update,
)
from mlx_diffusion_kit.cache.smooth_cache import (
    SmoothCacheConfig,
    SmoothCacheState,
    create_smooth_cache_state,
    smooth_cache_interpolate,
    smooth_cache_record,
)
from mlx_diffusion_kit.cache.teacache import (
    TeaCacheConfig,
    TeaCacheState,
    create_teacache_state,
    teacache_should_compute,
    teacache_update,
)
from mlx_diffusion_kit.gating.tgate import (
    TGateConfig,
    TGateState,
    create_tgate_state,
)
from mlx_diffusion_kit.quality.freeu import FreeUConfig
from mlx_diffusion_kit.tokens.ddit_scheduling import DDiTScheduleConfig, DDiTScheduler
from mlx_diffusion_kit.tokens.toca import (
    ToCaConfig,
    ToCaState,
    create_toca_state,
    toca_compose,
    toca_reset,
    toca_select_tokens,
    toca_update,
)
from mlx_diffusion_kit.tokens.tome import (
    MergeInfo,
    ToMeConfig,
    tome_merge,
    tome_unmerge,
)


class BlockStrategy(Enum):
    COMPUTE = "compute"
    SKIP = "skip"
    APPROXIMATE = "approx"


@dataclass
class PISAConfig:
    """Profile-Informed Selective Activation configuration.

    Blocks with sensitivity below the threshold are approximated
    with identity + scale instead of full forward passes.
    """

    approx_ratio: float = 0.3
    sensitivity_scores: Optional[dict[int, float]] = None
    enabled: bool = True


@dataclass
class OrchestratorConfig:
    teacache: Optional[TeaCacheConfig] = None
    fbcache: Optional[FBCacheConfig] = None
    spectral_cache: Optional[SpectralCacheConfig] = None
    smooth_cache: Optional[SmoothCacheConfig] = None
    tome: Optional[ToMeConfig] = None
    tgate: Optional[TGateConfig] = None
    freeu: Optional[FreeUConfig] = None
    pisa: Optional[PISAConfig] = None
    toca: Optional[ToCaConfig] = None
    ditfastattn: Optional[DiTFastAttnConfig] = None
    deep_cache: Optional[DeepCacheConfig] = None
    multigranular: Optional[MultiGranularConfig] = None
    ddit_schedule: Optional[DDiTScheduleConfig] = None
    encoder_sharing: Optional[EncoderSharingConfig] = None
    is_single_step: bool = False
    num_blocks: int = 24
    total_steps: int = 50


class DiffusionOptimizer:
    """Orchestrates all optimization components for a diffusion pipeline.

    Decision logic per step:
    1. TeaCache: should we compute this step at all? (multi-step only)
    2. ToMe: merge tokens before attention
    3. Per-block: COMPUTE / SKIP / APPROXIMATE (PISA)
    4. T-GATE: gate cross-attention after convergence (multi-step only)
    5. ToMe: unmerge tokens after attention
    6. TeaCache: update cache with new output
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()

        # Initialize sub-component states
        self._teacache_state: Optional[TeaCacheState] = None
        if self.config.teacache and not self.config.is_single_step:
            self._teacache_state = create_teacache_state()

        self._motion_tracker: Optional[MotionTracker] = None
        if (
            self.config.teacache
            and self.config.teacache.motion
            and not self.config.is_single_step
        ):
            self._motion_tracker = MotionTracker(self.config.teacache.motion)

        self._fbcache_state: Optional[FBCacheState] = None
        if self.config.fbcache and not self.config.is_single_step:
            self._fbcache_state = create_fbcache_state()

        self._spectral_cache_state: Optional[SpectralCacheState] = None
        if self.config.spectral_cache and not self.config.is_single_step:
            self._spectral_cache_state = create_spectral_cache_state()

        self._tgate_state: Optional[TGateState] = None
        if self.config.tgate and not self.config.is_single_step:
            self._tgate_state = create_tgate_state()

        self._smooth_cache_state: Optional[SmoothCacheState] = None
        if self.config.smooth_cache and not self.config.is_single_step:
            self._smooth_cache_state = create_smooth_cache_state()

        self._encoder_sharing_state: Optional[EncoderSharingState] = None
        if self.config.encoder_sharing and not self.config.is_single_step:
            self._encoder_sharing_state = create_encoder_sharing_state()

        self._ddit_scheduler: Optional[DDiTScheduler] = None
        if self.config.ddit_schedule and not self.config.is_single_step:
            self._ddit_scheduler = DDiTScheduler(
                self.config.total_steps, self.config.ddit_schedule
            )

        self._toca_state: Optional[ToCaState] = None
        if self.config.toca and not self.config.is_single_step:
            self._toca_state = create_toca_state()

        self._ditfastattn: Optional[DiTFastAttnManager] = None
        if self.config.ditfastattn and not self.config.is_single_step:
            self._ditfastattn = DiTFastAttnManager(
                self.config.num_blocks,
                self.config.num_blocks,  # num_heads — overridden by config if needed
                self.config.ditfastattn,
            )

        self._deep_cache_state: Optional[DeepCacheState] = None
        if self.config.deep_cache and not self.config.is_single_step:
            self._deep_cache_state = create_deepcache_state()

        self._multigranular: Optional[MultiGranularCache] = None
        if self.config.multigranular and self.config.multigranular.enabled:
            mg = self.config.multigranular
            self._multigranular = MultiGranularCache(
                bw_config=mg.bw,
                unicp_config=mg.unicp,
                quant_config=mg.quant,
                layer_shapes=mg.layer_shapes,
            )

        self._last_merge_info: Optional[MergeInfo] = None
        self._block_cache: dict[int, mx.array] = {}
        self._pisa_threshold = self._compute_pisa_threshold()

    def _compute_pisa_threshold(self) -> float:
        """Compute the sensitivity threshold for PISA approximation.

        Blocks with sensitivity <= threshold are approximated.
        The threshold is set so that approx_ratio fraction of blocks
        fall at or below it.
        """
        pisa = self.config.pisa
        if not pisa or not pisa.enabled or not pisa.sensitivity_scores:
            return float("-inf")

        scores = sorted(pisa.sensitivity_scores.values())
        if not scores:
            return float("-inf")

        n_approx = max(1, int(len(scores) * pisa.approx_ratio))
        # Index of the highest score that should still be approximated
        idx = min(n_approx - 1, len(scores) - 1)
        return scores[idx]

    def get_block_strategy(self, block_idx: int, step_idx: int) -> BlockStrategy:
        """Determine execution strategy for a transformer block.

        Args:
            block_idx: Index of the transformer block.
            step_idx: Current diffusion step.

        Returns:
            BlockStrategy indicating how to execute this block.
        """
        # SKIP: if we have a cached output from a previous step (multi-step)
        if (
            not self.config.is_single_step
            and block_idx in self._block_cache
            and self._teacache_state is not None
            and self._teacache_state.cached_residual is not None
        ):
            return BlockStrategy.SKIP

        # APPROXIMATE: PISA — low-sensitivity blocks
        pisa = self.config.pisa
        if pisa and pisa.enabled and pisa.sensitivity_scores:
            score = pisa.sensitivity_scores.get(block_idx)
            if score is not None and score <= self._pisa_threshold:
                return BlockStrategy.APPROXIMATE

        return BlockStrategy.COMPUTE

    def should_compute_step(
        self,
        step_idx: int,
        modulated_input: mx.array,
        sigma_t: float = 0.5,
        frame: Optional[mx.array] = None,
    ) -> bool:
        """Check if the full model forward should run for this step.

        Step-level cascade (first configured wins):
          1. TeaCache — best quality, requires calibrated coefficients
          2. SpectralCache — frequency-domain skip decision (requires sigma_t)

        FBCache is NOT in this cascade: it decides at the transformer-block
        boundary (skip blocks 2..N), not the whole-step boundary. Use
        :meth:`should_compute_remaining_blocks` instead.

        Single-step models always return True.

        Args:
            step_idx: Current diffusion step.
            modulated_input: Timestep-modulated input tensor.
            sigma_t: Current noise level 0→1 (for SpectralCache).
            frame: Current video frame (for WorldCache motion adjustment).

        Returns:
            True if the step should be computed, False to reuse cache.
        """
        if self.config.is_single_step:
            return True

        # Priority 1: TeaCache (with optional motion adjustment)
        if self._teacache_state is not None and self.config.teacache is not None:
            cfg = self.config.teacache
            if self._motion_tracker is not None and frame is not None:
                self._motion_tracker.update(frame)
                adjusted = self._motion_tracker.get_adjusted_threshold(cfg.rel_l1_thresh)
                original = cfg.rel_l1_thresh
                cfg.rel_l1_thresh = adjusted
                result = teacache_should_compute(
                    modulated_input, step_idx, cfg, self._teacache_state
                )
                cfg.rel_l1_thresh = original
                return result
            return teacache_should_compute(
                modulated_input, step_idx, cfg, self._teacache_state
            )

        # Priority 2: SpectralCache
        if self._spectral_cache_state is not None and self.config.spectral_cache is not None:
            return spectral_cache_should_compute(
                modulated_input, sigma_t, self.config.spectral_cache, self._spectral_cache_state
            )

        return True

    # --- B2 FBCache (block-level, not step-level) ---

    def should_compute_remaining_blocks(
        self, first_block_output: mx.array, step_idx: int
    ) -> bool:
        """FBCache decision: should the caller run blocks 2..N this step?

        Returns True if FBCache is not configured (caller must compute)
        or if the change in first-block output is above threshold.
        Returns False if the caller should reuse the cached residual via
        :meth:`fbcache_reconstruct`.
        """
        if (
            self.config.is_single_step
            or self._fbcache_state is None
            or self.config.fbcache is None
        ):
            return True
        return fbcache_should_compute_remaining(
            first_block_output, step_idx, self.config.fbcache, self._fbcache_state
        )

    def fbcache_update_residual(
        self, fb_output: mx.array, residual: mx.array
    ) -> None:
        """Record fresh (first-block-output, residual) after running all blocks."""
        if self._fbcache_state is not None:
            fbcache_update(fb_output, residual, self._fbcache_state)

    def fbcache_reconstruct_output(self, fb_output: mx.array) -> mx.array:
        """Return fb_output + cached_residual. Requires prior update."""
        if self._fbcache_state is None:
            raise RuntimeError("FBCache is not configured on this optimizer.")
        return fbcache_reconstruct(fb_output, self._fbcache_state)

    def update_step_cache(
        self,
        modulated_input: mx.array,
        output: mx.array,
        step_idx: int = 0,
    ) -> None:
        """Update step-level caches after a computed step.

        Covers TeaCache, SpectralCache, and SmoothCache. FBCache is NOT
        updated here because it caches a residual, not a full output;
        use :meth:`fbcache_update_residual` from within the block-level
        decision flow.

        Args:
            modulated_input: The input that was computed.
            output: The model's output.
            step_idx: Current step index (used by SmoothCache for interpolation).
        """
        if self._teacache_state is not None:
            teacache_update(modulated_input, output, self._teacache_state)
        if self._spectral_cache_state is not None:
            spectral_cache_update(modulated_input, output, self._spectral_cache_state)
        if self._smooth_cache_state is not None:
            smooth_cache_record(step_idx, output, self._smooth_cache_state)

    def get_cached_output(self, step_idx: int = 0) -> Optional[mx.array]:
        """Retrieve output for a skipped step (step-level caches only).

        If SmoothCache is configured, returns interpolated features.
        Otherwise, returns the raw TeaCache cached residual, then
        SpectralCache's cached output as a fallback.

        Note: this method does NOT return FBCache's cached data.
        FBCache stores a residual (not a full output), and reconstruction
        requires the current first-block output — call
        :meth:`fbcache_reconstruct_output(fb_output)` at the block level.

        Args:
            step_idx: The step being skipped (for SmoothCache interpolation).
        """
        if (
            self._smooth_cache_state is not None
            and self.config.smooth_cache is not None
            and self._smooth_cache_state.history
        ):
            return smooth_cache_interpolate(
                step_idx, self._smooth_cache_state, self.config.smooth_cache
            )
        if self._teacache_state is not None:
            return self._teacache_state.cached_residual
        if self._spectral_cache_state is not None:
            return self._spectral_cache_state.cached_output
        return None

    def merge_tokens(self, tokens: mx.array) -> mx.array:
        """Apply ToMe token merging if configured.

        Args:
            tokens: Input tokens [B, N, D].

        Returns:
            Merged tokens (or original if ToMe disabled/unconfigured).
        """
        if self.config.tome is None:
            self._last_merge_info = None
            return tokens

        merged, info = tome_merge(tokens, self.config.tome)
        self._last_merge_info = info
        return merged

    def unmerge_tokens(self, tokens: mx.array) -> mx.array:
        """Reverse ToMe token merging.

        Args:
            tokens: Merged tokens.

        Returns:
            Unmerged tokens at original resolution.
        """
        if self._last_merge_info is None:
            return tokens
        return tome_unmerge(tokens, self._last_merge_info)

    def should_compute_cross_attn(self, layer_idx: int, step_idx: int) -> bool:
        """Check if cross-attention should be computed or use cache (T-GATE).

        Args:
            layer_idx: Transformer layer index.
            step_idx: Current diffusion step.

        Returns:
            True if cross-attention should run, False to use cached value.
        """
        if self.config.is_single_step:
            return True

        tgate = self.config.tgate
        if tgate is None or not tgate.enabled:
            return True

        return step_idx < tgate.gate_step

    def cache_cross_attn(self, layer_idx: int, value: mx.array) -> None:
        """Store a cross-attention output in T-GATE cache."""
        if self._tgate_state is not None:
            self._tgate_state.cached_cross_attn[layer_idx] = value

    def get_cached_cross_attn(self, layer_idx: int) -> Optional[mx.array]:
        """Retrieve cached cross-attention output."""
        if self._tgate_state is not None:
            return self._tgate_state.cached_cross_attn.get(layer_idx)
        return None

    @property
    def tgate_state(self) -> Optional[TGateState]:
        return self._tgate_state

    @property
    def teacache_state(self) -> Optional[TeaCacheState]:
        return self._teacache_state

    @property
    def smooth_cache_state(self) -> Optional[SmoothCacheState]:
        return self._smooth_cache_state

    @property
    def encoder_sharing_state(self) -> Optional[EncoderSharingState]:
        return self._encoder_sharing_state

    @property
    def multigranular_cache(self) -> Optional[MultiGranularCache]:
        return self._multigranular

    @property
    def ddit_scheduler(self) -> Optional[DDiTScheduler]:
        return self._ddit_scheduler

    # --- B22 Encoder Sharing ---

    def should_recompute_encoder(self, step_idx: int) -> bool:
        """Check whether shared encoder blocks need recomputing.

        Returns True for single-step models or when encoder sharing is not configured.
        """
        if self.config.is_single_step:
            return True
        if self._encoder_sharing_state is None or self.config.encoder_sharing is None:
            return True
        return encoder_sharing_should_recompute(
            step_idx, self.config.encoder_sharing, self._encoder_sharing_state
        )

    def get_cached_encoder_output(self) -> Optional[mx.array]:
        """Retrieve cached encoder output."""
        if self._encoder_sharing_state is not None:
            return encoder_sharing_get_cached(self._encoder_sharing_state)
        return None

    def update_encoder_cache(self, step_idx: int, encoder_output: mx.array) -> None:
        """Update encoder sharing cache after computing encoder blocks."""
        if self._encoder_sharing_state is not None:
            encoder_sharing_update(step_idx, encoder_output, self._encoder_sharing_state)

    # --- B10 DDiT Scheduling ---

    def get_patch_stride(self, step_idx: int) -> int:
        """Get the dynamic patch stride for this step.

        Returns 1 (full resolution) if DDiT scheduling is not configured.
        """
        if self._ddit_scheduler is None:
            return 1
        return self._ddit_scheduler.get_patch_stride(step_idx)

    @property
    def toca_state(self) -> Optional[ToCaState]:
        return self._toca_state

    # --- B7 ToCa (per-layer token-level caching across steps) ---

    def toca_select(
        self, tokens: mx.array, layer_idx: int, step_idx: int
    ) -> Optional[tuple[mx.array, mx.array]]:
        """Partition tokens into (active, cached) indices for this layer.

        Returns None if ToCa is not configured. Otherwise returns the
        (active_indices, cached_indices) tuple from :func:`toca_select_tokens`.
        """
        if self._toca_state is None or self.config.toca is None:
            return None
        return toca_select_tokens(
            tokens, layer_idx, step_idx, self.config.toca, self._toca_state
        )

    def toca_record(self, layer_idx: int, tokens: mx.array) -> None:
        """Record post-block tokens for a layer after the caller assembled them."""
        if self._toca_state is not None:
            toca_update(layer_idx, tokens, self._toca_state)

    def toca_compose_tokens(
        self,
        active_features: mx.array,
        cached_features: mx.array,
        active_indices: mx.array,
        cached_indices: mx.array,
        total_n: int,
    ) -> mx.array:
        """Reassemble the full token tensor. Convenience passthrough."""
        return toca_compose(
            active_features, cached_features, active_indices, cached_indices, total_n
        )

    @property
    def ditfastattn_manager(self) -> Optional[DiTFastAttnManager]:
        return self._ditfastattn

    @property
    def deep_cache_state(self) -> Optional[DeepCacheState]:
        return self._deep_cache_state

    def should_recompute_deep(self, step_idx: int) -> bool:
        """Decide whether to run the UNet deep branch this step (DeepCache).

        Returns True if DeepCache is not configured (caller recomputes), or
        if the caching policy says this step must recompute.
        """
        if self._deep_cache_state is None or self.config.deep_cache is None:
            return True
        return deepcache_should_recompute(
            step_idx, self.config.deep_cache, self._deep_cache_state
        )

    def get_cached_deep_features(self) -> Optional[mx.array]:
        """Retrieve the cached deep-branch features, or None if absent."""
        if self._deep_cache_state is None:
            return None
        return deepcache_get(self._deep_cache_state)

    def store_deep_features(self, features: mx.array, step_idx: int) -> None:
        """Record freshly computed deep-branch features.

        Must be called by the model wrapper on every step that actually
        runs the deep branch, immediately after computing it.
        """
        if self._deep_cache_state is not None:
            deepcache_store(features, step_idx, self._deep_cache_state)

    @property
    def motion_tracker(self) -> Optional[MotionTracker]:
        return self._motion_tracker

    @property
    def fbcache_state(self) -> Optional[FBCacheState]:
        return self._fbcache_state

    @property
    def spectral_cache_state(self) -> Optional[SpectralCacheState]:
        return self._spectral_cache_state

    def reset(self) -> None:
        """Reset all internal state for a new inference run."""
        if self._teacache_state is not None:
            self._teacache_state = create_teacache_state()
        if self._motion_tracker is not None:
            self._motion_tracker.reset()
        if self._fbcache_state is not None:
            self._fbcache_state = create_fbcache_state()
        if self._spectral_cache_state is not None:
            self._spectral_cache_state = create_spectral_cache_state()
        if self._tgate_state is not None:
            self._tgate_state = create_tgate_state()
        if self._smooth_cache_state is not None:
            self._smooth_cache_state = create_smooth_cache_state()
        if self._encoder_sharing_state is not None:
            self._encoder_sharing_state = create_encoder_sharing_state()
        if self._toca_state is not None:
            toca_reset(self._toca_state)
        if self._ditfastattn is not None:
            self._ditfastattn.reset()
        if self._deep_cache_state is not None:
            deepcache_reset(self._deep_cache_state)
        if self._multigranular is not None:
            self._multigranular.clear()
        self._last_merge_info = None
        self._block_cache.clear()
