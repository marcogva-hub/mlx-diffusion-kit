"""B5 — DeepCache: Skip UNet deep-branch computation across diffusion steps.

Algorithm (Ma et al., CVPR 2024):
    In UNet architectures, the deep (bottleneck) features change slowly across
    diffusion steps because they encode high-level semantics that are stable
    over denoising. The shallow encoder/decoder layers, by contrast, change
    every step as they refine fine spatial detail. DeepCache exploits this
    by caching the bottleneck output for N consecutive steps and reusing it,
    recomputing only the shallow layers each step.

Module contract:
    This module does not know the structure of the user's UNet. It provides:
      1. A per-step decision: "must I recompute the deep branch this step?"
      2. A store for the most recent deep-branch output (single tensor).
      3. A retrieval accessor.

    The model wrapper is responsible for splitting its forward pass at the
    deep/shallow boundary and calling this module at those two points.

Applies to: UNet multi-step VSR models (DAM-VSR, DiffVSR, DLoRAL, UltraVSR,
    VEnhancer). The single-step flag on OrchestratorConfig disables it
    automatically for SeedVR2/DOVE/FlashVSR.

Reference: Ma, Fang, Wang. "DeepCache: Accelerating Diffusion Models for Free."
    CVPR 2024. https://arxiv.org/abs/2312.00858
"""

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx


@dataclass
class DeepCacheConfig:
    """Configuration for the DeepCache deep-branch caching policy.

    Attributes:
        cache_interval: Recompute the deep branch once every N steps
            (N=1 means no caching; N=3 is the paper's default and a
            conservative quality/speed tradeoff).
        start_step: Index of the first step at which caching is allowed.
            Before this step the deep branch is always recomputed so the
            cache is populated before any reuse happens.
        enabled: Master switch.
    """

    cache_interval: int = 3
    start_step: int = 0
    enabled: bool = True


@dataclass
class DeepCacheState:
    """Mutable state for DeepCache.

    Attributes:
        cached_deep_features: Most recently computed deep-branch output
            (single tensor, shape model-dependent). None before the first
            compute of this inference run.
        last_recompute_step: Step index at which the deep branch was last
            actually computed. -1 before the first compute.
        recompute_count: Number of real recomputes this run (for telemetry).
    """

    cached_deep_features: Optional[mx.array] = None
    last_recompute_step: int = -1
    recompute_count: int = 0


def create_deepcache_state() -> DeepCacheState:
    """Create a fresh DeepCacheState for a new inference run."""
    return DeepCacheState()


def deepcache_should_recompute(
    step_idx: int,
    config: DeepCacheConfig,
    state: DeepCacheState,
) -> bool:
    """Decide whether to recompute the deep branch at this step.

    Returns True if the caller must run the full deep branch this step.
    Returns False if the caller must reuse ``state.cached_deep_features``.

    The decision rules (checked in order):

      1. If caching is disabled → always recompute.
      2. If the step is before ``config.start_step`` → always recompute.
      3. If no cache exists yet (``cached_deep_features is None``) → recompute.
      4. If (step_idx − last_recompute_step) >= ``cache_interval`` → recompute.
      5. Otherwise → reuse cached deep features.

    Delta-based arithmetic is used (not modulo) so the policy remains
    correct when upstream components like TeaCache skip some steps and
    the caller calls DeepCache on a non-contiguous step sequence.
    """
    if not config.enabled:
        return True
    if step_idx < config.start_step:
        return True
    if state.cached_deep_features is None:
        return True
    return step_idx - state.last_recompute_step >= config.cache_interval


def deepcache_store(
    features: mx.array,
    step_idx: int,
    state: DeepCacheState,
) -> None:
    """Store freshly computed deep-branch features.

    Must be called immediately after the deep branch is computed on a
    recompute step. Updates ``last_recompute_step`` so the next call to
    ``deepcache_should_recompute`` measures the delta from this step.
    """
    state.cached_deep_features = features
    state.last_recompute_step = step_idx
    state.recompute_count += 1


def deepcache_get(state: DeepCacheState) -> Optional[mx.array]:
    """Return the cached deep-branch features, or None if not populated."""
    return state.cached_deep_features


def deepcache_reset(state: DeepCacheState) -> None:
    """Clear the cache (e.g., at the start of a new inference run)."""
    state.cached_deep_features = None
    state.last_recompute_step = -1
    state.recompute_count = 0
