"""B2 — First-Block Cache (FBCache): Skip remaining transformer blocks.

Algorithm:
    Run the first transformer block every step. If the first-block output
    changes little across consecutive steps, the aggregated residual that
    flows through blocks 2..N also changes little. FBCache caches that
    residual and, when the first-block output is stable, reconstructs the
    full output as::

        output ≈ first_block_output + cached_residual

    The model wrapper must therefore split its forward pass at the first
    block:

        fb = first_block(x)
        if fbcache_should_compute_remaining(fb, step, cfg, state):
            full = run_remaining_blocks(fb)
            residual = full - fb
            fbcache_update(fb, residual, state)
            return full
        else:
            return fbcache_reconstruct(fb, state)

    Unlike TeaCache, FBCache needs no calibration — the first-block output
    is a natural, model-agnostic proxy. That makes it the sensible default
    fallback when no TeaCache coefficients are published for a given model.

Edge cases:
    * First step always computes (no prior to compare to).
    * Reaching ``max_consecutive_cached`` forces a recompute to prevent
      indefinite drift.
    * Outside the ``[start_step, end_step)`` window, always computes.

Applies to: multi-step VSR models (SparkVSR, STAR, Vivid-VR, DAM-VSR,
    DiffVSR, VEnhancer).

Reference: "First-Block Cache" idea traces back to community
    implementations in Comfy-WaveSpeed (inspiration, not a formal cite).
    Closely related to TeaCache with a different change signal.
"""

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx


@dataclass
class FBCacheConfig:
    """Configuration for First-Block Cache.

    Attributes:
        rel_l1_thresh: Relative L1 change threshold on the first-block
            output. Below this → cache is used (skip remaining blocks).
            0.1 is the paper-aligned default.
        start_step: First step at which caching may kick in. The cache is
            populated starting at ``start_step``; steps before this always
            run the full model.
        end_step: Last step at which caching may be used. None means no
            upper bound. Useful for forcing full compute on the final few
            denoising steps where detail matters most.
        max_consecutive_cached: Hard ceiling on uninterrupted cache reuse
            to prevent drift.
        enabled: Master switch.
    """

    rel_l1_thresh: float = 0.1
    start_step: int = 0
    end_step: Optional[int] = None
    max_consecutive_cached: int = 5
    enabled: bool = True


@dataclass
class FBCacheState:
    """Mutable state for FBCache.

    Attributes:
        prev_fb_output: The first-block output from the most recent
            compute step. Used as the comparison baseline for the next
            decision.
        cached_residual: ``full_output - first_block_output`` from the
            most recent compute step. Reused when the decision says skip.
        step_counter: Monotonic counter of ``should_compute`` calls,
            currently informational.
        consecutive_cached: Number of consecutive cache reuses since the
            last real compute.
    """

    prev_fb_output: Optional[mx.array] = None
    cached_residual: Optional[mx.array] = None
    step_counter: int = 0
    consecutive_cached: int = 0


def create_fbcache_state() -> FBCacheState:
    """Create a fresh FBCacheState for a new inference run."""
    return FBCacheState()


def _rel_l1(curr: mx.array, prev: mx.array) -> float:
    diff = mx.mean(mx.abs(curr - prev))
    norm = mx.mean(mx.abs(prev)) + 1e-6
    return (diff / norm).item()


def fbcache_should_compute_remaining(
    fb_output: mx.array,
    step_idx: int,
    config: FBCacheConfig,
    state: FBCacheState,
) -> bool:
    """Decide whether to run the remaining transformer blocks.

    Returns True if the caller must run blocks 2..N. False means the
    caller must reconstruct via :func:`fbcache_reconstruct`.

    This function also **mutates** ``state.consecutive_cached`` — it is
    incremented on a cache reuse and reset on a compute.
    """
    state.step_counter += 1

    if not config.enabled:
        state.consecutive_cached = 0
        return True

    # Window bounds.
    if step_idx < config.start_step:
        state.consecutive_cached = 0
        return True
    if config.end_step is not None and step_idx >= config.end_step:
        state.consecutive_cached = 0
        return True

    # First step or no residual yet → always compute.
    if state.prev_fb_output is None or state.cached_residual is None:
        return True

    # Safety ceiling on consecutive reuses.
    if state.consecutive_cached >= config.max_consecutive_cached:
        state.consecutive_cached = 0
        return True

    # Actual decision: relative L1 change on first-block output.
    change = _rel_l1(fb_output, state.prev_fb_output)
    if change < config.rel_l1_thresh:
        state.consecutive_cached += 1
        return False

    state.consecutive_cached = 0
    return True


def fbcache_update(
    fb_output: mx.array,
    residual: mx.array,
    state: FBCacheState,
) -> None:
    """Record fresh first-block output and residual after a real compute.

    Must be called by the model wrapper after running blocks 2..N and
    computing ``residual = full_output - fb_output``.
    """
    state.prev_fb_output = fb_output
    state.cached_residual = residual


def fbcache_reconstruct(
    fb_output: mx.array,
    state: FBCacheState,
) -> mx.array:
    """Return ``fb_output + cached_residual``.

    Must only be called after ``fbcache_should_compute_remaining`` has
    returned False and the state therefore has a valid cached residual.

    Raises:
        RuntimeError: if no cached residual exists.
    """
    if state.cached_residual is None:
        raise RuntimeError(
            "FBCache has no cached residual; the caller must compute the "
            "remaining blocks on this step."
        )
    return fb_output + state.cached_residual


def fbcache_reset(state: FBCacheState) -> None:
    """Clear state for a new inference run."""
    state.prev_fb_output = None
    state.cached_residual = None
    state.step_counter = 0
    state.consecutive_cached = 0
