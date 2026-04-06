"""B2 — First-Block Cache (FBCache): Zero-calibration step caching.

Uses the first transformer block's output as a proxy for global change.
If it barely changes between steps, the entire model forward is skipped.
No per-model calibration needed — works immediately for all multi-step models.

Fallback for TeaCache when no calibrated coefficients are available.
"""

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx


@dataclass
class FBCacheConfig:
    threshold: float = 0.05
    max_consecutive_cached: int = 5
    enabled: bool = True


@dataclass
class FBCacheState:
    prev_first_block_output: Optional[mx.array] = None
    cached_full_output: Optional[mx.array] = None
    consecutive_cached: int = 0


def create_fbcache_state() -> FBCacheState:
    """Create a fresh FBCache state."""
    return FBCacheState()


def fbcache_should_compute(
    first_block_output: mx.array,
    config: FBCacheConfig,
    state: FBCacheState,
) -> bool:
    """Decide whether to compute the full model based on first-block output change.

    Args:
        first_block_output: Output of the first transformer block for this step.
        config: FBCache configuration.
        state: Mutable FBCache state.

    Returns:
        True if the full model should be computed, False to reuse cache.
    """
    if not config.enabled:
        return True

    if state.prev_first_block_output is None:
        return True

    diff = mx.mean(mx.abs(first_block_output - state.prev_first_block_output))
    norm = mx.mean(mx.abs(state.prev_first_block_output)) + 1e-6
    rel_l1 = (diff / norm).item()

    if rel_l1 < config.threshold and state.consecutive_cached < config.max_consecutive_cached:
        state.consecutive_cached += 1
        return False

    state.consecutive_cached = 0
    return True


def fbcache_update(
    first_block_output: mx.array,
    full_output: mx.array,
    state: FBCacheState,
) -> None:
    """Update FBCache state after a computed step.

    Args:
        first_block_output: First block's output for this step.
        full_output: Complete model output for this step.
        state: Mutable FBCache state.
    """
    state.prev_first_block_output = first_block_output
    state.cached_full_output = full_output
