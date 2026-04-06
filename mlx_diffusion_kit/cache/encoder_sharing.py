"""B22 — DDT Encoder Sharing: Cache early DiT blocks across steps.

In a multi-step DiT, the first K transformer blocks (the "encoder") change
little between consecutive denoising steps. This module caches their output
and recomputes only every M steps, saving K/N × (M-1)/M of DiT compute.

Complementary to TeaCache: TeaCache skips entire steps, encoder sharing
skips early blocks within a computed step. Both can be active simultaneously.
"""

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx


@dataclass
class EncoderSharingConfig:
    shared_blocks: int = 6
    recompute_interval: int = 3
    total_blocks: int = 48
    enabled: bool = True


@dataclass
class EncoderSharingState:
    cached_encoder_output: Optional[mx.array] = None
    last_computed_step: int = -1


def create_encoder_sharing_state() -> EncoderSharingState:
    """Create a fresh encoder sharing state."""
    return EncoderSharingState()


def encoder_sharing_should_recompute(
    step_idx: int,
    config: EncoderSharingConfig,
    state: EncoderSharingState,
) -> bool:
    """Decide whether to recompute the encoder blocks for this step.

    Returns True if:
      - Encoder sharing is disabled
      - No cached output exists yet
      - step_idx is on a recompute boundary (step_idx % interval == 0)

    Args:
        step_idx: Current denoising step index.
        config: Encoder sharing configuration.
        state: Mutable encoder sharing state.

    Returns:
        True if encoder blocks should be computed, False to use cache.
    """
    if not config.enabled:
        return True

    if state.cached_encoder_output is None:
        return True

    return step_idx % config.recompute_interval == 0


def encoder_sharing_get_cached(state: EncoderSharingState) -> Optional[mx.array]:
    """Retrieve the cached encoder output.

    Returns:
        Cached encoder block output, or None if not available.
    """
    return state.cached_encoder_output


def encoder_sharing_update(
    step_idx: int,
    encoder_output: mx.array,
    state: EncoderSharingState,
) -> None:
    """Update the encoder sharing cache after computing encoder blocks.

    Args:
        step_idx: Step at which the encoder was computed.
        encoder_output: Output tensor from the shared encoder blocks.
        state: Mutable encoder sharing state to update.
    """
    state.cached_encoder_output = encoder_output
    state.last_computed_step = step_idx
