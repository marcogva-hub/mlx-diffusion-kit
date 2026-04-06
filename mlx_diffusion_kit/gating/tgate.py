"""B11 — T-GATE: Cross-attention gating for multi-step diffusion models.

Freezes cross-attention outputs after convergence (typically ~5 steps).
Self-attention always runs. Training-free.

Reference: Zhang et al., "Cross-Attention Makes Inference Cumbersome
in Text-to-Image Diffusion Models" (2024).
"""

from dataclasses import dataclass, field
from typing import Callable

import mlx.core as mx


@dataclass
class TGateConfig:
    gate_step: int = 5
    enabled: bool = True


@dataclass
class TGateState:
    cached_cross_attn: dict[int, mx.array] = field(default_factory=dict)
    step_count: int = 0


def create_tgate_state() -> TGateState:
    """Create a fresh T-GATE state."""
    return TGateState()


def tgate_forward(
    layer_idx: int,
    step_idx: int,
    config: TGateConfig,
    state: TGateState,
    self_attn_fn: Callable[[mx.array], mx.array],
    cross_attn_fn: Callable[[mx.array, mx.array], mx.array],
    x: mx.array,
    context: mx.array,
) -> mx.array:
    """Execute a transformer block with T-GATE cross-attention gating.

    Args:
        layer_idx: Index of the transformer layer (for per-layer caching).
        step_idx: Current diffusion step index.
        config: T-GATE configuration.
        state: Mutable state holding cached cross-attention outputs.
        self_attn_fn: Self-attention callable, signature (x) -> x.
        cross_attn_fn: Cross-attention callable, signature (x, context) -> x.
        x: Input hidden states.
        context: Text encoder output (conditioning).

    Returns:
        Output hidden states after self-attn + cross-attn.
    """
    x = self_attn_fn(x)

    if not config.enabled or step_idx < config.gate_step:
        cross_out = cross_attn_fn(x, context)
        if config.enabled:
            state.cached_cross_attn[layer_idx] = cross_out
    else:
        cross_out = state.cached_cross_attn[layer_idx]

    return x + cross_out
