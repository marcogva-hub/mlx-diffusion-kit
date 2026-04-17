"""B12 — DiTFastAttn: Per-layer attention compression policy for multi-step DiT.

This module is a **decision layer**, not an attention kernel. Given the
current ``(layer_idx, step_idx)``, it returns one of four strategies
that the caller's attention wrapper must honor:

    FULL      Run the full dense attention, no compression.
    WINDOW    Run a windowed attention mask (``window_size`` diagonal
              band). Activates from ``window_start_step`` onward for
              every non-listed layer.
    SHARE     Reuse the post-softmax attention map from the previous
              step for this layer. Requires the caller to have called
              :func:`ditfastattn_record_attn_map` on the previous step.
    RESIDUAL  Reuse the attention block's residual contribution
              (``attn_output - residual_stream``) from the previous step.
              Cheaper than SHARE because it skips QK^T, softmax, and
              value-matmul entirely. Requires a prior
              :func:`ditfastattn_record_residual` call.

Integration boundary:
    This module never calls ``mlx_mfa.flash_attention``. The strategy
    is returned as an enum value; the caller's layer wrapper is
    responsible for executing it — typically via a small ``match`` on
    the returned strategy. A reference wrapper is **not** provided here
    because each user's transformer layer varies (pre/post-norm,
    fused QKV, custom projections), and a wrapper would necessarily
    pick a layout that wouldn't fit everyone.

Applies to: multi-step DiT (SparkVSR, STAR, Vivid-VR, DAM-VSR, DiffVSR,
    VEnhancer).

Reference: Yuan et al., "DiTFastAttn: Attention Compression for
    Diffusion Transformer Models", NeurIPS 2024.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import mlx.core as mx


class AttnStrategy(Enum):
    """Per-layer, per-step attention execution strategy."""

    FULL = "full"
    WINDOW = "window"
    SHARE = "share"
    RESIDUAL = "residual"


@dataclass
class DiTFastAttnConfig:
    """Configuration for DiTFastAttn.

    Attributes:
        window_start_step: Step at which non-specialized layers switch
            to WINDOW strategy. Layers before this step use FULL (unless
            listed in ``sharing_layers`` or ``residual_cache_layers``,
            which take precedence at their respective steps).
        window_size: Width of the local attention window (diagonal
            band). Semantics depend on the caller's mask builder.
        sharing_layers: Layer indices that should use SHARE strategy
            from step 1 onward. A cached attention map must be
            available (the caller records it on a compute step).
        residual_cache_layers: Layer indices that should use RESIDUAL
            strategy from step 1 onward. A cached residual must be
            available.
        enabled: Master switch.

    Precedence when a layer appears in multiple lists:
        RESIDUAL > SHARE > WINDOW > FULL. This reflects the compression
        order: RESIDUAL saves the most compute, so if the user asked
        for it they want it.
    """

    window_start_step: int = 10
    window_size: int = 64
    sharing_layers: list[int] = field(default_factory=list)
    residual_cache_layers: list[int] = field(default_factory=list)
    enabled: bool = True


@dataclass
class DiTFastAttnState:
    """Mutable per-layer caches for SHARE and RESIDUAL strategies.

    Attributes:
        cached_attn_maps: ``{layer_idx: post_softmax_attention_weights}``.
            Used by the SHARE strategy; populated by
            :func:`ditfastattn_record_attn_map`.
        cached_residuals: ``{layer_idx: attn_out - residual_stream}``.
            Used by the RESIDUAL strategy; populated by
            :func:`ditfastattn_record_residual`.
    """

    cached_attn_maps: dict[int, mx.array] = field(default_factory=dict)
    cached_residuals: dict[int, mx.array] = field(default_factory=dict)


def create_ditfastattn_state() -> DiTFastAttnState:
    """Create a fresh DiTFastAttnState for a new inference run."""
    return DiTFastAttnState()


def ditfastattn_decide(
    layer_idx: int,
    step_idx: int,
    config: DiTFastAttnConfig,
    state: DiTFastAttnState,
) -> AttnStrategy:
    """Return the attention strategy for ``(layer_idx, step_idx)``.

    Decision order (first matching rule wins):

      1. If ``enabled=False`` → FULL.
      2. If ``step_idx == 0`` → FULL (no cached state exists yet).
      3. If ``layer_idx`` is in ``residual_cache_layers`` AND a cached
         residual exists → RESIDUAL.
      4. If ``layer_idx`` is in ``sharing_layers`` AND a cached attn map
         exists → SHARE.
      5. If ``step_idx >= window_start_step`` → WINDOW.
      6. Otherwise → FULL.

    Falling back when caches are empty is a deliberate safety feature:
    it prevents stale decisions on the first eligible step, at the cost
    of one extra full compute at the beginning of the cache's lifetime.
    """
    if not config.enabled:
        return AttnStrategy.FULL

    if step_idx == 0:
        return AttnStrategy.FULL

    if (
        layer_idx in config.residual_cache_layers
        and layer_idx in state.cached_residuals
    ):
        return AttnStrategy.RESIDUAL

    if layer_idx in config.sharing_layers and layer_idx in state.cached_attn_maps:
        return AttnStrategy.SHARE

    if step_idx >= config.window_start_step:
        return AttnStrategy.WINDOW

    return AttnStrategy.FULL


def ditfastattn_record_attn_map(
    layer_idx: int,
    attn_map: mx.array,
    state: DiTFastAttnState,
) -> None:
    """Record a post-softmax attention map for later SHARE reuse."""
    state.cached_attn_maps[layer_idx] = attn_map


def ditfastattn_get_cached_attn(
    layer_idx: int,
    state: DiTFastAttnState,
) -> Optional[mx.array]:
    """Return the cached attention map for ``layer_idx``, or None."""
    return state.cached_attn_maps.get(layer_idx)


def ditfastattn_record_residual(
    layer_idx: int,
    residual: mx.array,
    state: DiTFastAttnState,
) -> None:
    """Record the attention block residual (``attn_out - residual_stream``)
    for later RESIDUAL reuse."""
    state.cached_residuals[layer_idx] = residual


def ditfastattn_get_cached_residual(
    layer_idx: int,
    state: DiTFastAttnState,
) -> Optional[mx.array]:
    """Return the cached attention residual for ``layer_idx``, or None."""
    return state.cached_residuals.get(layer_idx)


def ditfastattn_reset(state: DiTFastAttnState) -> None:
    """Clear all cached maps and residuals."""
    state.cached_attn_maps.clear()
    state.cached_residuals.clear()
