"""B7 — ToCa: Token-wise Feature Caching across diffusion steps.

Algorithm (Zou et al.):
    Between denoising steps, different tokens evolve at different rates.
    Tokens corresponding to stable regions of the latent change slowly;
    tokens in high-motion or high-detail regions change quickly.

    ToCa scores tokens by their **velocity** — the L1 change between the
    two most recent computed step outputs at this layer — then recomputes
    only the top-fraction by velocity (the "active" set) and reuses
    cached features for the rest (the "cached" set).

    Scoring modes:
      * ``"velocity"`` (default) — requires ≥ 2 prior cached steps. Score =
        ``||curr - prev|| / ||prev||`` per token.
      * ``"magnitude"`` — uses the current token's L2 norm. Available from
        step 1 onward and useful when no prior computed step exists yet.

    When history is insufficient for the chosen mode, the selection
    falls back to "all active" — a safe no-op that matches what the
    caller would do without ToCa.

Per-layer state:
    Each transformer block has independent token dynamics, so the cache
    is keyed by ``layer_idx``. A single ToCaState holds the per-layer
    history dicts.

Orthogonality:
    ToCa is independent of B8 ToMe. ToMe merges similar tokens **within
    a step**; ToCa caches tokens **across steps**. They compose: ToMe
    first reduces N → N/2 per step, then ToCa reduces further by only
    recomputing the fast-moving subset of those N/2 tokens.

Applies to: multi-step DiT models (SparkVSR, STAR, Vivid-VR).

Reference: Zou et al., "Accelerating Diffusion Transformers with
    Token-wise Feature Caching" (ToCa).
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

import mlx.core as mx


@dataclass
class ToCaConfig:
    """Configuration for ToCa.

    Attributes:
        recompute_ratio: Fraction of tokens to recompute each step. 0.5
            means half are recomputed, half are reused from cache.
        score_mode: ``"velocity"`` (requires 2-step history) or
            ``"magnitude"`` (requires only current tokens).
        enabled: Master switch.
    """

    recompute_ratio: float = 0.5
    score_mode: Literal["velocity", "magnitude"] = "velocity"
    enabled: bool = True


@dataclass
class ToCaLayerState:
    """Per-layer ToCa state: history of computed-step token tensors.

    Attributes:
        cached_tokens: Most recent full-block output tokens for this
            layer (shape ``[B, N, D]``). None before first compute.
        prev_tokens: Full-block output tokens from the step before
            ``cached_tokens`` (also ``[B, N, D]``). Needed for velocity.
        step_count: Number of times this layer's cache has been updated.
    """

    cached_tokens: Optional[mx.array] = None
    prev_tokens: Optional[mx.array] = None
    step_count: int = 0


@dataclass
class ToCaState:
    """Per-layer ToCa state container."""

    layers: dict[int, ToCaLayerState] = field(default_factory=dict)

    def layer(self, layer_idx: int) -> ToCaLayerState:
        """Get (or lazily create) the state for a layer."""
        s = self.layers.get(layer_idx)
        if s is None:
            s = ToCaLayerState()
            self.layers[layer_idx] = s
        return s


def create_toca_state() -> ToCaState:
    """Create a fresh ToCaState for a new inference run."""
    return ToCaState()


def _token_scores(
    current_tokens: mx.array,
    layer_state: ToCaLayerState,
    mode: str,
) -> Optional[mx.array]:
    """Per-token score of shape ``[B, N]`` (higher = needs recompute), or None
    if history is insufficient for the chosen mode."""
    if mode == "velocity":
        if layer_state.cached_tokens is None or layer_state.prev_tokens is None:
            return None
        # Velocity from the two most recent computed steps at this layer.
        delta = layer_state.cached_tokens - layer_state.prev_tokens
        # Per-token L1 change, normalized by magnitude of prev token so that
        # low-magnitude tokens don't dominate.
        abs_delta = mx.mean(mx.abs(delta), axis=-1)  # [B, N]
        prev_mag = mx.mean(mx.abs(layer_state.prev_tokens), axis=-1) + 1e-6  # [B, N]
        return abs_delta / prev_mag
    elif mode == "magnitude":
        # Available from step 1 (we just need current tokens).
        return mx.mean(mx.abs(current_tokens), axis=-1)  # [B, N]
    else:
        raise ValueError(f"Unknown score_mode: {mode}")


def toca_select_tokens(
    tokens: mx.array,
    layer_idx: int,
    step_idx: int,
    config: ToCaConfig,
    state: ToCaState,
) -> tuple[mx.array, mx.array]:
    """Partition tokens into (active, cached) sets for this layer+step.

    Args:
        tokens: Current step's input tokens at this layer, ``[B, N, D]``.
        layer_idx: Layer identifier (used as cache key).
        step_idx: Current diffusion step index (informational).
        config: ToCa configuration.
        state: Mutable ToCa state holding per-layer history.

    Returns:
        ``(active_indices, cached_indices)``:
          * ``active_indices``: ``[B, N_active]`` — token positions the
            caller must recompute this step.
          * ``cached_indices``: ``[B, N_cached]`` — token positions the
            caller should reuse from ``state.layer(layer_idx).cached_tokens``.

        Edge behavior — if ToCa is disabled, or the cache has no
        content yet, or history is insufficient for the chosen
        ``score_mode``, all N token positions are returned as active
        and ``cached_indices`` is empty.
    """
    B, N, _ = tokens.shape

    if not config.enabled:
        return _all_active(B, N)

    layer_state = state.layer(layer_idx)

    if layer_state.cached_tokens is None:
        return _all_active(B, N)

    scores = _token_scores(tokens, layer_state, config.score_mode)
    if scores is None:
        return _all_active(B, N)

    n_active = max(1, int(N * config.recompute_ratio))
    if n_active >= N:
        return _all_active(B, N)

    # Sort descending by score. argsort sorts ascending, so negate.
    sorted_idx = mx.argsort(-scores, axis=-1)  # [B, N]
    active = sorted_idx[:, :n_active]
    cached = sorted_idx[:, n_active:]
    # Sort each set by index to keep spatial order predictable for gather.
    active = mx.sort(active, axis=-1)
    cached = mx.sort(cached, axis=-1)
    return active, cached


def _all_active(B: int, N: int) -> tuple[mx.array, mx.array]:
    """Return ``(all_indices, empty_indices)`` for the fallback path."""
    arange = mx.arange(N, dtype=mx.int32)
    active = mx.broadcast_to(arange.reshape(1, N), (B, N))
    cached = mx.zeros((B, 0), dtype=mx.int32)
    return active, cached


def toca_compose(
    active_features: mx.array,
    cached_features: mx.array,
    active_indices: mx.array,
    cached_indices: mx.array,
    total_n: int,
) -> mx.array:
    """Reassemble a full ``[B, N, D]`` tensor from active+cached pieces.

    Args:
        active_features: ``[B, N_active, D]`` — freshly computed features
            for the positions in ``active_indices``.
        cached_features: ``[B, N_cached, D]`` — cached features for the
            positions in ``cached_indices``. May be ``[B, 0, D]`` if all
            tokens were recomputed.
        active_indices: ``[B, N_active]`` original positions (0..N-1) of
            the active tokens.
        cached_indices: ``[B, N_cached]`` original positions of the
            cached tokens.
        total_n: The total token count ``N``.

    Returns:
        ``[B, N, D]`` tensor with each token placed at its original index.
    """
    B, _, D = active_features.shape
    output = mx.zeros((B, total_n, D), dtype=active_features.dtype)
    # Scatter-add is safe here because active and cached index sets are disjoint.
    for b in range(B):
        output = output.at[b, active_indices[b]].add(active_features[b])
        if cached_indices.shape[1] > 0:
            output = output.at[b, cached_indices[b]].add(cached_features[b])
    return output


def toca_update(
    layer_idx: int,
    tokens: mx.array,
    state: ToCaState,
) -> None:
    """Record the full post-block token tensor for a layer after a step.

    Call after the caller has assembled the complete ``[B, N, D]`` output
    of a block (via :func:`toca_compose`). Shifts ``prev ← cached``,
    then ``cached ← tokens``, and increments the step count.
    """
    layer_state = state.layer(layer_idx)
    layer_state.prev_tokens = layer_state.cached_tokens
    layer_state.cached_tokens = tokens
    layer_state.step_count += 1


def toca_get_cached(
    layer_idx: int,
    state: ToCaState,
) -> Optional[mx.array]:
    """Return the last full token tensor recorded for ``layer_idx``, or None."""
    layer_state = state.layers.get(layer_idx)
    return layer_state.cached_tokens if layer_state is not None else None


def toca_reset(state: ToCaState) -> None:
    """Clear all per-layer history for a new inference run."""
    state.layers.clear()
