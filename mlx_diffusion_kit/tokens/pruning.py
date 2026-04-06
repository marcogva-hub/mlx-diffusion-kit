"""B8 — ToPi: Token Pruning for diffusion models.

Complementary to ToMe: instead of merging similar tokens, ToPi drops
the least important tokens before attention and restores them afterward.
Simpler and faster than ToMe, but less faithful.

All operations are fully vectorized — no .item() calls or Python loops.
"""

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx


@dataclass
class ToPiConfig:
    prune_ratio: float = 0.3
    importance: str = "norm"  # "attention" | "norm" | "random"
    restore_mode: str = "copy"  # "copy" | "zero" | "lerp"
    enabled: bool = True


@dataclass
class PruneInfo:
    kept_indices: mx.array  # [B, N_kept]
    pruned_indices: mx.array  # [B, N_pruned]
    original_n: int
    nearest_kept: mx.array  # [B, N_pruned] → index into kept tokens


def compute_token_importance(
    tokens: mx.array,
    method: str = "norm",
    attention_weights: Optional[mx.array] = None,
) -> mx.array:
    """Compute per-token importance scores.

    Args:
        tokens: Input tokens [B, N, D].
        method: Scoring method — "attention", "norm", or "random".
        attention_weights: Attention weight matrix [B, N, N] (required for "attention").

    Returns:
        Importance scores [B, N]. Higher = more important.
    """
    B, N, D = tokens.shape

    if method == "attention":
        if attention_weights is None:
            raise ValueError("attention_weights required for importance='attention'")
        # Column sum: how much each token is attended to
        return mx.sum(attention_weights, axis=-2)  # [B, N]

    elif method == "norm":
        return mx.linalg.norm(tokens, axis=-1)  # [B, N]

    elif method == "random":
        return mx.random.uniform(shape=(B, N))

    else:
        raise ValueError(f"Unknown importance method: {method}")


def _compute_nearest_kept(
    kept_indices: mx.array, pruned_indices: mx.array, n_kept: int
) -> mx.array:
    """For each pruned token, find the nearest kept token by index distance.

    Args:
        kept_indices: [B, N_kept] sorted indices of kept tokens.
        pruned_indices: [B, N_pruned] sorted indices of pruned tokens.
        n_kept: Number of kept tokens.

    Returns:
        [B, N_pruned] index into kept_indices (0..N_kept-1) of nearest neighbor.
    """
    B = kept_indices.shape[0]
    n_pruned = pruned_indices.shape[1]

    if n_pruned == 0:
        return mx.zeros((B, 0), dtype=mx.int32)

    # Expand for broadcasting: [B, N_pruned, 1] vs [B, 1, N_kept]
    pruned_exp = mx.expand_dims(pruned_indices, axis=-1)  # [B, N_pruned, 1]
    kept_exp = mx.expand_dims(kept_indices, axis=-2)  # [B, 1, N_kept]

    # Absolute distance
    dist = mx.abs(pruned_exp - kept_exp)  # [B, N_pruned, N_kept]

    # Nearest kept for each pruned token
    return mx.argmin(dist, axis=-1)  # [B, N_pruned]


def topi_prune(
    tokens: mx.array,
    config: ToPiConfig,
    attention_weights: Optional[mx.array] = None,
) -> tuple[mx.array, PruneInfo]:
    """Prune low-importance tokens.

    Args:
        tokens: Input tokens [B, N, D].
        config: Pruning configuration.
        attention_weights: Optional attention weights for importance="attention".

    Returns:
        (pruned_tokens [B, N_kept, D], prune_info).
    """
    B, N, D = tokens.shape

    if not config.enabled or config.prune_ratio <= 0.0:
        info = PruneInfo(
            kept_indices=mx.broadcast_to(mx.arange(N).reshape(1, N), (B, N)),
            pruned_indices=mx.zeros((B, 0), dtype=mx.int32),
            original_n=N,
            nearest_kept=mx.zeros((B, 0), dtype=mx.int32),
        )
        return tokens, info

    n_prune = int(N * config.prune_ratio)
    n_prune = min(n_prune, N - 1)  # Keep at least 1 token
    n_kept = N - n_prune

    if n_prune == 0:
        info = PruneInfo(
            kept_indices=mx.broadcast_to(mx.arange(N).reshape(1, N), (B, N)),
            pruned_indices=mx.zeros((B, 0), dtype=mx.int32),
            original_n=N,
            nearest_kept=mx.zeros((B, 0), dtype=mx.int32),
        )
        return tokens, info

    # Compute importance scores
    scores = compute_token_importance(tokens, config.importance, attention_weights)

    # Get indices sorted by importance (descending)
    sorted_indices = mx.argsort(-scores, axis=-1)  # [B, N] highest first

    kept_indices = sorted_indices[:, :n_kept]  # [B, N_kept]
    pruned_indices = sorted_indices[:, n_kept:]  # [B, N_pruned]

    # Sort kept/pruned by position to preserve spatial order
    kept_indices = mx.sort(kept_indices, axis=-1)
    pruned_indices = mx.sort(pruned_indices, axis=-1)

    # Compute nearest kept for each pruned token
    nearest_kept = _compute_nearest_kept(kept_indices, pruned_indices, n_kept)

    # Gather kept tokens
    # Expand indices for gather: [B, N_kept, 1] → broadcast with [B, N_kept, D]
    idx_exp = mx.expand_dims(kept_indices, axis=-1)  # [B, N_kept, 1]
    idx_exp = mx.broadcast_to(idx_exp, (B, n_kept, D))
    pruned_tokens = mx.take_along_axis(tokens, idx_exp, axis=1)

    info = PruneInfo(
        kept_indices=kept_indices,
        pruned_indices=pruned_indices,
        original_n=N,
        nearest_kept=nearest_kept,
    )
    return pruned_tokens, info


def topi_restore(
    pruned_output: mx.array,
    info: PruneInfo,
    config: ToPiConfig,
) -> mx.array:
    """Restore pruned tokens to original sequence length.

    Args:
        pruned_output: Output after attention [B, N_kept, D].
        info: PruneInfo from topi_prune.
        config: Pruning configuration (for restore_mode).

    Returns:
        Restored tokens [B, N_original, D].
    """
    B, N_kept, D = pruned_output.shape
    N = info.original_n

    if info.pruned_indices.shape[1] == 0:
        return pruned_output

    output = mx.zeros((B, N, D), dtype=pruned_output.dtype)

    # Place kept tokens at their original positions
    kept_idx_exp = mx.expand_dims(info.kept_indices, axis=-1)
    kept_idx_exp = mx.broadcast_to(kept_idx_exp, (B, N_kept, D))

    # Use scatter via a loop over batch (B is small: 1-8)
    for b in range(B):
        output = output.at[b, info.kept_indices[b]].add(pruned_output[b])

    # Restore pruned tokens
    n_pruned = info.pruned_indices.shape[1]

    if config.restore_mode == "zero":
        pass  # Already zeros

    elif config.restore_mode == "copy":
        # Each pruned token gets value of its nearest kept token
        # nearest_kept: [B, N_pruned] → index into kept dimension
        nearest_exp = mx.expand_dims(info.nearest_kept, axis=-1)
        nearest_exp = mx.broadcast_to(nearest_exp, (B, n_pruned, D))
        restored_vals = mx.take_along_axis(pruned_output, nearest_exp, axis=1)
        for b in range(B):
            output = output.at[b, info.pruned_indices[b]].add(restored_vals[b])

    elif config.restore_mode == "lerp":
        # Average of two nearest kept tokens
        # Use nearest_kept and the next one (clamped)
        nearest = info.nearest_kept  # [B, N_pruned]
        next_nearest = mx.minimum(nearest + 1, N_kept - 1)

        near_exp = mx.expand_dims(nearest, -1)
        next_exp = mx.expand_dims(next_nearest, -1)
        near_exp = mx.broadcast_to(near_exp, (B, n_pruned, D))
        next_exp = mx.broadcast_to(next_exp, (B, n_pruned, D))

        val_a = mx.take_along_axis(pruned_output, near_exp, axis=1)
        val_b = mx.take_along_axis(pruned_output, next_exp, axis=1)
        restored_vals = (val_a + val_b) * 0.5

        for b in range(B):
            output = output.at[b, info.pruned_indices[b]].add(restored_vals[b])

    return output
