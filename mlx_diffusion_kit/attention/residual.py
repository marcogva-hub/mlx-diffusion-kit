"""Residual connection utilities for compressed attention.

When using DiTFastAttn with mixed strategies (FULL + CACHED), residuals
must be correctly scaled to account for approximation error in cached heads.
"""

import math
from typing import Optional

import mlx.core as mx


def scaled_residual_add(
    x: mx.array,
    residual: mx.array,
    scale: float = 1.0,
    gate: Optional[mx.array] = None,
) -> mx.array:
    """Residual addition with optional scaling and gating.

    Args:
        x: Main tensor.
        residual: Residual tensor (same shape as x).
        scale: Global scale factor.
        gate: Optional per-element or broadcastable gate tensor.

    Returns:
        x + gate * scale * residual (or x + scale * residual if gate is None).
    """
    if gate is not None:
        return x + gate * scale * residual
    return x + scale * residual


def compute_residual_scale(
    layer_idx: int,
    total_layers: int,
    method: str = "inverse_sqrt",
) -> float:
    """Compute per-layer residual scaling factor.

    Args:
        layer_idx: Current layer index (0-based).
        total_layers: Total number of layers.
        method: "inverse_sqrt", "linear", or "constant".

    Returns:
        Scaling factor for the residual connection.
    """
    if method == "inverse_sqrt":
        return 1.0 / math.sqrt(layer_idx + 1)
    elif method == "linear":
        return 1.0 - layer_idx / max(total_layers, 1)
    elif method == "constant":
        return 1.0
    else:
        raise ValueError(f"Unknown method: {method}")


def residual_gate_from_sensitivity(
    sensitivity_scores: dict[int, float],
    layer_idx: int,
    default: float = 1.0,
) -> float:
    """Convert a sensitivity score to a residual gate factor.

    High sensitivity → gate ≈ 1.0 (full residual, don't attenuate).
    Low sensitivity → gate < 1.0 (attenuate residual from approximate layer).

    Args:
        sensitivity_scores: {layer_idx: sensitivity} mapping.
        layer_idx: Layer to get gate for.
        default: Default gate if layer not in scores.

    Returns:
        Gate factor in [0, 1].
    """
    score = sensitivity_scores.get(layer_idx, default)
    return max(0.0, min(1.0, score))
