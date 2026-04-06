"""B4 — SmoothCache: Interpolation for skipped diffusion steps.

When TeaCache decides to skip a step, SmoothCache interpolates between
computed features instead of reusing the stale cached residual. This
reduces "stutter" artifacts from consecutive identical features.

Three interpolation modes:
  LINEAR — lerp between bracketing computed features
  TAYLOR_1 — first-order Taylor extrapolation (linear, using derivative)
  TAYLOR_2 — second-order Taylor extrapolation (quadratic, using curvature)

Integration pattern (in orchestrator):
  if teacache_should_compute(step, ...):
      output = model_forward(...)
      smooth_cache_record(step, output, state)
      teacache_update(...)
  else:
      output = smooth_cache_interpolate(step, state, config)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import mlx.core as mx


class InterpolationMode(Enum):
    LINEAR = "linear"
    TAYLOR_1 = "taylor_1"
    TAYLOR_2 = "taylor_2"


@dataclass
class SmoothCacheConfig:
    mode: InterpolationMode = InterpolationMode.LINEAR
    enabled: bool = True


@dataclass
class SmoothCacheState:
    history: list[tuple[int, mx.array]] = field(default_factory=list)
    max_history: int = 3


def create_smooth_cache_state(max_history: int = 3) -> SmoothCacheState:
    """Create a fresh SmoothCache state."""
    return SmoothCacheState(max_history=max_history)


def smooth_cache_record(
    step_idx: int,
    features: mx.array,
    state: SmoothCacheState,
) -> None:
    """Record features from a computed (non-skipped) step.

    Args:
        step_idx: The step index that was computed.
        features: Output features from the model forward pass.
        state: Mutable SmoothCache state.
    """
    state.history.append((step_idx, features))
    if len(state.history) > state.max_history:
        state.history = state.history[-state.max_history:]


def _interpolate_linear(
    target_step: int,
    history: list[tuple[int, mx.array]],
) -> mx.array:
    """Linear interpolation or extrapolation from the two most recent entries."""
    if len(history) < 2:
        return history[-1][1]

    step_a, feat_a = history[-2]
    step_b, feat_b = history[-1]
    span = step_b - step_a
    if span == 0:
        return feat_b

    t = (target_step - step_a) / span
    return (1.0 - t) * feat_a + t * feat_b


def _interpolate_taylor_1(
    target_step: int,
    history: list[tuple[int, mx.array]],
) -> mx.array:
    """First-order Taylor extrapolation from the two most recent entries."""
    if len(history) < 2:
        return history[-1][1]

    step_a, feat_a = history[-2]
    step_b, feat_b = history[-1]
    span = step_b - step_a
    if span == 0:
        return feat_b

    d1 = (feat_b - feat_a) / span
    dt = target_step - step_b
    return feat_b + d1 * dt


def _interpolate_taylor_2(
    target_step: int,
    history: list[tuple[int, mx.array]],
) -> mx.array:
    """Second-order Taylor extrapolation using the three most recent entries.

    Falls back to Taylor-1 if only two entries are available.
    """
    if len(history) < 3:
        return _interpolate_taylor_1(target_step, history)

    step_0, feat_0 = history[-3]
    step_1, feat_1 = history[-2]
    step_2, feat_2 = history[-1]

    span_01 = step_1 - step_0
    span_12 = step_2 - step_1
    if span_01 == 0 or span_12 == 0:
        return _interpolate_taylor_1(target_step, history)

    # First derivatives
    d1_prev = (feat_1 - feat_0) / span_01
    d1 = (feat_2 - feat_1) / span_12

    # Second derivative (central difference)
    avg_span = (step_2 - step_0) / 2.0
    d2 = (d1 - d1_prev) / avg_span

    dt = target_step - step_2
    return feat_2 + d1 * dt + 0.5 * d2 * (dt * dt)


def smooth_cache_interpolate(
    target_step: int,
    state: SmoothCacheState,
    config: SmoothCacheConfig,
) -> mx.array:
    """Interpolate features for a skipped step.

    Args:
        target_step: The step index being skipped.
        state: SmoothCache state with history of computed features.
        config: Interpolation mode and settings.

    Returns:
        Interpolated features for the target step.

    Raises:
        ValueError: If no history is available.
    """
    if not state.history:
        raise ValueError("SmoothCache has no history — cannot interpolate")

    if not config.enabled or len(state.history) < 2:
        return state.history[-1][1]

    if config.mode == InterpolationMode.LINEAR:
        return _interpolate_linear(target_step, state.history)
    elif config.mode == InterpolationMode.TAYLOR_1:
        return _interpolate_taylor_1(target_step, state.history)
    elif config.mode == InterpolationMode.TAYLOR_2:
        return _interpolate_taylor_2(target_step, state.history)
    else:
        return state.history[-1][1]
