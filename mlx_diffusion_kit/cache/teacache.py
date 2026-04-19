"""B1 — TeaCache: Timestep-aware step caching for multi-step diffusion.

Computes relative L1 distance between consecutive modulated inputs.
If the distance (after polynomial rescaling) is below threshold,
the entire model forward pass is skipped and the cached residual is reused.

Reference: Liu et al., "Timestep Embedding Tells: It's Time to Cache
for Video Diffusion Model" (CVPR 2025 Highlight).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import mlx.core as mx

if TYPE_CHECKING:
    from mlx_diffusion_kit.cache.motion import MotionConfig


@dataclass
class TeaCacheConfig:
    rel_l1_thresh: float = 0.3
    poly_coeffs: Optional[list[float]] = None
    start_step: int = 0
    end_step: Optional[int] = None
    max_consecutive_cached: int = 5
    motion: Optional["MotionConfig"] = None
    enabled: bool = True


@dataclass
class TeaCacheState:
    prev_modulated_input: Optional[mx.array] = None
    cached_residual: Optional[mx.array] = None
    accumulated_distance: float = 0.0
    step_counter: int = 0
    consecutive_cached: int = 0


def create_teacache_state() -> TeaCacheState:
    """Create a fresh TeaCache state."""
    return TeaCacheState()


def _polyval(coeffs: list[float], x: float) -> float:
    """Evaluate polynomial with coefficients [a0, a1, a2, ...] at x.

    Result = a0 + a1*x + a2*x^2 + ...
    """
    result = 0.0
    for i, c in enumerate(coeffs):
        result += c * (x ** i)
    return result


def teacache_should_compute(
    modulated_input: mx.array,
    step_idx: int,
    config: TeaCacheConfig,
    state: TeaCacheState,
) -> bool:
    """Decide whether to compute the full model forward pass or reuse cache.

    Args:
        modulated_input: Current timestep-modulated input to the model.
        step_idx: Current diffusion step index.
        config: TeaCache configuration.
        state: Mutable TeaCache state.

    Returns:
        True if the model should be computed, False if cached residual should be used.
    """
    if not config.enabled:
        return True

    # Outside active window → always compute
    if step_idx < config.start_step:
        return True
    if config.end_step is not None and step_idx >= config.end_step:
        return True

    # First call — no previous input to compare
    if state.prev_modulated_input is None:
        return True

    # Compute relative L1 distance
    diff = mx.mean(mx.abs(modulated_input - state.prev_modulated_input)).item()
    norm = mx.mean(mx.abs(state.prev_modulated_input)).item() + 1e-6
    rel_l1 = diff / norm

    # Apply polynomial rescaling if coefficients provided
    if config.poly_coeffs is not None:
        rel_l1 = _polyval(config.poly_coeffs, rel_l1)

    state.accumulated_distance += rel_l1

    # Decide: skip if accumulated distance is small AND not too many consecutive skips
    if (
        state.accumulated_distance < config.rel_l1_thresh
        and state.consecutive_cached < config.max_consecutive_cached
    ):
        state.consecutive_cached += 1
        return False

    # Compute — reset accumulators
    state.accumulated_distance = 0.0
    state.consecutive_cached = 0
    return True


def teacache_update(
    modulated_input: mx.array,
    output: mx.array,
    state: TeaCacheState,
) -> None:
    """Update TeaCache state after a full model forward pass.

    Args:
        modulated_input: The input that was just computed.
        output: The model's output for this step.
        state: Mutable TeaCache state to update.
    """
    state.prev_modulated_input = modulated_input
    state.cached_residual = output
    state.step_counter += 1


_COEFFICIENTS_DIR = Path(__file__).parent / "coefficients"


def load_coefficients(model_name: str) -> TeaCacheConfig:
    """Load pre-calibrated TeaCache coefficients for a model.

    Args:
        model_name: Model identifier (e.g., "cogvideox"). Case-insensitive.

    Returns:
        TeaCacheConfig populated with the model's coefficients.

    Raises:
        FileNotFoundError: If no coefficients file exists for the model.
    """
    path = _COEFFICIENTS_DIR / f"{model_name.lower()}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No TeaCache coefficients for '{model_name}'. "
            f"Available: {[p.stem for p in _COEFFICIENTS_DIR.glob('*.json')]}"
        )

    with open(path) as f:
        data = json.load(f)

    return TeaCacheConfig(
        rel_l1_thresh=data["rel_l1_thresh"],
        poly_coeffs=data.get("poly_coeffs"),
    )
