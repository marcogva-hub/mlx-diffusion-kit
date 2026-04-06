"""B5 — DeepCache + MosaicDiff: Layer-level caching for UNet models.

Skips entire UNet layers between denoising steps. Middle (bottleneck) layers
change least → cache them and only recompute input/output layers.

Applicable to 3 UNet multi-step models: DAM-VSR, DiffVSR, VEnhancer.
Estimated impact: 2.2-3.2x on UNet backbone.

MosaicDiff provides principled layer selection via weight redundancy analysis.
"""

from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx


# ---------------------------------------------------------------------------
# DeepCache
# ---------------------------------------------------------------------------


@dataclass
class DeepCacheConfig:
    cache_interval: int = 3
    cached_layer_indices: Optional[list[int]] = None
    auto_cache_ratio: float = 0.5
    enabled: bool = True


@dataclass
class DeepCacheState:
    layer_cache: dict[int, mx.array] = field(default_factory=dict)
    last_full_step: int = -1
    skip_count: int = 0


class DeepCacheManager:
    """Manages per-layer caching for UNet architectures.

    Automatically selects bottleneck layers for caching if no explicit
    indices are provided. Cached layers are recomputed every cache_interval
    steps; between those steps, their cached output is reused.
    """

    def __init__(self, total_layers: int, config: Optional[DeepCacheConfig] = None):
        self.config = config or DeepCacheConfig()
        self.total_layers = total_layers
        self._state = DeepCacheState()

        if self.config.cached_layer_indices is not None:
            self._cached_set = set(self.config.cached_layer_indices)
        else:
            self._cached_set = self._auto_select_layers()

    def _auto_select_layers(self) -> set[int]:
        """Auto-select middle layers (bottleneck region) for caching."""
        n = self.total_layers
        if n <= 2:
            return set()

        # Middle region: N//4 to 3*N//4
        start = n // 4
        end = (3 * n) // 4
        candidates = list(range(start, end))

        n_cache = max(1, int(len(candidates) * self.config.auto_cache_ratio))
        # Pick from the center outward
        mid = len(candidates) // 2
        selected = sorted(candidates[mid - n_cache // 2: mid - n_cache // 2 + n_cache])
        return set(selected)

    def should_compute_layer(self, layer_idx: int, step_idx: int) -> bool:
        """Check if a layer should be computed at this step.

        Non-cached layers always compute. Cached layers compute only
        at cache_interval boundaries.

        Args:
            layer_idx: Layer index.
            step_idx: Current denoising step.

        Returns:
            True if the layer should be computed.
        """
        if not self.config.enabled:
            return True

        if layer_idx not in self._cached_set:
            return True

        # Always compute on first step or at interval boundaries
        if self._state.last_full_step < 0:
            return True

        return step_idx - self._state.last_full_step >= self.config.cache_interval

    def get_cached_layer(self, layer_idx: int) -> Optional[mx.array]:
        """Retrieve cached output for a layer."""
        return self._state.layer_cache.get(layer_idx)

    def update_layer(self, layer_idx: int, step_idx: int, output: mx.array) -> None:
        """Update the cache for a layer.

        Also updates last_full_step if this is a cached layer.
        """
        if layer_idx in self._cached_set:
            self._state.layer_cache[layer_idx] = output
            self._state.last_full_step = step_idx

    def get_cached_layers(self) -> set[int]:
        """Return indices of layers selected for caching."""
        return set(self._cached_set)

    def reset(self) -> None:
        """Clear all cached state."""
        self._state = DeepCacheState()


# ---------------------------------------------------------------------------
# MosaicDiff — Layer Redundancy Scoring
# ---------------------------------------------------------------------------


def analyze_layer_redundancy(
    layer_weights: dict[int, mx.array],
    method: str = "cosine",
) -> dict[int, float]:
    """Score each layer by its redundancy with adjacent layers.

    Layers with high redundancy are good candidates for caching — their
    computation is largely duplicated by their neighbors.

    Args:
        layer_weights: {layer_idx: weight_tensor} for each layer.
        method: "cosine" (cosine similarity) or "l2" (inverse L2 distance).

    Returns:
        {layer_idx: redundancy_score} normalized to [0, 1].
        Higher score = more redundant = better caching candidate.
    """
    if len(layer_weights) < 2:
        return {idx: 0.0 for idx in layer_weights}

    indices = sorted(layer_weights.keys())
    flat = {idx: layer_weights[idx].reshape(-1).astype(mx.float32) for idx in indices}

    raw_scores: dict[int, float] = {}
    for i, idx in enumerate(indices):
        sim_sum = 0.0
        count = 0

        for neighbor in [i - 1, i + 1]:
            if 0 <= neighbor < len(indices):
                n_idx = indices[neighbor]
                a = flat[idx]
                b = flat[n_idx]

                # Ensure same size for comparison
                min_len = min(a.shape[0], b.shape[0])
                a = a[:min_len]
                b = b[:min_len]

                if method == "cosine":
                    norm_a = mx.linalg.norm(a) + 1e-8
                    norm_b = mx.linalg.norm(b) + 1e-8
                    sim = (mx.sum(a * b) / (norm_a * norm_b)).item()
                elif method == "l2":
                    dist = mx.linalg.norm(a - b).item()
                    sim = 1.0 / (1.0 + dist)
                else:
                    raise ValueError(f"Unknown method: {method}")

                sim_sum += sim
                count += 1

        raw_scores[idx] = sim_sum / max(count, 1)

    # Normalize to [0, 1]
    if not raw_scores:
        return {}
    min_s = min(raw_scores.values())
    max_s = max(raw_scores.values())
    rng = max_s - min_s
    if rng < 1e-10:
        # All scores identical — if they're high (>0.5), all redundant
        avg = sum(raw_scores.values()) / len(raw_scores)
        return {k: 1.0 if avg > 0.5 else 0.0 for k in raw_scores}
    return {k: (v - min_s) / rng for k, v in raw_scores.items()}


def select_cacheable_layers(
    redundancy_scores: dict[int, float],
    ratio: float = 0.5,
) -> list[int]:
    """Select the most redundant layers for caching.

    Args:
        redundancy_scores: {layer_idx: redundancy_score}.
        ratio: Fraction of layers to select.

    Returns:
        Sorted list of layer indices to cache.
    """
    n_select = max(1, int(len(redundancy_scores) * ratio))
    sorted_by_score = sorted(redundancy_scores.items(), key=lambda x: -x[1])
    return sorted([idx for idx, _ in sorted_by_score[:n_select]])
