"""B6 — Multi-Granular Cache: BWCache + UniCP + QuantCache.

Combines bandwidth-aware allocation, unified caching policy, and
quantized feature compression for optimal per-layer, per-step caching.

Builds on B1 (TeaCache) and B4 (SmoothCache) for decision signals
and interpolation fallback.
"""

from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx


# ---------------------------------------------------------------------------
# Component 1 — BWCache (Bandwidth-Aware Cache Allocator)
# ---------------------------------------------------------------------------


@dataclass
class BWCacheConfig:
    memory_budget_gb: float = 8.0
    dtype_bytes: int = 2  # 2 for f16, 4 for f32
    prefer_quality: bool = True
    enabled: bool = True


class BWCacheAllocator:
    """Allocates per-layer cache budgets based on available memory.

    Larger layers get fewer cached steps; smaller layers get more.
    Total memory stays within the configured budget.
    """

    def __init__(
        self,
        config: BWCacheConfig,
        layer_shapes: dict[int, tuple[int, ...]],
    ):
        self.config = config
        self.layer_shapes = layer_shapes
        self._allocation = self._compute_allocation()

    def _layer_bytes(self, shape: tuple[int, ...]) -> int:
        """Bytes for one cached feature tensor at this layer."""
        elements = 1
        for s in shape:
            elements *= s
        return elements * self.config.dtype_bytes

    def _compute_allocation(self) -> dict[int, int]:
        """Compute {layer_idx: max_cached_steps} within budget."""
        if not self.config.enabled or not self.layer_shapes:
            return {}

        budget_bytes = self.config.memory_budget_gb * 1e9
        layer_sizes = {
            idx: self._layer_bytes(shape)
            for idx, shape in self.layer_shapes.items()
        }

        if not layer_sizes:
            return {}

        # Sort layers: prefer_quality = cache more layers (1 step each first),
        # else cache fewer layers with more steps
        if self.config.prefer_quality:
            # Allocate 1 step per layer, starting from smallest
            sorted_layers = sorted(layer_sizes.items(), key=lambda x: x[1])
        else:
            # Allocate to largest layers first (most compute to save)
            sorted_layers = sorted(layer_sizes.items(), key=lambda x: -x[1])

        allocation: dict[int, int] = {}
        remaining = budget_bytes

        # First pass: 1 step per layer until budget exhausted
        for idx, size in sorted_layers:
            if size <= remaining:
                allocation[idx] = 1
                remaining -= size
            else:
                allocation[idx] = 0

        # Second pass: distribute remaining budget as extra steps
        for idx, size in sorted_layers:
            if allocation.get(idx, 0) == 0 or size == 0:
                continue
            extra_steps = int(remaining // size)
            if extra_steps > 0:
                allocation[idx] += extra_steps
                remaining -= extra_steps * size

        return allocation

    def compute_allocation(self) -> dict[int, int]:
        """Return the computed allocation."""
        return dict(self._allocation)

    def should_cache_layer(
        self, layer_idx: int, step_idx: int, current_cached: int
    ) -> bool:
        """Check if this layer at this step can be cached."""
        if not self.config.enabled:
            return False
        max_steps = self._allocation.get(layer_idx, 0)
        return current_cached < max_steps


# ---------------------------------------------------------------------------
# Component 2 — UniCP (Unified Caching Policy)
# ---------------------------------------------------------------------------

# Decision constants
DECISION_COMPUTE = "compute"
DECISION_CACHE_REUSE = "cache_reuse"
DECISION_INTERPOLATE = "interpolate"

# Default threshold for TeaCache distance signal
_DEFAULT_DISTANCE_THRESHOLD = 0.3


@dataclass
class UniCPConfig:
    use_teacache_signal: bool = True
    use_smooth_interpolation: bool = True
    use_bw_budget: bool = True
    distance_threshold: float = _DEFAULT_DISTANCE_THRESHOLD
    layer_priority: Optional[dict[int, float]] = None
    enabled: bool = True


class UniCPPolicy:
    """Unified caching policy combining TeaCache, SmoothCache, and BWCache signals."""

    def __init__(
        self,
        config: UniCPConfig,
        bw_allocator: Optional[BWCacheAllocator] = None,
    ):
        self.config = config
        self.bw_allocator = bw_allocator

    def decide(
        self,
        layer_idx: int,
        step_idx: int,
        teacache_distance: Optional[float],
        bw_budget_ok: bool,
    ) -> str:
        """Decide caching strategy for a (layer, step) pair.

        Returns:
            "compute" | "cache_reuse" | "interpolate"
        """
        if not self.config.enabled:
            return DECISION_COMPUTE

        # Determine effective distance threshold (adjusted by layer priority)
        threshold = self.config.distance_threshold
        if self.config.layer_priority and layer_idx in self.config.layer_priority:
            priority = self.config.layer_priority[layer_idx]
            # Higher priority → lower threshold → more likely to compute
            threshold *= (1.0 - 0.5 * priority)

        # Budget check: if budget not OK, force reuse
        if self.config.use_bw_budget and not bw_budget_ok:
            return DECISION_CACHE_REUSE

        # TeaCache distance signal
        if self.config.use_teacache_signal and teacache_distance is not None:
            if teacache_distance > threshold:
                return DECISION_COMPUTE
            else:
                if self.config.use_smooth_interpolation:
                    return DECISION_INTERPOLATE
                return DECISION_CACHE_REUSE

        # No signal available → compute
        return DECISION_COMPUTE


# ---------------------------------------------------------------------------
# Component 3 — QuantCache (Quantized Feature Cache)
# ---------------------------------------------------------------------------


@dataclass
class QuantCacheConfig:
    bits: int = 8
    per_channel: bool = True
    enabled: bool = True


def quantcache_compress(
    features: mx.array, config: QuantCacheConfig
) -> tuple[mx.array, mx.array]:
    """Compress features to lower-bit representation.

    Args:
        features: Input features to compress.
        config: Quantization settings.

    Returns:
        (quantized, scales) where quantized is int8/int8-packed and
        scales enable reconstruction.
    """
    if not config.enabled:
        return features, mx.array([1.0])

    max_val = 2 ** (config.bits - 1) - 1

    if config.per_channel:
        # Scale per last dimension (channel)
        reduce_axes = tuple(range(features.ndim - 1))
        abs_max = mx.max(mx.abs(features), axis=reduce_axes, keepdims=True)
    else:
        abs_max = mx.max(mx.abs(features), keepdims=True)

    scales = abs_max / (max_val + 1e-8)
    quantized = mx.round(features / (scales + 1e-8))
    quantized = mx.clip(quantized, -max_val, max_val).astype(mx.int8)

    return quantized, scales


def quantcache_decompress(
    quantized: mx.array,
    scales: mx.array,
    config: QuantCacheConfig,
) -> mx.array:
    """Decompress quantized features back to float.

    Args:
        quantized: Quantized int representation.
        scales: Scale factors from compression.
        config: Quantization settings.

    Returns:
        Reconstructed float features.
    """
    if not config.enabled:
        return quantized

    return quantized.astype(mx.float16) * scales


# ---------------------------------------------------------------------------
# Unified Multi-Granular Cache
# ---------------------------------------------------------------------------


@dataclass
class MultiGranularConfig:
    """Configuration bundle for the multi-granular cache system."""

    bw: Optional[BWCacheConfig] = None
    unicp: Optional[UniCPConfig] = None
    quant: Optional[QuantCacheConfig] = None
    layer_shapes: Optional[dict[int, tuple[int, ...]]] = None
    enabled: bool = True


class MultiGranularCache:
    """Combined BWCache + UniCP + QuantCache system.

    Manages per-layer feature caching with bandwidth-aware allocation,
    unified policy decisions, and optional quantized storage.
    """

    def __init__(
        self,
        bw_config: Optional[BWCacheConfig] = None,
        unicp_config: Optional[UniCPConfig] = None,
        quant_config: Optional[QuantCacheConfig] = None,
        layer_shapes: Optional[dict[int, tuple[int, ...]]] = None,
    ):
        self.bw_config = bw_config or BWCacheConfig(enabled=False)
        self.unicp_config = unicp_config or UniCPConfig(enabled=False)
        self.quant_config = quant_config or QuantCacheConfig(enabled=False)

        self._bw_allocator: Optional[BWCacheAllocator] = None
        if self.bw_config.enabled and layer_shapes:
            self._bw_allocator = BWCacheAllocator(self.bw_config, layer_shapes)

        self._policy = UniCPPolicy(self.unicp_config, self._bw_allocator)

        # Per-layer cache: {layer_idx: (quantized_or_raw, scales, step_idx)}
        self._cache: dict[int, tuple[mx.array, mx.array, int]] = {}
        self._cached_counts: dict[int, int] = {}

        # Stats
        self._hits = 0
        self._misses = 0

    def process_layer(
        self,
        layer_idx: int,
        step_idx: int,
        features: Optional[mx.array],
        teacache_distance: Optional[float] = None,
    ) -> mx.array:
        """Process a layer through the multi-granular cache pipeline.

        Args:
            layer_idx: Transformer layer index.
            step_idx: Current denoising step.
            features: Freshly computed features (None if caller wants cache only).
            teacache_distance: L1 distance signal from TeaCache.

        Returns:
            Features to use (computed, cached, or interpolated).
        """
        # Determine budget status
        current_cached = self._cached_counts.get(layer_idx, 0)
        bw_ok = True
        if self._bw_allocator is not None:
            bw_ok = self._bw_allocator.should_cache_layer(
                layer_idx, step_idx, current_cached
            )

        decision = self._policy.decide(
            layer_idx, step_idx, teacache_distance, bw_ok
        )

        if decision == DECISION_COMPUTE:
            self._misses += 1
            if features is None:
                raise ValueError(
                    f"UniCP decided 'compute' for layer {layer_idx} step {step_idx} "
                    "but no features provided"
                )
            # Store in cache (optionally quantized)
            if self.quant_config.enabled:
                q, s = quantcache_compress(features, self.quant_config)
                self._cache[layer_idx] = (q, s, step_idx)
            else:
                self._cache[layer_idx] = (features, mx.array([1.0]), step_idx)
            self._cached_counts[layer_idx] = current_cached + 1
            return features

        elif decision == DECISION_CACHE_REUSE:
            self._hits += 1
            if layer_idx in self._cache:
                q, s, _ = self._cache[layer_idx]
                if self.quant_config.enabled:
                    return quantcache_decompress(q, s, self.quant_config)
                return q
            # Fallback: use provided features if no cache entry
            if features is not None:
                return features
            raise ValueError(
                f"Cache reuse requested for layer {layer_idx} but no cache entry exists"
            )

        else:  # DECISION_INTERPOLATE
            self._hits += 1
            # For interpolation, caller should use SmoothCache externally.
            # Here we return the cached value as base for interpolation.
            if layer_idx in self._cache:
                q, s, _ = self._cache[layer_idx]
                if self.quant_config.enabled:
                    return quantcache_decompress(q, s, self.quant_config)
                return q
            if features is not None:
                return features
            raise ValueError(
                f"Interpolation requested for layer {layer_idx} but no cache entry exists"
            )

    def get_stats(self) -> dict:
        """Return cache statistics."""
        total = self._hits + self._misses
        return {
            "cache_hits": self._hits,
            "cache_misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "cached_layers": len(self._cache),
            "total_cached_entries": sum(self._cached_counts.values()),
        }

    def clear(self) -> None:
        """Clear all cached features and reset stats."""
        self._cache.clear()
        self._cached_counts.clear()
        self._hits = 0
        self._misses = 0
