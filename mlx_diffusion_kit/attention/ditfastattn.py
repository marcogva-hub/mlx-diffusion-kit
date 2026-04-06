"""B12 — DiTFastAttn: Per-head attention compression for multi-step DiT.

Assigns each attention head a strategy (FULL, WINDOW, or CACHED) based on
its convergence behavior across denoising steps. Auto-profiles during the
first few steps, then applies the optimal strategy per head.

Applicable to: SparkVSR, STAR, Vivid-VR (multi-step DiT).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import mlx.core as mx


class HeadStrategy(Enum):
    FULL = "full"
    WINDOW = "window"
    CACHED = "cached"


@dataclass
class DiTFastAttnConfig:
    window_size: int = 256
    cache_start_step: int = 5
    head_sensitivity_scores: Optional[dict[tuple[int, int], float]] = None
    auto_profile_steps: int = 3
    sensitivity_threshold: float = 0.3
    enabled: bool = True


@dataclass
class DiTFastAttnState:
    cached_attention_weights: dict[tuple[int, int], mx.array] = field(default_factory=dict)
    head_strategies: dict[tuple[int, int], HeadStrategy] = field(default_factory=dict)
    profiling_data: list[dict[tuple[int, int], mx.array]] = field(default_factory=list)
    profiled: bool = False


class DiTFastAttnManager:
    """Manages per-head attention strategy assignment and caching.

    During the first auto_profile_steps steps, records attention weights
    to compute per-head variance. After profiling, assigns strategies:
    - Low variance heads → CACHED (reuse attention weights)
    - Medium variance heads → WINDOW (local attention)
    - High variance heads → FULL (no compression)
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        config: Optional[DiTFastAttnConfig] = None,
    ):
        self.config = config or DiTFastAttnConfig()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self._state = DiTFastAttnState()
        self._step_count = 0

        # If pre-computed sensitivity scores provided, assign strategies immediately
        if self.config.head_sensitivity_scores:
            self._assign_strategies_from_scores(self.config.head_sensitivity_scores)
            self._state.profiled = True

    def _assign_strategies_from_scores(
        self, scores: dict[tuple[int, int], float]
    ) -> None:
        """Assign strategies based on sensitivity scores."""
        threshold = self.config.sensitivity_threshold
        for key, score in scores.items():
            if score < threshold:
                self._state.head_strategies[key] = HeadStrategy.CACHED
            elif score < threshold * 2:
                self._state.head_strategies[key] = HeadStrategy.WINDOW
            else:
                self._state.head_strategies[key] = HeadStrategy.FULL

    def profile_step(
        self,
        layer_idx: int,
        head_idx: int,
        attention_weights: mx.array,
        step_idx: int,
    ) -> None:
        """Record attention weights for auto-profiling.

        Call during the first auto_profile_steps steps for each (layer, head).

        Args:
            layer_idx: Transformer layer index.
            head_idx: Attention head index.
            attention_weights: Attention weight matrix for this head.
            step_idx: Current denoising step.
        """
        if self._state.profiled or not self.config.enabled:
            return

        # Extend profiling_data list to cover this step
        while len(self._state.profiling_data) <= step_idx:
            self._state.profiling_data.append({})

        self._state.profiling_data[step_idx][(layer_idx, head_idx)] = attention_weights

        # Check if profiling is complete
        if step_idx + 1 >= self.config.auto_profile_steps:
            self.finalize_profiling()

    def finalize_profiling(self) -> None:
        """Compute strategies from profiling data.

        Sensitivity = mean variance of attention weights across profiled steps.
        High variance → FULL. Low variance → CACHED.
        """
        if self._state.profiled:
            return

        # Collect all (layer, head) pairs seen
        all_keys: set[tuple[int, int]] = set()
        for step_data in self._state.profiling_data:
            all_keys.update(step_data.keys())

        scores: dict[tuple[int, int], float] = {}
        for key in all_keys:
            weights_across_steps = []
            for step_data in self._state.profiling_data:
                if key in step_data:
                    weights_across_steps.append(step_data[key])

            if len(weights_across_steps) < 2:
                scores[key] = 1.0  # Insufficient data → assume dynamic
                continue

            # Variance across steps: mean of element-wise variance
            stacked = mx.stack(weights_across_steps, axis=0)  # [S, ...]
            variance = mx.var(stacked, axis=0)
            scores[key] = mx.mean(variance).item()

        self._assign_strategies_from_scores(scores)
        self._state.profiled = True

    def get_head_strategy(
        self, layer_idx: int, head_idx: int, step_idx: int
    ) -> HeadStrategy:
        """Return the strategy for a (layer, head) at a given step.

        Before profiling completes, all heads use FULL.
        After profiling, cached heads only activate after cache_start_step.
        """
        if not self.config.enabled:
            return HeadStrategy.FULL

        if not self._state.profiled:
            return HeadStrategy.FULL

        key = (layer_idx, head_idx)
        strategy = self._state.head_strategies.get(key, HeadStrategy.FULL)

        # CACHED only allowed after cache_start_step
        if strategy == HeadStrategy.CACHED and step_idx < self.config.cache_start_step:
            return HeadStrategy.FULL

        return strategy

    def get_window_mask(self, seq_len: int) -> mx.array:
        """Return a windowed attention mask [seq_len, seq_len].

        True = attend, False = masked. Each position attends to a local
        window of size config.window_size centered on itself.
        """
        indices = mx.arange(seq_len)
        row = mx.expand_dims(indices, axis=1)  # [N, 1]
        col = mx.expand_dims(indices, axis=0)  # [1, N]
        half_w = self.config.window_size // 2
        return mx.abs(row - col) <= half_w

    def cache_attention(
        self, layer_idx: int, head_idx: int, attention_weights: mx.array
    ) -> None:
        """Store attention weights for CACHED strategy reuse."""
        self._state.cached_attention_weights[(layer_idx, head_idx)] = attention_weights

    def get_cached_attention(
        self, layer_idx: int, head_idx: int
    ) -> Optional[mx.array]:
        """Retrieve cached attention weights."""
        return self._state.cached_attention_weights.get((layer_idx, head_idx))

    def reset(self) -> None:
        """Clear all state including profiling data."""
        self._state = DiTFastAttnState()
        if self.config.head_sensitivity_scores:
            self._assign_strategies_from_scores(self.config.head_sensitivity_scores)
            self._state.profiled = True
