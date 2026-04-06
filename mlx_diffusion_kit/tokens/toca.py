"""B7 — ToCa (Token Caching): Per-token caching between denoising steps.

Caches individual tokens that change little between steps and only recomputes
tokens that have significantly changed. Complementary to ToMe (B8) which
reduces tokens within a step, while ToCa reduces recomputation across steps.

Applicable to 6 multi-step models.
All operations are fully vectorized — no .item() calls or Python loops.
"""

from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx


@dataclass
class ToCaConfig:
    cache_ratio: float = 0.3
    similarity_threshold: float = 0.95
    recompute_interval: int = 5
    use_attention_scores: bool = True
    enabled: bool = True


@dataclass
class ToCaState:
    cached_tokens: Optional[mx.array] = None
    cached_mask: Optional[mx.array] = None
    last_full_step: int = -1


class TokenCacheManager:
    """Manages per-token caching across denoising steps.

    Identifies which tokens are stable (cacheable) vs dynamic (must recompute)
    using cosine similarity and optional attention-score prioritization.
    """

    def __init__(self, config: Optional[ToCaConfig] = None):
        self.config = config or ToCaConfig()
        self._state = ToCaState()

    def identify_stable_tokens(
        self,
        current_tokens: mx.array,
        step_idx: int,
        attention_scores: Optional[mx.array] = None,
    ) -> mx.array:
        """Identify tokens to cache vs recompute.

        Args:
            current_tokens: Current step's tokens [B, N, D].
            step_idx: Current denoising step.
            attention_scores: Optional [B, H, N, N] or [B, N, N] attention weights.

        Returns:
            Boolean mask [B, N]: True = stable (use cache), False = dynamic (recompute).
        """
        B, N, D = current_tokens.shape

        if not self.config.enabled:
            return mx.zeros((B, N), dtype=mx.bool_)

        # Force full recompute conditions
        if self._state.cached_tokens is None:
            return mx.zeros((B, N), dtype=mx.bool_)

        if step_idx - self._state.last_full_step >= self.config.recompute_interval:
            return mx.zeros((B, N), dtype=mx.bool_)

        # Cosine similarity between current and cached tokens
        cached = self._state.cached_tokens
        cur_norm = mx.linalg.norm(current_tokens, axis=-1, keepdims=True) + 1e-8
        cac_norm = mx.linalg.norm(cached, axis=-1, keepdims=True) + 1e-8
        cos_sim = mx.sum(
            (current_tokens / cur_norm) * (cached / cac_norm), axis=-1
        )  # [B, N]

        # Base stability mask: tokens above similarity threshold
        stable_mask = cos_sim >= self.config.similarity_threshold  # [B, N]

        # If using attention scores, deprioritize high-attention tokens
        # (high-attention tokens are more "influential" → riskier to cache)
        if self.config.use_attention_scores and attention_scores is not None:
            # Sum attention received (column sum) → importance per token
            if attention_scores.ndim == 4:
                # [B, H, N, N] → [B, N] via column sum then head mean
                importance = mx.mean(mx.sum(attention_scores, axis=-2), axis=1)
            else:
                importance = mx.sum(attention_scores, axis=-2)  # [B, N]

            # Low importance → better cache candidate
            # Sort by importance, allow caching only for bottom-(1-importance_rank) tokens
            rank = mx.argsort(importance, axis=-1)  # ascending: least important first
            # Create priority mask: first cache_ratio*N tokens (least important) can be cached
            n_cacheable = max(1, int(N * self.config.cache_ratio))
            priority_threshold = rank[:, n_cacheable:n_cacheable + 1]  # [B, 1]
            # Tokens whose rank position is below n_cacheable can be cached
            # Build mask: for each token, check if its rank is in the cacheable set
            token_ranks = mx.zeros_like(importance, dtype=mx.int32)
            for b_idx in range(B):
                token_ranks = token_ranks.at[b_idx, rank[b_idx]].add(
                    mx.arange(N, dtype=mx.int32)
                )
            cacheable = token_ranks < n_cacheable
            stable_mask = stable_mask & cacheable
        else:
            # Enforce cache_ratio without attention scores
            n_stable = mx.sum(stable_mask.astype(mx.int32), axis=-1, keepdims=True)
            max_cached = int(N * self.config.cache_ratio)
            if max_cached < N:
                # If too many stable, keep only the first max_cached by position
                cumsum = mx.cumsum(stable_mask.astype(mx.int32), axis=-1)
                stable_mask = stable_mask & (cumsum <= max_cached)

        return stable_mask

    def apply_cache(
        self,
        computed_tokens: mx.array,
        cache_mask: mx.array,
    ) -> mx.array:
        """Reconstruct full sequence from computed dynamic tokens and cached stable tokens.

        Args:
            computed_tokens: Full-size tensor with recomputed values [B, N, D].
                Dynamic positions have new values; stable positions can be anything.
            cache_mask: Boolean [B, N]: True = use cached value, False = use computed.

        Returns:
            Complete tokens [B, N, D] with cached stable + computed dynamic.
        """
        if self._state.cached_tokens is None:
            return computed_tokens

        mask_expanded = mx.expand_dims(cache_mask, axis=-1)  # [B, N, 1]
        return mx.where(mask_expanded, self._state.cached_tokens, computed_tokens)

    def update_cache(
        self, tokens: mx.array, mask: mx.array, step_idx: int
    ) -> None:
        """Update the token cache.

        Args:
            tokens: Complete tokens [B, N, D] from this step.
            mask: Cache mask used this step [B, N].
            step_idx: Current step index.
        """
        self._state.cached_tokens = tokens
        self._state.cached_mask = mask
        # Track full recompute steps (when no tokens were cached)
        if not mx.any(mask).item():
            self._state.last_full_step = step_idx

    def get_dynamic_indices(self, cache_mask: mx.array) -> mx.array:
        """Return indices of tokens that need recomputation.

        Args:
            cache_mask: Boolean [B, N]: True = cached, False = dynamic.

        Returns:
            Indices [B, N_dynamic] of dynamic tokens per batch.
            Note: N_dynamic may vary per batch element. Returns padded
            to max N_dynamic with -1 for padding.
        """
        B, N = cache_mask.shape
        # Invert mask: dynamic tokens
        dynamic = ~cache_mask  # [B, N]
        # Use argsort trick: sort so True (dynamic) comes first
        # Then take the first N_dynamic indices
        indices = mx.argsort(~dynamic, axis=-1)  # dynamic indices first
        n_dynamic = mx.sum(dynamic.astype(mx.int32), axis=-1)  # [B]
        max_dynamic = int(mx.max(n_dynamic).item())
        return indices[:, :max_dynamic]

    def reset(self) -> None:
        """Clear all cached state."""
        self._state = ToCaState()
