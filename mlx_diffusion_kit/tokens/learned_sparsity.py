"""B9 — DiffSparse: Learned token sparsity routing for DiT models.

Trains a lightweight per-layer router MLP to decide which tokens to keep.
Successor to ToMe for single-step DiT (SeedVR2, DOVE, FlashVSR).
Requires fine-tuning (~10K iterations) — this module provides the interface
and a deterministic stub implementation.

WITHOUT PRETRAINED WEIGHTS, this module operates in debug/testing mode only.
The stub fallback keeps the first N tokens deterministically, which is
spatially biased and will produce suboptimal results.

Reference: Adapted from E-DiT / DiffSparse approaches.
"""

import logging
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class DiffSparseConfig:
    budget: float = 0.5
    router_hidden_dim: int = 64
    strict: bool = False
    enabled: bool = True


class DiffSparseRouter(nn.Module):
    """Per-layer token routing via learned MLP.

    The router predicts a keep-probability for each token. The top-k
    tokens (by score) are retained.

    Without pretrained weights (the default state), behavior depends on
    the ``strict`` config flag:

    - ``strict=True``: raises RuntimeError. Use this in production pipelines
      to catch missing weights early.
    - ``strict=False`` (default): falls back to keeping the first budget*N
      tokens deterministically with a one-time warning. This is for
      testing/debugging only — results will be spatially biased.
    """

    def __init__(self, input_dim: int, config: DiffSparseConfig | None = None):
        super().__init__()
        self.config = config or DiffSparseConfig()
        self.input_dim = input_dim

        # Router MLP: input_dim -> hidden -> 1
        self.gate = nn.Sequential(
            nn.Linear(input_dim, self.config.router_hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.router_hidden_dim, 1),
        )
        self._pretrained = False
        self._warned = False

    def __call__(self, tokens: mx.array) -> tuple[mx.array, mx.array]:
        """Route tokens: select top-budget fraction by routing score.

        Args:
            tokens: Input tokens [B, N, D].

        Returns:
            (selected_tokens, routing_scores) where:
                selected_tokens: [B, N_kept, D]
                routing_scores: [B, N] raw routing logits

        Raises:
            RuntimeError: If strict=True and no pretrained weights are loaded.
        """
        if not self.config.enabled:
            scores = mx.zeros(tokens.shape[:2])
            return tokens, scores

        B, N, D = tokens.shape
        n_keep = max(1, int(N * self.config.budget))

        if not self._pretrained:
            if self.config.strict:
                raise RuntimeError(
                    "DiffSparse router has no pretrained weights. "
                    "Load weights with from_pretrained() or set strict=False for debug fallback."
                )
            if not self._warned:
                logger.warning(
                    "DiffSparse: no pretrained weights loaded. Using deterministic fallback "
                    "(keeps first N tokens). This is for testing only — results will be suboptimal."
                )
                self._warned = True
            return tokens[:, :n_keep, :], mx.zeros((B, N))

        # Full routing with learned weights
        scores = self.gate(tokens).squeeze(-1)  # [B, N]
        indices = mx.argpartition(-scores, kth=n_keep, axis=-1)[:, :n_keep]
        # Sort indices to preserve spatial order
        indices = mx.sort(indices, axis=-1)
        # Gather selected tokens
        selected = mx.take_along_axis(tokens, mx.expand_dims(indices, -1), axis=1)
        return selected, scores

    @staticmethod
    def from_pretrained(path: str) -> "DiffSparseRouter":
        """Load a pretrained router from disk.

        Args:
            path: Path to saved router weights.

        Raises:
            NotImplementedError: Router training not yet implemented.
        """
        raise NotImplementedError(
            "DiffSparse router weights required. "
            "Train with scripts/train_diffsparse_router.py (not yet available)."
        )
