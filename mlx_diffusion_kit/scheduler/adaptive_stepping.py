"""B14.2 — Adaptive Step Size for diffusion schedulers.

Detects convergence between consecutive denoising outputs and skips
redundant steps. Training-free, wraps any scheduler.
"""

from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx


@dataclass
class AdaptiveStepConfig:
    min_steps: int = 10
    max_steps: int = 50
    tolerance: float = 0.01
    reduction_factor: float = 0.5
    enabled: bool = True


class AdaptiveStepScheduler:
    """Adaptive step pruning for diffusion inference.

    Wraps a base set of timesteps and decides per-step whether the
    model forward pass can be skipped based on output convergence.
    """

    def __init__(
        self,
        base_timesteps: list[float],
        config: Optional[AdaptiveStepConfig] = None,
    ):
        self.config = config or AdaptiveStepConfig()
        self._base_timesteps = list(base_timesteps)
        self._skipped: set[int] = set()
        self._steps_computed: int = 0

    def should_skip_step(
        self,
        step_idx: int,
        prev_output: mx.array,
        curr_output: mx.array,
    ) -> bool:
        """Decide whether to skip the next step based on output convergence.

        Args:
            step_idx: Current step index (0-based).
            prev_output: Model output from the previous step.
            curr_output: Model output from the current step.

        Returns:
            True if the next step can be skipped, False otherwise.
        """
        if not self.config.enabled:
            return False

        total = len(self._base_timesteps)
        remaining = total - self._steps_computed

        # Never skip if we'd go below min_steps
        if self._steps_computed + remaining <= self.config.min_steps:
            return False

        # Already computed enough to be under min_steps with skipping?
        # Ensure min_steps are always respected
        if self._steps_computed < self.config.min_steps:
            self._steps_computed += 1
            return False

        # Compute MSE between consecutive outputs
        mse = mx.mean((curr_output - prev_output) ** 2).item()

        if mse < self.config.tolerance:
            self._skipped.add(step_idx + 1)
            return True

        self._steps_computed += 1
        return False

    def get_effective_timesteps(self) -> list[float]:
        """Return timesteps after pruning skipped steps."""
        return [
            t
            for i, t in enumerate(self._base_timesteps)
            if i not in self._skipped
        ]

    def reset(self) -> None:
        """Reset scheduler state for a new inference run."""
        self._skipped.clear()
        self._steps_computed = 0

    @property
    def num_skipped(self) -> int:
        """Number of steps skipped so far."""
        return len(self._skipped)

    @property
    def total_steps(self) -> int:
        """Total base steps."""
        return len(self._base_timesteps)
