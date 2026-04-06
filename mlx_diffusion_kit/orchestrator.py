"""B23 — Diffusion Optimization Orchestrator + PISA."""

from dataclasses import dataclass, field


@dataclass
class PISAConfig:
    """Profile-Informed Selective Activation configuration."""

    enabled: bool = True


@dataclass
class DiffusionOptimizer:
    """Orchestrates all optimization components for a diffusion pipeline.

    Manages step caching, token merging, gating, VAE optimization,
    and scheduling for both single-step and multi-step models.
    """

    is_single_step: bool = False
    pisa_config: PISAConfig = field(default_factory=PISAConfig)
