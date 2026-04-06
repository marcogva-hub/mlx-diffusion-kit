"""Component maturity tracking for mlx-diffusion-kit.

Each optimization component has a maturity level indicating its readiness
for production use. This registry is the single source of truth — keep it
updated as components evolve.

Levels:
  STABLE — Tested, integrated, production-ready. API frozen.
  BETA — Functional and tested, but API may change.
  EXPERIMENTAL — Functional, use with caution. Limited testing.
  STUB — Interface defined, implementation partial or absent.
"""

from enum import Enum


class Maturity(Enum):
    STABLE = "stable"
    BETA = "beta"
    EXPERIMENTAL = "experimental"
    STUB = "stub"


COMPONENT_MATURITY: dict[str, Maturity] = {
    "B1_TeaCache": Maturity.STABLE,
    "B2_FBCache": Maturity.BETA,
    "B3_SpectralCache": Maturity.BETA,
    "B4_SmoothCache": Maturity.STABLE,
    "B5_DeepCache": Maturity.BETA,
    "B6_MultiGranular": Maturity.BETA,
    "B7_ToCa": Maturity.BETA,
    "B8_ToMe": Maturity.STABLE,
    "B8_ToPi": Maturity.BETA,
    "B9_DiffSparse": Maturity.STUB,
    "B10_DDiT": Maturity.BETA,
    "B11_TGATE": Maturity.STABLE,
    "B12_DiTFastAttn": Maturity.EXPERIMENTAL,
    "B13_FreeU": Maturity.STABLE,
    "B14_DPMSolver": Maturity.STABLE,
    "B14_AdaptiveStep": Maturity.BETA,
    "B15_EmbedCache": Maturity.STABLE,
    "B17_WaveletVAE": Maturity.STABLE,
    "B18_SeparableConv3D": Maturity.STUB,
    "B19_DecoderDistill": Maturity.STUB,
    "B22_EncoderSharing": Maturity.BETA,
    "B23_Orchestrator": Maturity.STABLE,
    "WorldCache_Motion": Maturity.STUB,
}


def get_maturity(component_id: str) -> Maturity:
    """Get the maturity level of a component.

    Args:
        component_id: Component identifier (e.g., "B1_TeaCache").

    Returns:
        Maturity level. Returns EXPERIMENTAL for unknown components.
    """
    return COMPONENT_MATURITY.get(component_id, Maturity.EXPERIMENTAL)


def list_components(maturity: Maturity | None = None) -> dict[str, Maturity]:
    """List components, optionally filtered by maturity level.

    Args:
        maturity: If provided, only return components at this level.

    Returns:
        Dict of component_id -> Maturity.
    """
    if maturity is None:
        return dict(COMPONENT_MATURITY)
    return {k: v for k, v in COMPONENT_MATURITY.items() if v == maturity}
