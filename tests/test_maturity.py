"""Tests for maturity tracking system."""

from mlx_diffusion_kit.maturity import (
    Maturity,
    get_maturity,
    list_components,
)


def test_get_maturity_known():
    assert get_maturity("B1_TeaCache") == Maturity.STABLE
    assert get_maturity("B9_DiffSparse") == Maturity.STUB


def test_get_maturity_unknown():
    assert get_maturity("B99_Nonexistent") == Maturity.EXPERIMENTAL


def test_list_components_all():
    all_comps = list_components()
    assert len(all_comps) > 10
    assert "B1_TeaCache" in all_comps


def test_list_components_filtered():
    stable = list_components(Maturity.STABLE)
    assert all(v == Maturity.STABLE for v in stable.values())
    assert "B1_TeaCache" in stable
    assert "B9_DiffSparse" not in stable


def test_list_components_stub():
    stubs = list_components(Maturity.STUB)
    assert "B9_DiffSparse" in stubs
    assert "B1_TeaCache" not in stubs
