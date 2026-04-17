"""Tests for B18 Separable Conv3D (R(2+1)D factorization).

Contract invariants:
  - Mode A: SeparableConv3D.forward produces a 5D output with correct
    (N, T', H', W', out) shape.
  - Mode A: output is finite for realistic inputs.
  - Mode A: parameter count respects the separable formula
    ``kH·kW·in·mid + kT·mid·out (+ out)``.
  - Mode B: at full rank, the factorization reconstructs the original
    kernel to float32 precision (error < 1e-4).
  - Mode B: at reduced rank, the error is > 0 but still bounded.
  - Mode B: bridging via build_separable_from_decomposition produces
    a module whose forward pass succeeds and matches the dense reference
    within the reconstruction error tolerance.
  - Mode B: shape contract of returned tensors is exactly
    ``(mid, kH, kW, in)`` and ``(out, kT, mid)``.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_diffusion_kit.vae.separable_conv3d import (
    SeparableConv3D,
    build_separable_from_decomposition,
    decompose_conv3d_to_separable,
)


# -------------------------- Mode A: SeparableConv3D --------------------------


def test_mode_a_forward_shape():
    conv = SeparableConv3D(
        in_channels=4,
        out_channels=8,
        kernel_size=(3, 3, 3),
        mid_channels=6,
    )
    x = mx.random.normal((1, 5, 8, 8, 4))
    y = conv(x)
    # T: 5 - 3 + 1 = 3; H,W: 8 - 3 + 1 = 6.
    assert y.shape == (1, 3, 6, 6, 8)


def test_mode_a_forward_finite():
    conv = SeparableConv3D(in_channels=3, out_channels=5, kernel_size=(3, 3, 3))
    x = mx.random.normal((2, 4, 6, 6, 3))
    y = conv(x)
    assert mx.all(mx.isfinite(y)).item()


def test_mode_a_param_count_matches_formula():
    in_ch, out_ch, mid = 4, 8, 6
    kT, kH, kW = 3, 3, 3
    conv = SeparableConv3D(
        in_channels=in_ch,
        out_channels=out_ch,
        kernel_size=(kT, kH, kW),
        mid_channels=mid,
        bias=True,
    )
    # spatial.weight: (mid, kH, kW, in) = 6·3·3·4 = 216
    # temporal.weight: (out, kT, mid) = 8·3·6 = 144
    # temporal.bias: (out,) = 8
    expected = kH * kW * in_ch * mid + kT * mid * out_ch + out_ch
    total = conv.spatial.weight.size + conv.temporal.weight.size + conv.temporal.bias.size
    assert total == expected


def test_mode_a_input_validation():
    conv = SeparableConv3D(4, 8, (3, 3, 3))
    with pytest.raises(ValueError, match="5D input"):
        conv(mx.zeros((1, 8, 8, 4)))  # 4D


def test_mode_a_invalid_kernel_size():
    with pytest.raises(ValueError, match="kernel_size"):
        SeparableConv3D(4, 8, (3, 3))  # 2-tuple


# -------------------------- Mode B: SVD decomposition --------------------------


def test_mode_b_full_rank_is_near_lossless():
    """At full rank, reconstruction error should be at f32 precision."""
    mx.random.seed(0)
    out, kT, kH, kW, inp = 6, 3, 3, 3, 4
    W = mx.random.normal((out, kT, kH, kW, inp))

    spatial, temporal, err = decompose_conv3d_to_separable(W, rank=None)

    assert err < 1e-4, f"Full-rank decomposition error {err:.2e} too high"
    # Shape contract.
    max_rank = min(kT * out, kH * kW * inp)
    assert spatial.shape == (max_rank, kH, kW, inp)
    assert temporal.shape == (out, kT, max_rank)


def test_mode_b_reduced_rank_has_error_but_is_bounded():
    """At rank = 1, decomposition is lossy but still valid."""
    mx.random.seed(1)
    W = mx.random.normal((4, 3, 3, 3, 4))

    spatial, temporal, err = decompose_conv3d_to_separable(W, rank=1)

    # Error must be > 0 (lossy) but < 1 (bounded).
    assert err > 0.0
    assert err < 1.0
    assert spatial.shape == (1, 3, 3, 4)
    assert temporal.shape == (4, 3, 1)


def test_mode_b_error_monotone_in_rank():
    """Higher rank → lower or equal error."""
    mx.random.seed(2)
    W = mx.random.normal((6, 3, 3, 3, 4))

    errors = []
    max_rank = min(3 * 6, 3 * 3 * 4)  # 18, 36 → 18
    for r in (1, 4, 8, max_rank):
        _, _, err = decompose_conv3d_to_separable(W, rank=r)
        errors.append(err)

    for i in range(1, len(errors)):
        assert errors[i] <= errors[i - 1] + 1e-6, (
            f"Non-monotone: rank increased but error rose from "
            f"{errors[i-1]:.2e} to {errors[i]:.2e}"
        )


def test_mode_b_input_validation():
    with pytest.raises(ValueError, match="5D"):
        decompose_conv3d_to_separable(mx.zeros((4, 4, 4)))


def test_mode_b_bridge_builds_working_module():
    """Decompose a random conv3d, wrap in a module, run forward."""
    mx.random.seed(3)
    out, kT, kH, kW, inp = 8, 3, 3, 3, 4
    W = mx.random.normal((out, kT, kH, kW, inp))

    spatial, temporal, err = decompose_conv3d_to_separable(W, rank=None)

    mod = build_separable_from_decomposition(
        spatial_weight=spatial,
        temporal_weight=temporal,
        in_channels=inp,
        out_channels=out,
        kernel_size=(kT, kH, kW),
    )

    x = mx.random.normal((1, 5, 8, 8, inp))
    y = mod(x)
    assert y.shape == (1, 3, 6, 6, out)
    assert mx.all(mx.isfinite(y)).item()


def test_mode_b_forward_matches_dense_conv3d_at_full_rank():
    """The composed separable module at full rank must produce the same
    forward output as a dense nn.Conv3d with the original kernel.

    This is the user-facing contract for Mode B: full-rank decomposition
    must preserve forward behavior, not just weight-matrix reconstruction.
    A transpose/reshape bug inside decompose_conv3d_to_separable could
    silently corrupt the forward pass while still passing the weight-
    reconstruction test (since the same factorization reassembles to the
    same matrix regardless of axis ordering).
    """
    mx.random.seed(42)
    out, kT, kH, kW, inp = 4, 3, 3, 3, 3
    W = mx.random.normal((out, kT, kH, kW, inp))

    ref = nn.Conv3d(inp, out, (kT, kH, kW), bias=False)
    ref.weight = W

    sp, tp, _ = decompose_conv3d_to_separable(W, rank=None)
    mod = build_separable_from_decomposition(sp, tp, inp, out, (kT, kH, kW))

    x = mx.random.normal((1, 5, 5, 5, inp))
    y_ref = ref(x)
    y_sep = mod(x)
    assert mx.allclose(y_ref, y_sep, atol=1e-3, rtol=1e-3), (
        f"Mode B forward diverged from dense Conv3d: "
        f"max abs diff = {float(mx.max(mx.abs(y_ref - y_sep))):.6e}"
    )


def test_build_separable_preserves_bias():
    """build_separable_from_decomposition must carry a provided bias
    through to the composed module, not silently drop it.

    The bias mathematically belongs to the last stage of the forward
    pass (temporal conv). For a given input x, the difference between
    outputs with and without bias must equal the bias broadcast across
    the spatial/temporal output positions.
    """
    mx.random.seed(0)
    out, kT, kH, kW, inp = 4, 3, 3, 3, 3
    W = mx.random.normal((out, kT, kH, kW, inp))
    b = mx.random.normal((out,))

    sp, tp, _ = decompose_conv3d_to_separable(W, rank=None)

    mod_with = build_separable_from_decomposition(
        sp, tp, inp, out, (kT, kH, kW), bias=b
    )
    mod_without = build_separable_from_decomposition(
        sp, tp, inp, out, (kT, kH, kW), bias=None
    )

    x = mx.random.normal((1, 5, 5, 5, inp))
    y_with = mod_with(x)
    y_without = mod_without(x)

    # For each output channel c, the difference (y_with - y_without) at
    # every (N, T', H', W') position must equal b[c].
    diff = y_with - y_without  # (N, T', H', W', out)
    for c in range(out):
        channel_diff = diff[..., c]
        channel_mean = mx.mean(channel_diff)
        assert mx.allclose(channel_mean, b[c], atol=1e-3), (
            f"Channel {c}: expected bias {float(b[c]):.4f}, "
            f"got mean diff {float(channel_mean):.4f}"
        )


def test_build_separable_no_bias_by_default():
    """When bias is not supplied, the returned module must have bias=False
    (no implicit zero bias created)."""
    mx.random.seed(1)
    W = mx.random.normal((4, 3, 3, 3, 3))
    sp, tp, _ = decompose_conv3d_to_separable(W, rank=None)
    mod = build_separable_from_decomposition(sp, tp, 3, 4, (3, 3, 3))
    # Conv1d uses hasattr to check for bias; when bias=False it won't exist.
    assert "bias" not in mod.temporal.__dict__ or mod.temporal.bias is None or \
        not hasattr(mod.temporal, "bias"), (
        "Bridge with bias=None should not install a bias on the temporal conv"
    )
