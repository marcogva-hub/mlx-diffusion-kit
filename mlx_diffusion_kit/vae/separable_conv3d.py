"""B18 — Separable Conv3D: R(2+1)D factorization for VAE compute savings.

Two modes serve two users:

Mode A — :class:`SeparableConv3D` (``mlx.nn.Module``):
    For users building a model from scratch with R(2+1)D blocks. Applies
    a 2D spatial convolution followed by a 1D temporal convolution.
    Parameter count for a ``(kT, kH, kW)`` kernel with ``in → out`` and
    middle rank ``M``:

        separable = kH·kW·in·M + kT·M·out
        full      = kT·kH·kW·in·out

    When ``M`` is chosen to match ``full``, the two have similar cost
    and the factorization is "for free" in param count.

Mode B — :func:`decompose_conv3d_to_separable`:
    For users porting a pretrained model whose weights are stored as a
    dense Conv3D kernel. SVD on a reshape of the kernel yields a rank-R
    approximation ``(spatial_2d, temporal_1d)``. At full rank
    (``rank = min(kT·out, kH·kW·in)``) the reconstruction is exact up
    to numerical precision; at lower rank it is lossy and the returned
    ``reconstruction_error`` quantifies the cost.

    Rationale for SVD over other decompositions: R(2+1)D's natural
    bilinear form ``W[o,t,h,w,i] = Σ_m W_t[t,o,m] · W_s[m,h,w,i]`` is
    exactly the low-rank matrix factorization obtained by reshaping
    ``W → [kT·out, kH·kW·in]`` and applying SVD. No iterative solver,
    no training required.

MLX Conv layout reference (verified):
    * ``nn.Conv3d.weight`` has shape ``(out, kT, kH, kW, in)`` and
      expects input ``(N, T, H, W, in)``.
    * ``nn.Conv2d.weight``: ``(out, kH, kW, in)``, input ``(N, H, W, in)``.
    * ``nn.Conv1d.weight``: ``(out, kT, in)``, input ``(N, T, in)``.

Training-free note:
    Mode A is fully training-free — it's a new module. Mode B is
    training-free *but* lossy at reduced rank. For bit-exact deployment
    of pretrained models that used full Conv3d, the user must either
    accept the reconstruction error or retrain with a Mode A block.

Applies to: VAE 3D convolutions, most notably SeedVR2's temporal-
    aware encoder/decoder.

Reference: Tran et al., "A Closer Look at Spatiotemporal Convolutions
    for Action Recognition" (CVPR 2018) — the R(2+1)D factorization.
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Mode A — SeparableConv3D as an nn.Module
# ---------------------------------------------------------------------------


class SeparableConv3D(nn.Module):
    """R(2+1)D separable 3D convolution: spatial 2D then temporal 1D.

    Input layout: ``(N, T, H, W, in_channels)`` (MLX NDHWC).
    Output layout: ``(N, T', H', W', out_channels)`` where ``T'``, ``H'``,
    ``W'`` depend on the kernel size, stride, and padding.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        kernel_size: ``(kT, kH, kW)`` triple.
        mid_channels: Rank of the factorization. Defaults to
            ``out_channels`` — equivalent to ``out``-dim bottleneck.
            Smaller values reduce parameters and compute at the cost of
            representational capacity.
        spatial_stride: Stride for the spatial 2D convolution.
        temporal_stride: Stride for the temporal 1D convolution.
        spatial_padding: Padding for the spatial 2D convolution.
        temporal_padding: Padding for the temporal 1D convolution.
        bias: Whether the temporal conv carries a bias. The spatial conv
            never adds a bias in this factorization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int] = (3, 3, 3),
        mid_channels: Optional[int] = None,
        spatial_stride: int | tuple[int, int] = 1,
        temporal_stride: int = 1,
        spatial_padding: int | tuple[int, int] = 0,
        temporal_padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        if len(kernel_size) != 3:
            raise ValueError("kernel_size must be (kT, kH, kW)")
        kT, kH, kW = kernel_size
        if mid_channels is None:
            mid_channels = out_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.kernel_size = (kT, kH, kW)

        self.spatial = nn.Conv2d(
            in_channels,
            mid_channels,
            kernel_size=(kH, kW),
            stride=spatial_stride,
            padding=spatial_padding,
            bias=False,
        )
        self.temporal = nn.Conv1d(
            mid_channels,
            out_channels,
            kernel_size=kT,
            stride=temporal_stride,
            padding=temporal_padding,
            bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Apply spatial-then-temporal convolution.

        Args:
            x: ``(N, T, H, W, in_channels)``.

        Returns:
            ``(N, T', H', W', out_channels)``.
        """
        if x.ndim != 5:
            raise ValueError(
                f"SeparableConv3D expects 5D input (N,T,H,W,C); got shape {x.shape}"
            )
        N, T, H, W, Cin = x.shape
        # Spatial: fold T into batch, apply 2D conv.
        x2 = x.reshape(N * T, H, W, Cin)
        x2 = self.spatial(x2)                 # (N*T, H', W', mid)
        _, Hp, Wp, M = x2.shape
        # Temporal: put T on the time axis, fold spatial into batch.
        x3 = x2.reshape(N, T, Hp, Wp, M)
        x3 = mx.transpose(x3, (0, 2, 3, 1, 4))  # (N, H', W', T, M)
        x3 = x3.reshape(N * Hp * Wp, T, M)
        x3 = self.temporal(x3)                # (N*H'*W', T', out)
        _, Tp, Cout = x3.shape
        x3 = x3.reshape(N, Hp, Wp, Tp, Cout)
        x3 = mx.transpose(x3, (0, 3, 1, 2, 4))  # (N, T', H', W', out)
        return x3


# ---------------------------------------------------------------------------
# Mode B — SVD decomposition of a pretrained Conv3d kernel
# ---------------------------------------------------------------------------


def decompose_conv3d_to_separable(
    conv3d_weight: mx.array,
    rank: Optional[int] = None,
) -> tuple[mx.array, mx.array, float]:
    """Factor a pretrained 3D conv kernel into (spatial 2D, temporal 1D).

    The input weight has shape ``(out, kT, kH, kW, in)`` (MLX Conv3d
    layout). It is reshaped to a 2D matrix ``A`` of shape
    ``(kT·out, kH·kW·in)`` and factored via SVD. The resulting
    factors correspond to:

      * ``spatial_weight`` with shape ``(mid, kH, kW, in)`` — suitable
        as the weight of an ``nn.Conv2d(in, mid, (kH, kW), bias=False)``.
      * ``temporal_weight`` with shape ``(out, kT, mid)`` — suitable as
        the weight of an ``nn.Conv1d(mid, out, kT, bias=True or False)``.

    At ``rank = min(kT·out, kH·kW·in)`` (full rank) the reconstruction
    is exact up to numerical precision (``error < 1e-4`` at float32).
    Smaller ranks give a lossy approximation; the returned
    ``reconstruction_error`` is the relative Frobenius norm
    ``||W_full - W_reconstructed|| / ||W_full||``.

    Args:
        conv3d_weight: ``(out, kT, kH, kW, in)`` tensor.
        rank: Middle-channel count M. ``None`` → full rank (exact
            factorization up to numerical precision).

    Returns:
        ``(spatial_weight, temporal_weight, reconstruction_error)``.

    Raises:
        ValueError: If ``conv3d_weight`` is not 5D.
    """
    if conv3d_weight.ndim != 5:
        raise ValueError(
            f"conv3d_weight must be 5D (out, kT, kH, kW, in); got {conv3d_weight.shape}"
        )

    out, kT, kH, kW, inp = conv3d_weight.shape
    W = conv3d_weight.astype(mx.float32)

    # Reshape: swap out to group with kT on the row side, keep spatial+in on cols.
    # W[o, t, h, w, i] → A[t*out + o, h*kW*inp + w*inp + i]
    # Use transpose+reshape.
    W_t = mx.transpose(W, (1, 0, 2, 3, 4))   # (kT, out, kH, kW, in)
    A = W_t.reshape(kT * out, kH * kW * inp)

    # SVD on CPU (MLX SVD is CPU-only on M1 Max at current version; stream kwarg avoids GPU dispatch)
    U, S, Vt = mx.linalg.svd(A, stream=mx.cpu)
    # U: (kT*out, m), S: (m,), Vt: (m, kH*kW*in) where m = min(kT*out, kH*kW*in)
    full_rank = S.shape[0]
    if rank is None:
        r = full_rank
    else:
        r = max(1, min(rank, full_rank))

    U_r = U[:, :r]
    S_r = S[:r]
    Vt_r = Vt[:r, :]

    # Distribute sigma equally between U and V so both factors are scaled reasonably.
    sqrt_S = mx.sqrt(S_r + 1e-12)
    U_scaled = U_r * sqrt_S.reshape(1, r)              # (kT*out, r)
    Vt_scaled = sqrt_S.reshape(r, 1) * Vt_r            # (r, kH*kW*in)

    # Reconstruct A_r for the error measurement.
    A_r = U_scaled @ Vt_scaled
    err = mx.linalg.norm(A - A_r) / (mx.linalg.norm(A) + 1e-12)

    # Temporal weight: (out, kT, r).
    # From U_scaled shape (kT*out, r): reshape to (kT, out, r), transpose to (out, kT, r).
    temporal_weight = U_scaled.reshape(kT, out, r).transpose(1, 0, 2)  # (out, kT, r)

    # Spatial weight: (r, kH, kW, in).
    spatial_weight = Vt_scaled.reshape(r, kH, kW, inp)

    return spatial_weight, temporal_weight, float(err.item())


def build_separable_from_decomposition(
    spatial_weight: mx.array,
    temporal_weight: mx.array,
    in_channels: int,
    out_channels: int,
    kernel_size: tuple[int, int, int],
    spatial_stride: int | tuple[int, int] = 1,
    temporal_stride: int = 1,
    spatial_padding: int | tuple[int, int] = 0,
    temporal_padding: int = 0,
) -> SeparableConv3D:
    """Assemble a :class:`SeparableConv3D` using decomposed weights.

    Convenience helper that bridges Mode B (decompose) and Mode A
    (module). Sets the module's ``spatial.weight`` and ``temporal.weight``
    to the supplied tensors. The module is returned with no bias on
    the temporal conv (bias cannot be derived from a biasless Conv3d).

    Args:
        spatial_weight: Output of :func:`decompose_conv3d_to_separable`,
            shape ``(mid, kH, kW, in)``.
        temporal_weight: Output of :func:`decompose_conv3d_to_separable`,
            shape ``(out, kT, mid)``.
        in_channels, out_channels, kernel_size: Must match the dimensions
            implied by the decomposed weights.
        spatial_stride, temporal_stride, spatial_padding, temporal_padding:
            Passed through to :class:`SeparableConv3D`.

    Returns:
        A :class:`SeparableConv3D` with the decomposed weights installed.
    """
    mid = spatial_weight.shape[0]
    mod = SeparableConv3D(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        mid_channels=mid,
        spatial_stride=spatial_stride,
        temporal_stride=temporal_stride,
        spatial_padding=spatial_padding,
        temporal_padding=temporal_padding,
        bias=False,
    )
    mod.spatial.weight = spatial_weight
    mod.temporal.weight = temporal_weight
    return mod
