"""B8 — Token Merging (ToMe): Reduce token count via bipartite matching.

Applicable to ALL 11 VSR models. Primary intra-pass DiT lever for single-step
models. Uses cosine similarity bipartite matching to merge similar tokens,
with norm-preserving MLERP and static output shapes for MLX compile stability.

Reference: Bolya et al., "Token Merging: Your ViT But Faster" (ICLR 2023).
"""

import logging
from dataclasses import dataclass

import mlx.core as mx

logger = logging.getLogger(__name__)

_LCSA_MAX_MERGE_RATIO = 0.3


@dataclass
class ToMeConfig:
    merge_ratio: float = 0.5
    similarity: str = "cosine"
    use_mlerp: bool = True
    lcsa_compatible: bool = False
    enabled: bool = True


@dataclass
class MergeInfo:
    src_indices: mx.array
    dst_indices: mx.array
    merge_assignments: mx.array  # dst_idx -> matched src_idx (within src set)
    original_n: int
    batch_size: int


def _cosine_similarity(a: mx.array, b: mx.array) -> mx.array:
    """Pairwise cosine similarity between a [*, Na, D] and b [*, Nb, D].

    Returns [*, Na, Nb].
    """
    a_norm = a / (mx.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b_norm = b / (mx.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return a_norm @ mx.transpose(b_norm, (0, 2, 1)) if a.ndim == 3 else a_norm @ b_norm.T


def _l2_similarity(a: mx.array, b: mx.array) -> mx.array:
    """Negative L2 distance as similarity. Higher = more similar."""
    # a: [B, Na, D], b: [B, Nb, D]
    # diff: [B, Na, Nb, D]
    diff = mx.expand_dims(a, axis=-2) - mx.expand_dims(b, axis=-3)
    return -mx.sum(diff * diff, axis=-1)


def _mlerp(a: mx.array, b: mx.array, ratio: float = 0.5) -> mx.array:
    """Magnitude-preserving linear interpolation.

    Interpolates direction, then rescales to preserve geometric mean of norms.
    """
    merged = (1.0 - ratio) * a + ratio * b
    norm_a = mx.linalg.norm(a, axis=-1, keepdims=True) + 1e-8
    norm_b = mx.linalg.norm(b, axis=-1, keepdims=True) + 1e-8
    norm_merged = mx.linalg.norm(merged, axis=-1, keepdims=True) + 1e-8
    # Geometric mean of original norms
    target_norm = mx.sqrt(norm_a * norm_b)
    return merged * (target_norm / norm_merged)


def tome_merge(
    tokens: mx.array,
    config: ToMeConfig,
    spatial_dims: tuple[int, int, int] | None = None,
    spatial_weight: float = 0.3,
    temporal_weight: float = 0.5,
) -> tuple[mx.array, MergeInfo]:
    """Merge similar tokens via bipartite cosine matching.

    Args:
        tokens: Input tokens [B, N, D] or [B, H, N, D].
        config: ToMe configuration.
        spatial_dims: Optional (T, H, W) for video-aware spatiotemporal scoring.
            When provided, similarity combines cosine + spatial/temporal proximity.
        spatial_weight: Weight for spatial proximity (0-1). Only used with spatial_dims.
        temporal_weight: Weight for temporal proximity (0-1). Only used with spatial_dims.

    Returns:
        (merged_tokens, merge_info) where merged_tokens has N_merged = N - n_merge tokens.

    Note on LCSA interaction (FlashVSR):
        FlashVSR uses Local-Content Self-Attention (LCSA) with sparse patterns.
        Token merging changes the token count, which affects LCSA's local window
        boundaries. When using ToMe with FlashVSR:

        - Set ``lcsa_compatible=True`` in ToMeConfig to cap merge_ratio at 0.3
        - Provide ``spatial_dims`` for spatially-coherent merging
        - The LCSA mask (``mlx_mfa.make_lcsa_mask``) must be recomputed after
          merging with the new token count

        This interaction has NOT been validated end-to-end. Use with caution.
    """
    # LCSA compatibility guard
    effective_ratio = config.merge_ratio
    if config.lcsa_compatible:
        if effective_ratio > _LCSA_MAX_MERGE_RATIO:
            logger.warning(
                "ToMe lcsa_compatible=True: capping merge_ratio from %.2f to %.2f",
                effective_ratio,
                _LCSA_MAX_MERGE_RATIO,
            )
            effective_ratio = _LCSA_MAX_MERGE_RATIO
        if spatial_dims is None:
            logger.warning(
                "ToMe lcsa_compatible=True but spatial_dims not provided. "
                "Spatial-coherent merging recommended for LCSA models."
            )

    has_heads = tokens.ndim == 4
    if has_heads:
        B, H, N, D = tokens.shape
        # Merge on first head's tokens (shared matching across heads)
        tokens_flat = tokens.reshape(B * H, N, D)
    else:
        B, N, D = tokens.shape
        H = 1
        tokens_flat = tokens

    if not config.enabled or effective_ratio <= 0.0:
        # Passthrough — no merging
        info = MergeInfo(
            src_indices=mx.arange(N),
            dst_indices=mx.zeros(0, dtype=mx.int32),
            merge_assignments=mx.zeros(0, dtype=mx.int32),
            original_n=N,
            batch_size=B,
        )
        return tokens, info

    n_merge = int(N * effective_ratio)
    n_merge = min(n_merge, N // 2)  # Can't merge more than half

    if n_merge == 0:
        info = MergeInfo(
            src_indices=mx.arange(N),
            dst_indices=mx.zeros(0, dtype=mx.int32),
            merge_assignments=mx.zeros(0, dtype=mx.int32),
            original_n=N,
            batch_size=B,
        )
        return tokens, info

    # Bipartite partition: even indices = src, odd indices = dst
    n_src = N - n_merge
    n_dst = n_merge
    src_idx = mx.arange(0, N, 1)[:n_src]
    dst_idx = mx.arange(n_src, N, 1)[:n_dst]

    # Use first batch element (or BxH element) for matching
    # Matching is shared across the batch for static shapes
    if has_heads:
        match_tokens = tokens[:, 0, :, :]  # [B, N, D] — first head
    else:
        match_tokens = tokens_flat

    src_tokens_match = match_tokens[:1, :n_src, :]  # [1, n_src, D]
    dst_tokens_match = match_tokens[:1, n_src:n_src + n_dst, :]  # [1, n_dst, D]

    # Compute similarity
    if spatial_dims is not None:
        sim = compute_spatiotemporal_similarity(
            match_tokens[:1],
            spatial_dims,
            spatial_weight=spatial_weight,
            temporal_weight=temporal_weight,
            config=config,
        )
        # Extract dst×src submatrix from full [1, N, N]
        sim = sim[:, n_src:n_src + n_dst, :n_src]
    elif config.similarity == "cosine":
        sim = _cosine_similarity(dst_tokens_match, src_tokens_match)  # [1, n_dst, n_src]
    else:
        sim = _l2_similarity(dst_tokens_match, src_tokens_match)

    # Each dst matches to most similar src
    assignments = mx.argmax(sim[0], axis=-1)  # [n_dst]

    # Perform merge on all batch elements
    src_tok = tokens_flat[:, :n_src, :]  # [B*H, n_src, D]
    dst_tok = tokens_flat[:, n_src:n_src + n_dst, :]  # [B*H, n_dst, D]

    # Gather matched src tokens for each dst
    matched_src = src_tok[:, assignments, :]  # [B*H, n_dst, D]

    # Merge: MLERP or simple mean
    if config.use_mlerp:
        merged_dst = _mlerp(matched_src, dst_tok, ratio=0.5)
    else:
        merged_dst = (matched_src + dst_tok) / 2.0

    # Vectorized scatter-add: accumulate merged_dst into src positions
    # Count how many dst tokens map to each src
    dst_per_src = mx.zeros((n_src,))
    dst_per_src = dst_per_src.at[assignments].add(mx.ones((n_dst,)))

    # Weighted average: (src + sum_of_merged_dst) / (1 + count)
    total_count = 1.0 + dst_per_src  # [n_src]
    count_scale = mx.expand_dims(total_count, (0, -1))  # [1, n_src, 1]

    # Scatter-add merged_dst contributions into src positions per batch
    BH = tokens_flat.shape[0]
    contrib = mx.zeros_like(src_tok)  # [BH, n_src, D]
    for b in range(BH):
        contrib = contrib.at[b, assignments].add(merged_dst[b])

    output = (src_tok + contrib) / count_scale

    if has_heads:
        output = output.reshape(B, H, n_src, D)
    else:
        output = output.reshape(B, n_src, D)

    info = MergeInfo(
        src_indices=src_idx,
        dst_indices=dst_idx,
        merge_assignments=assignments,
        original_n=N,
        batch_size=B,
    )
    return output, info


def tome_unmerge(merged: mx.array, info: MergeInfo) -> mx.array:
    """Reconstruct full token sequence from merged tokens.

    Args:
        merged: Merged tokens [B, N_merged, D] or [B, H, N_merged, D].
        info: MergeInfo from tome_merge.

    Returns:
        Unmerged tokens [B, N_original, D] or [B, H, N_original, D].
    """
    has_heads = merged.ndim == 4

    if info.dst_indices.size == 0:
        return merged

    if has_heads:
        B, H, N_merged, D = merged.shape
        merged_flat = merged.reshape(B * H, N_merged, D)
    else:
        B, N_merged, D = merged.shape
        H = 1
        merged_flat = merged

    N = info.original_n
    BH = merged_flat.shape[0]

    # Reconstruct: src tokens stay, dst tokens copy from their matched src
    output = mx.zeros((BH, N, D), dtype=merged_flat.dtype)

    n_src = info.src_indices.shape[0]

    # Place src tokens
    output[:, :n_src, :] = merged_flat

    # Place dst tokens: each dst gets a copy of its matched src
    n_dst = info.dst_indices.shape[0]
    matched = merged_flat[:, info.merge_assignments, :]  # [BH, n_dst, D]
    output[:, n_src:n_src + n_dst, :] = matched

    if has_heads:
        output = output.reshape(B, H, N, D)
    else:
        output = output.reshape(B, N, D)

    return output


def compute_proportional_bias(info: MergeInfo) -> mx.array:
    """Compute token counts for proportional attention correction.

    Each merged src token represents itself + all dst tokens merged into it.
    Returns log-additive bias: log(count) for each src token.

    Args:
        info: MergeInfo from tome_merge.

    Returns:
        Bias vector [N_merged] with log(count) values.
    """
    n_src = info.src_indices.shape[0]

    if info.dst_indices.size == 0:
        return mx.zeros((info.original_n,))

    # Vectorized: count assignments without Python loops
    n_dst = info.merge_assignments.shape[0]
    counts = mx.ones((n_src,))
    counts = counts.at[info.merge_assignments].add(mx.ones((n_dst,)))

    return mx.log(counts)


def compute_spatiotemporal_similarity(
    tokens: mx.array,
    spatial_dims: tuple[int, int, int],
    spatial_weight: float = 0.3,
    temporal_weight: float = 0.5,
    config: ToMeConfig | None = None,
) -> mx.array:
    """Compute similarity combining cosine semantics + spatiotemporal proximity.

    For video models (9 of 11 VSR targets), tokens have structure [T, H, W].
    Pure cosine can merge spatially distant tokens; this adds proximity bias
    to encourage merging nearby tokens for spatially coherent regions.

    Final similarity = (1 - sw - tw) * cosine + sw * spatial_prox + tw * temporal_prox

    Args:
        tokens: Input tokens [B, N, D] where N = T*H*W.
        spatial_dims: (T, H, W) structure of the token grid.
        spatial_weight: Weight for 2D spatial proximity (0-1).
        temporal_weight: Weight for temporal proximity (0-1).
        config: Optional ToMeConfig (for similarity type fallback).

    Returns:
        Similarity matrix [B, N, N].
    """
    B, N, D = tokens.shape
    T, H, W = spatial_dims

    # Cosine similarity: [B, N, N]
    t_norm = tokens / (mx.linalg.norm(tokens, axis=-1, keepdims=True) + 1e-8)
    cosine_sim = t_norm @ mx.transpose(t_norm, (0, 2, 1))

    # Build coordinate grids for each token position
    # Token i corresponds to (t, h, w) = (i//(H*W), (i%(H*W))//W, i%W)
    indices = mx.arange(N)
    hw = H * W
    t_coords = (indices // hw).astype(mx.float32)  # [N]
    h_coords = ((indices % hw) // W).astype(mx.float32)
    w_coords = (indices % W).astype(mx.float32)

    # Spatial distance: Euclidean in (h, w) space
    h_diff = mx.expand_dims(h_coords, 0) - mx.expand_dims(h_coords, 1)  # [N, N]
    w_diff = mx.expand_dims(w_coords, 0) - mx.expand_dims(w_coords, 1)
    spatial_dist = mx.sqrt(h_diff * h_diff + w_diff * w_diff + 1e-8)
    spatial_prox = 1.0 / (1.0 + spatial_dist)  # [N, N]

    # Temporal distance: |t_i - t_j|
    t_diff = mx.abs(mx.expand_dims(t_coords, 0) - mx.expand_dims(t_coords, 1))
    temporal_prox = 1.0 / (1.0 + t_diff)  # [N, N]

    # Expand proximity matrices for batch dimension
    spatial_prox = mx.expand_dims(spatial_prox, 0)  # [1, N, N]
    temporal_prox = mx.expand_dims(temporal_prox, 0)

    cosine_weight = 1.0 - spatial_weight - temporal_weight
    sim = cosine_weight * cosine_sim + spatial_weight * spatial_prox + temporal_weight * temporal_prox

    return sim


def compute_attn_bias_for_mfa(info: MergeInfo) -> mx.array:
    """Build an attn_bias compatible with mlx-mfa for proportional attention.

    When tokens are merged, the resulting token represents >1 original token.
    Attention must weight merged tokens proportionally: tokens that represent
    more original tokens should receive proportionally more attention.

    The bias is log-additive: added to attention logits before softmax.
    bias[i] = log(count[i]) where count is how many original tokens token i represents.

    Output shape [1, 1, 1, N_merged] broadcasts with flash_attention's [B, H, N_q, N_kv].

    Usage with mlx-mfa::

        from mlx_mfa import flash_attention
        merged_tokens, info = tome_merge(tokens, config)
        q, k, v = compute_qkv(merged_tokens)
        bias = compute_attn_bias_for_mfa(info)
        attn_out = flash_attention(q, k, v, attn_bias=bias)

    Args:
        info: MergeInfo from tome_merge.

    Returns:
        attn_bias [1, 1, 1, N_merged] ready for flash_attention(attn_bias=...).
    """
    bias = compute_proportional_bias(info)  # [N_merged]
    return bias.reshape(1, 1, 1, -1)
