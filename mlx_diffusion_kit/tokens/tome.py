"""B8 — Token Merging (ToMe): Reduce token count via bipartite matching.

Applicable to ALL 11 VSR models. Primary intra-pass DiT lever for single-step
models. Uses cosine similarity bipartite matching to merge similar tokens,
with norm-preserving MLERP and static output shapes for MLX compile stability.

Reference: Bolya et al., "Token Merging: Your ViT But Faster" (ICLR 2023).
"""

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class ToMeConfig:
    merge_ratio: float = 0.5
    similarity: str = "cosine"
    use_mlerp: bool = True
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
) -> tuple[mx.array, MergeInfo]:
    """Merge similar tokens via bipartite cosine matching.

    Args:
        tokens: Input tokens [B, N, D] or [B, H, N, D].
        config: ToMe configuration.

    Returns:
        (merged_tokens, merge_info) where merged_tokens has N_merged = N - n_merge tokens.
    """
    has_heads = tokens.ndim == 4
    if has_heads:
        B, H, N, D = tokens.shape
        # Merge on first head's tokens (shared matching across heads)
        tokens_flat = tokens.reshape(B * H, N, D)
    else:
        B, N, D = tokens.shape
        H = 1
        tokens_flat = tokens

    if not config.enabled or config.merge_ratio <= 0.0:
        # Passthrough — no merging
        info = MergeInfo(
            src_indices=mx.arange(N),
            dst_indices=mx.zeros(0, dtype=mx.int32),
            merge_assignments=mx.zeros(0, dtype=mx.int32),
            original_n=N,
            batch_size=B,
        )
        return tokens, info

    n_merge = int(N * config.merge_ratio)
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
    if config.similarity == "cosine":
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

    # Scatter merged values back into src positions
    # For each assignment, we update the src token with the merged value
    # Build output: start with src tokens, then update with merged values
    output = mx.array(src_tok)  # copy

    # Accumulate merges into src tokens
    # Use a loop-free approach: for each unique src, average all merges
    # Since multiple dst can map to same src, we need scatter-add
    BH = tokens_flat.shape[0]
    for b in range(BH):
        for d in range(n_dst):
            src_pos = assignments[d].item()
            # Running average: blend current src with merged value
            output[b, src_pos] = (output[b, src_pos] + merged_dst[b, d]) / 2.0

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

    # Count how many dst tokens map to each src
    counts = mx.ones((n_src,))
    for d in range(info.merge_assignments.shape[0]):
        src_pos = info.merge_assignments[d].item()
        counts[src_pos] = counts[src_pos] + 1.0

    return mx.log(counts)
