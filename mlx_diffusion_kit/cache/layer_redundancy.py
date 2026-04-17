"""MosaicDiff — Layer redundancy analysis for UNet/DiT caching strategies.

Utility that scores each layer of a model by how similar its weights are
to its neighbors. Layers with high redundancy are good candidates for
caching (DeepCache, DiTFastAttn residual layers) because skipping their
computation has minimal impact on the output.

This is a tooling module, not a runtime cache. Typical use:

    scores = analyze_layer_redundancy(layer_weights, method="cosine")
    cacheable = select_cacheable_layers(scores, ratio=0.5)

The returned layer indices can then be passed as explicit config to
components that accept a per-layer caching list.

Reference inspiration: the "redundancy-aware layer selection" theme in
diffusion-model acceleration literature (e.g., the layer-importance work
feeding into DiTFastAttn and DeepCache follow-ups). Not a direct paper
reimplementation.
"""

from typing import Literal

import mlx.core as mx


def analyze_layer_redundancy(
    layer_weights: dict[int, mx.array],
    method: Literal["cosine", "l2"] = "cosine",
) -> dict[int, float]:
    """Score each layer by similarity to its adjacent layers.

    Args:
        layer_weights: Mapping from layer index to weight tensor. Keys are
            the layer identifiers the caller will later pass to a cache
            configuration. Tensors may have different shapes; the
            comparison is done on flattened, truncated vectors.
        method: Similarity metric.
            ``"cosine"`` → cosine similarity of flattened weight vectors.
            ``"l2"`` → 1/(1+L2 distance). Higher score means more redundant.

    Returns:
        Mapping from layer index to redundancy score in [0, 1], normalized
        across all layers in the input. Higher score = more redundant =
        better caching candidate.

    Edge cases:
        - If fewer than 2 layers are provided, every score is 0.0.
        - If all raw scores are identical, every score is 1.0 (all redundant)
          when the mean is > 0.5, else 0.0.
    """
    if len(layer_weights) < 2:
        return {idx: 0.0 for idx in layer_weights}

    indices = sorted(layer_weights.keys())
    flat = {idx: layer_weights[idx].reshape(-1).astype(mx.float32) for idx in indices}

    raw_scores: dict[int, float] = {}
    for i, idx in enumerate(indices):
        sim_sum = 0.0
        count = 0
        for neighbor in (i - 1, i + 1):
            if 0 <= neighbor < len(indices):
                n_idx = indices[neighbor]
                a = flat[idx]
                b = flat[n_idx]
                # Truncate to shorter length so unequal shapes can be compared.
                min_len = min(a.shape[0], b.shape[0])
                a = a[:min_len]
                b = b[:min_len]
                if method == "cosine":
                    norm_a = mx.linalg.norm(a) + 1e-8
                    norm_b = mx.linalg.norm(b) + 1e-8
                    sim = (mx.sum(a * b) / (norm_a * norm_b)).item()
                elif method == "l2":
                    dist = mx.linalg.norm(a - b).item()
                    sim = 1.0 / (1.0 + dist)
                else:
                    raise ValueError(f"Unknown method: {method}")
                sim_sum += sim
                count += 1
        raw_scores[idx] = sim_sum / max(count, 1)

    # Normalize to [0, 1]
    min_s = min(raw_scores.values())
    max_s = max(raw_scores.values())
    rng = max_s - min_s
    if rng < 1e-10:
        avg = sum(raw_scores.values()) / len(raw_scores)
        return {k: 1.0 if avg > 0.5 else 0.0 for k in raw_scores}
    return {k: (v - min_s) / rng for k, v in raw_scores.items()}


def select_cacheable_layers(
    redundancy_scores: dict[int, float],
    ratio: float = 0.5,
) -> list[int]:
    """Select the top-``ratio`` fraction of layers by redundancy score.

    Args:
        redundancy_scores: Output of :func:`analyze_layer_redundancy`.
        ratio: Fraction of layers to select. Values outside [0, 1] are
            clamped; at least one layer is always returned.

    Returns:
        Sorted list of layer indices recommended for caching.
    """
    ratio = max(0.0, min(1.0, ratio))
    n_select = max(1, int(len(redundancy_scores) * ratio))
    sorted_by_score = sorted(redundancy_scores.items(), key=lambda x: -x[1])
    return sorted(idx for idx, _ in sorted_by_score[:n_select])
