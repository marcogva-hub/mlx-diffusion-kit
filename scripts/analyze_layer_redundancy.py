"""Analyze layer-wise redundancy to inform caching and pruning strategies.

Usage:
    python scripts/analyze_layer_redundancy.py --weights model_weights.npz --output results.json

Loads model weights, extracts UNet-like layers, computes redundancy scores
via MosaicDiff analysis, and recommends layers for DeepCache.
"""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx

from mlx_diffusion_kit.cache.layer_redundancy import analyze_layer_redundancy, select_cacheable_layers


def main():
    parser = argparse.ArgumentParser(description="Analyze UNet layer redundancy for DeepCache")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights (.npz)")
    parser.add_argument("--output", type=str, default="redundancy_scores.json", help="Output JSON path")
    parser.add_argument("--method", type=str, default="cosine", choices=["cosine", "l2"])
    parser.add_argument("--cache-ratio", type=float, default=0.5, help="Fraction of layers to cache")
    parser.add_argument("--layer-prefix", type=str, default="", help="Prefix to filter layer names")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Error: weights file not found: {weights_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading weights from {weights_path}...")
    raw_weights = mx.load(str(weights_path))

    # Filter to layers matching prefix
    layer_weights: dict[int, mx.array] = {}
    layer_names: dict[int, str] = {}
    idx = 0
    for name in sorted(raw_weights.keys()):
        if args.layer_prefix and not name.startswith(args.layer_prefix):
            continue
        layer_weights[idx] = raw_weights[name]
        layer_names[idx] = name
        idx += 1

    if not layer_weights:
        print("Error: no layers found matching prefix", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing {len(layer_weights)} layers with method={args.method}...")
    scores = analyze_layer_redundancy(layer_weights, method=args.method)

    print("\nRedundancy scores (higher = more redundant):")
    for idx in sorted(scores.keys()):
        name = layer_names.get(idx, f"layer_{idx}")
        print(f"  {name}: {scores[idx]:.4f}")

    selected = select_cacheable_layers(scores, ratio=args.cache_ratio)
    print(f"\nRecommended layers to cache (ratio={args.cache_ratio}):")
    for idx in selected:
        name = layer_names.get(idx, f"layer_{idx}")
        print(f"  {name} (score: {scores[idx]:.4f})")

    # Save results
    output = {
        "method": args.method,
        "cache_ratio": args.cache_ratio,
        "scores": {layer_names.get(k, f"layer_{k}"): v for k, v in scores.items()},
        "recommended_cache_layers": [layer_names.get(i, f"layer_{i}") for i in selected],
    }
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
