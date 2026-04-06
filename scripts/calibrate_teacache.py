"""Calibrate TeaCache polynomial coefficients for a given model.

Offline mode: loads pre-saved features from a directory and computes
optimal polynomial rescaling coefficients + L1 threshold.

Usage:
    python scripts/calibrate_teacache.py \\
        --features-dir /path/to/saved_features/ \\
        --num-steps 50 \\
        --output coefficients_model.json

Feature directory format:
    Each file is step_NNN.npz with keys "modulated_input" (mx.array).
    Files are named step_000.npz, step_001.npz, etc.
"""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np


def calibrate_teacache_offline(
    features_dir: str | Path,
    poly_order: int = 4,
    target_skip_ratio: float = 0.4,
    model_name: str = "unknown",
) -> dict:
    """Calibrate TeaCache coefficients from pre-saved features.

    Args:
        features_dir: Directory containing step_NNN.npz files.
        poly_order: Order of the rescaling polynomial.
        target_skip_ratio: Target fraction of steps to skip.
        model_name: Name for the output metadata.

    Returns:
        Dict with "model", "rel_l1_thresh", "poly_coeffs", "notes".
    """
    features_dir = Path(features_dir)
    files = sorted(features_dir.glob("step_*.npz"))
    if len(files) < 2:
        raise ValueError(f"Need at least 2 feature files, found {len(files)}")

    # Load all modulated inputs
    inputs = []
    for f in files:
        data = mx.load(str(f))
        inputs.append(data["modulated_input"])

    # Compute pairwise relative L1 distances
    raw_distances = []
    for i in range(len(inputs) - 1):
        diff = mx.mean(mx.abs(inputs[i + 1] - inputs[i]))
        norm = mx.mean(mx.abs(inputs[i])) + 1e-6
        rel_l1 = (diff / norm).item()
        raw_distances.append(rel_l1)

    raw_np = np.array(raw_distances)

    # Compute cumulative distances (what TeaCache accumulates)
    cumulative = np.cumsum(raw_np)
    # Normalize to [0, 1] range for polynomial fitting
    x_norm = cumulative / (cumulative[-1] + 1e-8)

    # Fit polynomial: target is a linear CDF (uniform distribution)
    # This maps biased raw distances to calibrated uniform distances
    target = np.linspace(0, 1, len(x_norm))
    # Fit poly: calibrated = poly(raw)
    # We want poly(raw_distance) to produce a more uniform distribution
    coeffs_np = np.polyfit(raw_np, target, poly_order)
    # numpy polyfit returns highest degree first, we want [a0, a1, ..., an]
    poly_coeffs = list(reversed(coeffs_np.tolist()))

    # Compute calibrated distances using the polynomial
    calibrated = np.polyval(coeffs_np, raw_np)

    # Choose threshold: the value below which target_skip_ratio of steps are skipped
    sorted_cal = np.sort(calibrated)
    thresh_idx = int(len(sorted_cal) * target_skip_ratio)
    thresh_idx = min(thresh_idx, len(sorted_cal) - 1)
    rel_l1_thresh = float(sorted_cal[thresh_idx])

    return {
        "model": model_name,
        "rel_l1_thresh": round(rel_l1_thresh, 6),
        "poly_coeffs": [round(c, 6) for c in poly_coeffs],
        "notes": (
            f"Calibrated from {len(files)} steps, poly_order={poly_order}, "
            f"target_skip_ratio={target_skip_ratio}"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Calibrate TeaCache coefficients")
    parser.add_argument(
        "--features-dir", type=str, required=True,
        help="Directory with step_NNN.npz feature files",
    )
    parser.add_argument("--num-steps", type=int, default=50, help="Expected number of steps")
    parser.add_argument("--poly-order", type=int, default=4, help="Polynomial order")
    parser.add_argument("--target-skip", type=float, default=0.4, help="Target skip ratio")
    parser.add_argument("--model-name", type=str, default="unknown", help="Model name")
    parser.add_argument("--output", type=str, default="coefficients.json", help="Output JSON")
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    if not features_dir.exists():
        print(f"Error: features directory not found: {features_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Calibrating from {features_dir}...")
    result = calibrate_teacache_offline(
        features_dir,
        poly_order=args.poly_order,
        target_skip_ratio=args.target_skip,
        model_name=args.model_name,
    )

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Threshold: {result['rel_l1_thresh']}")
    print(f"Poly coeffs: {result['poly_coeffs']}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
