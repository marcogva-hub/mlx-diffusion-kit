"""Tests for calibrate_teacache.py offline calibration."""

import json

import mlx.core as mx
import pytest

from scripts.calibrate_teacache import calibrate_teacache_offline


def test_calibrate_synthetic_features(tmp_path):
    """Calibration with synthetic features should produce valid coefficients."""
    # Create 10 step files with gradually changing features
    for i in range(10):
        feat = mx.ones((4, 8, 32)) * (1.0 + 0.1 * i)
        mx.savez(str(tmp_path / f"step_{i:03d}"), modulated_input=feat)

    result = calibrate_teacache_offline(
        tmp_path, poly_order=4, target_skip_ratio=0.4, model_name="test_model"
    )

    assert result["model"] == "test_model"
    assert result["rel_l1_thresh"] > 0
    assert isinstance(result["poly_coeffs"], list)
    assert len(result["poly_coeffs"]) == 5  # order 4 → 5 coefficients
    assert "notes" in result


def test_calibrate_json_parseable(tmp_path):
    """Output should be valid JSON."""
    for i in range(5):
        feat = mx.random.normal((2, 4, 16)) * (i + 1)
        mx.savez(str(tmp_path / f"step_{i:03d}"), modulated_input=feat)

    result = calibrate_teacache_offline(tmp_path, model_name="json_test")
    json_str = json.dumps(result)
    parsed = json.loads(json_str)
    assert parsed["model"] == "json_test"
    assert isinstance(parsed["poly_coeffs"], list)


def test_calibrate_too_few_files(tmp_path):
    """Should raise with fewer than 2 files."""
    feat = mx.ones((2, 4))
    mx.savez(str(tmp_path / "step_000"), modulated_input=feat)

    with pytest.raises(ValueError, match="at least 2"):
        calibrate_teacache_offline(tmp_path)
