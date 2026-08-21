"""Tests for deterministic release-profile quality gates."""

import copy

import pytest

from src.quality_gates import QualityGateError, evaluate_release_profile, run_quality_gate
from src.release_config import load_release_profile


def test_v10_profile_passes_tracked_quality_gate(tmp_path) -> None:
    output = tmp_path / "quality_gate.json"
    report = run_quality_gate("models/configs/xgb_v10_candidate.json", output)

    assert output.is_file()
    assert report["passed"] is True
    assert report["model_name"] == "xgb_v10_regularized_fine_threshold"


def test_quality_gate_rejects_policy_metric_regression() -> None:
    profile = copy.deepcopy(load_release_profile("models/configs/xgb_v10_candidate.json"))
    profile["test_metrics"]["recall"] = 0.80

    with pytest.raises(QualityGateError, match="test recall"):
        evaluate_release_profile(profile)
