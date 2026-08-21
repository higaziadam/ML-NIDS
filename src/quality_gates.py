"""Deterministic release-quality gates for tracked ML-NIDS profiles.

The gate intentionally validates tracked release metadata rather than attempting
to retrain a model in CI. Large datasets and persisted model artifacts are kept
out of Git, while the policy, model contract, and recorded validation metrics
remain reviewable and enforceable on every change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from src.release_config import load_release_profile


class QualityGateError(ValueError):
    """Raised when a release profile fails a required quality policy."""


def _metric(metrics: Mapping[str, Any], *names: str) -> float:
    for name in names:
        value = metrics.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    raise QualityGateError(f"Missing numeric metric; expected one of {', '.join(names)}.")


def evaluate_release_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Validate release metadata and its documented alert-quality policy."""
    policy = profile["threshold_policy"]
    validation = profile.get("validation_selection")
    test_metrics = profile.get("test_metrics")
    if not isinstance(validation, Mapping):
        raise QualityGateError("Release profile must include validation_selection metrics.")
    if not isinstance(test_metrics, Mapping):
        raise QualityGateError("Release profile must include test_metrics.")

    selected_threshold = policy["selected_threshold"]
    if validation.get("policy_compliant") is not True:
        raise QualityGateError("validation_selection.policy_compliant must be true.")
    validation_recall = _metric(validation, "selected_threshold_recall", "recall")
    validation_fpr = _metric(
        validation, "selected_threshold_false_positive_rate", "false_positive_rate", "fpr"
    )
    test_recall = _metric(test_metrics, "recall")
    test_fpr = _metric(test_metrics, "false_positive_rate", "fpr")
    min_recall = float(policy["minimum_recall"])
    max_fpr = float(policy["maximum_false_positive_rate"])

    failures = []
    if selected_threshold not in policy["candidates"]:
        failures.append("selected threshold is not included in threshold candidates")
    if validation_recall < min_recall:
        failures.append(f"validation recall {validation_recall:.6f} is below {min_recall:.6f}")
    if validation_fpr > max_fpr:
        failures.append(f"validation FPR {validation_fpr:.6f} exceeds {max_fpr:.6f}")
    if test_recall < min_recall:
        failures.append(f"test recall {test_recall:.6f} is below {min_recall:.6f}")
    if test_fpr > max_fpr:
        failures.append(f"test FPR {test_fpr:.6f} exceeds {max_fpr:.6f}")

    artifacts = profile.get("artifacts")
    model_hash = artifacts.get("model_sha256") if isinstance(artifacts, Mapping) else None
    if (
        not isinstance(model_hash, str)
        or len(model_hash) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in model_hash)
    ):
        failures.append("release metadata does not contain a valid 64-character hexadecimal model SHA-256")
    if failures:
        raise QualityGateError("Release quality gate failed: " + "; ".join(failures))

    return {
        "gate": "release_profile_quality",
        "model_name": profile["model_name"],
        "status": profile["status"],
        "selected_threshold": selected_threshold,
        "minimum_recall": min_recall,
        "maximum_false_positive_rate": max_fpr,
        "validation_recall": validation_recall,
        "validation_false_positive_rate": validation_fpr,
        "test_recall": test_recall,
        "test_false_positive_rate": test_fpr,
        "passed": True,
        "note": (
            "This gate validates tracked release metadata and recorded metrics. "
            "It does not retrain a model or re-evaluate a large private dataset in CI."
        ),
    }


def run_quality_gate(profile_path: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    """Load one frozen profile, enforce gates, and optionally write a JSON report."""
    profile = load_release_profile(profile_path)
    report = evaluate_release_profile(profile)
    report["profile_path"] = str(profile_path)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Enforce tracked ML-NIDS release-quality gates")
    parser.add_argument("--profile", required=True, help="Frozen release-profile JSON file")
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()
    try:
        report = run_quality_gate(args.profile, args.output)
    except QualityGateError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
