"""
Reproducibility, calibration, drift, explainability, and external-evaluation tools.
These commands deliberately keep an independent labeled dataset separate from
training: it can validate a frozen artifact, but never tune its threshold or
hyperparameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss

from src.evaluate import ModelEvaluator
from src.predict import load_trained_model, make_predictions, preprocess_inference_data
from src.release_config import sha256_file


def _write_json(payload: Mapping[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def dataset_manifest(data_path: str | Path, label_column: str = "label") -> dict[str, Any]:
    """Return a content and schema fingerprint for a labeled or unlabeled CSV."""
    path = Path(data_path)
    data = pd.read_csv(path)
    schema_text = json.dumps([(name, str(dtype)) for name, dtype in data.dtypes.items()])
    manifest: dict[str, Any] = {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": len(data),
        "columns": data.columns.tolist(),
        "schema_sha256": hashlib.sha256(schema_text.encode("utf-8")).hexdigest(),
        "label_column": label_column if label_column in data.columns else None,
    }
    if label_column in data.columns:
        manifest["label_counts"] = {
            str(label): int(count)
            for label, count in data[label_column].value_counts(dropna=False).sort_index().items()
        }
    return manifest


def calibration_report(
    labels: Iterable[int], probabilities: Iterable[float], bins: int = 10
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Calculate reliability-curve data and Brier score for binary predictions."""
    y_true = np.asarray(labels)
    y_probability = np.asarray(probabilities, dtype=float)
    if bins < 2:
        raise ValueError("bins must be at least 2")
    if len(y_true) != len(y_probability) or set(np.unique(y_true).tolist()) != {0, 1}:
        raise ValueError("Calibration analysis requires aligned binary labels encoded as 0 and 1.")
    observed, predicted = calibration_curve(y_true, y_probability, n_bins=bins, strategy="quantile")
    table = pd.DataFrame({"mean_predicted_probability": predicted, "observed_attack_rate": observed})
    report = {
        "brier_score": float(brier_score_loss(y_true, y_probability)),
        "average_precision": float(average_precision_score(y_true, y_probability)),
        "bins_requested": bins,
        "bins_nonempty": len(table),
    }
    return report, table


def build_drift_baseline(data: pd.DataFrame, features: Iterable[str], bins: int = 10) -> dict[str, Any]:
    """Build a numeric feature baseline using fixed quantile-bin proportions."""
    if bins < 2:
        raise ValueError("bins must be at least 2")
    baseline: dict[str, Any] = {"workflow": "feature_drift_baseline", "bins": bins, "features": {}}
    for feature in features:
        if feature not in data.columns:
            raise ValueError(f"Missing feature for drift baseline: {feature}")
        values = pd.to_numeric(data[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            raise ValueError(f"Feature {feature!r} has no finite values for drift baseline")
        edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
        if len(edges) < 2:
            edges = np.array([float(values.iloc[0]) - 0.5, float(values.iloc[0]) + 0.5])
        counts, edges = np.histogram(values, bins=edges)
        proportions = (counts / counts.sum()).tolist()
        baseline["features"][feature] = {
            "edges": [float(edge) for edge in edges],
            "proportions": [float(value) for value in proportions],
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "finite_training_rows": int(len(values)),
        }
    return baseline


def drift_report(data: pd.DataFrame, baseline: Mapping[str, Any], psi_alert: float = 0.2) -> dict[str, Any]:
    """Compare new feature data with a saved baseline using population stability index."""
    if psi_alert <= 0:
        raise ValueError("psi_alert must be positive")
    results: dict[str, Any] = {"workflow": "feature_drift_report", "psi_alert": psi_alert, "features": {}}
    for feature, reference in baseline.get("features", {}).items():
        if feature not in data.columns:
            results["features"][feature] = {"status": "missing"}
            continue
        values = pd.to_numeric(data[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        edges = np.asarray(reference["edges"], dtype=float)
        expected = np.asarray(reference["proportions"], dtype=float)
        counts, _ = np.histogram(values, bins=edges)
        actual = counts / counts.sum() if counts.sum() else np.zeros_like(expected)
        epsilon = 1e-6
        psi = float(np.sum((actual - expected) * np.log((actual + epsilon) / (expected + epsilon))))
        results["features"][feature] = {
            "status": "ok",
            "psi": psi,
            "drift_detected": psi >= psi_alert,
            "finite_rows": int(len(values)),
        }
    detected = [name for name, result in results["features"].items() if result.get("drift_detected")]
    missing = [name for name, result in results["features"].items() if result["status"] == "missing"]
    results["summary"] = {"drifted_features": detected, "missing_features": missing, "requires_review": bool(detected or missing)}
    return results


def permutation_explainability(
    artifact: Mapping[str, Any], data: pd.DataFrame, label_column: str = "label", repeats: int = 5, random_state: int = 42
) -> pd.DataFrame:
    """Measure global feature importance as AP loss after feature permutation."""
    if repeats < 1 or label_column not in data.columns:
        raise ValueError("Explainability requires a labeled dataset and repeats >= 1.")
    labels = data[label_column].to_numpy()
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("Explainability labels must use the project 0/1 contract.")
    features = preprocess_inference_data(data.drop(columns=[label_column]), artifact["preprocessor"], artifact["feature_names"])
    model = artifact["model"]
    _, baseline_probability = make_predictions(artifact, features.values, return_probabilities=True, threshold=float(artifact["threshold"]))
    baseline_score = average_precision_score(labels, baseline_probability)
    rng = np.random.default_rng(random_state)
    rows = []
    for index, name in enumerate(features.columns):
        losses = []
        for _ in range(repeats):
            permuted = features.to_numpy(copy=True)
            permuted[:, index] = rng.permutation(permuted[:, index])
            probabilities = model.predict_proba(permuted)
            estimator = getattr(model, "model", model)
            classes = np.asarray(estimator.classes_)
            attack_probability = probabilities[:, int(np.flatnonzero(classes == 1)[0])]
            losses.append(float(baseline_score - average_precision_score(labels, attack_probability)))
        rows.append({"feature": name, "importance_mean": float(np.mean(losses)), "importance_std": float(np.std(losses)), "baseline_average_precision": float(baseline_score)})
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False, ignore_index=True)


def evaluate_representative_data(
    model_path: str | Path, data_path: str | Path, output_dir: str | Path, label_column: str = "label", psi_alert: float = 0.2
) -> dict[str, Any]:
    """Evaluate a frozen artifact once on compatible independent labeled traffic."""
    artifact = load_trained_model(model_path)
    if not isinstance(artifact, dict) or not {"model", "preprocessor", "feature_names", "threshold"}.issubset(artifact):
        raise ValueError("Representative evaluation requires a current ML-NIDS artifact.")
    data = pd.read_csv(data_path)
    if label_column not in data.columns:
        raise ValueError(f"Representative dataset is missing label column {label_column!r}.")
    labels = data[label_column].to_numpy()
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("Representative data labels must use 0=benign and 1=attack.")
    processed = preprocess_inference_data(data.drop(columns=[label_column]), artifact["preprocessor"], artifact["feature_names"])
    predictions, probabilities = make_predictions(artifact, processed.values, return_probabilities=True, threshold=float(artifact["threshold"]))
    metrics = ModelEvaluator().evaluate(labels, predictions, probabilities, positive_class=1, probability_classes=np.array([0, 1]))
    calibration, calibration_table = calibration_report(labels, probabilities)
    report: dict[str, Any] = {
        "workflow": "representative_target_validation",
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "model_version": artifact.get("model_version", Path(model_path).name),
        "dataset": dataset_manifest(data_path, label_column),
        "metrics": metrics,
        "calibration": calibration,
        "warning": "This report validates a frozen artifact only. Do not tune this artifact using this dataset.",
    }
    if artifact.get("drift_baseline"):
        report["drift"] = drift_report(data, artifact["drift_baseline"], psi_alert)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    calibration_table.to_csv(destination / "calibration.csv", index=False)
    _write_json(report, destination / "representative_validation.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="ML-NIDS governance and validation workflows")
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("dataset-manifest")
    manifest.add_argument("--data", required=True)
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--label", default="label")
    representative = commands.add_parser("representative-evaluate")
    representative.add_argument("--model", required=True)
    representative.add_argument("--data", required=True)
    representative.add_argument("--output", required=True)
    representative.add_argument("--label", default="label")
    representative.add_argument("--psi-alert", type=float, default=0.2)
    explain = commands.add_parser("explain", help="Write permutation importance for labeled evaluation data")
    explain.add_argument("--model", required=True)
    explain.add_argument("--data", required=True)
    explain.add_argument("--output", required=True)
    explain.add_argument("--label", default="label")
    explain.add_argument("--repeats", type=int, default=5)
    drift = commands.add_parser("drift-report", help="Compare compatible new flows with an artifact baseline")
    drift.add_argument("--model", required=True)
    drift.add_argument("--data", required=True)
    drift.add_argument("--output", required=True)
    drift.add_argument("--psi-alert", type=float, default=0.2)
    args = parser.parse_args()
    if args.command == "dataset-manifest":
        _write_json(dataset_manifest(args.data, args.label), args.output)
    elif args.command == "representative-evaluate":
        evaluate_representative_data(args.model, args.data, args.output, args.label, args.psi_alert)
    elif args.command == "explain":
        artifact = load_trained_model(args.model)
        if not isinstance(artifact, dict):
            raise ValueError("Explainability requires a current ML-NIDS artifact.")
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        permutation_explainability(artifact, pd.read_csv(args.data), args.label, args.repeats).to_csv(output, index=False)
    else:
        artifact = load_trained_model(args.model)
        if not isinstance(artifact, dict) or not artifact.get("drift_baseline"):
            raise ValueError("Drift monitoring requires an artifact with a saved drift_baseline.")
        _write_json(drift_report(pd.read_csv(args.data), artifact["drift_baseline"], args.psi_alert), args.output)


if __name__ == "__main__":
    main()
