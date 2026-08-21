"""Tests for ML workflow governance tools."""

import json

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.data_preprocessing import DataPreprocessor
from src.ml_workflow import (
    build_drift_baseline,
    calibration_report,
    dataset_manifest,
    drift_report,
    evaluate_representative_data,
    permutation_explainability,
)
from src.utils import save_model


def _artifact(tmp_path):
    features = pd.DataFrame({"flow_bytes": [1.0, 2.0, 9.0, 10.0], "packet_count": [1.0, 2.0, 9.0, 10.0]})
    preprocessor = DataPreprocessor().fit(features)
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(
        preprocessor.transform(features).values,
        [0, 0, 1, 1],
    )
    path = tmp_path / "model.pkl"
    save_model({"model": model, "preprocessor": preprocessor, "feature_names": features.columns.tolist(), "threshold": 0.5, "model_version": "workflow_test", "drift_baseline": build_drift_baseline(features, features.columns)}, path)
    return path, features.assign(label=[0, 0, 1, 1])


def test_manifest_calibration_and_drift_reports(tmp_path) -> None:
    data_path = tmp_path / "data.csv"
    pd.DataFrame({"feature": [0.0, 0.1, 0.9, 1.0], "label": [0, 0, 1, 1]}).to_csv(data_path, index=False)
    manifest = dataset_manifest(data_path)
    calibration, table = calibration_report([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    baseline = build_drift_baseline(pd.DataFrame({"feature": [0.0, 0.1, 0.2, 0.3]}), ["feature"])
    report = drift_report(pd.DataFrame({"feature": [9.0, 9.1, 9.2, 9.3]}), baseline, psi_alert=0.01)

    assert manifest["rows"] == 4 and manifest["label_counts"] == {"0": 2, "1": 2}
    assert calibration["brier_score"] >= 0 and not table.empty
    assert report["summary"]["requires_review"] is True


def test_representative_evaluation_and_permutation_explainability(tmp_path) -> None:
    model_path, data = _artifact(tmp_path)
    source = tmp_path / "representative.csv"
    data.to_csv(source, index=False)
    report = evaluate_representative_data(model_path, source, tmp_path / "report")
    artifact = json.loads((tmp_path / "report" / "representative_validation.json").read_text(encoding="utf-8"))
    import pickle
    with model_path.open("rb") as handle:
        model_artifact = pickle.load(handle)
    importance = permutation_explainability(model_artifact, data, repeats=2)

    assert report["workflow"] == "representative_target_validation"
    assert artifact["dataset"]["rows"] == 4
    assert (tmp_path / "report" / "calibration.csv").is_file()
    assert set(importance["feature"]) == {"flow_bytes", "packet_count"}
