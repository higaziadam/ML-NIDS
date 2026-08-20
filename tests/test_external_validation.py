"""Tests for independent-external-data schema preflight checks."""

import json

import pandas as pd

from src.external_validation import inspect_external_csv, prepare_cicids2017_external_data


def test_cicids2017_aliases_are_normalized_but_protocol_remains_required(tmp_path) -> None:
    source = tmp_path / "cicids2017.csv"
    pd.DataFrame(
        {
            " Total Length of Fwd Packets": [10],
            " min_seg_size_forward": [1],
            " Init_Win_bytes_forward": [2],
            " Init_Win_bytes_backward": [3],
            " Max Packet Length": [4],
            " Min Packet Length": [5],
            " Label": ["BENIGN"],
        }
    ).to_csv(source, index=False)

    report = inspect_external_csv(
        source,
        [
            "Fwd Packets Length Total",
            "Fwd Seg Size Min",
            "Init Fwd Win Bytes",
            "Init Bwd Win Bytes",
            "Packet Length Max",
            "Packet Length Min",
            "Protocol",
        ],
    )

    assert report["label_column_present"] is True
    assert report["missing_required_features"] == "Protocol"
    assert report["schema_compatible"] is False


def test_prepare_cicids2017_uses_fixed_feature_schema(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    pd.DataFrame(
        {
            " Total Length of Fwd Packets": [1, "Infinity"],
            " Label": ["BENIGN", "DDoS"],
        }
    ).to_csv(source_dir / "flows.csv", index=False)
    profile = {
        "model_name": "external_test",
        "status": "candidate",
        "dataset": {"label_mapping": {"0": "benign", "1": "attack"}},
        "split": {"train_size": 0.7, "validation_size": 0.15, "test_size": 0.15, "random_state": 42},
        "preprocessing": {
            "missing_value_strategy": "drop", "outlier_removal": False,
            "feature_selection": {"n_features": 1, "correlation_threshold": 0.95, "fixed_features": ["Fwd Packets Length Total"]},
            "normalization": {"method": "minmax"},
        },
        "model": {"type": "xgboost", "n_estimators": 3, "max_depth": 2, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1.0, "reg_lambda": 1.0, "scale_pos_weight": 1.0, "random_state": 42, "n_jobs": 1, "tree_method": "hist", "objective": "binary:logistic", "eval_metric": "logloss"},
        "threshold_policy": {"selected_threshold": 0.5, "candidates": [0.5], "minimum_recall": 0.5, "maximum_false_positive_rate": 1.0},
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    output_path = tmp_path / "prepared.csv"

    manifest = prepare_cicids2017_external_data(source_dir, profile_path, output_path, chunksize=1)
    prepared = pd.read_csv(output_path)

    assert manifest["rows_written"] == 1
    assert manifest["rows_dropped_nonfinite_or_unlabeled"] == 1
    assert prepared.columns.tolist() == ["Fwd Packets Length Total", "label"]
    assert prepared["label"].tolist() == [0]
