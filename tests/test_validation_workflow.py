"""Tests for final-holdout creation and nested cross-validation."""

import json

import numpy as np
import pandas as pd
import pytest

from src.validation import create_final_holdout, cross_validate_release_profile
from scripts.prepare_kaggle_data import KaggleDataProcessor


def _profile() -> dict:
    return {
        "model_name": "test_release",
        "status": "release_candidate",
        "dataset": {"label_mapping": {"0": "benign", "1": "attack"}},
        "split": {
            "train_size": 0.7,
            "validation_size": 0.15,
            "test_size": 0.15,
            "random_state": 42,
        },
        "preprocessing": {
            "remove_duplicates": True,
            "missing_value_strategy": "drop",
            "outlier_removal": False,
            "remove_constant_features": True,
            "feature_selection": {
                "method": "f_classif",
                "n_features": 2,
                "remove_correlated_features": True,
                "correlation_threshold": 0.95,
            },
            "normalization": {"method": "minmax", "fit_on_training_data_only": True},
        },
        "model": {
            "type": "xgboost",
            "n_estimators": 3,
            "max_depth": 2,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1.0,
            "reg_lambda": 1.0,
            "scale_pos_weight": 1.0,
            "random_state": 42,
            "n_jobs": 1,
            "tree_method": "hist",
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        },
        "threshold_policy": {
            "selected_threshold": 0.5,
            "candidates": [0.4, 0.5, 0.6],
            "minimum_recall": 0.5,
            "maximum_false_positive_rate": 1.0,
        },
    }


def _write_source(tmp_path) -> str:
    rng = np.random.default_rng(42)
    labels = np.array([0, 1] * 20)
    source = pd.DataFrame(
        {
            "feature_a": labels + rng.normal(0, 0.1, len(labels)),
            "feature_b": labels + rng.normal(0, 0.1, len(labels)),
            "label": labels,
        }
    )
    path = tmp_path / "source.csv"
    source.to_csv(path, index=False)
    return str(path)


def test_final_holdout_is_created_once(tmp_path) -> None:
    source_path = _write_source(tmp_path)
    output_dir = create_final_holdout(source_path, "final", output_root=tmp_path / "holdouts")

    assert (output_dir / "development.csv").is_file()
    assert (output_dir / "final_holdout.csv").is_file()
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["development_rows"] + manifest["final_holdout_rows"] == 40

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_final_holdout(source_path, "final", output_root=tmp_path / "holdouts")


def test_single_source_label_is_standardized_to_lowercase_name() -> None:
    source = pd.DataFrame({"Feature": [1, 2], "Label": ["Benign", "DDoS"]})
    standardized = KaggleDataProcessor.standardize_labels(None, source)

    assert "label" in standardized.columns
    assert "Label" not in standardized.columns
    assert standardized["label"].tolist() == [0, 1]


def test_nested_cross_validation_writes_summary_without_final_holdout(tmp_path) -> None:
    source_path = _write_source(tmp_path)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(_profile()), encoding="utf-8")

    output_dir = cross_validate_release_profile(
        source_path,
        profile_path,
        "cv",
        folds=2,
        output_root=tmp_path / "evaluation",
    )

    summary = pd.read_csv(output_dir / "summary.csv")
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert {"recall", "fpr"}.issubset(summary["metric"])
    assert metadata["workflow"] == "nested_stratified_cross_validation"
    assert metadata["final_holdout_accessed"] is False
