"""Tests for immutable release-profile handling and artifact protection."""

import copy
import json

import pytest

from src.config import CONFIG
from src.release_config import load_release_profile, runtime_config_from_profile
import src.train as train_module


def _v10_profile() -> dict:
    return {
        "model_name": "v10_test",
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
                "n_features": 20,
                "remove_correlated_features": True,
                "correlation_threshold": 0.95,
            },
            "normalization": {"method": "minmax", "fit_on_training_data_only": True},
        },
        "model": {
            "type": "xgboost",
            "n_estimators": 300,
            "max_depth": 10,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3.0,
            "reg_lambda": 1.0,
            "scale_pos_weight": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        },
        "threshold_policy": {
            "selected_threshold": 0.26,
            "candidates": [0.24, 0.25, 0.26, 0.27],
            "minimum_recall": 0.92,
            "maximum_false_positive_rate": 0.005,
        },
    }


def test_release_profile_overrides_active_experiment_config(tmp_path) -> None:
    profile_path = tmp_path / "v10.json"
    profile_path.write_text(json.dumps(_v10_profile()), encoding="utf-8")

    profile = load_release_profile(profile_path)
    runtime_config = runtime_config_from_profile(profile, CONFIG)

    assert runtime_config["model"]["model_type"] == "xgboost"
    assert runtime_config["model"]["hyperparameters"]["scale_pos_weight"] == 1.0
    assert runtime_config["threshold"]["candidates"] == [0.24, 0.25, 0.26, 0.27]
    assert CONFIG["model"]["hyperparameters"]["scale_pos_weight"] == 1.10


def test_release_profile_rejects_invalid_label_contract(tmp_path) -> None:
    invalid_profile = _v10_profile()
    invalid_profile["dataset"]["label_mapping"] = {"benign": 0, "attack": 1}
    profile_path = tmp_path / "invalid.json"
    profile_path.write_text(json.dumps(invalid_profile), encoding="utf-8")

    with pytest.raises(ValueError, match="label mapping"):
        load_release_profile(profile_path)


def test_candidate_status_is_accepted_for_development_profiles(tmp_path) -> None:
    candidate_profile = _v10_profile()
    candidate_profile["status"] = "candidate"
    profile_path = tmp_path / "candidate.json"
    profile_path.write_text(json.dumps(candidate_profile), encoding="utf-8")

    assert load_release_profile(profile_path)["status"] == "candidate"


def test_v13_feature_expansion_profile_is_valid() -> None:
    profile = load_release_profile("models/configs/xgb_v13_feature30_candidate.json")

    assert profile["status"] == "release_candidate"
    assert profile["preprocessing"]["feature_selection"]["n_features"] == 30
    assert profile["model"]["scale_pos_weight"] == 1.0


def test_frozen_profile_threshold_overrides_validation_selection() -> None:
    assert train_module.resolve_operating_threshold(0.30, 0.26) == 0.26
    assert train_module.resolve_operating_threshold(0.30) == 0.30


def test_v14_fixed_feature_schema_overrides_dynamic_selection() -> None:
    profile = load_release_profile("models/configs/xgb_v14_cicids2017_compatible_candidate.json")
    runtime = runtime_config_from_profile(profile, CONFIG)

    assert runtime["features"]["selected_features"] == profile["preprocessing"]["feature_selection"]["fixed_features"]
    assert len(runtime["features"]["selected_features"]) == 29


def test_artifact_paths_require_explicit_overwrite(tmp_path, monkeypatch) -> None:
    saved_dir = tmp_path / "saved"
    splits_dir = tmp_path / "splits"
    saved_dir.mkdir()
    splits_dir.mkdir()
    (saved_dir / "candidate.pkl").write_bytes(b"existing artifact")
    monkeypatch.setattr(train_module, "SAVED_MODELS_DIR", saved_dir)
    monkeypatch.setattr(train_module, "SPLITS_DIR", splits_dir)

    with pytest.raises(FileExistsError, match="--overwrite"):
        train_module.ensure_artifact_paths_available("candidate")

    train_module.ensure_artifact_paths_available("candidate", overwrite=True)
