"""Loading and validation for immutable model release profiles."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REQUIRED_STATUS = {"candidate", "release_candidate", "released"}
REQUIRED_MODEL_FIELDS = {
    "type",
    "n_estimators",
    "max_depth",
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "min_child_weight",
    "reg_lambda",
    "scale_pos_weight",
    "random_state",
    "n_jobs",
    "tree_method",
    "objective",
    "eval_metric",
}


def sha256_file(path: str | Path) -> str:
    """Return an uppercase SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Release profile field '{field_name}' must be an object.")
    return value


def validate_release_profile(profile: Mapping[str, Any]) -> None:
    """Validate the minimum contract for a reproducible binary candidate or release."""
    for field in ("model_name", "status", "dataset", "split", "preprocessing", "model", "threshold_policy"):
        if field not in profile:
            raise ValueError(f"Release profile is missing required field '{field}'.")
    if profile["status"] not in REQUIRED_STATUS:
        allowed = ", ".join(sorted(REQUIRED_STATUS))
        raise ValueError(f"Release profile status must be one of: {allowed}.")

    dataset = _require_mapping(profile["dataset"], "dataset")
    label_mapping = _require_mapping(dataset.get("label_mapping"), "dataset.label_mapping")
    if label_mapping != {"0": "benign", "1": "attack"}:
        raise ValueError("Release profiles require label mapping {'0': 'benign', '1': 'attack'}.")

    split = _require_mapping(profile["split"], "split")
    split_values = [split.get("train_size"), split.get("validation_size"), split.get("test_size")]
    if any(not isinstance(value, (int, float)) for value in split_values) or abs(sum(split_values) - 1.0) > 1e-9:
        raise ValueError("Release profile train_size, validation_size, and test_size must sum to 1.0.")

    model = _require_mapping(profile["model"], "model")
    missing_model_fields = REQUIRED_MODEL_FIELDS - set(model)
    if missing_model_fields:
        raise ValueError(f"Release profile model is missing fields: {sorted(missing_model_fields)}.")
    if model["type"] != "xgboost":
        raise ValueError("This release-profile loader currently supports XGBoost release profiles only.")
    if model["objective"] != "binary:logistic" or model["tree_method"] != "hist":
        raise ValueError("Release profile must use binary:logistic with the hist tree method.")

    threshold_policy = _require_mapping(profile["threshold_policy"], "threshold_policy")
    candidates = threshold_policy.get("candidates")
    selected = threshold_policy.get("selected_threshold")
    if not isinstance(candidates, list) or not candidates or selected not in candidates:
        raise ValueError("Release profile selected threshold must be listed in threshold candidates.")
    if any(not isinstance(value, (int, float)) or not 0.0 < value < 1.0 for value in candidates):
        raise ValueError("Release profile threshold candidates must be values between 0 and 1.")
    if not 0.0 < threshold_policy.get("minimum_recall", 0.0) <= 1.0:
        raise ValueError("Release profile minimum_recall must be in (0, 1].")
    if not 0.0 <= threshold_policy.get("maximum_false_positive_rate", -1.0) <= 1.0:
        raise ValueError("Release profile maximum_false_positive_rate must be in [0, 1].")


def load_release_profile(path: str | Path) -> dict[str, Any]:
    """Load, validate, and return a release profile without mutating it."""
    profile_path = Path(path)
    if not profile_path.is_file():
        raise FileNotFoundError(f"Release profile not found: {profile_path}")
    try:
        with profile_path.open("r", encoding="utf-8") as handle:
            profile = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Release profile is not valid JSON: {profile_path}") from exc
    if not isinstance(profile, dict):
        raise ValueError("Release profile root must be a JSON object.")
    validate_release_profile(profile)
    return copy.deepcopy(profile)


def runtime_config_from_profile(
    profile: Mapping[str, Any], base_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Build runtime training settings from a frozen profile and base defaults."""
    validate_release_profile(profile)
    runtime_config = copy.deepcopy(dict(base_config))
    preprocessing = _require_mapping(profile["preprocessing"], "preprocessing")
    feature_selection = _require_mapping(preprocessing["feature_selection"], "preprocessing.feature_selection")
    normalization = _require_mapping(preprocessing["normalization"], "preprocessing.normalization")
    split = _require_mapping(profile["split"], "split")
    threshold_policy = _require_mapping(profile["threshold_policy"], "threshold_policy")

    runtime_config["data"].update(
        {
            "train_size": split["train_size"],
            "val_size": split["validation_size"],
            "test_size": split["test_size"],
            "random_state": split["random_state"],
            "missing_value_strategy": preprocessing["missing_value_strategy"],
            "detect_outliers": preprocessing["outlier_removal"],
            "normalization_method": normalization["method"],
        }
    )
    runtime_config["features"].update(
        {
            "n_features": feature_selection["n_features"],
            "correlation_threshold": feature_selection["correlation_threshold"],
        }
    )
    runtime_config["model"] = {
        "model_type": profile["model"]["type"],
        "hyperparameters": copy.deepcopy(dict(profile["model"])),
    }
    runtime_config["model"]["hyperparameters"].pop("type")
    runtime_config["threshold"] = {
        "candidates": list(threshold_policy["candidates"]),
        "min_recall": threshold_policy["minimum_recall"],
        "max_fpr": threshold_policy["maximum_false_positive_rate"],
        "default": threshold_policy.get("selected_threshold", 0.5),
    }
    return runtime_config
