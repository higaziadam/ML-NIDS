"""Final-holdout and nested cross-validation workflow for release profiles."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config import CONFIG, EVALUATION_DIR, FINAL_HOLDOUT_DIR
from src.data_preprocessing import DataPreprocessor
from src.evaluate import ModelEvaluator, select_binary_threshold
from src.predict import load_trained_model, make_predictions, preprocess_inference_data
from src.release_config import (
    load_release_profile,
    runtime_config_from_profile,
    sha256_file,
)
from src.train import (
    _binary_probabilities,
    engineer_features_for_holdouts,
    preprocess_data,
    preprocess_holdout_data,
    train_model,
)
from src.utils import load_data, logger, save_data


def _validate_binary_labels(labels: pd.Series | np.ndarray) -> None:
    observed = set(np.asarray(labels).tolist())
    if observed != {0, 1}:
        raise ValueError(
            "Release validation requires labels encoded exactly as 0 (benign) and 1 (attack); "
            f"found {sorted(observed)}."
        )


def _load_labeled_data(data_path: str | Path, label_column: str) -> tuple[pd.DataFrame, pd.Series]:
    data = load_data(data_path)
    if label_column not in data.columns:
        raise ValueError(f"Label column '{label_column}' not found in {data_path}.")
    labels = data[label_column]
    _validate_binary_labels(labels)
    return data.drop(columns=[label_column]), labels


def _require_empty_directory(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing validation output: {path}. "
            "Choose a new --name or explicitly pass --overwrite."
        )
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, content: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2, sort_keys=True)
        handle.write("\n")


def create_final_holdout(
    data_path: str | Path,
    name: str,
    holdout_size: float = 0.15,
    label_column: str = "label",
    random_state: int = 42,
    output_root: Path = FINAL_HOLDOUT_DIR,
    overwrite: bool = False,
) -> Path:
    """Create one immutable development/final-holdout split from labeled data."""
    if not 0.0 < holdout_size < 1.0:
        raise ValueError("holdout_size must be between 0 and 1.")
    X, y = _load_labeled_data(data_path, label_column)
    development_X, final_X, development_y, final_y = train_test_split(
        X,
        y,
        test_size=holdout_size,
        random_state=random_state,
        stratify=y,
    )
    output_dir = Path(output_root) / name
    _require_empty_directory(output_dir, overwrite)

    development = development_X.copy()
    development[label_column] = development_y.values
    final_holdout = final_X.copy()
    final_holdout[label_column] = final_y.values
    development_path = output_dir / "development.csv"
    final_path = output_dir / "final_holdout.csv"
    save_data(development, development_path)
    save_data(final_holdout, final_path)
    _write_json(
        output_dir / "manifest.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_data_path": str(data_path),
            "source_data_sha256": sha256_file(data_path),
            "label_column": label_column,
            "label_contract": {"0": "benign", "1": "attack"},
            "holdout_size": holdout_size,
            "random_state": random_state,
            "development_rows": len(development),
            "final_holdout_rows": len(final_holdout),
            "development_sha256": sha256_file(development_path),
            "final_holdout_sha256": sha256_file(final_path),
            "warning": (
                "Do not use final_holdout.csv for threshold selection, hyperparameter tuning, "
                "or model comparison. Evaluate a frozen candidate on it once."
            ),
        },
    )
    logger.info(f"Created immutable final holdout at {output_dir}")
    return output_dir


def _fit_fold(
    X_train_raw: pd.DataFrame,
    y_train_raw: pd.Series,
    X_val_raw: pd.DataFrame,
    y_val_raw: pd.Series,
    X_outer_test_raw: pd.DataFrame,
    y_outer_test_raw: pd.Series,
    runtime_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit one fold and score both selected and frozen operating thresholds."""
    X_train, y_train = preprocess_data(X_train_raw, y_train_raw, runtime_config)
    X_val, y_val = preprocess_holdout_data(
        X_val_raw, y_val_raw, X_train.columns.tolist(), X_train_raw, runtime_config
    )
    X_outer_test, y_outer_test = preprocess_holdout_data(
        X_outer_test_raw,
        y_outer_test_raw,
        X_train.columns.tolist(),
        X_train_raw,
        runtime_config,
    )
    X_train, holdouts = engineer_features_for_holdouts(
        X_train, [X_val, X_outer_test], y_train, runtime_config
    )
    X_val, X_outer_test = holdouts

    preprocessor = DataPreprocessor(runtime_config["data"]["normalization_method"])
    X_train_np = preprocessor.fit_transform(X_train).to_numpy()
    X_val_np = preprocessor.transform(X_val).to_numpy()
    X_outer_test_np = preprocessor.transform(X_outer_test).to_numpy()
    model = train_model(
        X_train_np,
        y_train.to_numpy(),
        model_type=runtime_config["model"]["model_type"],
        **runtime_config["model"]["hyperparameters"],
    )
    validation_probabilities, negative_class, positive_class = _binary_probabilities(model, X_val_np)
    threshold, threshold_table = select_binary_threshold(
        y_val.to_numpy(),
        validation_probabilities,
        positive_class,
        runtime_config["threshold"]["candidates"],
        runtime_config["threshold"]["min_recall"],
        runtime_config["threshold"]["max_fpr"],
    )
    outer_probabilities, _, _ = _binary_probabilities(model, X_outer_test_np)
    outer_probability_matrix = model.predict_proba(X_outer_test_np)

    def evaluate_outer_threshold(operating_threshold: float) -> dict[str, Any]:
        outer_predictions = np.where(
            outer_probabilities >= operating_threshold, positive_class, negative_class
        )
        metrics = ModelEvaluator().evaluate(
            y_outer_test.to_numpy(), outer_predictions, outer_probability_matrix
        )
        metrics.update(
            {
                "selected_threshold": operating_threshold,
                "outer_policy_compliant": bool(
                    metrics["recall"] >= runtime_config["threshold"]["min_recall"]
                    and metrics["fpr"] <= runtime_config["threshold"]["max_fpr"]
                ),
            }
        )
        return metrics

    selected_metrics = evaluate_outer_threshold(threshold)
    frozen_threshold = runtime_config["threshold"]["default"]
    fixed_metrics = evaluate_outer_threshold(frozen_threshold)
    selected_validation = threshold_table.loc[threshold_table["threshold"] == threshold].iloc[0]
    selected_metrics.update(
        {
            "validation_recall": float(selected_validation["recall"]),
            "validation_fpr": float(selected_validation["fpr"]),
            "validation_policy_compliant": bool(selected_validation["meets_targets"]),
        }
    )
    fixed_metrics["frozen_profile_threshold"] = frozen_threshold
    return selected_metrics, fixed_metrics


def _metric_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize outer-fold metrics as mean and sample standard deviation."""
    metric_columns = [
        "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "specificity", "fpr", "fnr"
    ]
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "mean": metrics[metric].mean(),
                "std": metrics[metric].std(ddof=1),
            }
            for metric in metric_columns
            if metric in metrics
        ]
    )


def cross_validate_release_profile(
    data_path: str | Path,
    profile_path: str | Path,
    name: str,
    folds: int = 5,
    label_column: str = "label",
    output_root: Path = EVALUATION_DIR,
    overwrite: bool = False,
) -> Path:
    """Run nested stratified CV without accessing a final holdout partition."""
    if folds < 2:
        raise ValueError("folds must be at least 2.")
    profile = load_release_profile(profile_path)
    runtime_config = runtime_config_from_profile(profile, CONFIG)
    X, y = _load_labeled_data(data_path, label_column)
    class_counts = y.value_counts()
    if class_counts.min() < folds:
        raise ValueError("Each class must contain at least as many samples as the number of folds.")

    output_dir = Path(output_root) / name
    _require_empty_directory(output_dir, overwrite)
    outer_cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=runtime_config["data"]["random_state"],
    )
    inner_validation_size = runtime_config["data"]["val_size"] / (
        runtime_config["data"]["train_size"] + runtime_config["data"]["val_size"]
    )
    selected_fold_rows = []
    fixed_fold_rows = []
    for fold_number, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_outer_train = X.iloc[outer_train_idx]
        y_outer_train = y.iloc[outer_train_idx]
        X_inner_train, X_inner_val, y_inner_train, y_inner_val = train_test_split(
            X_outer_train,
            y_outer_train,
            test_size=inner_validation_size,
            random_state=runtime_config["data"]["random_state"] + fold_number,
            stratify=y_outer_train,
        )
        selected_metrics, fixed_metrics = _fit_fold(
            X_inner_train,
            y_inner_train,
            X_inner_val,
            y_inner_val,
            X.iloc[outer_test_idx],
            y.iloc[outer_test_idx],
            runtime_config,
        )
        selected_metrics["fold"] = fold_number
        fixed_metrics["fold"] = fold_number
        selected_fold_rows.append(selected_metrics)
        fixed_fold_rows.append(fixed_metrics)
        logger.info(f"Completed nested CV fold {fold_number}/{folds}")

    selected_fold_metrics = pd.DataFrame(selected_fold_rows)
    fixed_fold_metrics = pd.DataFrame(fixed_fold_rows)
    save_data(selected_fold_metrics, output_dir / "fold_metrics.csv")
    save_data(_metric_summary(selected_fold_metrics), output_dir / "summary.csv")
    save_data(fixed_fold_metrics, output_dir / "fixed_threshold_fold_metrics.csv")
    save_data(_metric_summary(fixed_fold_metrics), output_dir / "fixed_threshold_summary.csv")
    _write_json(
        output_dir / "metadata.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "workflow": "nested_stratified_cross_validation",
            "source_data_path": str(data_path),
            "source_data_sha256": sha256_file(data_path),
            "release_profile_path": str(profile_path),
            "release_profile_sha256": sha256_file(profile_path),
            "folds": folds,
            "all_validation_thresholds_policy_compliant": bool(
                selected_fold_metrics["validation_policy_compliant"].all()
            ),
            "frozen_profile_threshold": runtime_config["threshold"]["default"],
            "all_fixed_threshold_outer_folds_policy_compliant": bool(
                fixed_fold_metrics["outer_policy_compliant"].all()
            ),
            "final_holdout_accessed": False,
        },
    )
    logger.info(f"Nested cross-validation results saved to {output_dir}")
    return output_dir


def evaluate_final_holdout(
    data_path: str | Path,
    model_path: str | Path,
    profile_path: str | Path,
    name: str,
    label_column: str = "label",
    output_root: Path = EVALUATION_DIR,
    overwrite: bool = False,
) -> Path:
    """Evaluate a frozen artifact once on a final holdout and save the result."""
    profile = load_release_profile(profile_path)
    artifact = load_trained_model(model_path)
    if not isinstance(artifact, dict):
        raise ValueError("Final evaluation requires a current artifact containing model and preprocessor.")
    if artifact.get("threshold") != profile["threshold_policy"]["selected_threshold"]:
        raise ValueError("Artifact threshold does not match the frozen release profile.")
    X, y = _load_labeled_data(data_path, label_column)
    processed = preprocess_inference_data(
        X, artifact.get("preprocessor"), artifact.get("feature_names", [])
    )
    predictions, _ = make_predictions(artifact, processed.to_numpy(), return_probabilities=True)
    probabilities = artifact["model"].predict_proba(processed.to_numpy())
    metrics = ModelEvaluator().evaluate(y.to_numpy(), predictions, probabilities)
    metrics["selected_threshold"] = artifact["threshold"]

    output_dir = Path(output_root) / name
    _require_empty_directory(output_dir, overwrite)
    save_data(pd.DataFrame([metrics]), output_dir / "metrics.csv")
    save_data(
        pd.DataFrame(metrics["confusion_matrix"], index=["actual_benign", "actual_attack"], columns=["predicted_benign", "predicted_attack"]),
        output_dir / "confusion_matrix.csv",
    )
    _write_json(
        output_dir / "metadata.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "workflow": "final_holdout_evaluation",
            "warning": "This output is a final-holdout evaluation. Do not use it for further tuning.",
            "data_path": str(data_path),
            "data_sha256": sha256_file(data_path),
            "artifact_path": str(model_path),
            "artifact_sha256": sha256_file(model_path),
            "release_profile_path": str(profile_path),
            "release_profile_sha256": sha256_file(profile_path),
        },
    )
    logger.info(f"Final-holdout evaluation saved to {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="ML-NIDS validation workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    holdout_parser = subparsers.add_parser("create-holdout", help="Create one development/final split")
    holdout_parser.add_argument("--data", required=True)
    holdout_parser.add_argument("--name", required=True)
    holdout_parser.add_argument("--holdout-size", type=float, default=0.15)
    holdout_parser.add_argument("--label", default="label")
    holdout_parser.add_argument("--random-state", type=int, default=42)
    holdout_parser.add_argument("--overwrite", action="store_true")

    cv_parser = subparsers.add_parser("cross-validate", help="Run nested CV on development data only")
    cv_parser.add_argument("--data", required=True)
    cv_parser.add_argument("--config", required=True)
    cv_parser.add_argument("--name", required=True)
    cv_parser.add_argument("--folds", type=int, default=5)
    cv_parser.add_argument("--label", default="label")
    cv_parser.add_argument("--overwrite", action="store_true")

    final_parser = subparsers.add_parser("final-evaluate", help="Evaluate a frozen artifact once")
    final_parser.add_argument("--data", required=True)
    final_parser.add_argument("--model", required=True)
    final_parser.add_argument("--config", required=True)
    final_parser.add_argument("--name", required=True)
    final_parser.add_argument("--label", default="label")
    final_parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    if args.command == "create-holdout":
        create_final_holdout(args.data, args.name, args.holdout_size, args.label, args.random_state, overwrite=args.overwrite)
    elif args.command == "cross-validate":
        cross_validate_release_profile(args.data, args.config, args.name, args.folds, args.label, overwrite=args.overwrite)
    else:
        evaluate_final_holdout(args.data, args.model, args.config, args.name, args.label, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
