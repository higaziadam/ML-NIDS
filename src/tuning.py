"""Training-partition-only randomized hyperparameter search for ML-NIDS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from src.models import create_model
from src.train import _validate_binary_label_contract, model_hyperparameters
from src.config import CONFIG


SEARCH_SPACE: dict[str, dict[str, list[Any]]] = {
    "random_forest": {
        "n_estimators": [100, 200, 300], "max_depth": [8, 12, 16, None],
        "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"], "class_weight": ["balanced", "balanced_subsample"],
    },
    "xgboost": {
        "n_estimators": [150, 250, 350], "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.02, 0.05, 0.1], "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9], "min_child_weight": [1.0, 3.0, 5.0],
        "reg_lambda": [0.5, 1.0, 2.0], "scale_pos_weight": [1.0, 1.1, 1.25],
    },
}


def _sample_parameters(model_type: str, base: Mapping[str, Any], iterations: int, random_state: int) -> list[dict[str, Any]]:
    if model_type not in SEARCH_SPACE:
        raise ValueError(f"Random search is currently supported for {sorted(SEARCH_SPACE)}.")
    rng = np.random.default_rng(random_state)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    while len(candidates) < iterations:
        candidate = dict(base)
        candidate.update({name: values[int(rng.integers(len(values)))] for name, values in SEARCH_SPACE[model_type].items()})
        key = json.dumps(candidate, sort_keys=True, default=str)
        if key not in seen:
            candidates.append(candidate)
            seen.add(key)
    return candidates


def randomized_search(
    features: pd.DataFrame, labels: pd.Series, model_type: str, base_parameters: Mapping[str, Any], iterations: int = 10, folds: int = 3, random_state: int = 42
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Score sampled candidates with fold-local scaling and average precision."""
    if iterations < 1 or folds < 2:
        raise ValueError("iterations must be positive and folds must be at least 2.")
    _validate_binary_label_contract(labels)
    numeric = features.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    y = labels.loc[numeric.index].to_numpy()
    if min(np.bincount(y)) < folds:
        raise ValueError("Each class needs at least one sample in every tuning fold.")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    rows = []
    for candidate_index, parameters in enumerate(_sample_parameters(model_type, base_parameters, iterations, random_state), start=1):
        scores = []
        for train_index, validation_index in splitter.split(numeric, y):
            scaler = MinMaxScaler().fit(numeric.iloc[train_index])
            model = create_model(model_type, **parameters)
            model.fit(scaler.transform(numeric.iloc[train_index]), y[train_index])
            probabilities = model.predict_proba(scaler.transform(numeric.iloc[validation_index]))
            estimator = getattr(model, "model", model)
            attack_index = int(np.flatnonzero(np.asarray(estimator.classes_) == 1)[0])
            scores.append(average_precision_score(y[validation_index], probabilities[:, attack_index]))
        rows.append({"candidate": candidate_index, "mean_average_precision": float(np.mean(scores)), "std_average_precision": float(np.std(scores)), "parameters": json.dumps(parameters, sort_keys=True)})
    report = pd.DataFrame(rows).sort_values("mean_average_precision", ascending=False, ignore_index=True)
    return json.loads(report.iloc[0]["parameters"]), report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run randomized search on development/training data only")
    parser.add_argument("--data", required=True, help="Labeled development or training CSV; never final holdout")
    parser.add_argument("--output", required=True, help="CSV ranking output")
    parser.add_argument("--model", default="xgboost", choices=sorted(SEARCH_SPACE))
    parser.add_argument("--label", default="label")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()
    data = pd.read_csv(args.data)
    if args.label not in data.columns:
        raise ValueError(f"Missing label column {args.label!r}")
    best, report = randomized_search(data.drop(columns=[args.label]), data[args.label], args.model, model_hyperparameters(CONFIG, args.model), args.iterations, args.folds)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)
    with output.with_suffix(".best.json").open("w", encoding="utf-8") as handle:
        json.dump({"model_type": args.model, "best_parameters": best, "warning": "Promote parameters to a new candidate profile, then validate without reusing final-holdout results."}, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
