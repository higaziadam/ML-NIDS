"""Tests for training-only randomized hyperparameter search."""

import pandas as pd

from src.tuning import randomized_search


def test_randomized_search_returns_ranked_random_forest_candidate() -> None:
    labels = pd.Series([0, 1] * 8)
    features = pd.DataFrame({"feature_a": labels + 0.1, "feature_b": labels * 2 + 1})
    best, report = randomized_search(
        features,
        labels,
        "random_forest",
        {"random_state": 42, "n_jobs": 1},
        iterations=2,
        folds=2,
    )

    assert len(report) == 2
    assert "n_estimators" in best
