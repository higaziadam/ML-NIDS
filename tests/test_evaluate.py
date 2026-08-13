"""Regression tests for binary and multiclass evaluation."""

import numpy as np

from src.evaluate import ModelEvaluator


def test_evaluator_accepts_string_binary_labels() -> None:
    evaluator = ModelEvaluator()
    results = evaluator.evaluate(
        np.array(["Benign", "Attack", "Attack", "Benign"]),
        np.array(["Benign", "Attack", "Benign", "Benign"]),
        np.array([[0.9, 0.1], [0.1, 0.9], [0.6, 0.4], [0.8, 0.2]]),
    )
    assert results["labels"] == ["Attack", "Benign"]
    assert "f1" in results


def test_evaluator_accepts_multiclass_labels() -> None:
    evaluator = ModelEvaluator()
    results = evaluator.evaluate(
        np.array([0, 1, 2, 0, 1, 2]),
        np.array([0, 1, 1, 0, 2, 2]),
        np.array([
            [0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.6, 0.3],
            [0.8, 0.1, 0.1], [0.2, 0.2, 0.6], [0.1, 0.1, 0.8],
        ]),
    )
    assert results["labels"] == [0, 1, 2]
    assert len(results["confusion_matrix"]) == 3
