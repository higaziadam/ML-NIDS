"""Regression tests for binary and multiclass evaluation."""

import numpy as np

from src.evaluate import ModelEvaluator, select_binary_threshold


def test_evaluator_accepts_string_binary_labels() -> None:
    evaluator = ModelEvaluator()
    results = evaluator.evaluate(
        np.array(["Benign", "Attack", "Attack", "Benign"]),
        np.array(["Benign", "Attack", "Benign", "Benign"]),
        np.array([[0.9, 0.1], [0.1, 0.9], [0.6, 0.4], [0.8, 0.2]]),
        positive_class="Attack",
        probability_classes=np.array(["Benign", "Attack"]),
    )
    assert results["labels"] == ["Attack", "Benign"]
    assert "f1" in results
    assert results["roc_auc"] == 1.0


def test_evaluator_uses_supplied_probability_class_order() -> None:
    results = ModelEvaluator().evaluate(
        np.array([0, 1, 1, 0]),
        np.array([0, 1, 0, 0]),
        np.array([[0.1, 0.9], [0.9, 0.1], [0.4, 0.6], [0.2, 0.8]]),
        positive_class=1,
        probability_classes=np.array([1, 0]),
    )
    assert results["roc_auc"] == 1.0


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


def test_threshold_selection_prefers_highest_compliant_threshold() -> None:
    threshold, results = select_binary_threshold(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.3, 0.7, 0.9]),
        positive_class=1,
        candidates=[0.4, 0.5, 0.6],
        min_recall=1.0,
        max_fpr=0.0,
    )
    assert threshold == 0.6
    assert results["meets_targets"].all()
