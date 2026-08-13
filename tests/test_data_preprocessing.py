"""Regression tests for split validation."""

import pandas as pd
import pytest
import numpy as np

from src.data_preprocessing import DataSplitter
from src.train import train_model


def test_split_data_allows_no_validation_partition() -> None:
    X = pd.DataFrame({"feature": range(20)})
    y = pd.Series([0, 1] * 10)
    splits = DataSplitter.split_data(X, y, train_size=0.8, test_size=0.2, val_size=0)
    assert set(splits) == {"train", "test"}


def test_split_data_rejects_invalid_total() -> None:
    X = pd.DataFrame({"feature": range(20)})
    y = pd.Series([0, 1] * 10)
    with pytest.raises(ValueError, match="must equal 1.0"):
        DataSplitter.split_data(X, y, train_size=0.8, test_size=0.3, val_size=0)


def test_training_rejects_single_class_data() -> None:
    with pytest.raises(ValueError, match="only one class"):
        train_model(np.array([[0.0], [1.0]]), np.array([1, 1]))


def test_xgboost_model_trains_and_predicts() -> None:
    model = train_model(
        np.array([[0.0, 0.0], [0.1, 0.2], [0.9, 0.8], [1.0, 1.0]]),
        np.array([0, 0, 1, 1]),
        model_type="xgboost",
        n_estimators=5,
        max_depth=2,
        n_jobs=1,
    )
    assert model.predict(np.array([[0.0, 0.1]])).shape == (1,)
