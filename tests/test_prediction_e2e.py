"""End-to-end tests for persisted-artifact batch prediction."""

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.data_preprocessing import DataPreprocessor
from src.predict import make_predictions, predict_pipeline
from src.utils import save_model


def test_prediction_pipeline_loads_artifact_and_writes_csv(tmp_path) -> None:
    """A fitted artifact can score a CSV and persist predictions end to end."""
    feature_names = ["flow_bytes", "packet_count"]
    training_features = pd.DataFrame(
        {"flow_bytes": [1.0, 2.0, 9.0, 10.0], "packet_count": [1.0, 2.0, 9.0, 10.0]}
    )
    labels = [0, 0, 1, 1]
    preprocessor = DataPreprocessor(method="minmax").fit(training_features)
    classifier = RandomForestClassifier(n_estimators=5, random_state=42).fit(
        preprocessor.transform(training_features).values, labels
    )
    artifact_path = tmp_path / "model.pkl"
    save_model(
        {
            "model": classifier,
            "preprocessor": preprocessor,
            "feature_names": feature_names,
            "threshold": 0.5,
        },
        artifact_path,
    )

    input_path = tmp_path / "flows.csv"
    pd.DataFrame(
        {
            "flow_bytes": [1.5, 9.5],
            "packet_count": [1.5, 9.5],
            "label": [0, 1],
            "non_model_metadata": ["a", "b"],
        }
    ).to_csv(input_path, index=False)
    output_path = tmp_path / "predictions.csv"

    results = predict_pipeline(artifact_path, input_path, output_path=output_path)
    saved = pd.read_csv(output_path)

    assert output_path.is_file()
    assert len(results) == 2
    assert saved.columns.tolist() == ["prediction", "probability"]
    assert saved["prediction"].isin([0, 1]).all()
    assert saved["probability"].between(0, 1).all()


def test_inference_rejects_artifact_without_zero_one_label_contract() -> None:
    classifier = RandomForestClassifier(n_estimators=5, random_state=42).fit(
        [[0.0], [1.0]], ["benign", "attack"]
    )
    with pytest.raises(ValueError, match=r"classes 0 \(benign\) and 1 \(attack\)"):
        make_predictions(classifier, [[0.5]])
