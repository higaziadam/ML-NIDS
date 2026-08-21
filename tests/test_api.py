"""Tests for the ML-NIDS HTTP inference service."""

import pandas as pd
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

from src.api import create_app
from src.data_preprocessing import DataPreprocessor
from src.utils import save_model


def _write_artifact(tmp_path):
    features = pd.DataFrame({"flow_bytes": [1.0, 2.0, 9.0, 10.0], "packet_count": [1.0, 2.0, 9.0, 10.0]})
    preprocessor = DataPreprocessor(method="minmax").fit(features)
    classifier = RandomForestClassifier(n_estimators=5, random_state=42).fit(
        preprocessor.transform(features).values, [0, 0, 1, 1]
    )
    path = tmp_path / "api_model.pkl"
    save_model(
        {
            "model": classifier,
            "preprocessor": preprocessor,
            "feature_names": ["flow_bytes", "packet_count"],
            "threshold": 0.5,
            "model_version": "api_test",
        },
        path,
    )
    return path


def test_health_and_predict_use_the_loaded_artifact(tmp_path) -> None:
    with TestClient(create_app(_write_artifact(tmp_path))) as client:
        health = client.get("/health")
        schema = client.get("/schema")
        response = client.post(
            "/predict",
            json={"records": [{"flow_bytes": 1.5, "packet_count": 1.5}, {"flow_bytes": 9.5, "packet_count": 9.5}]},
        )

    assert health.status_code == 200
    assert health.json() == {"status": "ok", "model_version": "api_test", "feature_count": 2, "threshold": 0.5}
    assert schema.status_code == 200
    assert schema.json() == {"model_version": "api_test", "required_features": ["flow_bytes", "packet_count"]}
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "api_test"
    assert len(body["predictions"]) == 2
    assert all(0.0 <= item["probability"] <= 1.0 for item in body["predictions"])


def test_predict_rejects_a_record_missing_a_required_feature(tmp_path) -> None:
    with TestClient(create_app(_write_artifact(tmp_path))) as client:
        response = client.post("/predict", json={"records": [{"flow_bytes": 1.5}]})

    assert response.status_code == 422
    assert "missing required training columns" in response.json()["detail"]


def test_predict_can_require_an_api_key(tmp_path) -> None:
    with TestClient(create_app(_write_artifact(tmp_path), api_key="test-secret")) as client:
        unauthorized = client.post("/predict", json={"records": [{"flow_bytes": 1.5, "packet_count": 1.5}]})
        authorized = client.post(
            "/predict",
            headers={"X-API-Key": "test-secret"},
            json={"records": [{"flow_bytes": 1.5, "packet_count": 1.5}]},
        )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200


def test_predict_enforces_request_size_and_rate_limits(tmp_path) -> None:
    app = create_app(
        _write_artifact(tmp_path),
        max_request_bytes=1_000,
        rate_limit_requests=1,
        rate_limit_window_seconds=60,
    )
    with TestClient(app) as client:
        first = client.post("/predict", json={"records": [{"flow_bytes": 1.5, "packet_count": 1.5}]})
        limited = client.post("/predict", json={"records": [{"flow_bytes": 1.5, "packet_count": 1.5}]})

    assert first.status_code == 200
    assert first.headers["cache-control"] == "no-store"
    assert first.headers["x-content-type-options"] == "nosniff"
    assert limited.status_code == 429
    assert limited.headers["retry-after"]


def test_metrics_record_predictions_without_request_payloads(tmp_path) -> None:
    with TestClient(create_app(_write_artifact(tmp_path))) as client:
        response = client.post(
            "/predict",
            json={"records": [{"flow_bytes": 1.5, "packet_count": 1.5}, {"flow_bytes": 9.5, "packet_count": 9.5}]},
        )
        metrics = client.get("/metrics").text

    assert response.status_code == 200
    assert "ml_nids_api_requests_total" in metrics
    assert "ml_nids_api_request_duration_seconds" in metrics
    assert "ml_nids_flows_scored_total" in metrics
    assert "ml_nids_attack_predictions_total" in metrics
    assert "model_version=\"api_test\"" in metrics
    assert "flow_bytes" not in metrics and "packet_count" not in metrics
