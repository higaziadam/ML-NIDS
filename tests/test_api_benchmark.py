"""Tests for API latency benchmark report generation."""

import pandas as pd
import pytest

from src.api_benchmark import benchmark_api


def test_benchmark_records_latency_and_throughput(tmp_path) -> None:
    source = tmp_path / "flows.csv"
    pd.DataFrame({"flow_a": [1.0, 2.0], "flow_b": [3.0, 4.0]}).to_csv(source, index=False)
    output = tmp_path / "benchmark.json"

    def fake_api(method, url, payload, api_key, timeout):
        if url.endswith("/schema"):
            return {"model_version": "benchmark_test", "required_features": ["flow_a", "flow_b"]}
        if url.endswith("/health"):
            return {"status": "ok", "model_version": "benchmark_test", "feature_count": 2, "threshold": 0.5}
        return {"predictions": [{"prediction": 0, "probability": 0.1} for _ in payload["records"]]}

    report = benchmark_api(
        source,
        output,
        batch_sizes=[1, 2],
        requests_per_size=2,
        request_json=fake_api,
    )

    assert output.is_file()
    assert report["model_version"] == "benchmark_test"
    assert [row["batch_size"] for row in report["results"]] == [1, 2]
    assert all(row["requests_completed"] == 2 for row in report["results"])
    assert all(row["p95_ms"] >= 0 for row in report["results"])


def test_benchmark_rejects_a_changing_api_model_version(tmp_path) -> None:
    source = tmp_path / "flows.csv"
    pd.DataFrame({"flow_a": [1.0], "flow_b": [2.0]}).to_csv(source, index=False)

    def changing_api(method, url, payload, api_key, timeout):
        if url.endswith("/schema"):
            return {"model_version": "first", "required_features": ["flow_a", "flow_b"]}
        return {"model_version": "second", "feature_count": 2, "threshold": 0.5}

    with pytest.raises(RuntimeError, match="changed"):
        benchmark_api(source, tmp_path / "benchmark.json", request_json=changing_api)


def test_benchmark_rejects_a_nonpositive_timeout(tmp_path) -> None:
    with pytest.raises(ValueError, match="timeout"):
        benchmark_api(tmp_path / "flows.csv", tmp_path / "benchmark.json", timeout=0)


def test_benchmark_rejects_non_http_api_urls(tmp_path) -> None:
    with pytest.raises(ValueError, match="http"):
        benchmark_api(tmp_path / "flows.csv", tmp_path / "benchmark.json", api_url="file:///tmp")
