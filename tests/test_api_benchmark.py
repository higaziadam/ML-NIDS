"""Tests for API latency benchmark report generation."""

import pandas as pd

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
