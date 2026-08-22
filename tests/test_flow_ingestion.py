"""Tests for the CICFlowMeter CSV-to-API adapter."""

import pandas as pd
import pytest

from src.flow_ingestion import request_api_json, score_cicflowmeter_csv


def test_scores_normalized_cicflowmeter_records_and_retains_invalid_rows(tmp_path) -> None:
    source = tmp_path / "flows.csv"
    pd.DataFrame(
        {
            " Total Length of Fwd Packets": [10, 20, "Infinity"],
            " min_seg_size_forward": [1, 2, 3],
            "Label": ["BENIGN", "DDoS", "BENIGN"],
        }
    ).to_csv(source, index=False)
    output = tmp_path / "scores.csv"
    calls: list[tuple[str, str, dict | None]] = []

    def fake_api(method: str, url: str, payload: dict | None, api_key: str | None, timeout: float) -> dict:
        calls.append((method, url, payload))
        if url.endswith("/schema"):
            return {"model_version": "test_model", "required_features": ["Fwd Packets Length Total", "Fwd Seg Size Min"]}
        if url.endswith("/health"):
            return {"status": "ok", "model_version": "test_model", "feature_count": 2, "threshold": 0.26}
        return {
            "model_version": "test_model",
            "threshold": 0.26,
            "predictions": [
                {"index": index, "prediction": int(index == 1), "probability": 0.1 + index * 0.8}
                for index, _ in enumerate(payload["records"])
            ],
        }

    manifest = score_cicflowmeter_csv(source, output, batch_size=2, request_json=fake_api)
    scores = pd.read_csv(output)

    assert manifest["rows_read"] == 3
    assert manifest["rows_scored"] == 2
    assert manifest["rows_invalid_input"] == 1
    assert manifest["alerts"] == 1
    assert [call[0] for call in calls] == ["GET", "GET", "POST"]
    assert scores["source_row"].tolist() == [0, 1, 2]
    assert scores["model_version"].tolist() == ["test_model", "test_model", "test_model"]
    assert scores["threshold"].tolist() == [0.26, 0.26, 0.26]
    assert scores["score_status"].tolist() == ["scored", "scored", "invalid_input"]
    assert scores["is_alert"].tolist() == [False, True, False]

    score_cicflowmeter_csv(source, output, batch_size=2, overwrite=True, request_json=fake_api)
    assert len(pd.read_csv(output)) == 3


def test_refuses_to_replace_an_existing_score_output(tmp_path) -> None:
    source = tmp_path / "flows.csv"
    pd.DataFrame({"flow": [1]}).to_csv(source, index=False)
    output = tmp_path / "scores.csv"
    output.write_text("existing", encoding="utf-8")

    try:
        score_cicflowmeter_csv(source, output)
    except FileExistsError as exc:
        assert "Refusing to overwrite" in str(exc)
    else:
        raise AssertionError("Expected existing output to be protected")


def test_failed_scoring_preserves_existing_output_and_cleans_staging_files(tmp_path) -> None:
    source = tmp_path / "flows.csv"
    pd.DataFrame({"feature": [1.0, 2.0]}).to_csv(source, index=False)
    output = tmp_path / "scores.csv"
    output.write_text("previous complete result\n", encoding="utf-8")
    calls = 0

    def failing_api(method: str, url: str, payload: dict | None, api_key: str | None, timeout: float) -> dict:
        nonlocal calls
        if url.endswith("/schema"):
            return {"model_version": "test_model", "required_features": ["feature"]}
        if url.endswith("/health"):
            return {"status": "ok", "model_version": "test_model", "feature_count": 1, "threshold": 0.5}
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated API failure")
        return {"model_version": "test_model", "threshold": 0.5, "predictions": [{"prediction": 0, "probability": 0.1}]}

    try:
        score_cicflowmeter_csv(source, output, chunksize=1, overwrite=True, request_json=failing_api)
    except RuntimeError as exc:
        assert "simulated API failure" in str(exc)
    else:
        raise AssertionError("Expected scoring failure")

    assert output.read_text(encoding="utf-8") == "previous complete result\n"
    assert not list(tmp_path.glob(".scores.csv.*.partial"))


def test_rejects_non_http_api_urls_without_opening_them() -> None:
    with pytest.raises(ValueError, match="http"):
        request_api_json("GET", "file:///sensitive-data.csv")
