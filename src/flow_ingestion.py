"""Score CICFlowMeter-style CSV exports through the ML-NIDS HTTP API.

This adapter is flow-level only. It does not capture packets or create flows;
use a compatible flow exporter first, then provide its CSV output here.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from src.external_validation import normalized_columns


ApiRequest = Callable[[str, str, dict | None, str | None, float], dict]


def request_api_json(
    method: str,
    url: str,
    payload: dict | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> dict:
    """Send a JSON request to the inference API without logging flow contents."""
    headers = {"Accept": "application/json"}
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["X-API-Key"] = api_key
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API request to {url} failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise ConnectionError(f"Could not reach ML-NIDS API at {url}: {exc.reason}") from exc


def _normalize_and_validate_chunk(
    chunk: pd.DataFrame,
    required_features: list[str],
    source_rows: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return finite model-ready records and explicit invalid-input results."""
    data = chunk.copy()
    data.columns = normalized_columns(data.columns)
    if data.columns.duplicated().any():
        duplicates = data.columns[data.columns.duplicated()].tolist()
        raise ValueError(f"Normalized input schema has duplicate columns: {duplicates}")
    missing = [feature for feature in required_features if feature not in data.columns]
    if missing:
        raise ValueError(
            "Input CSV is missing features required by the loaded model: " + ", ".join(missing)
        )

    features = data.loc[:, required_features].apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    valid = features.notna().all(axis=1)
    valid_features = features.loc[valid].copy()
    valid_features.insert(0, "source_row", source_rows[valid.to_numpy()])
    invalid_rows = source_rows[~valid.to_numpy()]
    invalid = pd.DataFrame(
        {
            "source_row": invalid_rows,
            "prediction": pd.array([pd.NA] * len(invalid_rows), dtype="Int64"),
            "probability": np.full(len(invalid_rows), np.nan),
            "is_alert": np.zeros(len(invalid_rows), dtype=bool),
            "score_status": ["invalid_input"] * len(invalid_rows),
        }
    )
    return valid_features, invalid


def score_cicflowmeter_csv(
    input_path: str | Path,
    output_path: str | Path,
    api_url: str = "http://localhost:8000",
    *,
    api_key: str | None = None,
    batch_size: int = 100,
    chunksize: int = 10_000,
    timeout: float = 30.0,
    overwrite: bool = False,
    request_json: ApiRequest = request_api_json,
) -> dict[str, object]:
    """Normalize, validate, score, and record CICFlowMeter CSV flow records.

    The resulting CSV contains one row per source record. Invalid input rows are
    retained with ``score_status=invalid_input`` and are never submitted to the
    model. Existing output paths are protected unless ``overwrite`` is set.
    """
    source = Path(input_path)
    destination = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(f"Input CSV not found: {source}")
    manifest_path = destination.with_suffix(".manifest.json")
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output: {destination}")
        destination.unlink()
        if manifest_path.exists():
            manifest_path.unlink()
    if batch_size < 1 or chunksize < 1:
        raise ValueError("batch_size and chunksize must be positive integers")
    destination.parent.mkdir(parents=True, exist_ok=True)

    base_url = api_url.rstrip("/")
    schema = request_json("GET", f"{base_url}/schema", None, api_key, timeout)
    required_features = schema.get("required_features")
    if not isinstance(required_features, list) or not all(isinstance(item, str) for item in required_features):
        raise RuntimeError("API /schema response did not contain required_features")
    if not required_features:
        raise RuntimeError("API /schema response contains no required features")
    health = request_json("GET", f"{base_url}/health", None, api_key, timeout)
    model_version = health.get("model_version")
    threshold = health.get("threshold")
    if not isinstance(model_version, str) or not model_version:
        raise RuntimeError("API /health response did not contain a model_version")
    if not isinstance(threshold, (int, float)):
        raise RuntimeError("API /health response did not contain a numeric threshold")
    threshold = float(threshold)

    rows_read = 0
    rows_scored = 0
    rows_invalid = 0
    alerts = 0
    wrote_header = False

    for chunk in pd.read_csv(source, chunksize=chunksize, low_memory=False):
        source_rows = np.arange(rows_read, rows_read + len(chunk), dtype=int)
        rows_read += len(chunk)
        valid, invalid = _normalize_and_validate_chunk(chunk, required_features, source_rows)
        result_frames = [invalid]

        for start in range(0, len(valid), batch_size):
            batch = valid.iloc[start : start + batch_size]
            records = batch.loc[:, required_features].to_dict(orient="records")
            response = request_json("POST", f"{base_url}/predict", {"records": records}, api_key, timeout)
            predictions = response.get("predictions")
            if not isinstance(predictions, list) or len(predictions) != len(batch):
                raise RuntimeError("API /predict response count did not match the submitted batch")
            response_threshold = response.get("threshold")
            if not isinstance(response_threshold, (int, float)):
                raise RuntimeError("API /predict response did not contain a numeric threshold")
            response_model_version = response.get("model_version")
            if response_model_version != model_version or float(response_threshold) != threshold:
                raise RuntimeError("API model version or threshold changed during ingestion; refusing mixed output")
            scores = pd.DataFrame(predictions)
            if not {"prediction", "probability"}.issubset(scores.columns):
                raise RuntimeError("API /predict response did not contain prediction and probability")
            scores.insert(0, "source_row", batch["source_row"].to_numpy())
            scores["prediction"] = scores["prediction"].astype(int)
            scores["probability"] = scores["probability"].astype(float)
            scores["is_alert"] = scores["prediction"].eq(1)
            scores["score_status"] = "scored"
            result_frames.append(scores.loc[:, ["source_row", "prediction", "probability", "is_alert", "score_status"]])
            rows_scored += len(scores)
            alerts += int(scores["is_alert"].sum())

        results = pd.concat(result_frames, ignore_index=True).sort_values("source_row")
        results.insert(1, "model_version", model_version)
        results.insert(2, "threshold", threshold)
        results.to_csv(destination, mode="a", index=False, header=not wrote_header)
        wrote_header = True
        rows_invalid += len(invalid)

    if not wrote_header:
        raise ValueError("Input CSV contained no flow records")
    manifest = {
        "workflow": "cicflowmeter_csv_to_api",
        "input_path": str(source),
        "output_path": str(destination),
        "api_url": base_url,
        "model_version": model_version,
        "threshold": threshold,
        "required_features": required_features,
        "rows_read": rows_read,
        "rows_scored": rows_scored,
        "rows_invalid_input": rows_invalid,
        "alerts": alerts,
        "warning": "Scores are flow-level predictions. This adapter does not capture packets or create network flows.",
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Score CICFlowMeter-style CSV records through ML-NIDS API")
    parser.add_argument("--input", required=True, help="CICFlowMeter-style input CSV")
    parser.add_argument("--output", required=True, help="Output CSV with scores and alert status")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--api-key", default=os.getenv("ML_NIDS_API_KEY"))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--chunksize", type=int, default=10_000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = score_cicflowmeter_csv(
        args.input,
        args.output,
        args.api_url,
        api_key=args.api_key,
        batch_size=args.batch_size,
        chunksize=args.chunksize,
        timeout=args.timeout,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
