"""Measure local ML-NIDS API latency and throughput using flow CSV records."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
import pandas as pd

from src.external_validation import normalized_columns
from src.flow_ingestion import ApiRequest, normalize_api_base_url, request_api_json


def _load_records(input_path: str | Path, required_features: list[str], max_records: int) -> list[dict[str, float]]:
    data = pd.read_csv(input_path, nrows=max_records)
    data.columns = normalized_columns(data.columns)
    if data.columns.duplicated().any():
        raise ValueError("Input CSV has duplicate columns after header normalization")
    missing = [feature for feature in required_features if feature not in data.columns]
    if missing:
        raise ValueError("Input CSV is missing required model features: " + ", ".join(missing))
    features = data.loc[:, required_features].apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    if features.empty:
        raise ValueError("Input CSV contained no finite rows compatible with the model schema")
    return features.to_dict(orient="records")


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def benchmark_api(
    input_path: str | Path,
    output_path: str | Path,
    api_url: str = "http://localhost:8000",
    *,
    api_key: str | None = None,
    batch_sizes: Iterable[int] = (1, 10, 100),
    requests_per_size: int = 10,
    concurrency: int = 1,
    max_records: int = 1_000,
    timeout: float = 30.0,
    overwrite: bool = False,
    request_json: ApiRequest = request_api_json,
) -> dict[str, object]:
    """Benchmark API requests without tuning or modifying the loaded model."""
    destination = Path(output_path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing benchmark report: {destination}")
    if requests_per_size < 1 or concurrency < 1 or max_records < 1 or timeout <= 0:
        raise ValueError("requests_per_size, concurrency, max_records, and timeout must be positive")
    sizes = [int(size) for size in batch_sizes]
    if not sizes or any(size < 1 for size in sizes):
        raise ValueError("batch_sizes must contain positive integers")

    base_url = normalize_api_base_url(api_url)
    schema = request_json("GET", f"{base_url}/schema", None, api_key, timeout)
    health = request_json("GET", f"{base_url}/health", None, api_key, timeout)
    schema_model_version = schema.get("model_version")
    health_model_version = health.get("model_version")
    if not isinstance(schema_model_version, str) or not schema_model_version:
        raise RuntimeError("API /schema response did not contain a model_version")
    if not isinstance(health_model_version, str) or not health_model_version:
        raise RuntimeError("API /health response did not contain a model_version")
    if schema_model_version != health_model_version:
        raise RuntimeError("API model version changed between /schema and /health requests")
    required_features = schema.get("required_features")
    if not isinstance(required_features, list) or not all(isinstance(item, str) for item in required_features):
        raise RuntimeError("API /schema response did not contain required_features")
    records = _load_records(input_path, required_features, max_records)
    results: list[dict[str, object]] = []

    for batch_size in sizes:
        payloads = [
            {"records": [records[(request_index * batch_size + item_index) % len(records)] for item_index in range(batch_size)]}
            for request_index in range(requests_per_size)
        ]

        def call(payload: dict) -> float:
            start = perf_counter()
            response = request_json("POST", f"{base_url}/predict", payload, api_key, timeout)
            if len(response.get("predictions", [])) != batch_size:
                raise RuntimeError("API /predict response count did not match benchmark batch size")
            return perf_counter() - start

        start_total = perf_counter()
        latencies: list[float] = []
        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(call, payload) for payload in payloads]
            for future in as_completed(futures):
                try:
                    latencies.append(future.result())
                except Exception as exc:
                    errors.append(str(exc))
        elapsed = perf_counter() - start_total
        if not latencies:
            raise RuntimeError(f"All benchmark requests failed for batch size {batch_size}: {errors}")
        completed_records = len(latencies) * batch_size
        results.append(
            {
                "batch_size": batch_size,
                "requests_attempted": requests_per_size,
                "requests_completed": len(latencies),
                "errors": len(errors),
                "p50_ms": round(_percentile(latencies, 50) * 1_000, 3),
                "p95_ms": round(_percentile(latencies, 95) * 1_000, 3),
                "p99_ms": round(_percentile(latencies, 99) * 1_000, 3),
                "mean_ms": round(float(np.mean(latencies)) * 1_000, 3),
                "throughput_flows_per_second": round(completed_records / elapsed, 3),
            }
        )

    report = {
        "workflow": "api_latency_benchmark",
        "api_url": base_url,
        "model_version": health_model_version,
        "threshold": health.get("threshold"),
        "feature_count": health.get("feature_count"),
        "input_path": str(input_path),
        "input_records_available": len(records),
        "requests_per_size": requests_per_size,
        "concurrency": concurrency,
        "results": results,
        "note": "Latency includes local HTTP, request validation, preprocessing, and model inference. It excludes upstream flow creation and network capture.",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ML-NIDS API latency using a compatible flow CSV")
    parser.add_argument("--input", required=True, help="Compatible CICFlowMeter-style CSV")
    parser.add_argument("--output", required=True, help="JSON benchmark report path")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--api-key", default=os.getenv("ML_NIDS_API_KEY"))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 10, 100])
    parser.add_argument("--requests-per-size", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-records", type=int, default=1_000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = benchmark_api(
        args.input,
        args.output,
        args.api_url,
        api_key=args.api_key,
        batch_sizes=args.batch_sizes,
        requests_per_size=args.requests_per_size,
        concurrency=args.concurrency,
        max_records=args.max_records,
        timeout=args.timeout,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
