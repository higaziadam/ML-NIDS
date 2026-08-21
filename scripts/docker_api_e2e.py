"""Run an end-to-end Docker Compose smoke test for the inference API.

The test creates a tiny synthetic artifact, starts the API in its production
container image, then verifies /health, /schema, and /predict over HTTP. It
never uses a real dataset or a release model artifact.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from typing import Any
from http.client import RemoteDisconnected
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.data_preprocessing import DataPreprocessor
from src.utils import save_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = PROJECT_ROOT / "temp" / "docker-e2e"
MODEL_PATH = WORK_DIR / "models" / "nids_api_smoke.pkl"
MODEL_VERSION = "docker_e2e_smoke"
FEATURE_NAMES = ["flow_bytes", "packet_count"]


def _free_local_port() -> int:
    """Reserve an available local port long enough to configure Compose."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as socket_handle:
        socket_handle.bind(("127.0.0.1", 0))
        return int(socket_handle.getsockname()[1])


def _write_smoke_artifact(destination: Path) -> None:
    """Create a small valid binary artifact solely for container verification."""
    features = pd.DataFrame(
        {"flow_bytes": [1.0, 2.0, 9.0, 10.0], "packet_count": [1.0, 2.0, 9.0, 10.0]}
    )
    preprocessor = DataPreprocessor(method="minmax").fit(features)
    classifier = RandomForestClassifier(n_estimators=5, random_state=42).fit(
        preprocessor.transform(features).values,
        [0, 0, 1, 1],
    )
    save_model(
        {
            "model": classifier,
            "preprocessor": preprocessor,
            "feature_names": FEATURE_NAMES,
            "threshold": 0.5,
            "model_version": MODEL_VERSION,
        },
        destination,
    )


def _request_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_for_health(base_url: str, deadline_seconds: float = 75.0) -> dict[str, Any]:
    """Wait for the image to install, load its artifact, and become ready."""
    deadline = time.monotonic() + deadline_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            health = _request_json(f"{base_url}/health")
            if health.get("status") == "ok":
                return health
        except (URLError, RemoteDisconnected, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(1)
    raise RuntimeError(f"API did not become healthy within {deadline_seconds:.0f} seconds: {last_error}")


def _wait_for_prometheus_metric(base_url: str, query: str, deadline_seconds: float = 75.0) -> None:
    """Verify Prometheus has scraped a non-empty ML-NIDS metric series."""
    deadline = time.monotonic() + deadline_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            payload = _request_json(f"{base_url}/api/v1/query?{urlencode({'query': query})}")
            if payload.get("status") == "success" and payload.get("data", {}).get("result"):
                return
        except (URLError, RemoteDisconnected, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(1)
    raise RuntimeError(f"Prometheus did not scrape {query!r} within {deadline_seconds:.0f} seconds: {last_error}")


def _compose_command(project_name: str) -> list[str]:
    return [
        "docker",
        "compose",
        "--project-name",
        project_name,
        "-f",
        "compose.e2e.yaml",
    ]


def main() -> None:
    """Execute the Compose integration test and always clean up its resources."""
    if shutil.which("docker") is None:
        raise RuntimeError("Docker is required for the API end-to-end integration test.")

    port = _free_local_port()
    prometheus_port = _free_local_port()
    project_name = f"ml_nids_e2e_{os.getpid()}"
    environment = os.environ.copy()
    environment["ML_NIDS_E2E_PORT"] = str(port)
    environment["ML_NIDS_E2E_PROMETHEUS_PORT"] = str(prometheus_port)
    command = _compose_command(project_name)

    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    failure: BaseException | None = None
    try:
        _write_smoke_artifact(MODEL_PATH)
        subprocess.run(
            [*command, "up", "--build", "--detach", "nids-api", "prometheus"],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
        )
        base_url = f"http://127.0.0.1:{port}"
        health = _wait_for_health(base_url)
        if health.get("model_version") != MODEL_VERSION or health.get("feature_count") != len(FEATURE_NAMES):
            raise RuntimeError(f"Unexpected /health response: {health}")

        schema = _request_json(f"{base_url}/schema")
        if schema != {"model_version": MODEL_VERSION, "required_features": FEATURE_NAMES}:
            raise RuntimeError(f"Unexpected /schema response: {schema}")

        prediction = _request_json(
            f"{base_url}/predict",
            {"records": [{"flow_bytes": 1.5, "packet_count": 1.5}, {"flow_bytes": 9.5, "packet_count": 9.5}]},
        )
        predictions = prediction.get("predictions")
        if prediction.get("model_version") != MODEL_VERSION or not isinstance(predictions, list) or len(predictions) != 2:
            raise RuntimeError(f"Unexpected /predict response: {prediction}")
        if any(item.get("prediction") not in (0, 1) or not 0.0 <= float(item.get("probability", -1)) <= 1.0 for item in predictions):
            raise RuntimeError(f"Invalid prediction values returned by API: {prediction}")
        _wait_for_prometheus_metric(
            f"http://127.0.0.1:{prometheus_port}",
            "ml_nids_flows_scored_total",
        )
        print("Docker API end-to-end test passed.")
    except BaseException as exc:
        failure = exc
        raise
    finally:
        if failure is not None:
            subprocess.run([*command, "logs", "--no-color"], cwd=PROJECT_ROOT, env=environment, check=False)
        subprocess.run(
            [*command, "down", "--volumes", "--remove-orphans"],
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
        )
        if WORK_DIR.exists():
            shutil.rmtree(WORK_DIR)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Docker API end-to-end test failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
