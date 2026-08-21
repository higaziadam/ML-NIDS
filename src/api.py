"""HTTP inference service for a frozen ML-NIDS model artifact.

The service is intentionally inference-only: it loads one persisted artifact at
startup and never trains, tunes, or stores request payloads.
"""

from contextlib import asynccontextmanager
from collections import deque
import hmac
import json
import os
from pathlib import Path
import time
from typing import AsyncIterator

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, CONTENT_TYPE_LATEST, generate_latest

from src.predict import load_trained_model, make_predictions, preprocess_inference_data
from src.utils import logger, setup_logger


DEFAULT_MODEL_PATH = "models/saved/xgb_v10_regularized_fine_threshold.pkl"


class ApiMetrics:
    """Per-application Prometheus metrics with intentionally low-cardinality labels."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter("ml_nids_api_requests_total", "Completed API requests", ["endpoint", "method", "status_code"], registry=self.registry)
        self.latency = Histogram("ml_nids_api_request_duration_seconds", "API request duration", ["endpoint", "method"], registry=self.registry)
        self.errors = Counter("ml_nids_api_errors_total", "API responses with status >= 400", ["endpoint", "status_code"], registry=self.registry)
        self.flows_scored = Counter("ml_nids_flows_scored_total", "Network flows scored", ["model_version"], registry=self.registry)
        self.attack_predictions = Counter("ml_nids_attack_predictions_total", "Attack predictions returned", ["model_version"], registry=self.registry)
        self.batch_size = Histogram("ml_nids_prediction_batch_size", "Flow records per prediction request", ["model_version"], registry=self.registry)
        self.model_info = Gauge("ml_nids_model_info", "Active model metadata", ["model_version"], registry=self.registry)
        self.model_threshold = Gauge("ml_nids_model_threshold", "Active model decision threshold", ["model_version"], registry=self.registry)
        self.drift_status = Gauge("ml_nids_drift_status", "Current drift-review status", ["state"], registry=self.registry)
        self.set_drift_state("unknown")

    def set_drift_state(self, state: str) -> None:
        for candidate in ("healthy", "review_required", "unknown", "error"):
            self.drift_status.labels(state=candidate).set(1 if candidate == state else 0)

    def refresh_drift_state(self, report_path: str | None) -> None:
        if not report_path:
            return
        try:
            with Path(report_path).open("r", encoding="utf-8") as handle:
                report = json.load(handle)
            requires_review = bool(report.get("summary", {}).get("requires_review"))
            self.set_drift_state("review_required" if requires_review else "healthy")
        except (OSError, json.JSONDecodeError, AttributeError):
            self.set_drift_state("error")

V10_REQUEST_EXAMPLE = {
    "records": [
        {
            "Protocol": 6,
            "Fwd Packet Length Max": 60,
            "Fwd Packet Length Min": 20,
            "Fwd Packet Length Mean": 40,
            "Bwd Packet Length Min": 20,
            "Bwd Packet Length Mean": 40,
            "Bwd IAT Total": 1000,
            "Bwd IAT Std": 10,
            "Bwd IAT Max": 100,
            "Packet Length Min": 20,
            "Packet Length Max": 60,
            "Packet Length Mean": 40,
            "Packet Length Variance": 100,
            "RST Flag Count": 0,
            "PSH Flag Count": 0,
            "ACK Flag Count": 1,
            "Down/Up Ratio": 1,
            "Init Fwd Win Bytes": 1024,
            "Init Bwd Win Bytes": 1024,
            "Fwd Seg Size Min": 20,
        }
    ]
}


class PredictionRequest(BaseModel):
    """One or more numeric network-flow records to score."""

    model_config = ConfigDict(json_schema_extra={"example": V10_REQUEST_EXAMPLE})

    records: list[dict[str, float]] = Field(
        min_length=1,
        max_length=10_000,
        description=(
            "One or more numeric flow records. Retrieve GET /schema first and "
            "include every required feature in each record."
        ),
    )


class HealthResponse(BaseModel):
    """Readiness details for the currently loaded artifact."""

    status: str
    model_version: str
    feature_count: int
    threshold: float


class SchemaResponse(BaseModel):
    """Feature contract required by the currently loaded artifact."""

    model_version: str
    required_features: list[str]


class PredictionItem(BaseModel):
    """A thresholded label and attack probability for one input record."""

    index: int
    prediction: int
    probability: float


class PredictionResponse(BaseModel):
    """Response returned by the inference endpoint."""

    model_version: str
    threshold: float
    predictions: list[PredictionItem]


def _validate_artifact(artifact: object) -> dict:
    if not isinstance(artifact, dict):
        raise RuntimeError("Legacy model artifacts are not supported by the API. Retrain with the current pipeline.")
    required = {"model", "preprocessor", "feature_names", "threshold"}
    missing = sorted(required.difference(artifact))
    if missing:
        raise RuntimeError(f"Model artifact is missing required fields: {missing}")
    estimator = getattr(artifact["model"], "model", artifact["model"])
    classes = set(getattr(estimator, "classes_", []))
    if classes != {0, 1}:
        raise RuntimeError(
            "The API requires an artifact trained with classes 0 (benign) and 1 (attack)."
        )
    return artifact


def create_app(
    model_path: str | Path | None = None,
    *,
    api_key: str | None = None,
    max_request_bytes: int | None = None,
    rate_limit_requests: int | None = None,
    rate_limit_window_seconds: int | None = None,
) -> FastAPI:
    """Create an API application that loads the selected model during startup."""
    configured_path = Path(model_path or os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    configured_api_key = api_key if api_key is not None else os.getenv("API_KEY")
    request_limit = max_request_bytes if max_request_bytes is not None else int(os.getenv("MAX_REQUEST_BYTES", "1048576"))
    rate_limit = rate_limit_requests if rate_limit_requests is not None else int(os.getenv("RATE_LIMIT_REQUESTS", "60"))
    rate_window = rate_limit_window_seconds if rate_limit_window_seconds is not None else int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
    if request_limit <= 0 or rate_limit <= 0 or rate_window <= 0:
        raise ValueError("API request-size and rate-limit settings must be positive integers.")
    request_timestamps: dict[str, deque[float]] = {}
    metrics = ApiMetrics()
    drift_report_path = os.getenv("DRIFT_STATUS_PATH")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        setup_logger(__name__)
        artifact = _validate_artifact(load_trained_model(configured_path))
        app.state.artifact = artifact
        app.state.model_path = configured_path
        app.state.metrics = metrics
        model_version = artifact.get("model_version", configured_path.name)
        metrics.model_info.labels(model_version=model_version).set(1)
        metrics.model_threshold.labels(model_version=model_version).set(float(artifact["threshold"]))
        metrics.refresh_drift_state(drift_report_path)
        logger.info("ML-NIDS API started with model %s", artifact.get("model_version", configured_path.name))
        yield
        logger.info("ML-NIDS API stopped")

    app = FastAPI(
        title="ML-NIDS Inference API",
        version="1.0.0",
        description="Inference-only API for a frozen ML-NIDS artifact.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def apply_request_protections(request: Request, call_next):
        """Apply request safeguards and record privacy-safe operational metrics."""
        started = time.perf_counter()
        endpoint = request.url.path if request.url.path in {"/health", "/schema", "/predict"} else "other"
        response: Response | None = None
        try:
            if request.url.path == "/predict":
                content_length = request.headers.get("content-length")
                try:
                    declared_size = int(content_length) if content_length is not None else None
                except ValueError:
                    response = JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": "Invalid Content-Length header."})
                    return response
                if declared_size is not None and declared_size > request_limit:
                    response = JSONResponse(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, content={"detail": f"Request body exceeds the {request_limit}-byte limit."})
                    return response
                if len(await request.body()) > request_limit:
                    response = JSONResponse(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, content={"detail": f"Request body exceeds the {request_limit}-byte limit."})
                    return response
                client_host = request.client.host if request.client else "unknown"
                now = time.monotonic()
                for host, timestamps in list(request_timestamps.items()):
                    while timestamps and now - timestamps[0] >= rate_window:
                        timestamps.popleft()
                    if not timestamps:
                        del request_timestamps[host]
                timestamps = request_timestamps.setdefault(client_host, deque())
                if len(timestamps) >= rate_limit:
                    retry_after = max(1, int(rate_window - (now - timestamps[0])))
                    response = JSONResponse(status_code=status.HTTP_429_TOO_MANY_REQUESTS, content={"detail": "Prediction rate limit exceeded. Try again later."}, headers={"Retry-After": str(retry_after)})
                    return response
                timestamps.append(now)

            response = await call_next(request)
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            if request.url.path == "/predict":
                response.headers["Cache-Control"] = "no-store"
            return response
        except Exception:
            metrics.errors.labels(endpoint=endpoint, status_code="500").inc()
            raise
        finally:
            if request.url.path != "/metrics":
                status_code = str(response.status_code) if response is not None else "500"
                metrics.requests.labels(endpoint=endpoint, method=request.method, status_code=status_code).inc()
                metrics.latency.labels(endpoint=endpoint, method=request.method).observe(time.perf_counter() - started)
                if response is not None and response.status_code >= 400:
                    metrics.errors.labels(endpoint=endpoint, status_code=status_code).inc()

    def require_api_key(request: Request) -> None:
        """Require a configured API key for prediction without logging secrets."""
        if not configured_api_key:
            return
        submitted_key = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(submitted_key, configured_api_key):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key.")

    @app.get("/health", response_model=HealthResponse, summary="Service readiness")
    def health(request: Request) -> HealthResponse:
        artifact = request.app.state.artifact
        return HealthResponse(
            status="ok",
            model_version=artifact.get("model_version", request.app.state.model_path.name),
            feature_count=len(artifact["feature_names"]),
            threshold=float(artifact["threshold"]),
        )

    @app.get("/schema", response_model=SchemaResponse, summary="Loaded model input schema")
    def schema(request: Request) -> SchemaResponse:
        """Return the exact input fields required by the loaded artifact."""
        artifact = request.app.state.artifact
        return SchemaResponse(
            model_version=artifact.get("model_version", request.app.state.model_path.name),
            required_features=artifact["feature_names"],
        )

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics(request: Request) -> Response:
        """Expose aggregate service metrics; never emit request payloads or client identities."""
        request.app.state.metrics.refresh_drift_state(drift_report_path)
        return Response(generate_latest(request.app.state.metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        summary="Score network-flow records",
        description=(
            "Applies the saved preprocessor and frozen decision threshold. "
            "The example is valid for the default V10 artifact; use GET /schema "
            "when a different artifact is mounted."
        ),
    )
    def predict(
        payload: PredictionRequest,
        request: Request,
        _: None = Depends(require_api_key),
    ) -> PredictionResponse:
        artifact = request.app.state.artifact
        features = pd.DataFrame(payload.records)
        try:
            processed = preprocess_inference_data(
                features,
                artifact["preprocessor"],
                artifact["feature_names"],
            )
            labels, probabilities = make_predictions(
                artifact,
                processed.values,
                return_probabilities=True,
                threshold=float(artifact["threshold"]),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        model_version = artifact.get("model_version", request.app.state.model_path.name)
        request.app.state.metrics.flows_scored.labels(model_version=model_version).inc(len(labels))
        request.app.state.metrics.attack_predictions.labels(model_version=model_version).inc(int(np.sum(labels == 1)))
        request.app.state.metrics.batch_size.labels(model_version=model_version).observe(len(labels))
        return PredictionResponse(
            model_version=model_version,
            threshold=float(artifact["threshold"]),
            predictions=[
                PredictionItem(index=index, prediction=int(label), probability=float(probability))
                for index, (label, probability) in enumerate(zip(labels, probabilities))
            ],
        )

    return app


app = create_app()
