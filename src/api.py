"""HTTP inference service for a frozen ML-NIDS model artifact.

The service is intentionally inference-only: it loads one persisted artifact at
startup and never trains, tunes, or stores request payloads.
"""

from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import AsyncIterator

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.predict import load_trained_model, make_predictions, preprocess_inference_data
from src.utils import logger, setup_logger


DEFAULT_MODEL_PATH = "models/saved/xgb_v10_regularized_fine_threshold.pkl"


class PredictionRequest(BaseModel):
    """One or more numeric network-flow records to score."""

    records: list[dict[str, float]] = Field(min_length=1, max_length=10_000)


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
    return artifact


def create_app(model_path: str | Path | None = None) -> FastAPI:
    """Create an API application that loads the selected model during startup."""
    configured_path = Path(model_path or os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        setup_logger(__name__)
        artifact = _validate_artifact(load_trained_model(configured_path))
        app.state.artifact = artifact
        app.state.model_path = configured_path
        logger.info("ML-NIDS API started with model %s", artifact.get("model_version", configured_path.name))
        yield
        logger.info("ML-NIDS API stopped")

    app = FastAPI(
        title="ML-NIDS Inference API",
        version="1.0.0",
        description="Inference-only API for a frozen ML-NIDS artifact.",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health(request: Request) -> dict:
        artifact = request.app.state.artifact
        return {
            "status": "ok",
            "model_version": artifact.get("model_version", request.app.state.model_path.name),
            "feature_count": len(artifact["feature_names"]),
            "threshold": float(artifact["threshold"]),
        }

    @app.get("/schema")
    def schema(request: Request) -> dict:
        """Return the exact input fields required by the loaded artifact."""
        artifact = request.app.state.artifact
        return {
            "model_version": artifact.get("model_version", request.app.state.model_path.name),
            "required_features": artifact["feature_names"],
        }

    @app.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
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

        return PredictionResponse(
            model_version=artifact.get("model_version", request.app.state.model_path.name),
            threshold=float(artifact["threshold"]),
            predictions=[
                PredictionItem(index=index, prediction=int(label), probability=float(probability))
                for index, (label, probability) in enumerate(zip(labels, probabilities))
            ],
        )

    return app


app = create_app()
