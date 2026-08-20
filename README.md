# ML-NIDS

## 1. Project Name & Description

**ML-NIDS** is a machine-learning Network Intrusion Detection System project
for binary network-flow classification: `0 = benign` and `1 = attack`.

The project provides a reproducible workflow for data preparation, model
training, evaluation, batch prediction, and local HTTP inference. XGBoost
release profiles preserve selected features, a fitted preprocessor, and the
decision threshold with each saved model artifact.

Raw datasets and trained artifacts are intentionally not included in Git
because of their size and provenance.

> **Status:** V10 is the selected low-alert internal XGBoost candidate. V14
> demonstrated a substantial CICIDS2018-to-CIC-IDS2017 generalization gap, so
> this project does not claim a production-ready intrusion detector. See
> [RELEASE_V10.md](RELEASE_V10.md) and
> [experiments/experiments.md](experiments/experiments.md).

## 2. Project Features

- Binary benign-versus-attack classification for network-flow data.
- XGBoost and Random Forest experiment support.
- Frozen release-profile configuration for reproducible candidate runs.
- Training artifacts that retain the fitted preprocessor, ordered features, and
  decision threshold.
- Final-holdout and nested cross-validation workflows.
- Batch CSV prediction through Python or Docker.
- CICFlowMeter-style CSV ingestion that normalizes supported headers and submits
  flow records to the API in batches.
- FastAPI inference service with interactive OpenAPI documentation.
- API schema discovery, optional API-key protection, request-size limits,
  single-container rate limiting, and Docker health checks.
- Automated tests and GitHub Actions CI that tests the project and builds both
  Docker images.

## 3. Project Setup

### Prerequisites

- Python 3.14 (the recorded release environment).
- Docker Desktop for Docker batch inference or the API service.
- A local trained artifact for inference, such as
  `models/saved/xgb_v10_regularized_fine_threshold.pkl`.

### Install locally

```powershell
git clone https://github.com/higaziadam/ML-NIDS.git
cd ML-NIDS

python -m venv venv
.\venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

For runtime-only local dependencies, install `requirements.txt` instead of
`requirements-dev.txt`.

## 4. Project Structure

```text
ML-NIDS/
├── .github/workflows/ci.yml        # Test and Docker-build automation
├── data/                           # Local datasets and generated splits (ignored)
│   ├── raw/
│   ├── processed/
│   ├── splits/
│   └── final_holdout/
├── experiments/experiments.md      # Experiment history and interpretation
├── models/
│   ├── configs/                    # Frozen candidate profiles
│   ├── saved/                      # Local model artifacts (ignored)
│   └── evaluation/                 # Local evaluation outputs (ignored)
├── scripts/prepare_kaggle_data.py  # CICIDS2018 preparation utility
├── src/
│   ├── api.py                      # FastAPI inference service
│   ├── config.py                   # Default runtime configuration
│   ├── data_preprocessing.py       # Fitted preprocessing utilities
│   ├── evaluate.py                 # Metrics and threshold evaluation
│   ├── flow_ingestion.py           # CICFlowMeter CSV-to-API adapter
│   ├── predict.py                  # Batch prediction pipeline
│   ├── train.py                    # Training pipeline
│   └── validation.py               # Holdout and cross-validation workflow
├── tests/                          # Automated tests and small safe fixtures
├── Dockerfile                      # Batch-inference image
├── Dockerfile.api                  # FastAPI image
├── compose.yaml                    # Local API service
├── requirements.txt                # Python dependencies
└── requirements-runtime.txt        # Pinned Docker model runtime
```

## 5. Quickstart

The quickest path depends on whether you already have a model artifact.

### Reproduce a separate V10 artifact

Place prepared CICIDS2018 training data at `data/processed/train_data.csv`,
then run the frozen profile with a new name. This avoids overwriting V10.

```powershell
.\venv\Scripts\python.exe -m src.train `
  --config models\configs\xgb_v10_candidate.json `
  --data data\processed\train_data.csv `
  --name xgb_v10_reproduction
```

### Make a local batch prediction

```powershell
.\venv\Scripts\python.exe -m src.predict `
  --model models\saved\xgb_v10_regularized_fine_threshold.pkl `
  --data data\processed\your_flows.csv `
  --output output\predictions.csv
```

The input CSV must include every feature expected by the chosen artifact. The
API `GET /schema` endpoint can reveal that contract for a running API service.

## 6. Usage

### Prepare CICIDS2018 data

With supported source parquet files available locally, run:

```powershell
.\venv\Scripts\python.exe scripts\prepare_kaggle_data.py `
  --data-dir data `
  --output-dir data\processed
```

For an evaluation workflow that creates one development/final-holdout split,
produce one labeled source file instead:

```powershell
.\venv\Scripts\python.exe scripts\prepare_kaggle_data.py `
  --data-dir data `
  --output-dir data\processed `
  --single-source `
  --output-file cicids2018_labeled.csv
```

### Train an experiment

```powershell
.\venv\Scripts\python.exe -m src.train `
  --data data\processed\train_data.csv `
  --model xgboost `
  --name experiment_name
```

### Use the protected validation workflow

Create the holdout once, run cross-validation on development data only, and
evaluate the final holdout once after freezing the candidate. Do not use a
final-holdout result to tune that same candidate.

```powershell
.\venv\Scripts\python.exe -m src.validation create-holdout `
  --data data\processed\cicids2018_labeled.csv `
  --name candidate_holdout

.\venv\Scripts\python.exe -m src.validation cross-validate `
  --data data\final_holdout\candidate_holdout\development.csv `
  --config models\configs\xgb_v10_candidate.json `
  --name candidate_cv `
  --folds 5
```

For the full V10 safeguards and one-time final-evaluation command, see
[RELEASE_V10.md](RELEASE_V10.md).

## 7. Use It via API & Docker

### FastAPI service

The API loads one model artifact at startup. The default Compose configuration
expects this local file:

```text
models/saved/xgb_v10_regularized_fine_threshold.pkl
```

Start the service:

```powershell
docker compose up --build -d
```

Open interactive API documentation at <http://localhost:8000/docs>.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Returns service readiness, model version, feature count, and threshold. |
| `GET /schema` | Returns the exact required feature names for the loaded artifact. |
| `POST /predict` | Returns a label and attack probability for one or more flow records. |

Check the API and retrieve the feature contract:

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/schema
```

`POST /predict` requires every feature returned by `GET /schema`. The Swagger
page contains a valid V10 example. Stop the service with:

```powershell
docker compose down
```

### CICFlowMeter CSV-to-API adapter

The adapter takes an already-generated CICFlowMeter-style CSV, normalizes known
CIC-IDS2017 header variants, verifies the live API schema, and writes one output
row per input flow. Invalid rows are retained with `score_status=invalid_input`;
they are not sent to the model.

```powershell
.\venv\Scripts\python.exe -m src.flow_ingestion `
  --input path\to\cicflowmeter_export.csv `
  --output output\scored_flows.csv
```

The command writes `output\scored_flows.csv` and a corresponding manifest JSON.
The score file includes `source_row`, `model_version`, `threshold`,
`prediction`, `probability`, `is_alert`, and `score_status`. If the API
requires a key, set it outside source control:

```powershell
$env:ML_NIDS_API_KEY = 'your-api-key'
.\venv\Scripts\python.exe -m src.flow_ingestion `
  --input path\to\cicflowmeter_export.csv `
  --output output\scored_flows.csv
```

This adapter scores completed flows; it does not capture packets or generate
flows from a network interface or PCAP file.

### Docker batch prediction

Build the batch image:

```powershell
docker build -t ml-nids:local .
```

Run it with local model, input, and output directories mounted into the
container:

```powershell
New-Item -ItemType Directory -Force output

docker run --rm `
  -v "${PWD}/models:/app/models:ro" `
  -v "${PWD}/data/processed:/app/input:ro" `
  -v "${PWD}/output:/app/output" `
  ml-nids:local `
  --model /app/models/saved/xgb_v10_regularized_fine_threshold.pkl `
  --data /app/input/your_flows.csv `
  --output /app/output/predictions.csv
```

The image contains code and dependencies only. It does not embed datasets or
model artifacts.

## 8. Configuration

### Application and release profiles

- [src/config.py](src/config.py) contains default paths, model settings, logging,
  and dataset defaults.
- [models/configs/xgb_v10_candidate.json](models/configs/xgb_v10_candidate.json)
  is the frozen V10 profile. It records preprocessing, XGBoost parameters, and
  the selected threshold of `0.26`.
- Use a release profile with `--config`; do not override its model type, split,
  or threshold settings during reproduction.

### API environment variables

| Variable | Default | Purpose |
|---|---:|---|
| `MODEL_PATH` | V10 artifact path in the container | Artifact loaded by the API at startup. |
| `API_KEY` | Empty | If set, `POST /predict` requires it in the `X-API-Key` header. |
| `MAX_REQUEST_BYTES` | `1048576` | Maximum declared request size for predictions. |
| `RATE_LIMIT_REQUESTS` | `60` | Per-client prediction requests allowed in the window. |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | In-memory rate-limit window. |

Never commit real API keys or external datasets. The included rate limit is
per-container; use centralized controls for multi-instance deployment.

## 9. Results

| Version | Evaluation | Accuracy | Precision | Attack Recall | FPR | Interpretation |
|---|---|---:|---:|---:|---:|---|
| V10 | CICIDS2018 final holdout | 98.05% | 98.24% | 91.91% | 0.41% | Selected low-alert internal candidate; it narrowly missed the 92% recall policy. |
| V13 | CICIDS2018 5-fold fixed-threshold CV | 98.08% | 98.53% | 92.16% | 0.36% | Feature-expanded candidate with all reported folds policy-compliant. |
| V14 | CIC-IDS2017 external evaluation | 76.04% | 15.34% | 4.81% | 6.51% | Major external generalization gap. |

V10 was selected because its low false-positive operating point reduces alert
backlog. V14 shows that performance on CICIDS2018 does not transfer
automatically to CIC-IDS2017. Do not represent the system as production-ready
without validation on representative target-network traffic.

Detailed experiment history is available in
[experiments/experiments.md](experiments/experiments.md).

## 10. Testing

Run the complete suite:

```powershell
.\venv\Scripts\python.exe -m pytest -q --basetemp temp\pytest
```

The current suite contains 26 tests covering preprocessing, evaluation,
release-profile validation, holdout/cross-validation safeguards, external-data
schema preparation, batch prediction, flow ingestion, and API behavior/protections.

GitHub Actions runs this test suite and builds both Docker images on pushes and
pull requests to `main`.

## 11. Deployment

The local deployment path is Docker Compose plus the FastAPI service. The API
is inference-only: it loads a frozen artifact and does not train, tune, or
persist submitted flow records.

Before exposing the service outside a trusted local environment:

- Set a strong `API_KEY` and manage it outside source control.
- Put the service behind HTTPS and an authenticated reverse proxy.
- Use centralized rate limiting, monitoring, and secret management.
- Validate or retrain the model using representative target-network traffic.

The current Docker health check, request-size limit, and in-memory rate limiter
are useful local safeguards, but they are not substitutes for production
infrastructure controls.
