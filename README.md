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
  decision threshold, dataset/model fingerprints, calibration diagnostics, and
  a training-only feature-drift baseline.
- Training-only randomized hyperparameter search, calibration analysis,
  PSI-based drift reports, permutation-importance explainability, and
  independent representative-data validation.
- Final-holdout and nested cross-validation workflows.
- Batch CSV prediction through Python or Docker.
- CICFlowMeter-style CSV ingestion that normalizes supported headers and submits
  flow records to the API in batches.
- Optional Dockerized directory watcher for completed CICFlowMeter CSV exports.
- Measured API latency and throughput benchmark with JSON reports.
- FastAPI inference service with interactive OpenAPI documentation.
- API schema discovery, optional API-key protection, request-size limits,
  single-container rate limiting, and Docker health checks.
- Prometheus metrics for request volume, latency, errors, scored flows, attack
  predictions, batch size, model version/threshold, and drift-review status;
  with a provisioned Grafana dashboard.
- Automated tests and GitHub Actions quality gates for release metadata,
  dependencies, Dockerfiles, and container images.

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
│   ├── live_monitor.py             # Watches completed flow-export CSVs
│   ├── api_benchmark.py            # API latency and throughput measurement
│   ├── predict.py                  # Batch prediction pipeline
│   ├── train.py                    # Training pipeline
│   └── validation.py               # Holdout and cross-validation workflow
├── tests/                          # Automated tests and small safe fixtures
├── Dockerfile                      # Batch-inference image
├── Dockerfile.api                  # FastAPI image
├── compose.yaml                    # Local API service and optional watcher
├── runtime/                         # Local watcher input, alerts, archives (ignored)
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

### Govern the ML workflow

Create a content-addressed dataset manifest before training or evaluation:

```powershell
.\venv\Scripts\python.exe -m src.ml_workflow dataset-manifest `
  --data data\processed\train_data.csv `
  --output experiments\results\train_dataset_manifest.json
```

Run randomized hyperparameter search only on development/training data, never
on a final holdout or representative target-network dataset:

```powershell
.\venv\Scripts\python.exe -m src.tuning `
  --data data\final_holdout\candidate_holdout\development.csv `
  --model xgboost `
  --iterations 10 `
  --folds 3 `
  --output experiments\results\xgb_random_search.csv
```

Newly trained artifacts include a training-only drift baseline; their result
directories include calibration and drift files. Calibration measures
probability reliability using Brier score and reliability bins. Drift uses
Population Stability Index (PSI) to flag feature distribution changes.

Evaluate a frozen artifact once against compatible, independently collected,
labeled target-network traffic:

```powershell
.\venv\Scripts\python.exe -m src.ml_workflow representative-evaluate `
  --model models\saved\candidate.pkl `
  --data data\external\target_network_labeled.csv `
  --output models\evaluation\target_network_validation
```

The report writes metrics, calibration, dataset provenance, and drift results.
It cannot tune model parameters, features, or threshold.

For operational review, create a PSI drift report from compatible incoming-flow
data, or create a global permutation-importance report from labeled evaluation
data:

```powershell
.\venv\Scripts\python.exe -m src.ml_workflow drift-report `
  --model models\saved\candidate.pkl `
  --data runtime\recent_flows.csv `
  --output runtime\observability\drift_report.json

.\venv\Scripts\python.exe -m src.ml_workflow explain `
  --model models\saved\candidate.pkl `
  --data data\external\target_network_labeled.csv `
  --output models\evaluation\permutation_importance.csv
```

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

### Observability: Prometheus and Grafana

Start the API, Prometheus, and Grafana together. Set a non-default Grafana
password before exposing the dashboard outside your local machine:

```powershell
$env:GRAFANA_ADMIN_PASSWORD = "use-a-long-unique-password"
docker compose --profile observability up --build -d
```

Open Grafana at <http://localhost:3000> and sign in with the configured
`GRAFANA_ADMIN_USER` (default `admin`). The provisioned **ML-NIDS Overview**
dashboard shows request rate, error rate, p95 latency, attack predictions,
flow volume by model version, and drift-review status. Prometheus is available
at <http://localhost:9090> for local metric queries.

`GET /metrics` exposes only aggregate, low-cardinality labels (`endpoint`,
HTTP method/status, and model version). It never emits flow payloads, client
IP addresses, API keys, or feature values. To publish PSI drift state, write a
drift report to `runtime\observability\drift_report.json`; the API refreshes
the status when Prometheus scrapes `/metrics`.

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

### Live CICFlowMeter export watcher

The optional watcher provides a near-real-time handoff for a separate flow
exporter. It polls `runtime\incoming_flows`, waits for an export to remain
unchanged for two polling intervals, scores it through the API, writes the
scored CSV to `runtime\alerts`, and moves the source CSV to either
`runtime\processed_flows` or `runtime\failed_flows`. A failure also creates an
adjacent error JSON record. It never captures packets or monitors a NIC itself.

Start the API and watcher together:

```powershell
docker compose --profile live up --build -d
```

Place only completed CICFlowMeter-compatible CSV files in
`runtime\incoming_flows`. Stop both services with `docker compose down`.
The entire `runtime` directory is ignored by Git so local flow records and
alerts are not committed accidentally.

### Measure API latency

With the API running, benchmark the end-to-end inference request path using a
compatible flow CSV:

```powershell
.\venv\Scripts\python.exe -m src.api_benchmark `
  --input tests\fixtures\v10_cicflowmeter_sample.csv `
  --output benchmarks\reports\api_latency.json `
  --batch-sizes 1 10 100 `
  --requests-per-size 10
```

The report records p50/p95/p99 and mean request latency plus flow throughput
for each batch size. It includes local HTTP, API validation, preprocessing, and
model inference; it excludes upstream packet capture and flow-export time.
Keep the request count within the API rate-limit configuration or use a
separate controlled benchmark deployment.

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

The current suite contains 50 fast unit tests covering preprocessing, evaluation,
release-profile validation, holdout/cross-validation safeguards, external-data
schema preparation, batch prediction, flow ingestion, live-file handling, API
latency reporting, and API behavior/protections.

Run the deterministic V10 release-profile gate locally with:

```powershell
.\venv\Scripts\python.exe -m src.quality_gates `
  --profile models\configs\xgb_v10_candidate.json `
  --output temp\quality-gates\v10_release_profile.json
```

The gate validates the tracked V10 label contract, threshold policy, recorded
validation/test recall and false-positive rate, and artifact hash metadata. It
does not retrain or reevaluate the ignored full dataset in CI.

GitHub Actions runs the test suite, V10 quality gate, dependency audit,
Dockerfile linting, a Docker Compose API end-to-end test, and high/critical
vulnerability scans for both Docker images. The end-to-end test generates a
small synthetic artifact, starts the API container, and verifies `/health`,
`/schema`, and `/predict` through HTTP. Pull requests also receive a
dependency-change review.

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
