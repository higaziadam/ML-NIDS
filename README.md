# ML-NIDS: Machine Learning-based Network Intrusion Detection System

A machine learning-powered Network Intrusion Detection System (NIDS) for detecting network anomalies and cyber attacks in real-time.

> **Current status:** the implemented system is a batch-training and batch-
> inference ML pipeline. Its Docker image has been smoke-tested with a mounted
> model and input CSV, and a FastAPI inference service is available through
> Docker Compose.
> V10 is the selected low-alert internal candidate; V14's CIC-IDS2017 result
> exposed a major external-generalization gap, so no model is presented as
> production-ready. See [RELEASE_V10.md](RELEASE_V10.md) and
> [experiments/experiments.md](experiments/experiments.md).

> **Evaluation workflow:** use `python -m src.validation create-holdout`,
> `cross-validate`, and `final-evaluate` for single-split final evaluation.
> See [RELEASE_V10.md](RELEASE_V10.md) for the exact commands and safeguards.

## Features

- **Anomaly Detection**: Identify suspicious network traffic patterns
- **ML-Powered**: Built with scikit-learn, TensorFlow/PyTorch
- **Feature Extraction**: Advanced network flow feature engineering
- **Model Evaluation**: Comprehensive metrics (precision, recall, F1, ROC-AUC)
- **Reproducible Inference**: Dockerized CPU batch prediction
- **Logging & Monitoring**: Full audit trail of predictions

## Project Structure

```
ML_NIDS/
├── data/
│   ├── raw/          # Original datasets (CICIDS2018, NSL-KDD, etc.)
│   ├── processed/    # Cleaned & normalized data
│   └── splits/       # Train/test/validation splits
├── models/
│   ├── saved/        # Trained model checkpoints
│   ├── configs/      # Model architecture configurations
│   └── evaluation/   # Performance metrics & results
├── src/
│   ├── config.py              # Configuration management
│   ├── train.py               # Training pipeline
│   ├── predict.py             # Inference pipeline
│   ├── models.py              # Model definitions
│   ├── data_preprocessing.py  # Data cleaning & normalization
│   ├── feature_extraction.py  # Feature engineering
│   ├── evaluate.py            # Evaluation metrics
│   └── utils.py               # Helper functions & logging
├── tests/            # Unit tests
├── scripts/          # Utility scripts (download data, splits)
├── logs/             # Training & inference logs
├── deployment/       # FastAPI/Flask deployment
└── notebooks/        # Jupyter notebooks for EDA
```

## Installation

### Prerequisites
- Python 3.14 (the recorded training and release environment)
- pip or conda

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ML-NIDS.git
   cd ML-NIDS
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package (development mode)**
   ```bash
   pip install -e .
   ```

## Quick Start

### Reproduce a frozen candidate

The command below creates a new artifact and refuses to overwrite the tracked
V10 output paths. Use it only with the CICIDS2018 development data.

```powershell
.\venv\Scripts\python.exe -m src.train `
  --config models\configs\xgb_v10_candidate.json `
  --data data\processed\train_data.csv `
  --name xgb_v10_reproduction
```

### Batch prediction

```powershell
.\venv\Scripts\python.exe -m src.predict `
  --model models\saved\xgb_v10_regularized_fine_threshold.pkl `
  --data data\processed\your_flows.csv `
  --output output\predictions.csv
```

## Datasets

Supported datasets:
- **CICIDS2018**: Canadian Institute for Cybersecurity dataset
- **NSL-KDD**: Standard benchmark for intrusion detection
- **KDD99**: Classic DARPA dataset
- **UNSW-NB15**: Australian network traffic dataset

## Usage

### 1. Prepare Data

```bash
python scripts/download_data.py
python scripts/generate_splits.py
```

### 2. Train Model

```bash
python -m src.train --data data/processed/train_data.csv --model xgboost --name experiment_name
```

### 3. Evaluate a saved model

```bash
python -m src.validation final-evaluate --data data/processed/evaluation.csv --model models/saved/model.pkl --name evaluation_name
```

### 4. Make Predictions

```bash
python -m src.predict --model models/saved/model.pkl --data data/processed/test.csv --output output/predictions.csv
```

## Configuration

Edit `src/config.py` to configure:
- Data paths
- Model hyperparameters
- Training parameters
- Logging settings

Example:
```python
CONFIG = {
    'model_type': 'random_forest',
    'hyperparameters': {
        'n_estimators': 100,
        'max_depth': 20,
        'random_state': 42
    },
    'train_test_split': 0.8,
    'random_seed': 42
}
```

## Evaluation status and limitations

| Evaluation | Accuracy | Precision | Attack recall | FPR | Interpretation |
|---|---:|---:|---:|---:|---|
| V10 final CICIDS2018 holdout | 98.05% | 98.24% | 91.91% | 0.41% | Selected low-alert internal candidate; narrowly missed the project recall policy. |
| V14 CIC-IDS2017 external test | 76.04% | 15.34% | 4.81% | 6.51% | Major cross-dataset generalization gap; not suitable for production claims. |

V10 was selected for its low false-positive operating point, which reduces alert
backlog. The external result must not be used for further threshold/model tuning.
It demonstrates that a model trained on CICIDS2018 does not automatically
generalize to CIC-IDS2017. A production deployment requires fresh representative
labeled traffic, monitoring, and retraining/validation under the target network's
conditions.

## Testing

Run unit tests:
```bash
pytest tests/
pytest tests/ --cov=src  # With coverage
```

## Deployment

### FastAPI inference service

The API is inference-only: it loads one frozen artifact at startup and never
trains, tunes, or persists submitted flows. Start it with the mounted V10
artifact:

```powershell
docker compose up --build -d
Invoke-RestMethod http://localhost:8000/health
```

Open `http://localhost:8000/docs` for interactive OpenAPI documentation. The
service provides:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Confirms the service and reports model version, feature count, and threshold. |
| `GET /schema` | Returns the exact feature names required by the mounted model. |
| `POST /predict` | Scores one or more numeric flow records using the saved preprocessor and threshold. |

Before calling `POST /predict`, retrieve the model schema:

```powershell
Invoke-RestMethod http://localhost:8000/schema
```

Then send all of those numeric fields in each record. For example, create a
JSON request with the returned field names and submit it:

```powershell
Invoke-RestMethod http://localhost:8000/predict `
  -Method Post `
  -ContentType 'application/json' `
  -Body '{"records":[{"feature_name":123.4}]}'
```

The example payload is illustrative: `feature_name` must be replaced by every
feature returned by `GET /schema`. Missing, non-numeric, or non-finite fields
are rejected with HTTP 422. Stop the local service with:

```powershell
docker compose down
```

This local API has no authentication, TLS termination, persistent request log,
or horizontal scaling by default. Set an API key before exposing predictions:

```powershell
$env:API_KEY = 'replace-with-a-long-random-secret'
docker compose up --build -d
```

When `API_KEY` is set, callers must include the matching `X-API-Key` header on
`POST /predict`. Health and schema endpoints remain unauthenticated for local
orchestrator checks. Prediction requests are limited to 1 MiB and 60 requests
per client IP per 60 seconds by default; override `MAX_REQUEST_BYTES`,
`RATE_LIMIT_REQUESTS`, and `RATE_LIMIT_WINDOW_SECONDS` through environment
variables. These limits are in-memory and apply to one container only. Put the
service behind an authenticated HTTPS reverse proxy with centralized rate
limiting, monitoring, and secret management before public or multi-instance
deployment.

### Docker batch prediction

Build the image from the repository root:

```powershell
docker build -t ml-nids:local .
```

Create an output directory, then mount your model directory, input directory,
and output directory. The image contains code and dependencies only; it never
contains your datasets or trained model.

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

The selected model must be present locally under `models/saved/`, and the input
CSV must contain the feature columns expected by that model. Results are
written to `output/predictions.csv`. This is a batch CLI container, not an HTTP
service; no port mapping is required.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "Add your feature"`
4. Push to branch: `git push origin feature/your-feature`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{mlnids2024,
  author = {Your Name},
  title = {ML-NIDS: Machine Learning-based Network Intrusion Detection System},
  year = {2024},
  url = {https://github.com/yourusername/ML-NIDS}
}
```

## References

- [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html)
- [CICIDS2018 Dataset](https://www.unb.ca/cic/datasets/ids-2018.html)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow Documentation](https://www.tensorflow.org/)

## Support

For issues, questions, or suggestions, please open an [GitHub Issue](https://github.com/yourusername/ML-NIDS/issues).

## Authors

- Adam Higazi

---

**Last Updated**: August 2026
