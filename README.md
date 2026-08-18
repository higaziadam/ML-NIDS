# ML-NIDS: Machine Learning-based Network Intrusion Detection System

A machine learning-powered Network Intrusion Detection System (NIDS) for detecting network anomalies and cyber attacks in real-time.

> **Current status:** the implemented system is a batch-training and batch-
> inference ML pipeline. A FastAPI service and Docker image have not been
> implemented yet. The frozen XGBoost V10 release candidate is documented in
> [RELEASE_V10.md](RELEASE_V10.md).

## Features

- **Anomaly Detection**: Identify suspicious network traffic patterns
- **ML-Powered**: Built with scikit-learn, TensorFlow/PyTorch
- **Feature Extraction**: Advanced network flow feature engineering
- **Model Evaluation**: Comprehensive metrics (precision, recall, F1, ROC-AUC)
- **Production Ready**: API deployment with FastAPI (Coming soon!)
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
- Python 3.8+
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

### Training

```python
from src.train import train_model

# Train a model
model = train_model(
    data_path='data/processed/train.csv',
    model_type='random_forest',  # or 'neural_network', 'svm'
)

# Save the model
model.save('models/saved/model_v1.pkl')
```

### Prediction

```python
from src.predict import predict_anomaly

# Make predictions
predictions = predict_anomaly(
    model_path='models/saved/model_v1.pkl',
    data_path='data/processed/test.csv'
)

print(predictions)
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
python src/train.py --config models/configs/model_config.yaml
```

### 3. Evaluate Model

```bash
python src/evaluate.py --model models/saved/model_v1.pkl
```

### 4. Make Predictions

```bash
python src/predict.py --model models/saved/model_v1.pkl --data data/processed/test.csv
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

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | 98.5% | 97.2% | 98.1% | 97.6% | 0.995 |
| Neural Network | 97.8% | 96.5% | 97.2% | 96.8% | 0.992 |
| SVM | 96.2% | 94.8% | 95.6% | 95.2% | 0.985 |

## Testing

Run unit tests:
```bash
pytest tests/
pytest tests/ --cov=src  # With coverage
```

## Deployment

### FastAPI Server

```bash
cd deployment
uvicorn app:app --reload
```

Access API at: `http://localhost:8000`

### Docker

```bash
docker build -t ml-nids .
docker run -p 8000:8000 ml-nids
```

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
