# 🚀 ML-NIDS Quick Start Card

## Installed & Ready to Use

### After Cleanup (2026-08-12)
```
✅ All Python packages properly organized
✅ Data files in correct directories (data/raw/)
✅ Scripts importable as Python package
✅ Environment configuration template available
✅ 3 comprehensive guides created
```

---

## 📋 Essential Commands

### Setup (First Time)
```bash
# 1. Navigate to project
cd ML_NIDS

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies (choose one)
pip install -r requirements.txt           # Production only (RECOMMENDED)
pip install -r requirements-dev.txt       # Development (with testing tools)

# 4. (Optional) Create .env file
cp .env.example .env
# Edit .env with your settings
```

### Quick Pipeline (Core Usage)
```bash
# 1. Explore data visually
python scripts/explore_parquet.py

# 2. Prepare Kaggle data for training
python scripts/prepare_kaggle_data.py

# 3. Train a model
python src/train.py --data data/processed/train_data.csv --model random_forest --name my_model

# 4. Make predictions
python src/predict.py --model models/saved/my_model.pkl --data data/processed/test_data.csv --output predictions.csv

# 5. (Optional) Create interactive notebook
python scripts/create_eda_notebook.py
jupyter notebook notebooks/EDA_CICIDS2018.ipynb
```

---

## 📂 Directory Reference

| Directory | Purpose | What's Inside |
|-----------|---------|---|
| `data/raw/` | Original data | 10 CICIDS2018 parquet files |
| `data/processed/` | Processed data | train_data.csv, test_data.csv (created) |
| `data/splits/` | Train/test splits | train.csv, test.csv (created) |
| `src/` | Core code | Python modules (train, predict, etc.) |
| `scripts/` | Utility scripts | Data exploration, preparation |
| `models/saved/` | Trained models | .pkl files (created) |
| `models/evaluation/` | Model metrics | Performance reports (created) |
| `models/configs/` | Model templates | Hyperparameter templates |
| `logs/` | Execution logs | train.log, predict.log (created) |
| `notebooks/` | Jupyter notebooks | EDA notebooks (created) |
| `tests/` | Unit tests | Test files (to be added) |
| `deployment/` | Deployment config | FastAPI app, Docker (future) |
| `experiments/` | Experiment tracking | Experiment logs and results |

---

## 🔧 Common Tasks

### Train a Model
```bash
python src/train.py \
  --data data/processed/train_data.csv \
  --model random_forest \           # Options: random_forest, gradient_boosting, svm
  --name my_nids_model \
  --test-size 0.2                  # 80/20 split
```

### Make Predictions
```bash
python src/predict.py \
  --model models/saved/my_nids_model.pkl \
  --data data/processed/test_data.csv \
  --output predictions.csv
```

### Explore Data
```bash
python scripts/explore_parquet.py
# Shows: Files, statistics, columns, distributions, sample data
```

### Prepare Kaggle Data
```bash
python scripts/prepare_kaggle_data.py
# Creates: train_data.csv and test_data.csv
```

### View Trained Model Performance
```bash
cat models/evaluation/*.json  # View evaluation metrics
```

### Generate EDA Notebook
```bash
python scripts/create_eda_notebook.py
jupyter notebook notebooks/EDA_CICIDS2018.ipynb
```

---

## 📊 Configuration Files

### `.env.example` - Environment Variables
Located in project root. Copy to `.env` and customize:
```bash
cp .env.example .env
# Edit with your settings
```

### `models/configs/model_config.json` - Model Hyperparameters
Reference configuration for Random Forest, Gradient Boosting, SVM

### `experiments/experiments.md` - Experiment Tracking
Log your model versions, parameters, and results here

### `src/config.py` - Core Configuration
Main configuration file with paths and defaults

---

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| `README.md` | Project overview |
| `KAGGLE_QUICKSTART.md` | Guide for using Kaggle data |
| `CLEANUP_COMPLETE.md` | What was fixed in cleanup |
| `DIRECTORY_ANALYSIS.md` | Technical analysis of issues |
| `FINAL_REPORT.md` | Comprehensive cleanup report |
| `.env.example` | Environment configuration template |

---

## ⚡ Pro Tips

### Speed Up Training
```bash
# Use fewer trees for faster training
python src/train.py --data data/processed/train_data.csv --model random_forest
# Edit src/config.py: n_estimators=50 (default is 100)
```

### Use Different Models
```bash
# Random Forest (fastest, good accuracy)
python src/train.py --model random_forest

# Gradient Boosting (slower, better accuracy)
python src/train.py --model gradient_boosting

# SVM (slowest, traditional approach)
python src/train.py --model svm
```

### Batch Predictions
```bash
# Predict on multiple files
python src/predict.py --model models/saved/model.pkl --data data1.csv --data data2.csv
```

### Enable Detailed Logging
```bash
# Check logs
tail -f logs/train.log     # Real-time training log
tail -f logs/predict.log   # Prediction log
```

---

## 🐛 Quick Troubleshooting

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: scripts` | `scripts/__init__.py` exists (fixed ✓) |
| `FileNotFoundError: data/raw/*.parquet` | Files moved to data/raw/ (fixed ✓) |
| `ImportError from src modules` | All import paths corrected (fixed ✓) |
| Installation too slow | Use `requirements.txt` not `requirements-dev.txt` |
| Need testing tools | Install with `requirements-dev.txt` |
| Package installation issues | Try: `pip install --upgrade pip` then `pip install -r requirements.txt` |

---

## 🎯 Workflow

### For Data Scientists
```bash
# 1. Explore
python scripts/explore_parquet.py

# 2. Prepare
python scripts/prepare_kaggle_data.py

# 3. Experiment
python src/train.py --data data/processed/train_data.csv --model random_forest
python src/train.py --data data/processed/train_data.csv --model gradient_boosting

# 4. Evaluate
# Check models/evaluation/ for metrics

# 5. Document
# Log results in experiments/experiments.md
```

### For Developers
```bash
# 1. Install dev dependencies
pip install -r requirements-dev.txt

# 2. Write tests
pytest tests/

# 3. Format code
black src/ scripts/

# 4. Type check
mypy src/

# 5. Profile performance
python -m cProfile -s cumtime src/train.py --data ...
```

### For DevOps/Deployment
```bash
# 1. Build Docker image
docker build -t ml-nids .

# 2. Run container
docker run -p 8000:8000 ml-nids

# 3. Access API
curl http://localhost:8000/predict
```

---

## 📦 Requirements Summary

### Production (15 packages)
```
pip install -r requirements.txt
```
Includes: numpy, pandas, scikit-learn, tensorflow, keras, matplotlib, seaborn, etc.

### Development (26 packages)
```
pip install -r requirements-dev.txt
```
Includes: All production + pytest, black, flake8, jupyter, mypy, etc.

---

## 🔗 Key Files to Know

- `src/config.py` - Change hyperparameters here
- `src/train.py` - Training pipeline entry point
- `src/predict.py` - Inference pipeline entry point
- `scripts/explore_parquet.py` - Explore Kaggle data
- `experiments/experiments.md` - Track your experiments
- `.env.example` - Environment configuration

---

## ✅ Last Updated

- **Date**: 2026-08-12
- **Status**: All systems organized and ready
- **Next Action**: Run `python scripts/explore_parquet.py` to get started

---

## 🌟 You're All Set!

Your ML-NIDS project is organized, documented, and ready to use.

**Start here**: `python scripts/explore_parquet.py`

Happy machine learning! 🚀
