"""
Configuration management for ML-NIDS.

This module contains all configuration settings including paths, hyperparameters,
model configurations, and logging settings.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# ============================================================================
# DATA PATHS
# ============================================================================
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
FINAL_HOLDOUT_DIR = DATA_DIR / "final_holdout"

# Do not create directories during import. Inference containers intentionally
# mount model data read-only, and configuration must remain safe to import in
# those deployments. Write-oriented operations create their own destination
# directories immediately before saving output.

# ============================================================================
# MODEL PATHS
# ============================================================================
MODELS_DIR = PROJECT_ROOT / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved"
MODEL_CONFIGS_DIR = MODELS_DIR / "configs"
EVALUATION_DIR = MODELS_DIR / "evaluation"

# ============================================================================
# LOGGING PATHS
# ============================================================================
LOGS_DIR = PROJECT_ROOT / "logs"

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
DATA_CONFIG = {
    "train_size": 0.70,
    "test_size": 0.15,
    "val_size": 0.15,
    "random_state": 42,
    "missing_value_strategy": "drop",  # or "mean", "median"
    "outlier_method": "iqr",  # or "zscore"
    # Do not discard unusual traffic in supervised NIDS training: attacks can
    # legitimately look like statistical outliers.
    "detect_outliers": False,
    "normalization_method": "minmax",  # or "zscore", "robust"
}

# ============================================================================
# FEATURE CONFIGURATION
# ============================================================================
FEATURE_CONFIG = {
    "selected_features": None,  # None = use all, or provide list of feature names
    "n_features": 20,
    "correlation_threshold": 0.95,  # Remove highly correlated features
}

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
MODEL_CONFIG = {
    "model_type": "xgboost",  # Options: random_forest, xgboost, svm
    "hyperparameters": {
        "random_forest": {
            "n_estimators": 300,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": -1,
            "class_weight": "balanced",
        },
        "svm": {"kernel": "rbf", "C": 1.0, "gamma": "scale", "random_state": 42},
        "xgboost": {
            "n_estimators": 300,
            "max_depth": 10,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3.0,
            "reg_lambda": 1.0,
            "scale_pos_weight": 1.10,
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        },
    },
}

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================
EVALUATION_CONFIG = {
    "save_results": True,
}

# Threshold selection uses validation data only.  The final test partition is
# held back until a threshold has been selected.
THRESHOLD_CONFIG = {
    "candidates": [0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35],
    "min_recall": 0.92,
    "max_fpr": 0.005,
    "default": 0.50,
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOGGING_CONFIG = {
    "level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
    "log_file": LOGS_DIR / "ml_nids.log",
    "max_bytes": 10485760,  # 10 MB
    "backup_count": 5,
}

# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================
INFERENCE_CONFIG = {
    "prediction_threshold": 0.5,
}

# ============================================================================
# COMBINED CONFIG DICT
# ============================================================================
CONFIG = {
    "data": DATA_CONFIG,
    "features": FEATURE_CONFIG,
    "model": MODEL_CONFIG,
    "evaluation": EVALUATION_CONFIG,
    "threshold": THRESHOLD_CONFIG,
    "logging": LOGGING_CONFIG,
    "inference": INFERENCE_CONFIG,
    "paths": {
        "project_root": PROJECT_ROOT,
        "data_dir": DATA_DIR,
        "raw_data_dir": RAW_DATA_DIR,
        "processed_data_dir": PROCESSED_DATA_DIR,
        "splits_dir": SPLITS_DIR,
        "final_holdout_dir": FINAL_HOLDOUT_DIR,
        "models_dir": MODELS_DIR,
        "saved_models_dir": SAVED_MODELS_DIR,
        "model_configs_dir": MODEL_CONFIGS_DIR,
        "evaluation_dir": EVALUATION_DIR,
        "logs_dir": LOGS_DIR,
    },
}


def get_config():
    """Return the full configuration dictionary."""
    return CONFIG


def print_config():
    """Print the current configuration."""
    import json
    
    config_copy = CONFIG.copy()
    # Convert Path objects to strings for JSON serialization
    config_copy["paths"] = {k: str(v) for k, v in config_copy["paths"].items()}
    
    print(json.dumps(config_copy, indent=2))


if __name__ == "__main__":
    print_config()
