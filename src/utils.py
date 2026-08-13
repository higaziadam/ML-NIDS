"""
Utility functions for ML-NIDS including logging, metrics, and data loading.
"""

import logging
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

from src.config import LOGGING_CONFIG, LOGS_DIR


# ============================================================================
# LOGGING SETUP
# ============================================================================
def setup_logger(name: str, log_level: str = None) -> logging.Logger:
    """
    Setup and configure logger for the application.
    
    Args:
        name: Logger name (usually __name__)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    if log_level is None:
        log_level = LOGGING_CONFIG["level"]
    
    logger_instance = logging.getLogger(name)
    
    # Prevent duplicate handlers by checking if already configured
    if logger_instance.handlers:
        return logger_instance
    
    logger_instance.setLevel(getattr(logging, log_level))
    logger_instance.propagate = False
    
    # Create formatter
    formatter = logging.Formatter(
        LOGGING_CONFIG["format"],
        datefmt=LOGGING_CONFIG["date_format"]
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(formatter)
    logger_instance.addHandler(console_handler)
    
    # File handler
    log_file = LOGGING_CONFIG["log_file"]
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level))
        file_handler.setFormatter(formatter)
        logger_instance.addHandler(file_handler)
    except Exception as e:
        logger_instance.warning(f"Could not set up file logging: {e}")
    
    return logger_instance


logger = setup_logger(__name__)


# ============================================================================
# FILE I/O UTILITIES
# ============================================================================
def save_model(model: Any, filepath: Union[str, Path]) -> None:
    """
    Save model to pickle file.
    
    Args:
        model: Model object to save
        filepath: Path to save the model
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    
    logger.info(f"Model saved to {filepath}")


def load_model(filepath: Union[str, Path]) -> Any:
    """
    Load model from pickle file.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded model object
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    
    logger.info(f"Model loaded from {filepath}")
    return model


def save_data(data: pd.DataFrame, filepath: Union[str, Path], format: str = "csv") -> None:
    """
    Save data to file.
    
    Args:
        data: DataFrame to save
        filepath: Path to save the data
        format: File format ('csv', 'parquet', 'xlsx')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == "csv":
        data.to_csv(filepath, index=False)
    elif format.lower() == "parquet":
        data.to_parquet(filepath, index=False)
    elif format.lower() == "xlsx":
        data.to_excel(filepath, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Data saved to {filepath} ({len(data)} rows, {len(data.columns)} columns)")


def load_data(filepath: Union[str, Path], format: str = None) -> pd.DataFrame:
    """
    Load data from file.
    
    Args:
        filepath: Path to the data file
        format: File format (auto-detected if None)
        
    Returns:
        Loaded DataFrame
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    if format is None:
        format = filepath.suffix.lower().strip(".")
    
    try:
        if format == "csv":
            data = pd.read_csv(filepath)
        elif format == "parquet":
            data = pd.read_parquet(filepath)
        elif format in ["xlsx", "xls"]:
            data = pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
    except ValueError:
        raise
    except Exception as exc:
        raise OSError(f"Failed to load {format} data from {filepath}: {exc}") from exc
    
    logger.info(f"Data loaded from {filepath} ({len(data)} rows, {len(data.columns)} columns)")
    return data


def save_config(config: Dict, filepath: Union[str, Path]) -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        filepath: Path to save the config
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert Path objects to strings
    config_copy = {}
    for key, value in config.items():
        if isinstance(value, Path):
            config_copy[key] = str(value)
        else:
            config_copy[key] = value
    
    with open(filepath, "w") as f:
        json.dump(config_copy, f, indent=2)
    
    logger.info(f"Configuration saved to {filepath}")


# ============================================================================
# EVALUATION METRICS
# ============================================================================
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels (0/1)
        y_pred_proba: Predicted probabilities (for AUC)
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_pred_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")
    
    return metrics


def print_metrics(metrics: Dict[str, float]) -> None:
    """Print formatted metrics."""
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    for metric_name, value in metrics.items():
        print(f"{metric_name.upper():.<20} {value:.4f}")
    print("="*50 + "\n")


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Get confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Confusion matrix
    """
    return confusion_matrix(y_true, y_pred)


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print classification report."""
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_true, y_pred, digits=4))
    print("="*50 + "\n")


# ============================================================================
# DATA UTILITIES
# ============================================================================
def get_data_info(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Get information about a DataFrame.
    
    Args:
        data: Input DataFrame
        
    Returns:
        Dictionary with data information
    """
    return {
        "shape": data.shape,
        "columns": data.columns.tolist(),
        "dtypes": data.dtypes.to_dict(),
        "missing_values": data.isnull().sum().to_dict(),
        "duplicates": data.duplicated().sum(),
        "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024**2,
    }


def print_data_info(data: pd.DataFrame) -> None:
    """Print data information."""
    info = get_data_info(data)
    print("\n" + "="*50)
    print("DATA INFORMATION")
    print("="*50)
    print(f"Shape: {info['shape']}")
    print(f"Columns: {len(info['columns'])}")
    print(f"Missing values: {sum(info['missing_values'].values())}")
    print(f"Duplicates: {info['duplicates']}")
    print(f"Memory usage: {info['memory_usage_mb']:.2f} MB")
    print("="*50 + "\n")


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    train_size: float = 0.7,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets.
    
    Args:
        X: Features
        y: Labels
        train_size: Training set proportion
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=train_size,
        random_state=random_state,
        stratify=y
    )
    
    logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test")
    return X_train, X_test, y_train, y_test


def balance_dataset(X: np.ndarray, y: np.ndarray, strategy: str = "smote") -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance imbalanced dataset.
    
    Args:
        X: Features
        y: Labels
        strategy: Balancing strategy ('smote', 'oversample', 'undersample')
        
    Returns:
        Balanced X, y
    """
    try:
        if strategy.lower() == "smote":
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y)
        elif strategy.lower() == "oversample":
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(random_state=42)
            X_balanced, y_balanced = ros.fit_resample(X, y)
        elif strategy.lower() == "undersample":
            from imblearn.under_sampling import RandomUnderSampler
            rus = RandomUnderSampler(random_state=42)
            X_balanced, y_balanced = rus.fit_resample(X, y)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        logger.info(f"Dataset balanced using {strategy}: {len(X_balanced)} samples")
        return X_balanced, y_balanced
    except ImportError:
        logger.warning("imbalanced-learn not installed. Skipping dataset balancing.")
        return X, y


# ============================================================================
# TIMING UTILITIES
# ============================================================================
class Timer:
    """Context manager for timing code execution."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = pd.Timestamp.now()
        return self
    
    def __exit__(self, *args):
        self.end_time = pd.Timestamp.now()
        duration = (self.end_time - self.start_time).total_seconds()
        logger.info(f"{self.name} completed in {duration:.2f} seconds")
    
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        if self.end_time is None:
            return (pd.Timestamp.now() - self.start_time).total_seconds()
        return (self.end_time - self.start_time).total_seconds()


if __name__ == "__main__":
    logger.info("Utils module loaded successfully")
