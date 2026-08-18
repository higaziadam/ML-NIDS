"""
Training pipeline for ML-NIDS.

This module handles the complete training workflow including data loading,
preprocessing, feature engineering, model training, and evaluation.
"""

import argparse
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config import CONFIG, SAVED_MODELS_DIR, SPLITS_DIR
from src.utils import logger, save_model, load_data, save_data, Timer, setup_logger
from src.data_preprocessing import DataCleaner, DataPreprocessor, preprocess_pipeline, DataSplitter
from src.feature_extraction import feature_engineering_pipeline
from src.models import create_model, ModelFactory
from src.evaluate import comprehensive_evaluation, select_binary_threshold


def load_training_data(
    data_path: str,
    label_column: str = "label",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load training data.
    
    Args:
        data_path: Path to data file
        label_column: Name of label column
        
    Returns:
        Features and labels
    """
    logger.info(f"Loading data from {data_path}")
    
    data = load_data(data_path)
    
    if label_column not in data.columns:
        raise ValueError(f"Label column '{label_column}' not found in data")
    
    X = data.drop(columns=[label_column])
    y = data[label_column]
    
    logger.info(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    return X, y


def preprocess_data(
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess data.
    
    Args:
        X: Input features
        y: Labels
        
    Returns:
        Preprocessed features and labels
    """
    logger.info("Starting data preprocessing")
    
    X_processed, y_processed = preprocess_pipeline(
        X, y,
        remove_duplicates=CONFIG["data"].get("remove_duplicates", True),
        handle_missing=CONFIG["data"].get("missing_value_strategy", "drop"),
        detect_outliers=CONFIG["data"].get("detect_outliers", False),
        outlier_method=CONFIG["data"].get("outlier_method", "iqr"),
        remove_constant=True,
        # Scaling is fitted after the train/test split so test and inference data
        # never influence the learned normalization parameters.
        normalize=False,
        normalization_method=CONFIG["data"].get("normalization_method", "minmax"),
    )
    
    logger.info(f"Preprocessing complete: {X_processed.shape}")
    
    return X_processed, y_processed


def preprocess_holdout_data(
    X: pd.DataFrame,
    y: pd.Series,
    training_columns: list[str],
    training_reference: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare held-out data without fitting or learning from it.

    Imputation statistics, outlier thresholds, constant columns, feature
    selection, and scaling are deliberately learned from training data only.
    """
    X = X.copy().replace([np.inf, -np.inf], np.nan)
    strategy = CONFIG["data"].get("missing_value_strategy", "drop")
    if strategy == "drop":
        X = X.dropna()
    elif strategy in {"mean", "median"}:
        numeric_reference = training_reference.select_dtypes(include=[np.number])
        statistics = numeric_reference.mean() if strategy == "mean" else numeric_reference.median()
        X = X.fillna(statistics)
    elif strategy == "forward_fill":
        X = X.ffill().bfill()
    else:
        raise ValueError(f"Unknown missing-value strategy: {strategy}")
    y = y.loc[X.index]
    missing_columns = [column for column in training_columns if column not in X.columns]
    if missing_columns:
        raise ValueError(f"Holdout data is missing training columns: {missing_columns}")
    return X.loc[:, training_columns], y


def engineer_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Engineer features.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        
    Returns:
        Engineered train and test features
    """
    logger.info("Starting feature engineering")
    
    X_train_eng, selected_features = feature_engineering_pipeline(
        X_train, y_train,
        create_interactions=False,
        select_features=True,
        n_features=CONFIG["features"].get("n_features", 20),
        remove_correlated=True,
        correlation_threshold=CONFIG["features"].get("correlation_threshold", 0.95),
    )
    
    # Apply same features to test set with validation
    if selected_features:
        # Validate that selected features exist in test set
        valid_features = [f for f in selected_features if f in X_test.columns]
        if not valid_features:
            logger.warning("No selected features found in test set, using original features")
            X_test_eng = X_test
        else:
            X_test_eng = X_test[valid_features]
    else:
        X_test_eng = X_test
    
    logger.info(f"Feature engineering complete: {X_train_eng.shape[1]} features")
    
    return X_train_eng, X_test_eng


def engineer_features_for_holdouts(
    X_train: pd.DataFrame,
    holdouts: Sequence[pd.DataFrame],
    y_train: pd.Series,
) -> Tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Fit feature engineering on training data and apply that schema to holdouts."""
    X_train_eng, selected_features = feature_engineering_pipeline(
        X_train, y_train,
        create_interactions=False,
        select_features=True,
        n_features=CONFIG["features"].get("n_features", 20),
        remove_correlated=True,
        correlation_threshold=CONFIG["features"].get("correlation_threshold", 0.95),
    )
    feature_names = selected_features or X_train_eng.columns.tolist()
    transformed = []
    for data in holdouts:
        missing = [column for column in feature_names if column not in data.columns]
        if missing:
            raise ValueError(f"Holdout data is missing engineered training features: {missing}")
        transformed.append(data.loc[:, feature_names])
    return X_train_eng, transformed


def _binary_probabilities(model: object, X: np.ndarray) -> Tuple[np.ndarray, Any, Any]:
    """Return positive-class probabilities and ordered classes for a binary model."""
    probabilities = model.predict_proba(X)
    estimator = getattr(model, "model", model)
    classes = np.asarray(getattr(estimator, "classes_", []))
    if probabilities.ndim != 2 or probabilities.shape[1] != 2 or len(classes) != 2:
        raise ValueError("Threshold tuning requires a fitted binary classifier with two probability columns.")
    return probabilities[:, 1], classes[0], classes[1]


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "random_forest",
    **model_kwargs
) -> object:
    """
    Train model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_type: Type of model
        **model_kwargs: Model-specific hyperparameters
        
    Returns:
        Trained model
    """
    logger.info(f"Training {model_type} model")

    classes, counts = np.unique(y_train, return_counts=True)
    if len(classes) < 2:
        distribution = dict(zip(classes.tolist(), counts.tolist()))
        raise ValueError(
            "Training data contains only one class after preprocessing: "
            f"{distribution}. Disable destructive cleaning (especially outlier removal) "
            "or provide training data containing both classes."
        )
    
    # Use hyperparameters from config if not provided
    if not model_kwargs:
        model_kwargs = CONFIG["model"]["hyperparameters"]
    
    with Timer(f"Training {model_type} model"):
        model = create_model(model_type, **model_kwargs)
        model.fit(X_train, y_train)
    
    logger.info(f"Model training complete")
    
    return model


def evaluate_model(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "model",
    threshold: Optional[float] = None,
) -> dict:
    """
    Evaluate model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Model name for logging
        
    Returns:
        Evaluation results
    """
    logger.info(f"Evaluating {model_name}")
    
    y_pred_proba = model.predict_proba(X_test)
    if threshold is None:
        y_pred = model.predict(X_test)
    else:
        positive_probabilities, negative_class, positive_class = _binary_probabilities(model, X_test)
        y_pred = np.where(positive_probabilities >= threshold, positive_class, negative_class)
    
    # Comprehensive evaluation
    results = comprehensive_evaluation(
        y_test,
        y_pred,
        y_pred_proba,
        selected_threshold=threshold,
        save_results=CONFIG["evaluation"].get("save_results", True),
        results_path=SAVED_MODELS_DIR / f"{model_name}_results",
    )
    if threshold is not None:
        results["selected_threshold"] = threshold
    
    return results


def save_training_artifacts(
    model: object,
    preprocessor: DataPreprocessor,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    threshold: float,
    threshold_results: pd.DataFrame,
    model_name: str = "model",
) -> None:
    """
    Save training artifacts.
    
    Args:
        model: Trained model
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        y_train: Training labels
        y_val: Validation labels
        y_test: Test labels
        threshold: Threshold selected from validation data
        threshold_results: Metrics for all validation threshold candidates
        model_name: Model name
    """
    logger.info("Saving training artifacts")
    
    # Save the model together with the fitted inference schema and scaler.  These
    # objects must be reused at inference time to avoid feature-order drift and
    # data leakage from fitting a scaler on incoming traffic.
    model_path = SAVED_MODELS_DIR / f"{model_name}.pkl"
    save_model(
        {
            "model": model,
            "preprocessor": preprocessor,
            "feature_names": X_train.columns.tolist(),
            "threshold": threshold,
            "model_version": model_name,
            "threshold_policy": CONFIG["threshold"].copy(),
        },
        model_path,
    )
    
    # Save datasets
    # Keep each run's reproducibility splits separate instead of overwriting a
    # previous experiment's train/test files.
    splits_dir = SPLITS_DIR / model_name
    
    X_train_with_label = X_train.copy()
    X_train_with_label["label"] = y_train.values
    save_data(X_train_with_label, splits_dir / "train.csv")

    X_val_with_label = X_val.copy()
    X_val_with_label["label"] = y_val.values
    save_data(X_val_with_label, splits_dir / "validation.csv")
    
    X_test_with_label = X_test.copy()
    X_test_with_label["label"] = y_test.values
    save_data(X_test_with_label, splits_dir / "test.csv")
    save_data(threshold_results, SAVED_MODELS_DIR / f"{model_name}_results" / "threshold_selection.csv")
    
    logger.info(f"Training artifacts saved to {SAVED_MODELS_DIR}")


def train_pipeline(
    data_path: str,
    label_column: str = "label",
    model_type: str = "random_forest",
    model_name: str = "nids_model",
    save_artifacts: bool = True,
    test_size: Optional[float] = None,
) -> Tuple[object, dict]:
    """
    Complete training pipeline.
    
    Args:
        data_path: Path to training data
        label_column: Name of label column
        model_type: Type of model to train
        model_name: Name for saving artifacts
        save_artifacts: Save model and data splits
        test_size: Optional final-test proportion override. The configured
            validation split is always retained.
        
    Returns:
        Trained model, evaluation results
    """
    logger.info("Starting training pipeline")
    
    with Timer("Complete training pipeline"):
        # Load data
        X, y = load_training_data(data_path, label_column)
        
        # Split raw data first.  Every learned preprocessing operation below is
        # then fitted only on the training partition.
        splitter = DataSplitter()
        configured_test_size = CONFIG["data"]["test_size"] if test_size is None else test_size
        val_size = CONFIG["data"]["val_size"]
        train_size = 1.0 - configured_test_size - val_size
        splits = splitter.split_data(
            X, y,
            train_size=train_size,
            test_size=configured_test_size,
            val_size=val_size,
            random_state=CONFIG["data"]["random_state"],
        )
        
        X_train_raw, y_train_raw = splits["train"]
        X_val_raw, y_val_raw = splits["val"]
        X_test_raw, y_test_raw = splits["test"]

        X_train, y_train = preprocess_data(X_train_raw, y_train_raw)
        X_val, y_val = preprocess_holdout_data(
            X_val_raw, y_val_raw, X_train.columns.tolist(), X_train_raw
        )
        X_test, y_test = preprocess_holdout_data(
            X_test_raw,
            y_test_raw,
            X_train.columns.tolist(),
            X_train_raw,
        )
        
        # Feature selection is fitted on training data only; validation and
        # test data receive the exact same learned feature schema.
        X_train_eng, holdout_features = engineer_features_for_holdouts(
            X_train, [X_val, X_test], y_train
        )
        X_val_eng, X_test_eng = holdout_features
        
        # Fit one scaler on training features only, then use it for every later
        # split and for persisted inference.
        inference_preprocessor = DataPreprocessor(
            method=CONFIG["data"].get("normalization_method", "minmax")
        )
        X_train_eng = inference_preprocessor.fit_transform(X_train_eng)
        X_val_eng = inference_preprocessor.transform(X_val_eng)
        X_test_eng = inference_preprocessor.transform(X_test_eng)

        # Convert to numpy for model training
        X_train_np = X_train_eng.values if isinstance(X_train_eng, pd.DataFrame) else X_train_eng
        y_train_np = y_train.values if isinstance(y_train, pd.Series) else y_train
        X_val_np = X_val_eng.values if isinstance(X_val_eng, pd.DataFrame) else X_val_eng
        y_val_np = y_val.values if isinstance(y_val, pd.Series) else y_val
        X_test_np = X_test_eng.values if isinstance(X_test_eng, pd.DataFrame) else X_test_eng
        y_test_np = y_test.values if isinstance(y_test, pd.Series) else y_test
        
        # Train model
        model = train_model(X_train_np, y_train_np, model_type=model_type)

        validation_probabilities, _, positive_class = _binary_probabilities(model, X_val_np)
        selected_threshold, threshold_results = select_binary_threshold(
            y_val_np,
            validation_probabilities,
            positive_class,
            CONFIG["threshold"]["candidates"],
            CONFIG["threshold"]["min_recall"],
            CONFIG["threshold"]["max_fpr"],
        )
        logger.info(f"Selected validation threshold: {selected_threshold:.2f}")
        
        # Evaluate the chosen threshold once on the untouched test partition.
        results = evaluate_model(model, X_test_np, y_test_np, model_name, threshold=selected_threshold)
        
        # Save artifacts
        if save_artifacts:
            save_training_artifacts(
                model,
                inference_preprocessor,
                X_train_eng,
                X_val_eng,
                X_test_eng,
                y_train,
                y_val,
                y_test,
                selected_threshold,
                threshold_results,
                model_name,
            )
        
        logger.info("Training pipeline complete")
    
    return model, results


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train ML-NIDS model")
    parser.add_argument(
        "--data",
        type=str,
        default=str(CONFIG["paths"]["processed_data_dir"] / "train_data.csv"),
        help="Path to training data"
    )
    parser.add_argument(
        "--label",
        type=str,
        default="label",
        help="Name of label column"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=CONFIG["model"]["model_type"],
        choices=ModelFactory.get_available_models(),
        help="Model type"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="nids_model",
        help="Model name for saving"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=None,
        help="Optional final-test proportion override; validation split is retained"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save artifacts"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger(__name__)
    
    # Train
    model, results = train_pipeline(
        data_path=args.data,
        label_column=args.label,
        model_type=args.model,
        model_name=args.name,
        save_artifacts=not args.no_save,
        test_size=args.test_size,
    )
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
