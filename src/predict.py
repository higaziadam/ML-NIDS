"""
Prediction/Inference pipeline for ML-NIDS.

This module handles making predictions on new data using trained models.
"""

import argparse
from pathlib import Path
from typing import Union, Tuple

import numpy as np
import pandas as pd

from src.config import CONFIG, SAVED_MODELS_DIR
from src.utils import logger, load_model, load_data, save_data, Timer, setup_logger
from src.data_preprocessing import DataPreprocessor


def load_trained_model(model_path: Union[str, Path]) -> object:
    """
    Load trained model.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Loaded model
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = load_model(model_path)
    logger.info(f"Model loaded from {model_path}")
    
    return model


def preprocess_inference_data(
    X: pd.DataFrame,
    preprocessor: DataPreprocessor,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Preprocess data for inference.
    
    Args:
        X: Input features
        preprocessor: Preprocessor fitted on the training features
        feature_names: Ordered training feature names
        
    Returns:
        Preprocessed features
    """
    logger.info("Preprocessing inference data")
    
    if not isinstance(preprocessor, DataPreprocessor) or not preprocessor.is_fitted:
        raise ValueError("Model artifact has no fitted preprocessor; retrain the model with the current pipeline.")
    if "label" in X.columns:
        X = X.drop(columns="label")
    missing = [column for column in feature_names if column not in X.columns]
    if missing:
        raise ValueError(f"Inference schema is missing required training columns: {missing}")
    unexpected = [column for column in X.columns if column not in feature_names]
    if unexpected:
        logger.info(f"Ignoring {len(unexpected)} non-model input columns: {unexpected}")
    X_processed = preprocessor.transform(X.loc[:, feature_names])
    
    logger.info(f"Inference data preprocessed: {X_processed.shape}")
    
    return X_processed


def make_predictions(
    model: object,
    X: np.ndarray,
    return_probabilities: bool = True,
    threshold: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions.
    
    Args:
        model: Trained model
        X: Input features
        return_probabilities: Return probability predictions
        threshold: Custom prediction threshold
        
    Returns:
        Predictions, probabilities
    """
    if threshold is None:
        threshold = CONFIG["inference"].get("prediction_threshold", 0.5)
    
    trained_model = model.get("model") if isinstance(model, dict) else model
    if trained_model is None:
        raise ValueError("Invalid model artifact: missing model")
    logger.info(f"Making predictions on {X.shape[0]} samples")
    
    with Timer("Inference"):
        y_pred = trained_model.predict(X)
        
        if return_probabilities:
            y_proba = trained_model.predict_proba(X)
            # If 2D, extract positive class probabilities
            if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
                y_proba = y_proba[:, 1]
        else:
            y_proba = None
    
    logger.info(f"Predictions complete")
    
    return y_pred, y_proba


def format_predictions(
    predictions: np.ndarray,
    probabilities: np.ndarray = None,
    data_index: pd.Index = None,
) -> pd.DataFrame:
    """
    Format predictions into DataFrame.
    
    Args:
        predictions: Predicted labels
        probabilities: Predicted probabilities
        data_index: Original data index
        
    Returns:
        DataFrame with predictions
    """
    results = pd.DataFrame({
        "prediction": predictions,
    }, index=data_index)
    
    if probabilities is not None:
        results["probability"] = probabilities
    
    return results


def predict_pipeline(
    model_path: Union[str, Path],
    data_path: Union[str, Path],
    output_path: Union[str, Path] = None,
    return_probabilities: bool = True,
    save_results: bool = True,
) -> pd.DataFrame:
    """
    Complete prediction pipeline.
    
    Args:
        model_path: Path to trained model
        data_path: Path to input data
        output_path: Path to save predictions
        return_probabilities: Return probability predictions
        save_results: Save predictions to file
        
    Returns:
        DataFrame with predictions
    """
    logger.info("Starting prediction pipeline")
    
    with Timer("Complete prediction pipeline"):
        # Load model
        model = load_trained_model(model_path)
        
        # Load data
        logger.info(f"Loading data from {data_path}")
        X = load_data(data_path)
        original_index = X.index
        
        if not isinstance(model, dict):
            raise ValueError("Legacy model file detected. Retrain it with the current pipeline before inference.")
        # Preprocess using the fitted training-time transformer and schema.
        X_processed = preprocess_inference_data(
            X,
            model.get("preprocessor"),
            model.get("feature_names", []),
        )
        
        # Make predictions
        X_np = X_processed.values if isinstance(X_processed, pd.DataFrame) else X_processed
        y_pred, y_proba = make_predictions(
            model, X_np,
            return_probabilities=return_probabilities,
        )
        
        # Format results
        results = format_predictions(y_pred, y_proba, original_index)
        
        # Save results
        if save_results and output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_data(results, output_path)
            logger.info(f"Predictions saved to {output_path}")
        
        logger.info("Prediction pipeline complete")
    
    return results


def batch_predict(
    model_path: Union[str, Path],
    data_paths: list,
    output_dir: Union[str, Path] = None,
    return_probabilities: bool = True,
) -> list:
    """
    Batch prediction on multiple files.
    
    Args:
        model_path: Path to trained model
        data_paths: List of data file paths
        output_dir: Directory to save predictions
        return_probabilities: Return probability predictions
        
    Returns:
        List of prediction DataFrames
    """
    logger.info(f"Starting batch prediction on {len(data_paths)} files")
    
    results = []
    
    for data_path in data_paths:
        output_path = None
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{Path(data_path).stem}_predictions.csv"
        
        result = predict_pipeline(
            model_path,
            data_path,
            output_path=output_path,
            return_probabilities=return_probabilities,
            save_results=output_path is not None,
        )
        
        results.append(result)
    
    logger.info(f"Batch prediction complete")
    
    return results


def realtime_predict(
    model_path: Union[str, Path],
    single_sample: dict,
    return_probability: bool = True,
) -> Tuple[int, float]:
    """
    Make single prediction (realtime inference).
    
    Args:
        model_path: Path to trained model
        single_sample: Single sample as dictionary
        return_probability: Return probability
        
    Returns:
        Prediction, probability
    """
    # Load model once (cache this in production)
    model = load_trained_model(model_path)
    
    # Convert to DataFrame
    X = pd.DataFrame([single_sample])
    
    if not isinstance(model, dict):
        raise ValueError("Legacy model file detected. Retrain it with the current pipeline before inference.")
    # Preprocess with the fitted training-time transformer.
    X_processed = preprocess_inference_data(
        X,
        model.get("preprocessor"),
        model.get("feature_names", []),
    )
    X_np = X_processed.values
    
    # Predict
    y_pred, y_proba = make_predictions(
        model, X_np,
        return_probabilities=return_probability,
    )
    
    return y_pred[0], y_proba[0] if y_proba is not None else None


def main():
    """Main prediction script."""
    parser = argparse.ArgumentParser(description="Run ML-NIDS predictions")
    parser.add_argument(
        "--model",
        type=str,
        default=str(SAVED_MODELS_DIR / "nids_model.pkl"),
        help="Path to trained model"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to input data for prediction"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save predictions"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Enable batch prediction mode"
    )
    parser.add_argument(
        "--no-proba",
        action="store_true",
        help="Don't return probabilities"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger(__name__)
    
    # Predict
    if args.batch:
        # Batch mode
        data_paths = args.data.split(",")
        results = batch_predict(
            args.model,
            data_paths,
            output_dir=args.output,
            return_probabilities=not args.no_proba,
        )
    else:
        # Single file mode
        results = predict_pipeline(
            args.model,
            args.data,
            output_path=args.output,
            return_probabilities=not args.no_proba,
            save_results=args.output is not None,
        )
    
    if isinstance(results, list):
        for i, result in enumerate(results):
            print(f"\nPredictions {i+1}:")
            print(result.head(10))
    else:
        print("\nPredictions:")
        print(results.head(10))
    
    logger.info("Prediction completed successfully!")


if __name__ == "__main__":
    main()
