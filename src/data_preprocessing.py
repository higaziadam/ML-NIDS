"""
Data preprocessing utilities for ML-NIDS.

Handles data cleaning, normalization, missing values, and outlier detection.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from src.utils import logger, Timer


class DataPreprocessor:
    """Data preprocessing and normalization."""
    
    def __init__(self, method: str = "minmax", handle_missing: str = "drop"):
        """
        Initialize DataPreprocessor.
        
        Args:
            method: Normalization method ('minmax', 'zscore', 'robust')
            handle_missing: How to handle missing values ('drop', 'mean', 'median')
        """
        self.method = method
        self.handle_missing = handle_missing
        self.scaler = None
        self.feature_names = None
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fit preprocessor on training data.
        
        Args:
            X: Input features
            
        Returns:
            Self for chaining
        """
        with Timer("Data preprocessing fitting"):
            self._validate_features(X, require_expected_columns=False)
            self.feature_names = X.columns.tolist()
            
            # Initialize scaler
            if self.method == "minmax":
                self.scaler = MinMaxScaler()
            elif self.method == "zscore":
                self.scaler = StandardScaler()
            elif self.method == "robust":
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown normalization method: {self.method}")
            
            # Fit scaler
            self.scaler.fit(X)
            self.is_fitted = True
            logger.info(f"Preprocessor fitted with {self.method} normalization")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data.
        
        Args:
            X: Input features
            
        Returns:
            Transformed features
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted yet")
        
        self._validate_features(X, require_expected_columns=True)
        X_transformed = pd.DataFrame(
            self.scaler.transform(X),
            columns=self.feature_names,
            index=X.index
        )
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform data."""
        return self.fit(X).transform(X)

    def _validate_features(self, X: pd.DataFrame, require_expected_columns: bool) -> None:
        """Validate that features are numeric, finite, and schema-compatible."""
        if not isinstance(X, pd.DataFrame) or X.empty:
            raise ValueError("Features must be a non-empty pandas DataFrame")
        if require_expected_columns and X.columns.tolist() != self.feature_names:
            raise ValueError(
                "Inference feature columns do not match the fitted training schema. "
                f"Expected {self.feature_names}, got {X.columns.tolist()}."
            )
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            raise ValueError(f"All model features must be numeric. Non-numeric columns: {non_numeric}")
        if not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError("Features contain missing or infinite values; clean them before scaling")


class DataCleaner:
    """Data cleaning utilities."""
    
    @staticmethod
    def remove_duplicates(data: pd.DataFrame, subset: Optional[list] = None) -> pd.DataFrame:
        """
        Remove duplicate rows.
        
        Args:
            data: Input DataFrame
            subset: Column names to consider for duplicates
            
        Returns:
            Data without duplicates
        """
        initial_shape = data.shape[0]
        data_cleaned = data.drop_duplicates(subset=subset)
        removed = initial_shape - data_cleaned.shape[0]
        
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")
        
        return data_cleaned
    
    @staticmethod
    def handle_missing_values(
        data: pd.DataFrame,
        strategy: str = "drop"
    ) -> pd.DataFrame:
        """
        Handle missing values.
        
        Args:
            data: Input DataFrame
            strategy: 'drop', 'mean', 'median', or 'forward_fill'
            
        Returns:
            Data with handled missing values
            
        Raises:
            ValueError: If strategy is not valid
        """
        # Validate strategy parameter
        valid_strategies = ["drop", "mean", "median", "forward_fill"]
        if strategy not in valid_strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Valid options: {valid_strategies}")
        
        missing_count = data.isnull().sum().sum()
        
        if missing_count == 0:
            return data
        
        if strategy == "drop":
            data_cleaned = data.dropna()
        elif strategy == "mean":
            data_cleaned = data.fillna(data.mean())
        elif strategy == "median":
            data_cleaned = data.fillna(data.median())
        elif strategy == "forward_fill":
            data_cleaned = data.ffill().bfill()  # Updated to pandas 2.0+ compatible syntax
        
        logger.info(f"Handled {missing_count} missing values using {strategy} strategy")
        return data_cleaned
    
    @staticmethod
    def detect_outliers(
        data: pd.DataFrame,
        method: str = "iqr",
        threshold: float = 1.5
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Detect outliers using IQR or Z-score.
        
        Args:
            data: Input DataFrame
            method: 'iqr' or 'zscore'
            threshold: IQR multiplier (1.5) or z-score threshold (3)
            
        Returns:
            Data without outliers, boolean mask of outliers
            
        Raises:
            ValueError: If method is not valid
        """
        # Validate method parameter
        valid_methods = ["iqr", "zscore"]
        if method not in valid_methods:
            raise ValueError(f"Unknown method: {method}. Valid options: {valid_methods}")
        
        if method == "iqr":
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = ((data < (Q1 - threshold * IQR)) | (data > (Q3 + threshold * IQR))).any(axis=1)
        
        elif method == "zscore":
            from scipy import stats
            z_scores = np.abs(stats.zscore(data.select_dtypes(include=[np.number])))
            outlier_mask = (z_scores > threshold).any(axis=1)
        
        data_cleaned = data[~outlier_mask]
        removed = outlier_mask.sum()
        
        if removed > 0:
            logger.info(f"Detected and removed {removed} outliers using {method}")
        
        return data_cleaned, outlier_mask
    
    @staticmethod
    def remove_constant_features(data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove features with constant values.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Data without constant features
        """
        constant_features = [col for col in data.columns if data[col].nunique() == 1]
        
        if constant_features:
            logger.info(f"Removed {len(constant_features)} constant features")
            data = data.drop(columns=constant_features)
        
        return data
    
    @staticmethod
    def remove_quasi_constant_features(
        data: pd.DataFrame,
        threshold: float = 0.95
    ) -> pd.DataFrame:
        """
        Remove quasi-constant features (dominated by one value).
        
        Args:
            data: Input DataFrame
            threshold: Threshold for removing features
            
        Returns:
            Data without quasi-constant features
        """
        quasi_constant = []
        
        for col in data.columns:
            if data[col].dtype in ['object', 'category']:
                continue
            
            dominant_freq = data[col].value_counts().iloc[0] / len(data)
            if dominant_freq > threshold:
                quasi_constant.append(col)
        
        if quasi_constant:
            logger.info(f"Removed {len(quasi_constant)} quasi-constant features")
            data = data.drop(columns=quasi_constant)
        
        return data


class DataSplitter:
    """Data splitting utilities."""
    
    @staticmethod
    def split_data(
        X: pd.DataFrame,
        y: pd.Series,
        train_size: float = 0.7,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """
        Split data into train, test, and validation sets.
        
        Args:
            X: Features
            y: Labels
            train_size: Training proportion
            test_size: Testing proportion
            val_size: Validation proportion
            random_state: Random seed
            
        Returns:
            Dictionary with split data
        """
        from sklearn.model_selection import train_test_split
        
        if not all(0 <= size < 1 for size in (train_size, test_size, val_size)):
            raise ValueError("Split sizes must be in the range [0, 1).")
        if not np.isclose(train_size + test_size + val_size, 1.0):
            raise ValueError(
                "train_size + test_size + val_size must equal 1.0; "
                f"got {train_size + test_size + val_size:.3f}."
            )
        if y.nunique() < 2:
            raise ValueError("At least two target classes are required for a stratified split.")

        # Split into train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        if val_size == 0:
            logger.info(f"Data split: Train={len(X_temp)}, Test={len(X_test)}")
            return {"train": (X_temp, y_temp), "test": (X_test, y_test)}

        # Split train+val into train and val
        val_ratio = val_size / (train_size + val_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=random_state,
            stratify=y_temp
        )
        
        logger.info(
            f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}"
        )
        
        return {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        }


def preprocess_pipeline(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    remove_duplicates: bool = True,
    handle_missing: str = "drop",
    detect_outliers: bool = True,
    outlier_method: str = "iqr",
    remove_constant: bool = True,
    normalize: bool = True,
    normalization_method: str = "minmax",
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Complete preprocessing pipeline.
    
    Args:
        X: Input features
        y: Labels (optional)
        remove_duplicates: Remove duplicate rows
        handle_missing: Strategy for missing values
        detect_outliers: Detect and remove outliers
        outlier_method: Method for outlier detection
        remove_constant: Remove constant features
        normalize: Normalize features
        normalization_method: Normalization method
        
    Returns:
        Preprocessed features and labels
    """
    with Timer("Complete preprocessing pipeline"):
        X = X.copy().replace([np.inf, -np.inf], np.nan)
        # Remove duplicates
        if remove_duplicates:
            if y is not None:
                combined = pd.concat([X, y], axis=1)
                combined = DataCleaner.remove_duplicates(combined)
                X = combined.iloc[:, :-1]
                y = combined.iloc[:, -1]
            else:
                X = DataCleaner.remove_duplicates(X)
        
        # Handle missing values
        X = DataCleaner.handle_missing_values(X, strategy=handle_missing)
        if y is not None:
            y = y[X.index]
        
        # Detect outliers
        if detect_outliers:
            X, _ = DataCleaner.detect_outliers(X, method=outlier_method)
            if y is not None:
                y = y[X.index]
        
        # Remove constant features
        if remove_constant:
            X = DataCleaner.remove_constant_features(X)
            X = DataCleaner.remove_quasi_constant_features(X)
        
        # Normalize
        if normalize:
            preprocessor = DataPreprocessor(method=normalization_method)
            X = preprocessor.fit_transform(X)
        
        logger.info(f"Preprocessing complete: {X.shape[0]} rows, {X.shape[1]} features")
    
    return X, y


if __name__ == "__main__":
    logger.info("Data preprocessing module loaded")
