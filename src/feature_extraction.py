"""
Feature extraction and engineering for network traffic analysis.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

from src.utils import logger, Timer


class FeatureExtractor:
    """Extract and engineer features from network traffic."""
    
    @staticmethod
    def extract_basic_statistics(data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract basic statistical features.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with extracted features
        """
        features = pd.DataFrame(index=data.index)
        
        # Rolling statistics retain one feature row per flow.  Fixed index groups
        # are not meaningful when a DataFrame has a non-RangeIndex.
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        window_size = 10
        for col in numeric_cols:
            rolling = data[col].rolling(window=window_size, min_periods=1)
            features[f"{col}_mean"] = rolling.mean()
            features[f"{col}_std"] = rolling.std().fillna(0.0)
            features[f"{col}_min"] = rolling.min()
            features[f"{col}_max"] = rolling.max()
        
        logger.info(f"Extracted {features.shape[1]} basic statistical features")
        return features
    
    @staticmethod
    def extract_network_features(data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract network-specific features.
        
        Assumes data has columns like: src_ip, dst_ip, src_port, dst_port, protocol, bytes
        
        Args:
            data: Input DataFrame with network traffic data
            
        Returns:
            DataFrame with network features
        """
        features = pd.DataFrame(index=data.index)
        
        # Port-based features
        if 'src_port' in data.columns and 'dst_port' in data.columns:
            features['port_range'] = data['dst_port'] - data['src_port']
            features['src_port_high'] = (data['src_port'] > 1024).astype(int)
            features['dst_port_high'] = (data['dst_port'] > 1024).astype(int)
        
        # Protocol-based features
        if 'protocol' in data.columns:
            features['protocol'] = pd.Categorical(data['protocol']).codes
        
        # Byte-based features
        if 'bytes_sent' in data.columns and 'bytes_received' in data.columns:
            features['total_bytes'] = data['bytes_sent'] + data['bytes_received']
            features['byte_ratio'] = (
                (data['bytes_sent'] + 1) / (data['bytes_received'] + 1)
            ).astype(float)
        
        logger.info(f"Extracted {features.shape[1]} network-specific features")
        return features
    
    @staticmethod
    def create_interaction_features(X: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create interaction features between columns.
        
        Args:
            X: Input DataFrame
            columns: Columns to create interactions for (None = all numeric)
            
        Returns:
            DataFrame with interaction features
        """
        features = X.copy()
        
        if columns is None:
            columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(columns) > 20:  # Limit to avoid explosion
            columns = columns[:20]
        
        interaction_count = 0
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                if col1 in features.columns and col2 in features.columns:
                    features[f"{col1}_x_{col2}"] = features[col1] * features[col2]
                    interaction_count += 1
        
        logger.info(f"Created {interaction_count} interaction features")
        return features


class FeatureSelector:
    """Select most important features."""
    
    def __init__(self, n_features: int = 20, method: str = "f_classif"):
        """
        Initialize FeatureSelector.
        
        Args:
            n_features: Number of features to select
            method: Feature selection method ('f_classif', 'mutual_info')
        """
        self.n_features = n_features
        self.method = method
        self.selector = None
        self.selected_features = None
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureSelector':
        """
        Fit feature selector.
        
        Args:
            X: Input features
            y: Target labels
            
        Returns:
            Self for chaining
        """
        with Timer("Feature selection"):
            if self.method not in {"f_classif", "mutual_info"}:
                raise ValueError("Unknown selection method. Valid options: f_classif, mutual_info")
            if not isinstance(self.n_features, int) or self.n_features < 1:
                raise ValueError("n_features must be a positive integer")
            score_func = f_classif if self.method == "f_classif" else mutual_info_classif
            
            self.selector = SelectKBest(score_func=score_func, k=min(self.n_features, X.shape[1]))
            self.selector.fit(X, y)
            
            # Get selected feature names
            mask = self.selector.get_support()
            self.selected_features = X.columns[mask].tolist()
            self.is_fitted = True
            
            logger.info(f"Selected {len(self.selected_features)} features using {self.method}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using selected features.
        
        Args:
            X: Input features
            
        Returns:
            Data with selected features
        """
        if not self.is_fitted:
            raise ValueError("Selector not fitted yet")
        missing = [column for column in self.selected_features if column not in X.columns]
        if missing:
            raise ValueError(f"Input is missing selected features: {missing}")
        return X[self.selected_features]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform."""
        return self.fit(X, y).transform(X)
    
    def get_feature_scores(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Get feature importance scores."""
        score_func = f_classif if self.method == "f_classif" else mutual_info_classif
        scores = score_func(X, y)
        # f_classif returns (scores, p_values), whereas mutual_info_classif
        # returns the score array directly.
        values = scores[0] if isinstance(scores, tuple) else scores
        return dict(zip(X.columns, values))


class FeatureAnalyzer:
    """Analyze features for quality and correlation."""
    
    @staticmethod
    def check_feature_variance(X: pd.DataFrame, threshold: float = 0.01) -> List[str]:
        """
        Check for low-variance features.
        
        Args:
            X: Input features
            threshold: Variance threshold
            
        Returns:
            List of low-variance features
        """
        from sklearn.feature_selection import VarianceThreshold
        
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)
        
        low_variance = X.columns[~selector.get_support()].tolist()
        
        if low_variance:
            logger.info(f"Found {len(low_variance)} low-variance features")
        
        return low_variance
    
    @staticmethod
    def check_feature_correlation(
        X: pd.DataFrame,
        threshold: float = 0.95
    ) -> List[Tuple[str, str, float]]:
        """
        Find highly correlated features.
        
        Args:
            X: Input features
            threshold: Correlation threshold
            
        Returns:
            List of (feature1, feature2, correlation) tuples
        """
        correlation_matrix = X.corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        correlated_features = [
            (col, row, upper_triangle.loc[row, col])
            for col in upper_triangle.columns
            for row in upper_triangle.index
            if upper_triangle.loc[row, col] > threshold
        ]
        
        if correlated_features:
            logger.info(f"Found {len(correlated_features)} highly correlated feature pairs")
        
        return correlated_features
    
    @staticmethod
    def remove_correlated_features(
        X: pd.DataFrame,
        threshold: float = 0.95
    ) -> pd.DataFrame:
        """
        Remove highly correlated features.
        
        Args:
            X: Input features
            threshold: Correlation threshold
            
        Returns:
            Data without highly correlated features
        """
        correlation_matrix = X.corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > threshold)]
        
        if to_drop:
            X_filtered = X.drop(columns=to_drop)
            logger.info(f"Removed {len(to_drop)} highly correlated features")
            return X_filtered
        
        return X
    
    @staticmethod
    def print_feature_stats(X: pd.DataFrame) -> None:
        """Print feature statistics."""
        print("\n" + "="*60)
        print("FEATURE STATISTICS")
        print("="*60)
        print(f"Number of features: {X.shape[1]}")
        print(f"Number of samples: {X.shape[0]}")
        print(f"\nData types:\n{X.dtypes.value_counts()}")
        print(f"\nMissing values: {X.isnull().sum().sum()}")
        print("\nFeature ranges:")
        print(X.describe())
        print("="*60 + "\n")


def feature_engineering_pipeline(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    create_interactions: bool = False,
    select_features: bool = False,
    n_features: int = 20,
    remove_correlated: bool = True,
    correlation_threshold: float = 0.95,
) -> Tuple[pd.DataFrame, Optional[List[str]]]:
    """
    Complete feature engineering pipeline.
    
    Args:
        X: Input features
        y: Target labels (required for feature selection)
        create_interactions: Create interaction features
        select_features: Select top features
        n_features: Number of features to select
        remove_correlated: Remove highly correlated features
        correlation_threshold: Correlation threshold
        
    Returns:
        Engineered features, selected feature names
    """
    with Timer("Feature engineering pipeline"):
        features = X.copy()
        
        # Remove correlated features
        if remove_correlated:
            features = FeatureAnalyzer.remove_correlated_features(
                features,
                threshold=correlation_threshold
            )
        
        # Create interaction features
        if create_interactions and len(features.columns) < 50:
            features = FeatureExtractor.create_interaction_features(features)
        
        # Select top features
        selected_features = None
        if select_features and y is not None:
            selector = FeatureSelector(n_features=n_features)
            features = selector.fit_transform(features, y)
            selected_features = selector.selected_features
        
        logger.info(f"Feature engineering complete: {features.shape[1]} features")
    
    return features, selected_features


if __name__ == "__main__":
    logger.info("Feature extraction module loaded")
