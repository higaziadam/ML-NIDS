"""
Model definitions for ML-NIDS including Random Forest, XGBoost, and SVM.
"""

from typing import Optional
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils import logger, save_model, load_model

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - exercised when optional dependency is absent
    XGBClassifier = None


class BaseModel:
    """Base class for all models."""
    
    def __init__(self, model_type: str = "random_forest", **kwargs):
        """
        Initialize base model.
        
        Args:
            model_type: Type of model
            **kwargs: Additional arguments
        """
        self.model_type = model_type
        self.model = None
        self.is_fitted = False
        self.n_features_in_: Optional[int] = None
        self.logger = logger
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model."""
        raise NotImplementedError("Subclasses must implement fit()")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        raise NotImplementedError("Subclasses must implement predict()")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        raise NotImplementedError("Subclasses must implement predict_proba()")
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute model score."""
        raise NotImplementedError("Subclasses must implement score()")
    
    def save(self, filepath: str) -> None:
        """Save model to file."""
        if self.model is None:
            raise ValueError("Model not fitted yet")
        save_model(self.model, filepath)
    
    def load(self, filepath: str) -> None:
        """Load model from file."""
        self.model = load_model(filepath)
        self.is_fitted = True
        self.n_features_in_ = getattr(self.model, "n_features_in_", None)

    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        """Validate feature arrays before passing them to scikit-learn."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Expected a 2D feature matrix, got shape {X.shape}.")
        if self.n_features_in_ is not None and X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}. "
                "Use the same feature schema used for training."
            )
        if not np.isfinite(X).all():
            raise ValueError("Feature matrix contains missing or infinite values.")
        return X


class RandomForestModel(BaseModel):
    """Random Forest classifier for NIDS."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 20,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features: str = "sqrt",
        random_state: int = 42,
        n_jobs: int = -1,
        class_weight: str = "balanced",
        **kwargs
    ):
        """
        Initialize Random Forest model.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split
            min_samples_leaf: Minimum samples per leaf
            max_features: Number of features to consider
            random_state: Random seed
            n_jobs: Number of jobs for parallel processing
            class_weight: Handle class imbalance
        """
        super().__init__(model_type="random_forest")
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight=class_weight,
            verbose=0,
        )
        
        self.logger.info(f"Random Forest model initialized with {n_estimators} estimators")
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Fit Random Forest model."""
        X = self._validate_X(X)
        self.model.fit(X, y)
        self.n_features_in_ = X.shape[1]
        self.is_fitted = True
        self.logger.info("Random Forest model fitted successfully")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.predict(self._validate_X(X))
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.predict_proba(self._validate_X(X))
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute model accuracy."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.score(self._validate_X(X), y)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importances."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.feature_importances_


class XGBoostModel(BaseModel):
    """Native XGBoost classifier for NIDS."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        scale_pos_weight: float = 1.0,
        n_jobs: int = -1,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize XGBoost model.
        
        Args:
            n_estimators: Number of boosting trees
            learning_rate: Learning rate
            max_depth: Maximum tree depth
            subsample: Fraction of rows sampled per tree
            colsample_bytree: Fraction of features sampled per tree
            min_child_weight: Minimum Hessian weight required in a child
            reg_lambda: L2 regularization strength
            scale_pos_weight: Positive-class weight for imbalanced data
            n_jobs: Native training threads (-1 uses all available cores)
            random_state: Random seed
        """
        super().__init__(model_type="xgboost")
        if XGBClassifier is None:
            raise ImportError(
                "XGBoost is required for model_type='xgboost'. "
                "Install it with: python -m pip install xgboost"
            )

        self.model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            reg_lambda=reg_lambda,
            scale_pos_weight=scale_pos_weight,
            n_jobs=n_jobs,
            random_state=random_state,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
        )

        self.logger.info(f"XGBoost model initialized with {n_estimators} estimators")
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Fit XGBoost model."""
        X = self._validate_X(X)
        self.model.fit(X, y)
        self.n_features_in_ = X.shape[1]
        self.is_fitted = True
        self.logger.info("XGBoost model fitted successfully")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.predict(self._validate_X(X))
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.predict_proba(self._validate_X(X))
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute model accuracy."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.score(self._validate_X(X), y)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importances."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.feature_importances_


class SVMModel(BaseModel):
    """Support Vector Machine classifier for NIDS."""
    
    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str = "scale",
        probability: bool = True,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize SVM model.
        
        Args:
            kernel: Kernel type (linear, rbf, poly, sigmoid)
            C: Regularization parameter
            gamma: Kernel coefficient
            probability: Enable probability estimates
            random_state: Random seed
        """
        super().__init__(model_type="svm")
        
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                kernel=kernel,
                C=C,
                gamma=gamma,
                probability=probability,
                random_state=random_state,
                verbose=0,
            ))
        ])
        
        self.logger.info(f"SVM model initialized with {kernel} kernel")
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Fit SVM model."""
        X = self._validate_X(X)
        self.model.fit(X, y)
        self.n_features_in_ = X.shape[1]
        self.is_fitted = True
        self.logger.info("SVM model fitted successfully")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.predict(self._validate_X(X))
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.predict_proba(self._validate_X(X))
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute model accuracy."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.score(self._validate_X(X), y)


class ModelFactory:
    """Factory class for creating models."""
    
    _models = {
        "random_forest": RandomForestModel,
        "xgboost": XGBoostModel,
        "svm": SVMModel,
    }
    
    @classmethod
    def create_model(cls, model_type: str, **kwargs) -> BaseModel:
        """
        Create a model instance.
        
        Args:
            model_type: Type of model
            **kwargs: Model-specific arguments
            
        Returns:
            Model instance
        """
        if model_type not in cls._models:
            available = ", ".join(cls.get_available_models())
            raise ValueError(f"Unknown model type: {model_type}. Available models: {available}")
        
        return cls._models[model_type](**kwargs)
    
    @classmethod
    def get_available_models(cls) -> list:
        """Get list of available models."""
        return list(cls._models.keys())


def create_model(model_type: str, **kwargs) -> BaseModel:
    """
    Convenience function to create a model.
    
    Args:
        model_type: Type of model
        **kwargs: Model-specific arguments
        
    Returns:
        Model instance
    """
    return ModelFactory.create_model(model_type, **kwargs)


if __name__ == "__main__":
    print("Available models:", ModelFactory.get_available_models())
