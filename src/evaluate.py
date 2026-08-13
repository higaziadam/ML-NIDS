"""
Model evaluation metrics and analysis for NIDS.
"""

from typing import Any, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from pathlib import Path

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
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize

from src.utils import logger, save_data, Timer


class ModelEvaluator:
    """Evaluate model performance."""
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize ModelEvaluator.
        
        Args:
            threshold: Classification threshold for binary classification
        """
        self.threshold = threshold
        self.results = {}
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels (0/1)
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        with Timer("Model evaluation"):
            metrics = {}
            
            labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
            is_binary = len(labels) == 2
            average = "binary" if is_binary else "weighted"
            metric_kwargs: Dict[str, Any] = {"average": average, "zero_division": 0}
            if is_binary:
                # scikit-learn defaults to ``pos_label=1``, which fails for
                # legitimate string labels such as Benign/Attack.
                metric_kwargs["pos_label"] = labels[-1]

            # Classification metrics
            metrics["accuracy"] = accuracy_score(y_true, y_pred)
            metrics["precision"] = precision_score(y_true, y_pred, **metric_kwargs)
            metrics["recall"] = recall_score(y_true, y_pred, **metric_kwargs)
            metrics["f1"] = f1_score(y_true, y_pred, **metric_kwargs)
            metrics["labels"] = labels.tolist()
            
            # ROC-AUC
            if y_pred_proba is not None:
                try:
                    if is_binary and len(y_pred_proba.shape) > 1:
                        y_proba = y_pred_proba[:, 1]
                    elif not is_binary and len(y_pred_proba.shape) > 1:
                        metrics["roc_auc"] = roc_auc_score(
                            y_true, y_pred_proba, labels=labels, multi_class="ovr", average="weighted"
                        )
                        y_true_binarized = label_binarize(y_true, classes=labels)
                        metrics["pr_auc"] = average_precision_score(
                            y_true_binarized, y_pred_proba, average="weighted"
                        )
                        y_proba = None
                    else:
                        y_proba = y_pred_proba
                    if y_proba is not None:
                        y_true_binary = (np.asarray(y_true) == labels[-1]).astype(int)
                        metrics["roc_auc"] = roc_auc_score(y_true_binary, y_proba)
                        metrics["pr_auc"] = average_precision_score(y_true_binary, y_proba)
                except Exception as e:
                    logger.warning(f"Could not compute AUC: {e}")
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            metrics["confusion_matrix"] = cm.tolist()
            if is_binary:
                tn, fp, fn, tp = cm.ravel()
                metrics["true_negatives"] = int(tn)
                metrics["false_positives"] = int(fp)
                metrics["false_negatives"] = int(fn)
                metrics["true_positives"] = int(tp)
                metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
                metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0
                metrics["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0
                metrics["fnr"] = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            self.results = metrics
            logger.info("Model evaluation complete")
        
        return metrics
    
    def print_metrics(self) -> None:
        """Print formatted metrics."""
        if not self.results:
            logger.warning("No evaluation results available")
            return
        
        print("\n" + "="*60)
        print("MODEL EVALUATION METRICS")
        print("="*60)
        print(f"Accuracy ............... {self.results.get('accuracy', 0):.4f}")
        print(f"Precision .............. {self.results.get('precision', 0):.4f}")
        print(f"Recall ................. {self.results.get('recall', 0):.4f}")
        print(f"F1-Score ............... {self.results.get('f1', 0):.4f}")
        print(f"ROC-AUC ................ {self.results.get('roc_auc', 0):.4f}")
        print(f"PR-AUC ................. {self.results.get('pr_auc', 0):.4f}")
        if len(self.results.get("labels", [])) == 2:
            print(f"\nSpecificity ............ {self.results.get('specificity', 0):.4f}")
            print(f"Sensitivity ............ {self.results.get('sensitivity', 0):.4f}")
            print(f"False Positive Rate .... {self.results.get('fpr', 0):.4f}")
            print(f"False Negative Rate .... {self.results.get('fnr', 0):.4f}")
            print("\nConfusion Matrix:")
            print(f"  True Negatives ....... {self.results.get('true_negatives', 0)}")
            print(f"  False Positives ...... {self.results.get('false_positives', 0)}")
            print(f"  False Negatives ...... {self.results.get('false_negatives', 0)}")
            print(f"  True Positives ....... {self.results.get('true_positives', 0)}")
        else:
            print(f"\nClasses ............... {self.results.get('labels', [])}")
        print("="*60 + "\n")
    
    def get_metrics_dataframe(self) -> pd.DataFrame:
        """Get metrics as DataFrame."""
        return pd.DataFrame([self.results])


class ConfusionMatrixAnalyzer:
    """Analyze confusion matrix."""
    
    @staticmethod
    def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute confusion matrix."""
        return confusion_matrix(y_true, y_pred)
    
    @staticmethod
    def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Print confusion matrix."""
        labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        if len(labels) != 2:
            print("\nCONFUSION MATRIX")
            print(pd.DataFrame(cm, index=labels, columns=labels))
            return
        print("\n" + "="*40)
        print("CONFUSION MATRIX")
        print("="*40)
        print("                Predicted")
        print("                Negative  Positive")
        print(f"Actual Negative  {cm[0,0]:>6}    {cm[0,1]:>6}")
        print(f"       Positive  {cm[1,0]:>6}    {cm[1,1]:>6}")
        print("="*40 + "\n")


class ClassificationReportAnalyzer:
    """Analyze classification report."""
    
    @staticmethod
    def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Get classification report as dictionary."""
        return classification_report(y_true, y_pred, output_dict=True)
    
    @staticmethod
    def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Print classification report."""
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(y_true, y_pred, digits=4))
        print("="*60 + "\n")


class ROCAnalyzer:
    """Analyze ROC curve."""
    
    @staticmethod
    def compute_roc_curve(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            False positive rates, true positive rates, thresholds
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        return fpr, tpr, thresholds
    
    @staticmethod
    def compute_roc_auc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Compute ROC AUC score."""
        return roc_auc_score(y_true, y_pred_proba)


class PrecisionRecallAnalyzer:
    """Analyze precision-recall curve."""
    
    @staticmethod
    def compute_pr_curve(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute precision-recall curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Precisions, recalls, thresholds
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        return precision, recall, thresholds
    
    @staticmethod
    def compute_pr_auc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Compute PR AUC score."""
        return average_precision_score(y_true, y_pred_proba)


def comprehensive_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    save_results: bool = False,
    results_path: Optional[Path] = None,
) -> Dict[str, any]:
    """
    Comprehensive model evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        save_results: Save results to file
        results_path: Path to save results
        
    Returns:
        Dictionary with all evaluation results
    """
    with Timer("Comprehensive evaluation"):
        results = {}
        
        # Evaluator
        evaluator = ModelEvaluator()
        results["metrics"] = evaluator.evaluate(y_true, y_pred, y_pred_proba)
        evaluator.print_metrics()
        
        # Confusion matrix
        labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        ConfusionMatrixAnalyzer.print_confusion_matrix(y_true, y_pred)
        results["confusion_matrix"] = cm.tolist()
        
        # Classification report
        ClassificationReportAnalyzer.print_classification_report(y_true, y_pred)
        results["classification_report"] = ClassificationReportAnalyzer.get_classification_report(
            y_true, y_pred
        )
        
        # ROC curve
        if y_pred_proba is not None and len(labels) == 2:
            if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] >= 2:
                y_proba = y_pred_proba[:, 1]
            else:
                logger.warning("Skipping ROC/PR curves: model returned fewer than two probability columns.")
                y_proba = None

            if y_proba is not None:
                y_true_binary = (np.asarray(y_true) == labels[-1]).astype(int)
                fpr, tpr, thresholds = ROCAnalyzer.compute_roc_curve(y_true_binary, y_proba)
                results["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
                
                # PR curve
                precision, recall, thresholds = PrecisionRecallAnalyzer.compute_pr_curve(y_true_binary, y_proba)
                results["pr_curve"] = {"precision": precision.tolist(), "recall": recall.tolist()}
        
        # Save results
        if save_results and results_path:
            results_path = Path(results_path)
            results_path.mkdir(parents=True, exist_ok=True)
            
            # Save metrics
            metrics_df = evaluator.get_metrics_dataframe()
            save_data(metrics_df, results_path / "metrics.csv")
            
            # Save confusion matrix
            cm_df = pd.DataFrame(cm, index=labels, columns=labels)
            save_data(cm_df, results_path / "confusion_matrix.csv")
            
            logger.info(f"Results saved to {results_path}")
        
        return results


if __name__ == "__main__":
    logger.info("Evaluation module loaded")
