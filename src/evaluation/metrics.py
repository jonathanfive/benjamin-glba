import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve, matthews_corrcoef,
    balanced_accuracy_score, cohen_kappa_score
)
from sklearn.preprocessing import label_binarize
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class GLBAMetrics:
    """
    Comprehensive evaluation metrics for GLBA violation detection model.
    
    Provides both standard ML metrics and domain-specific evaluation
    for financial compliance text classification.
    """
    
    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names or [
            'No Violation', 
            'GLBA Violation', 
            'Safeguards Violation', 
            'Pretexting Violation'
        ]
        self.num_classes = len(self.class_names)
        
    def compute_basic_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = 'weighted'
    ) -> Dict[str, float]:
        """Compute basic classification metrics."""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
            'cohen_kappa': cohen_kappa_score(y_true, y_pred)
        }
        
        return metrics
    
    def compute_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-class metrics."""
        
        precision_scores = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_scores = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_scores = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        per_class_metrics = {}
        
        for i, class_name in enumerate(self.class_names):
            if i < len(precision_scores):
                per_class_metrics[class_name] = {
                    'precision': precision_scores[i],
                    'recall': recall_scores[i],
                    'f1_score': f1_scores[i]
                }
        
        return per_class_metrics
    
    def compute_confusion_matrix_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Compute confusion matrix and derived metrics."""
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate per-class metrics from confusion matrix
        class_metrics = {}
        
        for i, class_name in enumerate(self.class_names):
            if i < cm.shape[0]:
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                tn = cm.sum() - tp - fp - fn
                
                # Handle division by zero
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                
                class_metrics[class_name] = {
                    'true_positives': int(tp),
                    'false_positives': int(fp),
                    'true_negatives': int(tn),
                    'false_negatives': int(fn),
                    'precision': precision,
                    'recall': recall,
                    'specificity': specificity
                }
        
        return {
            'confusion_matrix': cm.tolist(),
            'class_metrics': class_metrics
        }
    
    def compute_roc_auc_metrics(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, float]:
        """Compute ROC AUC scores."""
        
        if len(y_proba.shape) == 1 or y_proba.shape[1] == 2:
            # Binary classification
            if y_proba.shape[1] == 2:
                y_scores = y_proba[:, 1]
            else:
                y_scores = y_proba
            
            auc_score = roc_auc_score(y_true, y_scores)
            return {'roc_auc': auc_score}
        
        else:
            # Multi-class classification
            try:
                # Compute one-vs-rest AUC
                auc_ovr = roc_auc_score(y_true, y_proba, average='weighted', multi_class='ovr')
                
                # Compute per-class AUC
                y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
                per_class_auc = {}
                
                for i, class_name in enumerate(self.class_names):
                    if i < y_proba.shape[1] and i < y_true_bin.shape[1]:
                        try:
                            auc = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
                            per_class_auc[f'auc_{class_name.lower().replace(" ", "_")}'] = auc
                        except ValueError:
                            # Handle case where class doesn't appear in y_true
                            per_class_auc[f'auc_{class_name.lower().replace(" ", "_")}'] = 0.0
                
                return {
                    'roc_auc_weighted': auc_ovr,
                    **per_class_auc
                }
                
            except ValueError as e:
                logger.warning(f"Could not compute ROC AUC: {e}")
                return {'roc_auc_weighted': 0.0}
    
    def compute_domain_specific_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute domain-specific metrics for GLBA violation detection.
        
        Focus on metrics important for compliance and regulatory use cases.
        """
        
        metrics = {}
        
        # Violation detection rate (recall for violation classes)
        violation_classes = [1, 2, 3]  # GLBA, Safeguards, Pretexting
        
        violation_mask = np.isin(y_true, violation_classes)
        if violation_mask.sum() > 0:
            violation_recall = recall_score(
                y_true[violation_mask], 
                y_pred[violation_mask], 
                labels=violation_classes,
                average='weighted',
                zero_division=0
            )
            metrics['violation_detection_rate'] = violation_recall
        
        # False positive rate for "No Violation" class
        no_violation_mask = y_true == 0
        if no_violation_mask.sum() > 0:
            fp_rate = (y_pred[no_violation_mask] != 0).mean()
            metrics['false_positive_rate'] = fp_rate
        
        # Precision for each violation type (critical for compliance)
        glba_precision = precision_score(y_true, y_pred, labels=[1], average='weighted', zero_division=0)
        safeguards_precision = precision_score(y_true, y_pred, labels=[2], average='weighted', zero_division=0)
        pretexting_precision = precision_score(y_true, y_pred, labels=[3], average='weighted', zero_division=0)
        
        metrics.update({
            'glba_precision': glba_precision,
            'safeguards_precision': safeguards_precision,
            'pretexting_precision': pretexting_precision
        })
        
        # High confidence prediction rate
        if y_proba is not None:
            high_confidence_threshold = 0.8
            max_probs = np.max(y_proba, axis=1)
            high_confidence_rate = (max_probs >= high_confidence_threshold).mean()
            metrics['high_confidence_prediction_rate'] = high_confidence_rate
            
            # Calibration: accuracy of high-confidence predictions
            high_conf_mask = max_probs >= high_confidence_threshold
            if high_conf_mask.sum() > 0:
                high_conf_accuracy = accuracy_score(
                    y_true[high_conf_mask], 
                    y_pred[high_conf_mask]
                )
                metrics['high_confidence_accuracy'] = high_conf_accuracy
        
        return metrics
    
    def compute_comprehensive_metrics(
        self,
        y_true: Union[np.ndarray, List],
        y_pred: Union[np.ndarray, List],
        y_proba: Optional[Union[np.ndarray, List]] = None
    ) -> Dict[str, Any]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            
        Returns:
            Dictionary containing all computed metrics
        """
        
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if y_proba is not None:
            y_proba = np.array(y_proba)
        
        # Validate inputs
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        if y_proba is not None and len(y_proba) != len(y_true):
            raise ValueError("y_proba must have the same length as y_true")
        
        metrics = {}
        
        # Basic metrics
        metrics['basic'] = self.compute_basic_metrics(y_true, y_pred)
        
        # Per-class metrics
        metrics['per_class'] = self.compute_per_class_metrics(y_true, y_pred)
        
        # Confusion matrix metrics
        metrics['confusion_matrix'] = self.compute_confusion_matrix_metrics(y_true, y_pred)
        
        # ROC AUC metrics (if probabilities provided)
        if y_proba is not None:
            metrics['roc_auc'] = self.compute_roc_auc_metrics(y_true, y_proba)
        
        # Domain-specific metrics
        metrics['domain_specific'] = self.compute_domain_specific_metrics(
            y_true, y_pred, y_proba
        )
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        return metrics
    
    def compute_metrics(
        self,
        y_true: Union[np.ndarray, List],
        y_pred: Union[np.ndarray, List],
        y_proba: Optional[Union[np.ndarray, List]] = None
    ) -> Dict[str, float]:
        """
        Compute key metrics for model evaluation (flattened format).
        
        Returns a flat dictionary suitable for logging and comparison.
        """
        
        comprehensive = self.compute_comprehensive_metrics(y_true, y_pred, y_proba)
        
        # Flatten important metrics
        flat_metrics = {}
        
        # Basic metrics
        flat_metrics.update(comprehensive['basic'])
        
        # Domain-specific metrics
        flat_metrics.update(comprehensive['domain_specific'])
        
        # Key per-class metrics
        for class_name, metrics in comprehensive['per_class'].items():
            clean_name = class_name.lower().replace(' ', '_').replace('-', '_')
            flat_metrics[f'{clean_name}_f1'] = metrics['f1_score']
            flat_metrics[f'{clean_name}_precision'] = metrics['precision']
            flat_metrics[f'{clean_name}_recall'] = metrics['recall']
        
        # ROC AUC if available
        if 'roc_auc' in comprehensive:
            flat_metrics.update(comprehensive['roc_auc'])
        
        return flat_metrics
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: bool = False,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (8, 6)
    ) -> plt.Figure:
        """Plot confusion matrix."""
        
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax
        )
        
        ax.set_title(title)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        return fig
    
    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """Plot ROC curves for each class."""
        
        if len(y_proba.shape) == 1 or y_proba.shape[1] == 2:
            # Binary classification
            fig, ax = plt.subplots(figsize=figsize)
            
            if y_proba.shape[1] == 2:
                y_scores = y_proba[:, 1]
            else:
                y_scores = y_proba
            
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            auc_score = roc_auc_score(y_true, y_scores)
            
            ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
            ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        else:
            # Multi-class classification
            y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
            
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            axes = axes.ravel()
            
            for i, class_name in enumerate(self.class_names):
                if i >= len(axes) or i >= y_proba.shape[1]:
                    break
                
                if i < y_true_bin.shape[1]:
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                    auc_score = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
                    
                    axes[i].plot(fpr, tpr, label=f'{class_name} (AUC = {auc_score:.3f})')
                    axes[i].plot([0, 1], [0, 1], 'k--', alpha=0.5)
                    axes[i].set_xlabel('False Positive Rate')
                    axes[i].set_ylabel('True Positive Rate')
                    axes[i].set_title(f'ROC Curve - {class_name}')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")
        
        return fig
    
    def plot_precision_recall_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """Plot precision-recall curves for each class."""
        
        if len(y_proba.shape) == 1 or y_proba.shape[1] == 2:
            # Binary classification
            fig, ax = plt.subplots(figsize=figsize)
            
            if y_proba.shape[1] == 2:
                y_scores = y_proba[:, 1]
            else:
                y_scores = y_proba
            
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            
            ax.plot(recall, precision, label='Precision-Recall Curve')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        else:
            # Multi-class classification
            y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
            
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            axes = axes.ravel()
            
            for i, class_name in enumerate(self.class_names):
                if i >= len(axes) or i >= y_proba.shape[1]:
                    break
                
                if i < y_true_bin.shape[1]:
                    precision, recall, _ = precision_recall_curve(
                        y_true_bin[:, i], y_proba[:, i]
                    )
                    
                    axes[i].plot(recall, precision, label=f'{class_name}')
                    axes[i].set_xlabel('Recall')
                    axes[i].set_ylabel('Precision')
                    axes[i].set_title(f'Precision-Recall Curve - {class_name}')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-recall curves saved to {save_path}")
        
        return fig
    
    def save_metrics_report(
        self,
        metrics: Dict[str, Any],
        output_path: Path,
        include_plots: bool = True
    ):
        """Save comprehensive metrics report."""
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics as JSON
        metrics_file = output_path / 'metrics.json'
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = self._make_json_serializable(metrics)
        
        with open(metrics_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        logger.info(f"Metrics report saved to {metrics_file}")
        
        # Save human-readable report
        report_file = output_path / 'metrics_report.txt'
        self._write_human_readable_report(metrics, report_file)
        
        logger.info(f"Human-readable report saved to {report_file}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to serializable format."""
        
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        else:
            return obj
    
    def _write_human_readable_report(self, metrics: Dict[str, Any], output_path: Path):
        """Write human-readable metrics report."""
        
        with open(output_path, 'w') as f:
            f.write("GLBA Violation Detection Model - Evaluation Report\n")
            f.write("=" * 60 + "\n\n")
            
            # Basic metrics
            if 'basic' in metrics:
                f.write("BASIC METRICS\n")
                f.write("-" * 20 + "\n")
                for metric, value in metrics['basic'].items():
                    f.write(f"{metric:25s}: {value:.4f}\n")
                f.write("\n")
            
            # Domain-specific metrics
            if 'domain_specific' in metrics:
                f.write("DOMAIN-SPECIFIC METRICS\n")
                f.write("-" * 25 + "\n")
                for metric, value in metrics['domain_specific'].items():
                    f.write(f"{metric:30s}: {value:.4f}\n")
                f.write("\n")
            
            # Per-class metrics
            if 'per_class' in metrics:
                f.write("PER-CLASS METRICS\n")
                f.write("-" * 20 + "\n")
                for class_name, class_metrics in metrics['per_class'].items():
                    f.write(f"\n{class_name}:\n")
                    for metric, value in class_metrics.items():
                        f.write(f"  {metric:15s}: {value:.4f}\n")
                f.write("\n")
            
            # Confusion matrix summary
            if 'confusion_matrix' in metrics:
                f.write("CONFUSION MATRIX SUMMARY\n")
                f.write("-" * 25 + "\n")
                cm = metrics['confusion_matrix']['confusion_matrix']
                f.write("Confusion Matrix:\n")
                for i, row in enumerate(cm):
                    row_str = " ".join(f"{val:6d}" for val in row)
                    f.write(f"  {self.class_names[i]:20s}: {row_str}\n")
                f.write("\n")


def create_glba_metrics(class_names: Optional[List[str]] = None) -> GLBAMetrics:
    """Create GLBAMetrics instance with default or custom class names."""
    
    return GLBAMetrics(class_names=class_names)