"""
Advanced Evaluation Metrics for PhD-Level Research
Comprehensive evaluation suite with statistical rigor
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, cohen_kappa_score
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json


class ScoliosisEvaluationMetrics:
    """
    PhD-Level Evaluation Metrics Suite
    
    Implements all standard medical AI evaluation metrics:
    - Classification metrics (accuracy, precision, recall, F1)
    - Clinical metrics (sensitivity, specificity, PPV, NPV)
    - Statistical significance (p-values, confidence intervals)
    - Agreement metrics (Cohen's kappa, ICC)
    - Cobb angle metrics (MAE, RMSE, Bland-Altman)
    """
    
    def __init__(self, save_dir: str = "outputs/evaluation"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def evaluate_classification(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        class_names: List[str] = None
    ) -> Dict:
        """
        Comprehensive classification evaluation
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (for ROC/PR curves)
            class_names: Names of classes
            
        Returns:
            Dictionary with all metrics
        """
        if class_names is None:
            class_names = ['1-derece', '2-derece', '3-derece', 'saglikli']
        
        results = {}
        
        # Basic metrics
        results['accuracy'] = np.mean(y_true == y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm.tolist()
        
        # Per-class metrics
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        results['classification_report'] = report
        
        # Cohen's Kappa (inter-rater agreement)
        results['cohens_kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # Clinical metrics (for binary: scoliosis vs healthy)
        y_true_binary = (y_true != 3).astype(int)  # 3 = 'saglikli'
        y_pred_binary = (y_pred != 3).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
        
        results['clinical_metrics'] = {
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,  # Recall
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,  # Precision
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        }
        
        # ROC-AUC (if probabilities provided)
        if y_prob is not None:
            results['roc_auc'] = self._calculate_roc_auc(y_true, y_prob, class_names)
        
        # Bootstrap confidence intervals
        results['confidence_intervals'] = self._bootstrap_ci(y_true, y_pred)
        
        # Save results
        self.results['classification'] = results
        self._save_confusion_matrix(cm, class_names)
        
        return results
    
    def evaluate_cobb_angle(
        self, 
        true_angles: np.ndarray, 
        pred_angles: np.ndarray,
        create_bland_altman: bool = True
    ) -> Dict:
        """
        Evaluate Cobb angle measurement accuracy
        
        PhD requirement: Clinical agreement analysis
        
        Metrics:
        - MAE (Mean Absolute Error)
        - RMSE (Root Mean Squared Error)
        - Pearson correlation
        - Bland-Altman agreement
        - Within ±5° threshold (clinical standard)
        """
        results = {}
        
        # Basic error metrics
        errors = pred_angles - true_angles
        results['mae'] = np.mean(np.abs(errors))
        results['rmse'] = np.sqrt(np.mean(errors ** 2))
        results['mean_error'] = np.mean(errors)
        results['std_error'] = np.std(errors)
        
        # Correlation
        pearson_r, pearson_p = stats.pearsonr(true_angles, pred_angles)
        results['pearson_r'] = pearson_r
        results['pearson_p'] = pearson_p
        
        # Intraclass Correlation Coefficient (ICC)
        results['icc'] = self._calculate_icc(true_angles, pred_angles)
        
        # Clinical accuracy thresholds
        results['within_5deg'] = np.mean(np.abs(errors) <= 5) * 100  # %
        results['within_10deg'] = np.mean(np.abs(errors) <= 10) * 100
        
        # Bland-Altman analysis
        if create_bland_altman:
            bland_altman_metrics = self._bland_altman_analysis(true_angles, pred_angles)
            results['bland_altman'] = bland_altman_metrics
            self._plot_bland_altman(true_angles, pred_angles)
        
        # Regression plot
        self._plot_regression(true_angles, pred_angles, pearson_r)
        
        self.results['cobb_angle'] = results
        return results
    
    def _calculate_roc_auc(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray,
        class_names: List[str]
    ) -> Dict:
        """Calculate ROC-AUC for multi-class"""
        from sklearn.preprocessing import label_binarize
        
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
        
        # Calculate ROC-AUC for each class
        roc_auc = {}
        
        for i, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc[class_name] = auc(fpr, tpr)
            
            # Plot ROC curve
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc[class_name]:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {class_name}')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(self.save_dir / f'roc_curve_{class_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Macro average
        roc_auc['macro_avg'] = np.mean(list(roc_auc.values()))
        
        return roc_auc
    
    def _bootstrap_ci(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> Dict:
        """
        Bootstrap confidence intervals for accuracy
        
        PhD requirement: Report uncertainty
        """
        np.random.seed(42)
        
        n_samples = len(y_true)
        accuracies = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            acc = np.mean(y_true_boot == y_pred_boot)
            accuracies.append(acc)
        
        # Calculate confidence interval
        alpha = 1 - confidence
        lower = np.percentile(accuracies, alpha/2 * 100)
        upper = np.percentile(accuracies, (1 - alpha/2) * 100)
        
        return {
            'mean': np.mean(accuracies),
            'std': np.std(accuracies),
            'ci_lower': lower,
            'ci_upper': upper,
            'confidence_level': confidence
        }
    
    def _calculate_icc(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate Intraclass Correlation Coefficient (ICC)
        
        Used in clinical studies to assess agreement
        """
        # Combine into 2D array (raters)
        data = np.column_stack([x, y])
        
        # Grand mean
        grand_mean = np.mean(data)
        
        # Between-subject variance
        subject_means = np.mean(data, axis=1)
        bs_var = np.var(subject_means, ddof=1)
        
        # Within-subject variance
        ws_var = np.mean([np.var(row, ddof=1) for row in data])
        
        # ICC(2,1) - Two-way random effects, single measure
        icc = bs_var / (bs_var + ws_var)
        
        return icc
    
    def _bland_altman_analysis(
        self, 
        true_vals: np.ndarray, 
        pred_vals: np.ndarray
    ) -> Dict:
        """
        Bland-Altman analysis for clinical agreement
        
        Standard in medical literature for comparing measurements
        """
        mean_vals = (true_vals + pred_vals) / 2
        diff_vals = pred_vals - true_vals
        
        mean_diff = np.mean(diff_vals)
        std_diff = np.std(diff_vals, ddof=1)
        
        # Limits of agreement (95%)
        loa_upper = mean_diff + 1.96 * std_diff
        loa_lower = mean_diff - 1.96 * std_diff
        
        return {
            'mean_difference': mean_diff,
            'std_difference': std_diff,
            'loa_upper': loa_upper,
            'loa_lower': loa_lower,
            'within_loa': np.mean((diff_vals >= loa_lower) & (diff_vals <= loa_upper)) * 100
        }
    
    def _plot_bland_altman(self, true_vals: np.ndarray, pred_vals: np.ndarray):
        """Create Bland-Altman plot"""
        mean_vals = (true_vals + pred_vals) / 2
        diff_vals = pred_vals - true_vals
        
        mean_diff = np.mean(diff_vals)
        std_diff = np.std(diff_vals, ddof=1)
        loa_upper = mean_diff + 1.96 * std_diff
        loa_lower = mean_diff - 1.96 * std_diff
        
        plt.figure(figsize=(10, 6))
        plt.scatter(mean_vals, diff_vals, alpha=0.5, s=30)
        plt.axhline(mean_diff, color='blue', linestyle='--', label=f'Mean: {mean_diff:.2f}°')
        plt.axhline(loa_upper, color='red', linestyle='--', label=f'Upper LoA: {loa_upper:.2f}°')
        plt.axhline(loa_lower, color='red', linestyle='--', label=f'Lower LoA: {loa_lower:.2f}°')
        plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
        
        plt.xlabel('Mean of True and Predicted Cobb Angle (°)', fontsize=12)
        plt.ylabel('Difference (Predicted - True) (°)', fontsize=12)
        plt.title('Bland-Altman Plot: Agreement Analysis', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.savefig(self.save_dir / 'bland_altman_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_regression(self, true_vals: np.ndarray, pred_vals: np.ndarray, r: float):
        """Create regression plot"""
        plt.figure(figsize=(8, 8))
        
        # Scatter plot
        plt.scatter(true_vals, pred_vals, alpha=0.5, s=30, label='Predictions')
        
        # Perfect prediction line
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect prediction')
        
        # Regression line
        z = np.polyfit(true_vals, pred_vals, 1)
        p = np.poly1d(z)
        plt.plot(true_vals, p(true_vals), 'r-', label=f'Linear fit (r={r:.3f})')
        
        plt.xlabel('True Cobb Angle (°)', fontsize=12)
        plt.ylabel('Predicted Cobb Angle (°)', fontsize=12)
        plt.title('Cobb Angle Prediction Accuracy', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.axis('equal')
        
        plt.savefig(self.save_dir / 'regression_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_confusion_matrix(self, cm: np.ndarray, class_names: List[str]):
        """Save confusion matrix visualization"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, report_name: str = "evaluation_report.json"):
        """
        Generate comprehensive evaluation report
        
        PhD requirement: Detailed documentation
        """
        report_path = self.save_dir / report_name
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n{'='*70}")
        print("EVALUATION REPORT")
        print('='*70)
        
        if 'classification' in self.results:
            cls_results = self.results['classification']
            print(f"\nClassification Metrics:")
            print(f"  Accuracy: {cls_results['accuracy']:.4f}")
            print(f"  Cohen's Kappa: {cls_results['cohens_kappa']:.4f}")
            print(f"\nClinical Metrics:")
            for metric, value in cls_results['clinical_metrics'].items():
                print(f"  {metric.capitalize()}: {value:.4f}")
            
            print(f"\nConfidence Interval (95%):")
            ci = cls_results['confidence_intervals']
            print(f"  Accuracy: {ci['mean']:.4f} [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
        
        if 'cobb_angle' in self.results:
            cobb_results = self.results['cobb_angle']
            print(f"\nCobb Angle Metrics:")
            print(f"  MAE: {cobb_results['mae']:.2f}°")
            print(f"  RMSE: {cobb_results['rmse']:.2f}°")
            print(f"  Pearson r: {cobb_results['pearson_r']:.4f} (p={cobb_results['pearson_p']:.4e})")
            print(f"  ICC: {cobb_results['icc']:.4f}")
            print(f"  Within ±5°: {cobb_results['within_5deg']:.1f}%")
            print(f"  Within ±10°: {cobb_results['within_10deg']:.1f}%")
        
        print(f"\n{'='*70}")
        print(f"Full report saved to: {report_path}")
        print(f"Plots saved to: {self.save_dir}/")
        print('='*70 + "\n")


def demo_evaluation():
    """Demonstrate evaluation metrics"""
    print("="*70)
    print("PhD-Level Evaluation Metrics Demo")
    print("="*70)
    
    # Simulate predictions
    np.random.seed(42)
    n_samples = 200
    
    # Classification (4 classes)
    y_true = np.random.randint(0, 4, n_samples)
    y_pred = y_true.copy()
    y_pred[np.random.choice(n_samples, 30, replace=False)] = np.random.randint(0, 4, 30)  # 15% error
    
    # Probabilities
    y_prob = np.random.dirichlet(np.ones(4), n_samples)
    
    # Cobb angles
    true_angles = np.random.uniform(5, 60, n_samples)
    pred_angles = true_angles + np.random.normal(0, 3, n_samples)  # ±3° error
    
    # Evaluate
    evaluator = ScoliosisEvaluationMetrics(save_dir="outputs/demo_evaluation")
    
    cls_results = evaluator.evaluate_classification(y_true, y_pred, y_prob)
    cobb_results = evaluator.evaluate_cobb_angle(true_angles, pred_angles)
    
    evaluator.generate_report()


if __name__ == "__main__":
    demo_evaluation()
