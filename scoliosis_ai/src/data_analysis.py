"""
Data Science Analysis Module for Scoliosis AI
Provides statistical analysis, visualizations, and metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, mean_absolute_error, mean_squared_error, r2_score
)
from scipy import stats
import json


class ScoliosisDataAnalyzer:
    """Data analysis and visualization for scoliosis detection"""
    
    def __init__(self, output_dir="outputs/analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10
    
    def analyze_dataset(self, data_yaml_path):
        """Analyze dataset statistics"""
        try:
            import yaml
            with open(data_yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
            
            stats = {
                'num_classes': data_config.get('nc', 0),
                'class_names': data_config.get('names', []),
                'train_path': data_config.get('train', ''),
                'val_path': data_config.get('val', ''),
            }
            
            # Count images in train/val
            train_images = list(Path(data_config.get('train', '')).glob('*.jpg')) if Path(data_config.get('train', '')).exists() else []
            val_images = list(Path(data_config.get('val', '')).glob('*.jpg')) if Path(data_config.get('val', '')).exists() else []
            
            stats['train_images'] = len(train_images)
            stats['val_images'] = len(val_images)
            stats['total_images'] = len(train_images) + len(val_images)
            
            return stats
        except Exception as e:
            print(f"Error analyzing dataset: {e}")
            return None
    
    def plot_training_history(self, results_csv_path, save_path=None):
        """Plot training metrics over epochs"""
        try:
            df = pd.read_csv(results_csv_path)
            df.columns = df.columns.str.strip()  # Remove whitespace
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Training History', fontsize=16, fontweight='bold')
            
            # Loss curves
            if 'train/box_loss' in df.columns:
                axes[0, 0].plot(df.index, df['train/box_loss'], label='Box Loss', linewidth=2)
                axes[0, 0].set_title('Training Losses')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            # Precision/Recall
            if 'metrics/precision(B)' in df.columns and 'metrics/recall(B)' in df.columns:
                axes[0, 1].plot(df.index, df['metrics/precision(B)'], label='Precision', linewidth=2)
                axes[0, 1].plot(df.index, df['metrics/recall(B)'], label='Recall', linewidth=2)
                axes[0, 1].set_title('Precision & Recall')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Score')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # mAP
            if 'metrics/mAP50(B)' in df.columns and 'metrics/mAP50-95(B)' in df.columns:
                axes[1, 0].plot(df.index, df['metrics/mAP50(B)'], label='mAP@50', linewidth=2)
                axes[1, 0].plot(df.index, df['metrics/mAP50-95(B)'], label='mAP@50-95', linewidth=2)
                axes[1, 0].set_title('Mean Average Precision (mAP)')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('mAP')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # Learning rate
            if 'lr/pg0' in df.columns:
                axes[1, 1].plot(df.index, df['lr/pg0'], linewidth=2, color='green')
                axes[1, 1].set_title('Learning Rate')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Learning Rate')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.savefig(self.output_dir / 'training_history.png', dpi=300, bbox_inches='tight')
            
            plt.close()
            return True
        except Exception as e:
            print(f"Error plotting training history: {e}")
            return False
    
    def generate_confusion_matrix(self, y_true, y_pred, class_names, save_path=None):
        """Generate confusion matrix visualization"""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        
        plt.close()
        return cm
    
    def generate_roc_curves(self, y_true, y_scores, class_names, save_path=None):
        """Generate ROC curves for multi-class classification"""
        from sklearn.preprocessing import label_binarize
        
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['blue', 'red', 'green', 'orange']
        for i, (color, class_name) in enumerate(zip(colors, class_names)):
            if i < y_scores.shape[1]:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=color, lw=2,
                       label=f'{class_name} (AUC = {roc_auc:.2f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - Multi-Class Classification', fontsize=16, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def bland_altman_plot(self, method1, method2, save_path=None):
        """Generate Bland-Altman plot for agreement analysis"""
        mean = np.mean([method1, method2], axis=0)
        diff = method1 - method2
        md = np.mean(diff)
        sd = np.std(diff, ddof=1)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(mean, diff, alpha=0.6, s=50)
        ax.axhline(md, color='red', linestyle='--', linewidth=2, label=f'Mean Diff = {md:.2f}')
        ax.axhline(md + 1.96*sd, color='gray', linestyle='--', linewidth=2, label=f'+1.96 SD = {md + 1.96*sd:.2f}')
        ax.axhline(md - 1.96*sd, color='gray', linestyle='--', linewidth=2, label=f'-1.96 SD = {md - 1.96*sd:.2f}')
        
        ax.set_xlabel('Mean of Two Methods', fontsize=12)
        ax.set_ylabel('Difference Between Methods', fontsize=12)
        ax.set_title('Bland-Altman Plot - Agreement Analysis', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'bland_altman.png', dpi=300, bbox_inches='tight')
        
        plt.close()
        
        # Calculate ICC
        icc = self.calculate_icc(method1, method2)
        
        return {
            'mean_difference': md,
            'std_difference': sd,
            'upper_limit': md + 1.96*sd,
            'lower_limit': md - 1.96*sd,
            'icc': icc
        }
    
    def calculate_icc(self, method1, method2):
        """Calculate Intraclass Correlation Coefficient"""
        data = np.column_stack([method1, method2])
        
        # Mean squares
        msr = np.var(np.mean(data, axis=1), ddof=1) * data.shape[1]
        msw = np.mean(np.var(data, axis=1, ddof=1))
        
        # ICC(2,1)
        icc = (msr - msw) / (msr + msw)
        
        return icc
    
    def cobb_angle_statistics(self, angles_dict, save_path=None):
        """Analyze Cobb angle statistics and distribution"""
        angles = list(angles_dict.values())
        
        stats_dict = {
            'count': len(angles),
            'mean': np.mean(angles),
            'median': np.median(angles),
            'std': np.std(angles, ddof=1),
            'min': np.min(angles),
            'max': np.max(angles),
            'q25': np.percentile(angles, 25),
            'q75': np.percentile(angles, 75)
        }
        
        # Severity classification
        severity = {
            'Normal (< 10°)': sum(1 for a in angles if a < 10),
            'Mild (10-25°)': sum(1 for a in angles if 10 <= a < 25),
            'Moderate (25-40°)': sum(1 for a in angles if 25 <= a < 40),
            'Severe (≥ 40°)': sum(1 for a in angles if a >= 40)
        }
        
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Histogram
        axes[0].hist(angles, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(stats_dict['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean = {stats_dict['mean']:.1f}°")
        axes[0].axvline(stats_dict['median'], color='green', linestyle='--', linewidth=2, label=f"Median = {stats_dict['median']:.1f}°")
        axes[0].set_xlabel('Cobb Angle (degrees)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Cobb Angle Distribution', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot(angles, vert=True)
        axes[1].set_ylabel('Cobb Angle (degrees)', fontsize=12)
        axes[1].set_title('Cobb Angle Box Plot', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Severity pie chart
        axes[2].pie(severity.values(), labels=severity.keys(), autopct='%1.1f%%',
                   colors=['green', 'yellow', 'orange', 'red'], startangle=90)
        axes[2].set_title('Severity Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'cobb_angle_stats.png', dpi=300, bbox_inches='tight')
        
        plt.close()
        
        return {**stats_dict, 'severity_distribution': severity}
    
    def model_comparison(self, models_results, save_path=None):
        """Compare multiple models performance"""
        df = pd.DataFrame(models_results)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Accuracy comparison
        if 'accuracy' in df.columns:
            axes[0, 0].bar(df['model'], df['accuracy'], color='skyblue', edgecolor='black')
            axes[0, 0].set_ylabel('Accuracy', fontsize=12)
            axes[0, 0].set_title('Model Accuracy', fontsize=14)
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Precision/Recall comparison
        if 'precision' in df.columns and 'recall' in df.columns:
            x = np.arange(len(df))
            width = 0.35
            axes[0, 1].bar(x - width/2, df['precision'], width, label='Precision', color='green', alpha=0.7)
            axes[0, 1].bar(x + width/2, df['recall'], width, label='Recall', color='blue', alpha=0.7)
            axes[0, 1].set_ylabel('Score', fontsize=12)
            axes[0, 1].set_title('Precision & Recall', fontsize=14)
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(df['model'], rotation=45)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # F1 Score comparison
        if 'f1_score' in df.columns:
            axes[1, 0].bar(df['model'], df['f1_score'], color='orange', edgecolor='black')
            axes[1, 0].set_ylabel('F1 Score', fontsize=12)
            axes[1, 0].set_title('F1 Score', fontsize=14)
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Inference time comparison
        if 'inference_time' in df.columns:
            axes[1, 1].bar(df['model'], df['inference_time'], color='red', edgecolor='black')
            axes[1, 1].set_ylabel('Time (ms)', fontsize=12)
            axes[1, 1].set_title('Inference Time', fontsize=14)
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def generate_report(self, data, report_path=None):
        """Generate comprehensive statistical report"""
        report_text = f"""
SCOLIOSIS AI - DATA ANALYSIS REPORT
{'='*60}

Dataset Statistics:
------------------
Total Samples: {data.get('total_samples', 'N/A')}
Training Samples: {data.get('train_samples', 'N/A')}
Validation Samples: {data.get('val_samples', 'N/A')}

Model Performance:
-----------------
Accuracy: {data.get('accuracy', 'N/A')}
Precision: {data.get('precision', 'N/A')}
Recall: {data.get('recall', 'N/A')}
F1 Score: {data.get('f1_score', 'N/A')}
mAP@50: {data.get('mAP50', 'N/A')}
mAP@50-95: {data.get('mAP50_95', 'N/A')}

Cobb Angle Analysis:
-------------------
Mean Angle: {data.get('mean_angle', 'N/A')}°
Median Angle: {data.get('median_angle', 'N/A')}°
Std Deviation: {data.get('std_angle', 'N/A')}°
Range: {data.get('min_angle', 'N/A')}° - {data.get('max_angle', 'N/A')}°

Agreement Analysis:
------------------
ICC: {data.get('icc', 'N/A')}
Mean Difference: {data.get('mean_diff', 'N/A')}°
95% Limits of Agreement: [{data.get('loa_lower', 'N/A')}°, {data.get('loa_upper', 'N/A')}°]

{'='*60}
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        if report_path:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_text.strip())
        else:
            report_path = self.output_dir / 'analysis_report.txt'
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_text.strip())
        
        return report_text.strip()


# Convenience functions
def quick_analysis(results_csv_path):
    """Quick training analysis"""
    analyzer = ScoliosisDataAnalyzer()
    return analyzer.plot_training_history(results_csv_path)


def analyze_cobb_angles(angles_file):
    """Analyze Cobb angles from JSON file"""
    with open(angles_file, 'r') as f:
        angles_dict = json.load(f)
    
    analyzer = ScoliosisDataAnalyzer()
    return analyzer.cobb_angle_statistics(angles_dict)
