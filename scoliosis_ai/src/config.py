"""
Configuration module for Scoliosis Detection System
Contains all hyperparameters, paths, and model configurations
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"

# Dataset configuration
DATASET_CONFIG = {
    "root": r"C:\Users\ARUNN\Downloads\Dataset\archive (1)\scoliosis.v1-new-version-1.yolov5pytorch",
    "yaml_path": r"C:\Users\ARUNN\Downloads\Dataset\archive (1)\scoliosis.v1-new-version-1.yolov5pytorch\data.yaml",
    "classes": ['1-derece', '2-derece', '3-derece', 'saglikli'],
    "num_classes": 4,
    "img_size": 640,
}

# YOLOv8 Detection Configuration - 1 HOUR FAST MODE ⚡
YOLO_CONFIG = {
    "model_name": "yolov8s.pt",  # SMALL model for speed (11.2M params)
    "epochs": 30,  # Optimized for 1-hour training
    "batch_size": 16,  # Increased for faster training
    "img_size": 640,  # Standard size for speed/accuracy balance
    "patience": 50,  # More patience for better training
    "optimizer": "AdamW",  # Better optimizer than SGD for accuracy
    "lr0": 0.001,  # Lower LR for stable convergence (vs 0.01)
    "lrf": 0.01,  # Final LR = 0.001 * 0.01 = 0.00001
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 5.0,  # More warmup for stability
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "conf_threshold": 0.25,
    "iou_threshold": 0.45,
    # ADVANCED augmentation for accuracy
    "hsv_h": 0.015,  # Color augmentation
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 5.0,  # Small rotation for robustness
    "translate": 0.1,  # Translation
    "scale": 0.5,  # Scale variation
    "shear": 2.0,  # Shear augmentation
    "perspective": 0.0001,  # Perspective transform
    "flipud": 0.0,  # No vertical flip (spines don't flip vertically)
    "fliplr": 0.5,  # Horizontal flip (left/right is valid)
    "mosaic": 1.0,  # Full mosaic augmentation
    "mixup": 0.15,  # Mixup for better generalization
    "copy_paste": 0.1,  # Copy-paste augmentation
    "auto_augment": "randaugment",  # Automatic augmentation policy
    "erasing": 0.4,  # Random erasing
    # Advanced training techniques
    "multi_scale": False,  # Multi-scale training disabled (causing issues)
    "rect": False,  # Rectangular training (set True if images have similar AR)
    "cos_lr": True,  # Cosine LR scheduling
    "close_mosaic": 20,  # Close mosaic in last 20 epochs
    "amp": False,  # AMP (set True if using GPU)
    "fraction": 1.0,  # USE 100% OF DATASET for maximum accuracy
    "profile": False,
    "freeze": None,  # Freeze layers: None, [0] (backbone), or [0,1,2] (more layers)
    "save_dir": MODELS_DIR / "detection",
    # Label smoothing for better generalization
    "label_smoothing": 0.0,  # 0.0-0.1, helps prevent overconfidence
    # Dropout (if supported)
    "dropout": 0.0,  # Experimental, 0.0-0.5
}

# Vision Transformer Configuration
VIT_CONFIG = {
    "model_name": "google/vit-base-patch16-224",
    "image_size": 224,
    "num_labels": 1,  # regression for Cobb angle
    "epochs": 50,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_steps": 500,
    "max_grad_norm": 1.0,
    "dropout": 0.1,
    "save_dir": MODELS_DIR / "vit",
}

# Quantum Circuit Configuration
QUANTUM_CONFIG = {
    "n_qubits": 4,
    "n_layers": 3,
    "learning_rate": 0.001,
    "epochs": 30,
    "batch_size": 8,
    "device": "default.qubit",
    "interface": "torch",
    "save_dir": MODELS_DIR / "quantum",
}

# Cobb Angle Calculation Configuration
COBB_CONFIG = {
    "method": "landmark_based",  # or "segmentation_based"
    "min_angle_threshold": 10,  # degrees
    "vertebra_levels": ["T1", "T12", "L1", "L5"],
    "save_dir": MODELS_DIR / "cobb",
}

# Segmentation Configuration (nnU-Net - Optional)
SEGMENTATION_CONFIG = {
    "enabled": False,  # Set to True if using segmentation
    "model_path": "path/to/nnunet/model",
    "save_dir": MODELS_DIR / "segmentation",
}

# Gemma LLM Report Generation Configuration
GEMMA_CONFIG = {
    "model_name": "google/gemma-2b-it",
    "max_length": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "device": "cpu",
    "load_in_8bit": True,  # Quantization for CPU efficiency
    "save_dir": MODELS_DIR / "report",
}

# Training Configuration
TRAINING_CONFIG = {
    "seed": 42,
    "num_workers": 4,
    "pin_memory": True,
    "mixed_precision": False,  # Set True if using GPU with AMP
    "gradient_accumulation_steps": 1,
    "early_stopping_patience": 10,
    "save_frequency": 5,  # Save checkpoint every N epochs
}

# Evaluation Metrics
METRICS_CONFIG = {
    "detection_metrics": ["mAP@0.5", "mAP@0.5:0.95", "precision", "recall"],
    "cobb_metrics": ["MAE", "RMSE", "R2"],
    "classification_metrics": ["accuracy", "f1_score", "confusion_matrix"],
}

# Visualization Configuration
VIZ_CONFIG = {
    "plot_predictions": True,
    "save_plots": True,
    "dpi": 150,
    "font_size": 10,
}

# Logging Configuration
LOGGING_CONFIG = {
    "log_level": "INFO",
    "log_file": LOGS_DIR / "training.log",
    "tensorboard_dir": LOGS_DIR / "tensorboard",
    "wandb_project": "scoliosis-detection",
    "wandb_enabled": False,  # Set True to enable W&B logging
}

# Clinical Report Template
REPORT_TEMPLATE = """
SPINE X-RAY ANALYSIS REPORT

Patient Information:
- Image ID: {image_id}
- Analysis Date: {date}

Findings:
{findings}

Measurements:
- Primary Cobb Angle: {primary_cobb}°
- Secondary Cobb Angle: {secondary_cobb}°
- Severity Classification: {severity_class}
- Confidence: {confidence}%

Interpretation:
{interpretation}

Recommendation:
{recommendation}

Automated Analysis System: Scoliosis AI v1.0
This report is generated by AI and should be reviewed by a qualified medical professional.
"""

def create_directories():
    """Create all necessary directories if they don't exist"""
    for dir_path in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, LOGS_DIR,
                     YOLO_CONFIG["save_dir"], VIT_CONFIG["save_dir"],
                     QUANTUM_CONFIG["save_dir"], COBB_CONFIG["save_dir"],
                     GEMMA_CONFIG["save_dir"], LOGGING_CONFIG["tensorboard_dir"]]:
        os.makedirs(dir_path, exist_ok=True)

if __name__ == "__main__":
    create_directories()
    print("Configuration loaded successfully!")
    print(f"Base directory: {BASE_DIR}")
    print(f"Dataset path: {DATASET_CONFIG['root']}")
