"""
Advanced YOLO Training Script with High Accuracy Settings
Includes: Advanced augmentation, callbacks, validation, model selection
"""

import torch
from ultralytics import YOLO
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))
from src.config import YOLO_CONFIG, DATASET_CONFIG
from src.utils import setup_logging

def train_high_accuracy():
    """Train YOLOv8 with high accuracy configuration"""
    
    print("="*70)
    print("🎯 YOLOV8 HIGH ACCURACY TRAINING")
    print("="*70)
    print(f"\n📊 Training Configuration:")
    print(f"   Model: {YOLO_CONFIG['model_name']}")
    print(f"   Epochs: {YOLO_CONFIG['epochs']}")
    print(f"   Image Size: {YOLO_CONFIG['img_size']}px")
    print(f"   Batch Size: {YOLO_CONFIG['batch_size']}")
    print(f"   Optimizer: {YOLO_CONFIG['optimizer']}")
    print(f"   Dataset: {YOLO_CONFIG['fraction']*100:.0f}% (Full dataset)")
    print(f"   Augmentation: Advanced (Mosaic, Mixup, Copy-Paste, RandAugment)")
    print(f"\n" + "="*70 + "\n")
    
    # Load model
    model = YOLO(YOLO_CONFIG['model_name'])
    
    # Training parameters optimized for accuracy
    results = model.train(
        # Data
        data=DATASET_CONFIG['yaml_path'],
        epochs=YOLO_CONFIG['epochs'],
        batch=YOLO_CONFIG['batch_size'],
        imgsz=YOLO_CONFIG['img_size'],
        
        # Device
        device='cpu',  # Use 'cuda' or '0' for GPU
        workers=8,
        
        # Optimizer
        optimizer=YOLO_CONFIG['optimizer'],
        lr0=YOLO_CONFIG['lr0'],
        lrf=YOLO_CONFIG['lrf'],
        momentum=YOLO_CONFIG['momentum'],
        weight_decay=YOLO_CONFIG['weight_decay'],
        cos_lr=YOLO_CONFIG['cos_lr'],
        
        # Warmup
        warmup_epochs=YOLO_CONFIG['warmup_epochs'],
        warmup_momentum=YOLO_CONFIG['warmup_momentum'],
        warmup_bias_lr=YOLO_CONFIG['warmup_bias_lr'],
        
        # Augmentation - Color
        hsv_h=YOLO_CONFIG['hsv_h'],
        hsv_s=YOLO_CONFIG['hsv_s'],
        hsv_v=YOLO_CONFIG['hsv_v'],
        
        # Augmentation - Geometric
        degrees=YOLO_CONFIG['degrees'],
        translate=YOLO_CONFIG['translate'],
        scale=YOLO_CONFIG['scale'],
        shear=YOLO_CONFIG['shear'],
        perspective=YOLO_CONFIG['perspective'],
        flipud=YOLO_CONFIG['flipud'],
        fliplr=YOLO_CONFIG['fliplr'],
        
        # Augmentation - Advanced
        mosaic=YOLO_CONFIG['mosaic'],
        mixup=YOLO_CONFIG['mixup'],
        copy_paste=YOLO_CONFIG['copy_paste'],
        auto_augment=YOLO_CONFIG.get('auto_augment'),
        erasing=YOLO_CONFIG.get('erasing', 0.4),
        
        # Multi-scale & Rectangle
        multi_scale=YOLO_CONFIG.get('multi_scale', True),
        rect=YOLO_CONFIG.get('rect', False),
        close_mosaic=YOLO_CONFIG.get('close_mosaic', 20),
        
        # Regularization
        label_smoothing=YOLO_CONFIG.get('label_smoothing', 0.0),
        dropout=YOLO_CONFIG.get('dropout', 0.0),
        
        # Training control
        patience=YOLO_CONFIG['patience'],
        fraction=YOLO_CONFIG['fraction'],
        freeze=YOLO_CONFIG.get('freeze'),
        
        # Saving
        project=str(YOLO_CONFIG['save_dir']),
        name='scoliosis_yolo_high_accuracy',
        exist_ok=True,
        save=True,
        save_period=10,  # Save every 10 epochs
        
        # Validation & Logging
        val=True,
        plots=True,
        verbose=True,
        amp=YOLO_CONFIG.get('amp', False),
        
        # Model
        pretrained=True,
        
        # Advanced features
        profile=False,
        resume=False,  # Set True to resume from last checkpoint
    )
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📊 Best Model: models/detection/scoliosis_yolo_high_accuracy/weights/best.pt")
    print(f"📊 Last Model: models/detection/scoliosis_yolo_high_accuracy/weights/last.pt")
    
    # Print final metrics
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print(f"\n📈 Final Metrics:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"   {key}: {value:.4f}")
    
    print("\n" + "="*70)
    return results


def validate_model(model_path=None):
    """Validate trained model"""
    if model_path is None:
        model_path = "models/detection/scoliosis_yolo_high_accuracy/weights/best.pt"
    
    print(f"\n🔍 Validating model: {model_path}\n")
    
    model = YOLO(model_path)
    
    metrics = model.val(
        data=DATASET_CONFIG['yaml_path'],
        imgsz=YOLO_CONFIG['img_size'],
        batch=YOLO_CONFIG['batch_size'],
        device='cpu',
        plots=True,
        save_json=True,
        save_hybrid=True,
    )
    
    print("\n" + "="*70)
    print("📊 VALIDATION RESULTS")
    print("="*70)
    print(f"   mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"   mAP@0.5:     {metrics.box.map50:.4f}")
    print(f"   Precision:   {metrics.box.mp:.4f}")
    print(f"   Recall:      {metrics.box.mr:.4f}")
    print("="*70 + "\n")
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="High Accuracy YOLO Training")
    parser.add_argument('--mode', choices=['train', 'val', 'both'], default='both',
                       help='Mode: train, val, or both')
    parser.add_argument('--model', type=str, default=None,
                       help='Model path for validation')
    
    args = parser.parse_args()
    
    if args.mode in ['train', 'both']:
        results = train_high_accuracy()
    
    if args.mode in ['val', 'both']:
        validate_model(args.model)
    
    print("\n✅ All operations completed successfully!")
