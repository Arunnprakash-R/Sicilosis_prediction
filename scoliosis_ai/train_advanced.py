"""
Advanced YOLO Training Script with High Accuracy Settings
Includes: Advanced augmentation, callbacks, validation, model selection
"""

import torch
from ultralytics import YOLO
from pathlib import Path
import sys
import json
import platform
import subprocess
from datetime import datetime

import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent))
from src.config import YOLO_CONFIG, DATASET_CONFIG
from src.utils import setup_logging


def _get_git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def save_run_metadata(cfg: dict):
    """Save reproducibility metadata for a training run."""
    run_dir = Path(cfg['save_dir']) / cfg.get('name', 'scoliosis_yolo_high_accuracy')
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "profile": cfg.get("profile", "unknown"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _get_git_commit(),
        "dataset_yaml": DATASET_CONFIG.get("yaml_path"),
        "config": {
            "model_name": cfg.get("model_name"),
            "epochs": cfg.get("epochs"),
            "img_size": cfg.get("img_size"),
            "batch_size": cfg.get("batch_size"),
            "optimizer": cfg.get("optimizer"),
            "fraction": cfg.get("fraction"),
            "patience": cfg.get("patience"),
        },
        "versions": {
            "torch": getattr(torch, "__version__", "unknown"),
        },
    }

    with open(run_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _summarize_results_csv(csv_path: Path):
    """Build compact summary and diagnostics from a YOLO results.csv."""
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    m50_col = 'metrics/mAP50(B)' if 'metrics/mAP50(B)' in df.columns else None
    m95_col = 'metrics/mAP50-95(B)' if 'metrics/mAP50-95(B)' in df.columns else None

    if m50_col is None:
        return None

    best_idx = int(df[m50_col].idxmax())
    best_row = df.loc[best_idx]
    last_row = df.iloc[-1]

    best_epoch = int(best_row['epoch']) if 'epoch' in df.columns else best_idx
    total_epochs = len(df)

    # Failure / quality diagnostics
    val_cls_gap = None
    if 'val/cls_loss' in df.columns and 'train/cls_loss' in df.columns:
        val_cls_gap = float(last_row['val/cls_loss'] - last_row['train/cls_loss'])

    precision = float(last_row['metrics/precision(B)']) if 'metrics/precision(B)' in df.columns else None
    recall = float(last_row['metrics/recall(B)']) if 'metrics/recall(B)' in df.columns else None
    pr_gap = abs(precision - recall) if precision is not None and recall is not None else None

    summary = {
        "run_name": csv_path.parent.name,
        "results_csv": str(csv_path),
        "epochs_logged": total_epochs,
        "best_epoch_by_map50": best_epoch,
        "best_map50": float(best_row[m50_col]),
        "best_map50_95": float(best_row[m95_col]) if m95_col else None,
        "last_map50": float(last_row[m50_col]),
        "last_map50_95": float(last_row[m95_col]) if m95_col else None,
        "last_precision": precision,
        "last_recall": recall,
        "diagnostics": {
            "precision_recall_gap": pr_gap,
            "val_minus_train_cls_loss": val_cls_gap,
            "convergence_drop_from_best_map50": float(best_row[m50_col] - last_row[m50_col]),
        },
    }
    return summary


def generate_comparison_report(output_dir: Path = Path("outputs/analysis")):
    """Generate comparison report across available YOLO runs."""
    detection_dir = Path("models/detection")
    runs = sorted(detection_dir.glob("*/results.csv"))
    summaries = []

    for csv_file in runs:
        summary = _summarize_results_csv(csv_file)
        if summary is not None:
            summaries.append(summary)

    if not summaries:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "training_comparison.json"
    md_path = output_dir / "training_comparison.md"

    ranked = sorted(
        summaries,
        key=lambda x: (
            x["best_map50_95"] if x["best_map50_95"] is not None else -1,
            x["best_map50"],
        ),
        reverse=True,
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"generated_at": datetime.now().isoformat(), "runs": ranked}, f, indent=2)

    lines = []
    lines.append("# YOLO Training Comparison")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Ranking (best first)")
    lines.append("")
    lines.append("| Run | Best mAP50 | Best mAP50-95 | Last P | Last R | Best Epoch | Epochs |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for row in ranked:
        lines.append(
            f"| {row['run_name']} | {row['best_map50']:.4f} | "
            f"{(row['best_map50_95'] if row['best_map50_95'] is not None else 0.0):.4f} | "
            f"{(row['last_precision'] if row['last_precision'] is not None else 0.0):.4f} | "
            f"{(row['last_recall'] if row['last_recall'] is not None else 0.0):.4f} | "
            f"{row['best_epoch_by_map50']} | {row['epochs_logged']} |"
        )

    lines.append("")
    lines.append("## Diagnostics")
    lines.append("")
    for row in ranked:
        diag = row["diagnostics"]
        lines.append(f"### {row['run_name']}")
        lines.append(f"- Precision-Recall gap: {diag['precision_recall_gap']:.4f}" if diag['precision_recall_gap'] is not None else "- Precision-Recall gap: N/A")
        lines.append(f"- Val-Train cls loss gap: {diag['val_minus_train_cls_loss']:.4f}" if diag['val_minus_train_cls_loss'] is not None else "- Val-Train cls loss gap: N/A")
        lines.append(f"- mAP50 drop from best to last: {diag['convergence_drop_from_best_map50']:.4f}")
        lines.append("")

    lines.append(f"JSON report: {json_path}")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return {"json": str(json_path), "markdown": str(md_path), "top_run": ranked[0]["run_name"]}

def get_training_profile(profile: str):
    """Return training overrides for the selected profile."""
    profile = (profile or "fast").lower()

    profiles = {
        "fast": {
            "profile_name": "FAST CPU TUNING",
            "model_name": "yolov8n.pt",
            "epochs": 12,
            "max_training_hours": 1.0,
            "img_size": 416,
            "batch_size": 12,
            "fraction": 0.40,
            "patience": 8,
            "name": "scoliosis_yolo_fast",
            "workers": 4,
            "save_period": 5,
        },
        "balanced": {
            "profile_name": "BALANCED TRAINING",
            "model_name": "yolov8s.pt",
            "epochs": 20,
            "max_training_hours": 1.0,
            "img_size": 512,
            "batch_size": 12,
            "fraction": 0.70,
            "patience": 15,
            "name": "scoliosis_yolo_balanced",
            "workers": 4,
            "save_period": 5,
        },
        "final": {
            "profile_name": "HIGH ACCURACY FINAL",
            "model_name": YOLO_CONFIG.get("model_name", "yolov8s.pt"),
            "epochs": YOLO_CONFIG.get("epochs", 30),
            "max_training_hours": YOLO_CONFIG.get("max_training_hours", 1.0),
            "img_size": YOLO_CONFIG.get("img_size", 640),
            "batch_size": YOLO_CONFIG.get("batch_size", 16),
            "fraction": YOLO_CONFIG.get("fraction", 1.0),
            "patience": YOLO_CONFIG.get("patience", 50),
            "name": "scoliosis_yolo_high_accuracy",
            "workers": 8,
            "save_period": 10,
        },
    }

    if profile not in profiles:
        raise ValueError(f"Unknown profile: {profile}. Use one of: {list(profiles.keys())}")

    cfg = YOLO_CONFIG.copy()
    cfg.update(profiles[profile])
    return cfg


def train_high_accuracy(profile: str = "fast", max_training_hours: float = 1.0):
    """Train YOLOv8 with profile-based configuration."""
    cfg = get_training_profile(profile)
    cfg['max_training_hours'] = max_training_hours
    cfg["profile"] = profile

    save_run_metadata(cfg)
    
    print("="*70)
    print(f"🎯 YOLOV8 TRAINING ({cfg['profile_name']})")
    print("="*70)
    print(f"\n📊 Training Configuration:")
    print(f"   Profile: {profile}")
    print(f"   Model: {cfg['model_name']}")
    print(f"   Epochs: {cfg['epochs']}")
    print(f"   Max Training Time: {cfg.get('max_training_hours', 1.0)} hour(s)")
    print(f"   Image Size: {cfg['img_size']}px")
    print(f"   Batch Size: {cfg['batch_size']}")
    print(f"   Optimizer: {cfg['optimizer']}")
    print(f"   Dataset Fraction: {cfg['fraction']*100:.0f}%")
    print(f"   Augmentation: Advanced (Mosaic, Mixup, Copy-Paste, RandAugment)")
    print(f"\n" + "="*70 + "\n")
    
    # Load model
    model = YOLO(cfg['model_name'])
    
    # Training parameters optimized for accuracy
    results = model.train(
        # Data
        data=DATASET_CONFIG['yaml_path'],
        epochs=cfg['epochs'],
        time=cfg.get('max_training_hours', 1.0),
        batch=cfg['batch_size'],
        imgsz=cfg['img_size'],
        
        # Device
        device='cpu',  # Use 'cuda' or '0' for GPU
        workers=cfg.get('workers', 4),
        
        # Optimizer
        optimizer=cfg['optimizer'],
        lr0=cfg['lr0'],
        lrf=cfg['lrf'],
        momentum=cfg['momentum'],
        weight_decay=cfg['weight_decay'],
        cos_lr=cfg['cos_lr'],
        
        # Warmup
        warmup_epochs=cfg['warmup_epochs'],
        warmup_momentum=cfg['warmup_momentum'],
        warmup_bias_lr=cfg['warmup_bias_lr'],
        
        # Augmentation - Color
        hsv_h=cfg['hsv_h'],
        hsv_s=cfg['hsv_s'],
        hsv_v=cfg['hsv_v'],
        
        # Augmentation - Geometric
        degrees=cfg['degrees'],
        translate=cfg['translate'],
        scale=cfg['scale'],
        shear=cfg['shear'],
        perspective=cfg['perspective'],
        flipud=cfg['flipud'],
        fliplr=cfg['fliplr'],
        
        # Augmentation - Advanced
        mosaic=cfg['mosaic'],
        mixup=cfg['mixup'],
        copy_paste=cfg['copy_paste'],
        auto_augment=cfg.get('auto_augment'),
        erasing=cfg.get('erasing', 0.4),
        
        # Multi-scale & Rectangle
        multi_scale=cfg.get('multi_scale', True),
        rect=cfg.get('rect', False),
        close_mosaic=cfg.get('close_mosaic', 20),
        
        # Regularization
        label_smoothing=cfg.get('label_smoothing', 0.0),
        dropout=cfg.get('dropout', 0.0),
        
        # Training control
        patience=cfg['patience'],
        fraction=cfg['fraction'],
        freeze=cfg.get('freeze'),
        
        # Saving
        project=str(cfg['save_dir']),
        name=cfg.get('name', 'scoliosis_yolo_high_accuracy'),
        exist_ok=True,
        save=True,
        save_period=cfg.get('save_period', 10),
        
        # Validation & Logging
        val=True,
        plots=True,
        verbose=True,
        amp=cfg.get('amp', False),
        
        # Model
        pretrained=True,
        
        # Advanced features
        profile=False,
        resume=False,  # Set True to resume from last checkpoint
    )
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    run_name = cfg.get('name', 'scoliosis_yolo_high_accuracy')
    print(f"\n📊 Best Model: models/detection/{run_name}/weights/best.pt")
    print(f"📊 Last Model: models/detection/{run_name}/weights/last.pt")
    
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
    
    parser = argparse.ArgumentParser(description="Profile-based YOLO Training")
    parser.add_argument('--mode', choices=['train', 'val', 'both', 'compare'], default='both',
                       help='Mode: train, val, or both')
    parser.add_argument('--profile', choices=['fast', 'balanced', 'final'], default='fast',
                       help='Training profile: fast (CPU tuning), balanced, or final (high accuracy)')
    parser.add_argument('--time-hours', type=float, default=1.0,
                       help='Maximum training time in hours (hard stop)')
    parser.add_argument('--model', type=str, default=None,
                       help='Model path for validation')
    
    args = parser.parse_args()

    if args.mode == 'compare':
        report = generate_comparison_report()
        if report:
            print(f"\n📊 Comparison report generated: {report['markdown']}")
            print(f"📊 Comparison JSON: {report['json']}")
            print(f"🏆 Top run: {report['top_run']}")
        else:
            print("\n⚠️ No runs found to compare.")
        print("\n✅ All operations completed successfully!")
        sys.exit(0)
    
    if args.mode in ['train', 'both']:
        results = train_high_accuracy(args.profile, max_training_hours=args.time_hours)
    
    if args.mode in ['val', 'both']:
        validate_model(args.model)

    report = generate_comparison_report()
    if report:
        print(f"\n📊 Comparison report generated: {report['markdown']}")
        print(f"📊 Comparison JSON: {report['json']}")
        print(f"🏆 Top run: {report['top_run']}")
    
    print("\n✅ All operations completed successfully!")
