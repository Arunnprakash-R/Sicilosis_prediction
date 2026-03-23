"""
Model Validation and Inventory Script
Validates all available models (YOLO, ViT, Quantum, Segmentation)
and provides a detailed report of installed weights and frameworks.
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def check_model_file(path):
    """Check if a model file exists and get its size."""
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        return True, f"{size_mb:.1f} MB"
    return False, "NOT FOUND"

def validate_detection_models():
    """Validate YOLO detection models."""
    print("\n" + "="*70)
    print("DETECTION MODELS (YOLO)")
    print("="*70)
    
    base_path = "models/detection"
    
    # Check base YOLO weights
    yolov8n_path = os.path.join(base_path, "yolov8n.pt")
    exists, size = check_model_file(yolov8n_path)
    print(f"\n✓ YOLOv8n (nano):     {yolov8n_path}")
    print(f"  Status: {'✓ READY' if exists else '✗ MISSING'} ({size})")
    
    # Check trained models
    trained_models = [
        "scoliosis_yolo",
        "scoliosis_yolo_enhanced", 
        "scoliosis_yolo_high_accuracy"
    ]
    
    for model_name in trained_models:
        model_dir = os.path.join(base_path, model_name, "weights")
        if os.path.exists(model_dir):
            checkpoint_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
            print(f"\n✓ {model_name}:")
            print(f"  Location: {model_dir}")
            print(f"  Checkpoints ({len(checkpoint_files)}):")
            for ckpt in sorted(checkpoint_files):
                path = os.path.join(model_dir, ckpt)
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"    - {ckpt:20} ({size_mb:6.1f} MB)")
        else:
            print(f"\n✗ {model_name}: weights directory not found")

def validate_segmentation_models():
    """Validate segmentation models."""
    print("\n" + "="*70)
    print("SEGMENTATION MODELS (U-Net, Attention U-Net)")
    print("="*70)
    
    base_path = "models/segmentation"
    if os.path.exists(base_path):
        contents = os.listdir(base_path)
        if contents:
            print(f"\n✓ Segmentation directory exists: {base_path}")
            model_files = [f for f in contents if f.endswith(('.pt', '.pth', '.ckpt'))]
            if model_files:
                print(f"  Model files ({len(model_files)}):")
                for mf in model_files:
                    path = os.path.join(base_path, mf)
                    size_mb = os.path.getsize(path) / (1024 * 1024)
                    print(f"    - {mf:30} ({size_mb:6.1f} MB)")
            else:
                print("  Note: No pretrained weights found (can train or download U-Net++)")
        else:
            print(f"✗ Segmentation directory is empty")
    else:
        print(f"✗ Segmentation directory not found: {base_path}")

def validate_vit_models():
    """Validate Vision Transformer models."""
    print("\n" + "="*70)
    print("VISION TRANSFORMER MODELS (ViT)")
    print("="*70)
    
    base_path = "models/vit"
    if os.path.exists(base_path):
        print(f"\n✓ ViT directory exists: {base_path}")
        model_files = [f for f in os.listdir(base_path) if f.endswith(('.pt', '.pth', '.safetensors'))]
        if model_files:
            print(f"  Model files ({len(model_files)}):")
            for mf in model_files:
                path = os.path.join(base_path, mf)
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"    - {mf:30} ({size_mb:6.1f} MB)")
        else:
            print("  Note: No finetuned weights stored locally (timm auto-downloads base models)")
    else:
        print(f"✓ ViT directory will be created on first use")

def validate_quantum_models():
    """Validate quantum models."""
    print("\n" + "="*70)
    print("QUANTUM MODELS (PennyLane/Qiskit)")
    print("="*70)
    
    base_path = "models/quantum"
    if os.path.exists(base_path):
        print(f"\n✓ Quantum directory exists: {base_path}")
        model_files = [f for f in os.listdir(base_path) if f.endswith(('.pt', '.pth', '.json', '.pkl'))]
        if model_files:
            print(f"  Model files ({len(model_files)}):")
            for mf in model_files:
                path = os.path.join(base_path, mf)
                if os.path.isfile(path):
                    size_mb = os.path.getsize(path) / (1024 * 1024)
                    print(f"    - {mf:30} ({size_mb:6.1f} MB)")
        else:
            print("  Note: No trained quantum models found (can train via train_quantum.py)")
    else:
        print(f"✓ Quantum directory will be created on first use")

def validate_frameworks():
    """Validate that all required frameworks are installed."""
    print("\n" + "="*70)
    print("FRAMEWORK VALIDATION")
    print("="*70)
    
    frameworks = {
        'ultralytics': 'YOLO detection',
        'torch': 'PyTorch (deep learning)',
        'torchvision': 'Computer vision utilities',
        'timm': 'Vision Transformer models',
        'transformers': 'Hugging Face transformers',
        'segmentation_models_pytorch': 'Segmentation models',
        'pennylane': 'Quantum machine learning',
        'qiskit': 'Quantum computing',
        'cv2': 'OpenCV (image processing)',
        'sklearn': 'Scikit-learn (ML utilities)',
        'pandas': 'Data manipulation',
        'numpy': 'Numerical computing',
        'matplotlib': 'Visualization',
        'PIL': 'Image processing (Pillow)',
    }
    
    for module_name, description in frameworks.items():
        try:
            __import__(module_name)
            print(f"✓ {module_name:30} - {description}")
        except ImportError:
            print(f"✗ {module_name:30} - {description} [MISSING]")

def main():
    """Run full model validation."""
    print("\n" + "="*70)
    print("SCOLIOSIS AI - MODEL & FRAMEWORK VALIDATION")
    print("="*70)
    
    validate_detection_models()
    validate_segmentation_models()
    validate_vit_models()
    validate_quantum_models()
    validate_frameworks()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Key Models:
  • YOLO Detection:     Used for spine vertebra detection
  • Segmentation:       U-Net/Attention U-Net for spine mask (optional)
  • ViT:                Vision Transformer for Cobb angle regression
  • Quantum:            PennyLane hybrid models for classification

Quick Start:
  1. python scripts/validate_models.py  (this script)
  2. python launcher_simple.py           (single diagnosis)
  3. python run_research_pipeline.py     (full analysis pipeline)

Detailed Documentation:
  • README.md           - Project overview
  • START_HERE.md       - Quick start guide
  • PHD_ROADMAP.md      - Research objectives
  • QUICK_START_PHD.md  - PhD reproducibility setup
""")

if __name__ == "__main__":
    main()
