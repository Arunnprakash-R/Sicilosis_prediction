"""
Model Configuration and Loader
Centralized configuration for all models (YOLO, ViT, Quantum, Segmentation)
Provides unified interface for loading and managing model weights.
"""

import os
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import warnings

class ModelConfig:
    """Configuration for all available models."""
    
    # Base paths
    MODELS_ROOT = Path(__file__).parent.parent / "models"
    DETECTION_PATH = MODELS_ROOT / "detection"
    SEGMENTATION_PATH = MODELS_ROOT / "segmentation"
    VIT_PATH = MODELS_ROOT / "vit"
    QUANTUM_PATH = MODELS_ROOT / "quantum"
    COBB_PATH = MODELS_ROOT / "cobb"
    
    # YOLO Detection Models
    YOLO = {
        "pretrained_nano": {
            "path": DETECTION_PATH / "yolov8n.pt",
            "description": "YOLOv8 Nano (6.2 MB) - Fast inference, good for real-time",
            "resolution": 416,
        },
        "scoliosis_enhanced": {
            "path": DETECTION_PATH / "scoliosis_yolo_enhanced" / "weights" / "best.pt",
            "description": "Fine-tuned on scoliosis dataset (enhanced version, 5.9 MB)",
            "resolution": 416,
            "training_epochs": 25,
        },
        "scoliosis_high_accuracy": {
            "path": DETECTION_PATH / "scoliosis_yolo_high_accuracy" / "weights" / "best.pt",
            "description": "Fine-tuned for high accuracy (64.1 MB, larger model)",
            "resolution": 640,
            "training_epochs": 20,
        },
    }
    
    # Segmentation Models (U-Net variants)
    SEGMENTATION = {
        "mps_net": {
            "model_name": "mps_net",
            "num_classes": 1,
            "encoder": "resnet50",
            "description": "Medical segmentation network on ResNet50 backbone",
        },
        "unetplusplus": {
            "model_name": "unetplusplus",
            "num_classes": 1,
            "encoder": "resnet50",
            "description": "U-Net++ with nested skip connections",
        },
        "attention_unet": {
            "model_name": "Custom AttentionUNet",
            "num_classes": 1,
            "description": "Custom Attention U-Net with attention gates",
        },
    }
    
    # Vision Transformer for Regression
    VIT = {
        "base": {
            "model_name": "vit_base_patch16_224",
            "pretrained": True,
            "num_classes": 1,  # Outputs: Cobb angle
            "description": "ViT Base (224x224, ImageNet pretrained)",
            "auto_download": True,
        },
        "small": {
            "model_name": "vit_small_patch16_224",
            "pretrained": True,
            "num_classes": 1,
            "description": "ViT Small (lighter, faster)",
            "auto_download": True,
        },
    }
    
    # Quantum Models
    QUANTUM = {
        "hybrid_classifier": {
            "backend": "qiskit",
            "n_qubits": 4,
            "n_layers": 2,
            "circuit_type": "ansatz",
            "description": "Hybrid quantum-classical classifier (PennyLane + Qiskit)",
        },
        "quantum_kernel": {
            "backend": "qiskit",
            "n_qubits": 6,
            "description": "Quantum kernel for SVM classification",
        },
    }
    
    @classmethod
    def get_yolo_path(cls, model_name: str = "scoliosis_enhanced") -> Path:
        """Get path to YOLO model weights."""
        if model_name not in cls.YOLO:
            raise ValueError(f"Unknown YOLO model: {model_name}. Available: {list(cls.YOLO.keys())}")
        return cls.YOLO[model_name]["path"]
    
    @classmethod
    def get_vit_config(cls, model_name: str = "base") -> Dict[str, Any]:
        """Get ViT configuration."""
        if model_name not in cls.VIT:
            raise ValueError(f"Unknown ViT model: {model_name}. Available: {list(cls.VIT.keys())}")
        return cls.VIT[model_name].copy()
    
    @classmethod
    def get_segmentation_config(cls, model_name: str = "unetplusplus") -> Dict[str, Any]:
        """Get segmentation model configuration."""
        if model_name not in cls.SEGMENTATION:
            raise ValueError(f"Unknown segmentation model: {model_name}. Available: {list(cls.SEGMENTATION.keys())}")
        return cls.SEGMENTATION[model_name].copy()
    
    @classmethod
    def get_quantum_config(cls, model_name: str = "hybrid_classifier") -> Dict[str, Any]:
        """Get quantum model configuration."""
        if model_name not in cls.QUANTUM:
            raise ValueError(f"Unknown quantum model: {model_name}. Available: {list(cls.QUANTUM.keys())}")
        return cls.QUANTUM[model_name].copy()
    
    @classmethod
    def list_available_yolo_models(cls) -> Dict[str, str]:
        """List all available YOLO models."""
        return {name: cfg["description"] for name, cfg in cls.YOLO.items()}
    
    @classmethod
    def list_available_vit_models(cls) -> Dict[str, str]:
        """List all available ViT models."""
        return {name: cfg["description"] for name, cfg in cls.VIT.items()}
    
    @classmethod
    def list_available_segmentation_models(cls) -> Dict[str, str]:
        """List all available segmentation models."""
        return {name: cfg["description"] for name, cfg in cls.SEGMENTATION.items()}
    
    @classmethod
    def list_available_quantum_models(cls) -> Dict[str, str]:
        """List all available quantum models."""
        return {name: cfg["description"] for name, cfg in cls.QUANTUM.items()}
    
    @classmethod
    def verify_model_availability(cls, verbose: bool = True) -> Dict[str, Tuple[bool, str]]:
        """Verify all model weights exist."""
        status = {}
        
        # Check YOLO models
        for model_name, cfg in cls.YOLO.items():
            path = cfg["path"]
            exists = path.exists()
            status[f"yolo_{model_name}"] = (exists, str(path))
            if verbose and exists:
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"✓ YOLO {model_name:25} ({size_mb:6.1f} MB)")
            elif verbose:
                print(f"✗ YOLO {model_name:25} - NOT FOUND")
        
        return status


def print_model_summary():
    """Print available models summary."""
    print("\n" + "="*80)
    print("AVAILABLE MODELS")
    print("="*80)
    
    print("\nYOLO Detection (Spine Vertebra Detection):")
    for name, desc in ModelConfig.list_available_yolo_models().items():
        print(f"  • {name:25} - {desc}")
    
    print("\nSegmentation (Spine Mask Generation):")
    for name, desc in ModelConfig.list_available_segmentation_models().items():
        print(f"  • {name:25} - {desc}")
    
    print("\nVision Transformer (Cobb Angle Regression):")
    for name, desc in ModelConfig.list_available_vit_models().items():
        print(f"  • {name:25} - {desc}")
    
    print("\nQuantum Models (Severity Classification):")
    for name, desc in ModelConfig.list_available_quantum_models().items():
        print(f"  • {name:25} - {desc}")
    
    print("\n" + "="*80)


def print_model_status():
    """Print availability status of all models."""
    print("\n" + "="*80)
    print("MODEL AVAILABILITY STATUS")
    print("="*80)
    
    ModelConfig.verify_model_availability(verbose=True)
    print("\n" + "="*80)


if __name__ == "__main__":
    print("\nModel Configuration Module")
    print("="*80)
    print("\nUsage Example:")
    print("-" * 80)
    print("""
from models_config import ModelConfig

# Get YOLO model path
yolo_path = ModelConfig.get_yolo_path("scoliosis_enhanced")

# Get ViT configuration
vit_cfg = ModelConfig.get_vit_config("base")

# List all available models
print(ModelConfig.list_available_yolo_models())
print(ModelConfig.list_available_vit_models())

# Check model availability
status = ModelConfig.verify_model_availability()
    """)
    print("-" * 80)
    
    print_model_summary()
    print_model_status()
