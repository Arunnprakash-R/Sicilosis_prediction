"""
Model selector and registry for scoliosis_ai.

Provides:
- Rich terminal listing with statuses and file checks
- Interactive model selection
- Programmatic model selection by ID
- Loader function mapping by model type
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, Tuple, Any

try:
    from colorama import init as _colorama_init

    _colorama_init()
except Exception:
    pass


# ANSI colors
RESET = "\033[0m"
BOLD = "\033[1m"

COLORS = {
    "Detection": "\033[96m",  # cyan
    "Segmentation": "\033[95m",  # magenta
    "Vision Transformer": "\033[93m",  # yellow
    "Quantum ML": "\033[38;5;214m",  # orange-ish
    "Report Generation": "\033[92m",  # green
}

STATUS_BADGES = {
    "ready": "\033[92m● ready\033[0m",
    "config": "\033[93m◐ config\033[0m",
    "empty": "\033[91m○ empty\033[0m",
    "planned": "\033[95m◌ planned\033[0m",
}


def _not_implemented(*_args, **_kwargs):
    raise NotImplementedError("Loader not implemented for this model type yet.")


def _load_yolo(model_path: str):
    from ultralytics import YOLO

    return YOLO(model_path)


def _load_unet(model_path: str):
    import torch
    from src.segmentation_model import UNet

    model = UNet()
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model


def _load_attention_unet(model_path: str):
    import torch
    from src.segmentation_model import AttentionUNet

    model = AttentionUNet()
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model


def _load_vit(model_path: str):
    from transformers import ViTForImageClassification

    if os.path.isdir(model_path):
        return ViTForImageClassification.from_pretrained(model_path)
    return ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=1)


def _load_gemma(model_path: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return model, tokenizer


def _load_quantum_hybrid(model_path: str):
    import torch
    from src.train_quantum import HybridQuantumClassicalModel

    model = HybridQuantumClassicalModel()
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model


LOADERS: Dict[str, Callable[..., Any]] = {
    "yolo": _load_yolo,
    "unet": _load_unet,
    "attention_unet": _load_attention_unet,
    "unetplusplus": _not_implemented,
    "mps_net": _not_implemented,
    "vit": _load_vit,
    "quantum_hybrid": _load_quantum_hybrid,
    "quantum_kernel": _not_implemented,
    "qsum": _not_implemented,
    "gemma": _load_gemma,
}


MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Detection
    "det_yolov8n": {
        "category": "Detection",
        "name": "YOLOv8n Base",
        "file": "yolov8n.pt",
        "type": "yolo",
        "tag": "baseline",
        "status": "ready",
        "speed": "★★★★★",
        "accuracy": "★★★☆☆",
        "desc": "Ultralytics YOLOv8n baseline detector for quick inference.",
    },
    "det_yolo_enhanced": {
        "category": "Detection",
        "name": "Scoliosis YOLO Enhanced",
        "file": "models/detection/scoliosis_yolo_enhanced.pt",
        "type": "yolo",
        "tag": "recommended",
        "status": "ready",
        "speed": "★★★★☆",
        "accuracy": "★★★★★",
        "desc": "Fine-tuned scoliosis detector; best default quality/speed balance.",
        "default": True,
    },
    "det_yolo_fast": {
        "category": "Detection",
        "name": "Scoliosis YOLO Fast",
        "file": "models/detection/scoliosis_yolo_fast.pt",
        "type": "yolo",
        "tag": "fast",
        "status": "ready",
        "speed": "★★★★★",
        "accuracy": "★★★☆☆",
        "desc": "Latency-optimized detector for near real-time use.",
    },
    "det_yolo_highacc": {
        "category": "Detection",
        "name": "Scoliosis YOLO High Accuracy",
        "file": "models/detection/scoliosis_yolo_high_accuracy.pt",
        "type": "yolo",
        "tag": "high-accuracy",
        "status": "ready",
        "speed": "★★★☆☆",
        "accuracy": "★★★★★",
        "desc": "Higher-capacity detector for maximum detection fidelity.",
    },
    # Added model 1
    "det_yolov8s": {
        "category": "Detection",
        "name": "YOLOv8s Base",
        "file": "yolov8s.pt",
        "type": "yolo",
        "tag": "new-balanced",
        "status": "ready",
        "speed": "★★★★☆",
        "accuracy": "★★★★☆",
        "desc": "New option: stronger baseline than v8n with moderate compute.",
    },
    # Added model 2
    "det_yolov8m": {
        "category": "Detection",
        "name": "YOLOv8m Base",
        "file": "yolov8m.pt",
        "type": "yolo",
        "tag": "new-high-capacity",
        "status": "ready",
        "speed": "★★★☆☆",
        "accuracy": "★★★★☆",
        "desc": "New option: larger detector for improved robustness on hard cases.",
    },

    # Segmentation
    "seg_unet": {
        "category": "Segmentation",
        "name": "UNet",
        "file": "models/segmentation/unet.pt",
        "type": "unet",
        "tag": "medical",
        "status": "ready",
        "speed": "★★★☆☆",
        "accuracy": "★★★★☆",
        "desc": "Classic U-Net for binary spine mask segmentation.",
    },
    "seg_attention_unet": {
        "category": "Segmentation",
        "name": "Attention UNet",
        "file": "models/segmentation/attention_unet.pt",
        "type": "attention_unet",
        "tag": "attention",
        "status": "ready",
        "speed": "★★★☆☆",
        "accuracy": "★★★★★",
        "desc": "Attention-gated U-Net for focused spine-region segmentation.",
    },
    "seg_unetpp": {
        "category": "Segmentation",
        "name": "UNet++",
        "file": "models/segmentation/unetplusplus.pt",
        "type": "unetplusplus",
        "tag": "nested-skips",
        "status": "config",
        "speed": "★★☆☆☆",
        "accuracy": "★★★★★",
        "desc": "Configured architecture with nested skip connections; training pending.",
    },
    "seg_mpsnet": {
        "category": "Segmentation",
        "name": "MPSNet",
        "file": "models/segmentation/mps_net.pt",
        "type": "mps_net",
        "tag": "research",
        "status": "config",
        "speed": "★★☆☆☆",
        "accuracy": "★★★★☆",
        "desc": "Configured medical prior segmentation model; weights pending.",
    },

    # Vision Transformer
    "vit_base": {
        "category": "Vision Transformer",
        "name": "ViT Base",
        "file": "models/vit/vit-base-patch16-224",
        "type": "vit",
        "tag": "regression",
        "status": "empty",
        "speed": "★★★☆☆",
        "accuracy": "★★★★☆",
        "desc": "Cobb angle regression backbone (base variant).",
    },
    "vit_small": {
        "category": "Vision Transformer",
        "name": "ViT Small",
        "file": "models/vit/vit-small-patch16-224",
        "type": "vit",
        "tag": "lightweight",
        "status": "empty",
        "speed": "★★★★☆",
        "accuracy": "★★★☆☆",
        "desc": "Smaller ViT variant for faster Cobb regression experiments.",
    },

    # Quantum ML
    "q_hybrid": {
        "category": "Quantum ML",
        "name": "Hybrid Quantum-Classical",
        "file": "models/quantum/quantum_hybrid.pt",
        "type": "quantum_hybrid",
        "tag": "vqnn",
        "status": "empty",
        "speed": "★☆☆☆☆",
        "accuracy": "★★★☆☆",
        "desc": "Variational hybrid quantum-classical scoliosis classifier.",
    },
    "q_kernel": {
        "category": "Quantum ML",
        "name": "Quantum Kernel Classifier",
        "file": "models/quantum/quantum_kernel.pt",
        "type": "quantum_kernel",
        "tag": "kernel",
        "status": "empty",
        "speed": "★☆☆☆☆",
        "accuracy": "★★★☆☆",
        "desc": "Quantum feature map + kernel-based classification pipeline.",
    },
    "q_qsum": {
        "category": "Quantum ML",
        "name": "QSum Model",
        "file": "models/quantum/qsum.pt",
        "type": "qsum",
        "tag": "multi-scale",
        "status": "planned",
        "speed": "★☆☆☆☆",
        "accuracy": "★★★★☆",
        "desc": "Planned quantum summation network for multi-scale spine feature aggregation.",
    },

    # Report Generation
    "report_gemma_2b": {
        "category": "Report Generation",
        "name": "Gemma 2B Instruct",
        "file": "models/report/gemma-2b-instruct",
        "type": "gemma",
        "tag": "llm-report",
        "status": "config",
        "speed": "★★☆☆☆",
        "accuracy": "★★★★☆",
        "desc": "Clinical-style report generation with instruction tuning.",
    },
}


class ModelSelector:
    def __init__(self, base_dir: str | Path | None = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent
        self.registry = MODEL_REGISTRY
        self.loaders = LOADERS
        self._ordered_ids = list(self.registry.keys())

    def _full_path(self, rel_path: str) -> str:
        return str((self.base_dir / rel_path).resolve())

    def _row(self, idx: int, model_id: str, model: Dict[str, Any]) -> str:
        category_color = COLORS.get(model["category"], "")
        status_badge = STATUS_BADGES.get(model["status"], model["status"])
        full_path = self._full_path(model["file"])
        exists_icon = "✔" if os.path.exists(full_path) else "✘"

        return (
            f"{category_color}{idx:>2}) {model_id:<16}{RESET} "
            f"{model['name']:<35} "
            f"{status_badge:<18} "
            f"{exists_icon} "
            f"{model['file']:<55} "
            f"[{model['tag']}]"
        )

    def _print_legend(self):
        print(f"\n{BOLD}Legend:{RESET}")
        print("  ✔ file exists   ✘ file missing")
        print("  " + " | ".join(f"{k}: {v}" for k, v in STATUS_BADGES.items()))

    def list_models(self):
        grouped: Dict[str, list[tuple[str, Dict[str, Any]]]] = {}
        for model_id in self._ordered_ids:
            model = self.registry[model_id]
            grouped.setdefault(model["category"], []).append((model_id, model))

        print(f"\n{BOLD}Model Registry ({len(self.registry)} models){RESET}")
        print("-" * 140)

        index = 1
        for category in [
            "Detection",
            "Segmentation",
            "Vision Transformer",
            "Quantum ML",
            "Report Generation",
        ]:
            items = grouped.get(category, [])
            if not items:
                continue
            print(f"\n{COLORS.get(category, '')}{BOLD}{category}{RESET}")
            for model_id, model in items:
                print(self._row(index, model_id, model))
                index += 1

        self._print_legend()

    def _resolve_model(self, model_id: str) -> Tuple[str, str, Callable[..., Any]]:
        if model_id not in self.registry:
            raise KeyError(f"Unknown model_id: {model_id}")

        model = self.registry[model_id]
        model_type = model["type"]
        loader_fn = self.loaders.get(model_type, _not_implemented)
        return self._full_path(model["file"]), model_type, loader_fn

    def select_by_id(self, model_id: str) -> Tuple[str, str, Callable[..., Any]]:
        return self._resolve_model(model_id)

    def get_default(self) -> Tuple[str, str, Callable[..., Any]]:
        for model_id, model in self.registry.items():
            if model.get("default"):
                return self._resolve_model(model_id)
        raise RuntimeError("No default model defined in registry.")

    def interactive_select(self) -> Tuple[str, str, Callable[..., Any]]:
        self.list_models()

        indexed = {str(i + 1): model_id for i, model_id in enumerate(self._ordered_ids)}
        selected_id = None

        while selected_id is None:
            choice = input("\nEnter model number: ").strip()
            if choice not in indexed:
                print("Invalid selection. Try again.")
                continue
            selected_id = indexed[choice]

        model = self.registry[selected_id]
        full_path, model_type, loader_fn = self._resolve_model(selected_id)
        exists = os.path.exists(full_path)

        print(f"\n{BOLD}Selected Model Details{RESET}")
        print(f"ID       : {selected_id}")
        print(f"Name     : {model['name']}")
        print(f"Category : {model['category']}")
        print(f"Type     : {model_type}")
        print(f"File     : {model['file']}")
        print(f"FullPath : {full_path}")
        print(f"Exists   : {'yes' if exists else 'no'}")
        print(f"Status   : {model['status']}")
        print(f"Speed    : {model['speed']}")
        print(f"Accuracy : {model['accuracy']}")
        print(f"Desc     : {model['desc']}")

        confirm = input("\nConfirm selection? [y/N]: ").strip().lower()
        if confirm not in {"y", "yes"}:
            print("Selection cancelled. Restarting selection...")
            return self.interactive_select()

        return full_path, model_type, loader_fn


if __name__ == "__main__":
    selector = ModelSelector()
    model_path, model_type, loader = selector.interactive_select()
    print("\nSelection result:")
    print(f"Path      : {model_path}")
    print(f"Type      : {model_type}")
    print(f"Loader fn : {loader.__name__}")
