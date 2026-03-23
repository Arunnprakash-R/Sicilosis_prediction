"""
Integration guide snippets for model_selector.py.

Run:
    python integration_guide.py
"""

GUIDE = r'''
============================================================
MODEL SELECTOR INTEGRATION GUIDE
============================================================

1) inference.py (replace hard-coded model load)
------------------------------------------------------------
from model_selector import ModelSelector


def load_model(interactive: bool = False, model_id: str | None = None):
    selector = ModelSelector()

    if interactive:
        model_path, model_type, loader_fn = selector.interactive_select()
    elif model_id:
        model_path, model_type, loader_fn = selector.select_by_id(model_id)
    else:
        model_path, model_type, loader_fn = selector.get_default()

    model = loader_fn(model_path)
    return model, model_path, model_type


# Example inside your init / startup path:
# model, model_path, model_type = load_model(interactive=args.select_model, model_id=args.model)


2) launcher.py (add flags: --model, --select-model, --list-models)
------------------------------------------------------------
import argparse
from model_selector import ModelSelector


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=None,
                    help="Model id, e.g. det_yolo_enhanced, det_yolov8s, det_yolov8m")
parser.add_argument("--select-model", action="store_true",
                    help="Interactive model selector")
parser.add_argument("--list-models", action="store_true",
                    help="List all registered models and exit")
args = parser.parse_args()

selector = ModelSelector()
if args.list_models:
    selector.list_models()
    raise SystemExit(0)

if args.select_model:
    model_path, model_type, loader_fn = selector.interactive_select()
elif args.model:
    model_path, model_type, loader_fn = selector.select_by_id(args.model)
else:
    model_path, model_type, loader_fn = selector.get_default()

loaded_model = loader_fn(model_path)


3) launcher_simple.py (one-liner model usage)
------------------------------------------------------------
from model_selector import ModelSelector; model_path, model_type, loader_fn = ModelSelector().get_default()


4) CLI usage examples
------------------------------------------------------------
# List all models
python launcher.py --list-models

# Interactive selection
python launcher.py --select-model

# Use default (Scoliosis YOLO Enhanced)
python launcher.py

# Use new model 1 (YOLOv8s Base)
python launcher.py --model det_yolov8s

# Use new model 2 (YOLOv8m Base)
python launcher.py --model det_yolov8m

# Use the selector directly
python model_selector.py


New models added and how users can access them
------------------------------------------------------------
1. YOLOv8s Base  -> model id: det_yolov8s  -> file: yolov8s.pt
2. YOLOv8m Base  -> model id: det_yolov8m  -> file: yolov8m.pt

Access methods:
- CLI id selection: --model det_yolov8s / --model det_yolov8m
- Interactive mode: --select-model and choose the numbered item
- Programmatic: ModelSelector().select_by_id("det_yolov8s")

============================================================
'''


def main():
    print(GUIDE)


if __name__ == "__main__":
    main()
