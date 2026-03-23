"""
Reproducibility pack for Scoliosis AI.
Captures environment, config, and git metadata.
"""

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def get_git_commit():
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


def get_pip_freeze():
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip().splitlines()
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser(description="Capture reproducibility metadata")
    parser.add_argument("--out", default="outputs/research/run_manifest.json")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    try:
        from src.config import YOLO_CONFIG, DATASET_CONFIG
        config = {"YOLO_CONFIG": YOLO_CONFIG, "DATASET_CONFIG": DATASET_CONFIG}
    except Exception:
        config = {}

    payload = {
        "timestamp": datetime.now().isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": get_git_commit(),
        "notes": args.notes,
        "config": config,
        "pip_freeze": get_pip_freeze(),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved reproducibility manifest: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
