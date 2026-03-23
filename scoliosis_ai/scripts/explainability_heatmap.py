"""
Generate a localization heatmap using the predicted bounding box.
This provides a simple, deterministic explainability visualization.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Generate bbox heatmap")
    parser.add_argument("--image", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", default="outputs/analysis")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print("Could not read image.")
        return 1

    with open(args.summary, "r", encoding="utf-8") as f:
        data = json.load(f)

    detections = data.get("detections", [])
    if len(detections) != 1:
        print("Summary must contain exactly one detection.")
        return 1

    bbox = detections[0].get("bbox", [])
    if len(bbox) != 4:
        print("Invalid bbox.")
        return 1

    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = img.shape[:2]

    heatmap = np.zeros((h, w), dtype=np.float32)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    sx = max(1, (x2 - x1) // 2)
    sy = max(1, (y2 - y1) // 2)

    for y in range(max(0, y1), min(h, y2)):
        for x in range(max(0, x1), min(w, x2)):
            dx = (x - cx) / float(sx)
            dy = (y - cy) / float(sy)
            heatmap[y, x] = np.exp(-0.5 * (dx * dx + dy * dy))

    heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8) if heatmap.max() > 0 else heatmap.astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 0.65, heatmap_color, 0.35, 0)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(args.image).stem}_heatmap.jpg"
    cv2.imwrite(str(out_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"Saved heatmap: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
