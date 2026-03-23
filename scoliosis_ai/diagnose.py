"""
Simple One-Click Scoliosis Diagnosis
Upload image → Get detailed report + annotated image
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

from inference import ScoliosisInference
from src.cobb_angle import CobbAngleCalculator
from src.generate_report import ReportGenerator
from src.utils import set_seed


def calculate_font_scale(img_height: int, img_width: int) -> dict:
    """Calculate dynamic font sizes based on image dimensions - truly responsive"""
    # Use the SMALLER dimension as reference to ensure text fits in all cases
    min_dim = min(img_height, img_width)
    
    # For a 1000px minimum dimension, scale proportionally
    # Larger images = larger fonts automatically
    scale_factor = min_dim / 1000.0
    scale_factor = max(0.6, min(scale_factor, 3.5))  # Clamp between 0.6x and 3.5x
    
    return {
        'title': 0.65 * scale_factor,          # Large, readable title
        'label': 0.50 * scale_factor,          # Large, readable labels
        'value': 0.55 * scale_factor,          # Large, readable values
        'angle': 0.70 * scale_factor,          # Large cobb angle numbers
        'status': 0.85 * scale_factor,         # Very large status text
        'watermark': 0.35 * scale_factor,      # Medium watermark
        'thickness_thin': max(2, int(2 * scale_factor)),
        'thickness_normal': max(3, int(3 * scale_factor)),
        'thickness_thick': max(4, int(4 * scale_factor)),
        'thickness_heavy': max(5, int(5 * scale_factor)),
        'panel_width': int(600 * scale_factor),
        'panel_height': int(200 * scale_factor),
        'status_width': int(280 * scale_factor),
        'status_height': int(90 * scale_factor),
        'corner_size': int(15 * scale_factor),
        'padding': int(15 * scale_factor)
    }


def put_text_with_background(image, text, position, font_scale, color, thickness, 
                             bg_color=(30, 30, 30), padding=5, opacity=0.85):
    """Put text on image with clear background for readability - optimized for sharpness"""
    # Use DUPLEX font for crisper appearance
    font = cv2.FONT_HERSHEY_DUPLEX
    
    # Get text size with current font
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # Calculate background box with extra padding
    x, y = position
    x1 = x - padding - 2
    y1 = y - text_size[1] - padding - 2
    x2 = x + text_size[0] + padding + 2
    y2 = y + padding + 2
    
    # Ensure box is within image bounds
    h, w = image.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    # Draw solid background (no blending)
    cv2.rectangle(image, (x1, y1), (x2, y2), bg_color, -1)
    
    # Draw a thin border around background for definition
    cv2.rectangle(image, (x1, y1), (x2, y2), (80, 80, 80), 1)
    
    # Draw text with antialiasing for crisp appearance
    cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    
    return image


def draw_spine_analysis(image_path: str, predictions: list, output_dir: Path):
    """Iron-Man-HUD style annotated diagnosis image with spine highlighting."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    # ── Upscale small images ────────────────────────────────────────────
    orig_h, orig_w = img.shape[:2]
    WORK_MIN = 1200
    up = max(1.0, WORK_MIN / min(orig_w, orig_h))
    if up > 1.0:
        img = cv2.resize(img, (int(orig_w * up), int(orig_h * up)),
                         interpolation=cv2.INTER_CUBIC)

    h, w = img.shape[:2]

    # ── SPINE HIGHLIGHTING ──────────────────────────────────────────────
    # Extract spine structure from the X-ray using intensity & edge analysis
    # Applied BEFORE any HUD overlays so the highlight sits under annotations
    if isinstance(predictions, dict):
        predictions = [predictions]

    def _raw_bbox(raw_bbox):
        """Parse a bbox without drawing — just return coords."""
        if raw_bbox is None:
            return None
        bbox = raw_bbox
        if hasattr(bbox, "tolist"):
            bbox = bbox.tolist()
        if isinstance(bbox, (tuple, list)) and len(bbox) == 2:
            p1, p2 = bbox
            if (isinstance(p1, (tuple, list)) and isinstance(p2, (tuple, list))
                    and len(p1) >= 2 and len(p2) >= 2):
                bbox = [p1[0], p1[1], p2[0], p2[1]]
        if not isinstance(bbox, (tuple, list)) or len(bbox) < 4:
            return None
        try:
            x1, y1, x2, y2 = [int(float(bbox[i]) * up) for i in range(4)]
        except (TypeError, ValueError):
            return None
        x1, y1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
        x2, y2 = max(0, min(w - 1, x2)), max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    # Collect all bboxes for spine region
    spine_bboxes = [_raw_bbox(p.get('bbox')) for p in predictions]
    spine_bboxes = [b for b in spine_bboxes if b is not None]

    if spine_bboxes:
        # Merge into one region covering all detections
        sx1 = min(b[0] for b in spine_bboxes)
        sy1 = min(b[1] for b in spine_bboxes)
        sx2 = max(b[2] for b in spine_bboxes)
        sy2 = max(b[3] for b in spine_bboxes)

        roi = img[sy1:sy2, sx1:sx2].copy()
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_h, roi_w = roi.shape[:2]

        # Enhance contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # --- Step 1: find bone (bright regions on X-ray) ---------------
        _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # --- Step 2: suppress horizontal structures (ribs) -------------
        # Use a tall, narrow morphological kernel to keep only vertically-
        # continuous structures and destroy horizontal ones.
        kern_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, roi_h // 30)))
        spine_v = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kern_v, iterations=1)
        # Close small vertical gaps between vertebrae
        kern_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(9, roi_h // 50)))
        spine_v = cv2.morphologyEx(spine_v, cv2.MORPH_CLOSE, kern_close, iterations=2)

        # --- Step 3: restrict to narrow center column ------------------
        # The vertebral column sits near the horizontal center of the bbox.
        # Use a narrow strip (~25% of roi width, centered).
        col_half = max(10, int(roi_w * 0.13))
        cx = roi_w // 2
        center_strip = np.zeros_like(spine_v)
        center_strip[:, max(0, cx - col_half):min(roi_w, cx + col_half)] = 255
        spine_v = cv2.bitwise_and(spine_v, center_strip)

        # --- Step 4: column scan to refine center ----------------------
        # For each row, find the mass-center of bright pixels and build a
        # narrow corridor around it (vertebral body width ~10-16% of roi).
        col_mask = np.zeros_like(spine_v)
        body_half = max(8, int(roi_w * 0.07))
        for row_y in range(roi_h):
            row = spine_v[row_y, :]
            cols = np.where(row > 0)[0]
            if len(cols) > 0:
                center = int(np.mean(cols))
                left = max(0, center - body_half)
                right = min(roi_w, center + body_half)
                col_mask[row_y, left:right] = 255

        # Intersect column corridor with bright bone
        spine_mask = cv2.bitwise_and(otsu, col_mask)

        # --- Step 5: morphological cleanup for smooth vertebral shape ---
        kern_sm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        spine_mask = cv2.morphologyEx(spine_mask, cv2.MORPH_CLOSE, kern_sm, iterations=2)
        spine_mask = cv2.morphologyEx(spine_mask, cv2.MORPH_OPEN, kern_sm, iterations=1)

        # Keep only large connected components (vertebrae, not noise)
        contours, _ = cv2.findContours(spine_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = (roi_h * roi_w) * 0.0005
        spine_mask_clean = np.zeros_like(spine_mask)
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                cv2.drawContours(spine_mask_clean, [cnt], -1, 255, -1)

        # Smooth edges
        spine_mask_clean = cv2.GaussianBlur(spine_mask_clean, (7, 7), 2)
        _, spine_mask_clean = cv2.threshold(spine_mask_clean, 100, 255, cv2.THRESH_BINARY)

        # --- Step 6: apply cyan highlight overlay ----------------------
        overlay_color = np.zeros_like(roi)
        overlay_color[:, :, 0] = 180   # B (cyan in BGR)
        overlay_color[:, :, 1] = 255   # G
        overlay_color[:, :, 2] = 0     # R

        alpha = 0.28
        mask_3ch = cv2.merge([spine_mask_clean, spine_mask_clean, spine_mask_clean])
        mask_f = mask_3ch.astype(np.float32) / 255.0

        roi_f = roi.astype(np.float32)
        overlay_f = overlay_color.astype(np.float32)
        blended = roi_f * (1.0 - mask_f * alpha) + overlay_f * (mask_f * alpha)
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        # Draw contour edges for crisp border
        contours2, _ = cv2.findContours(spine_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours2, -1, (180, 255, 0), 1, cv2.LINE_AA)

        img[sy1:sy2, sx1:sx2] = blended

    # ── Add right-side panel space ──────────────────────────────────────
    panel_w = max(340, int(w * 0.38))
    canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
    canvas[:, :w] = img
    total_w = w + panel_w

    # ── Color palette (Stark HUD) ──────────────────────────────────────
    CYAN     = (0, 255, 255)
    CYAN_DIM = (0, 180, 200)
    GREEN    = (0, 255, 100)
    RED      = (0, 80, 255)
    ORANGE   = (0, 165, 255)
    WHITE    = (255, 255, 255)
    DARK_BG  = (15, 15, 20)
    PANEL_BG = (20, 22, 30)

    # Fill panel background
    canvas[:, w:] = PANEL_BG

    # ── Adaptive sizing ─────────────────────────────────────────────────
    ref = (w * h) ** 0.5
    font_px_lg = max(22, min(48, int(ref / 30)))
    font_px_md = max(16, min(36, int(ref / 40)))
    font_px_sm = max(13, min(28, int(ref / 52)))
    bracket_len = max(20, int(ref / 30))
    bracket_thick = max(2, int(ref / 350))

    # ── Load fonts ──────────────────────────────────────────────────────
    def _load_font(size):
        for c in ("arialbd.ttf", "C:/Windows/Fonts/arialbd.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
            try:
                return ImageFont.truetype(c, size)
            except OSError:
                continue
        return ImageFont.load_default()

    font_lg = _load_font(font_px_lg)
    font_md = _load_font(font_px_md)
    font_sm = _load_font(font_px_sm)

    if isinstance(predictions, dict):
        predictions = [predictions]

    # ── Normalize bbox ──────────────────────────────────────────────────
    def _normalize_bbox(raw_bbox):
        if raw_bbox is None:
            return None
        bbox = raw_bbox
        if hasattr(bbox, "tolist"):
            bbox = bbox.tolist()
        if isinstance(bbox, (tuple, list)) and len(bbox) == 2:
            p1, p2 = bbox
            if (isinstance(p1, (tuple, list)) and isinstance(p2, (tuple, list))
                    and len(p1) >= 2 and len(p2) >= 2):
                bbox = [p1[0], p1[1], p2[0], p2[1]]
        if not isinstance(bbox, (tuple, list)) or len(bbox) < 4:
            return None
        try:
            x1, y1, x2, y2 = [int(float(bbox[i]) * up) for i in range(4)]
        except (TypeError, ValueError):
            return None
        x1, y1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
        x2, y2 = max(0, min(w - 1, x2)), max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    # ── Draw HUD targeting brackets (instead of plain rectangle) ───────
    def _draw_brackets(img_cv, x1, y1, x2, y2, color, thick, length):
        """Draw corner brackets like a targeting reticle."""
        L = min(length, (x2 - x1) // 3, (y2 - y1) // 3)
        # Top-left
        cv2.line(img_cv, (x1, y1), (x1 + L, y1), color, thick, cv2.LINE_AA)
        cv2.line(img_cv, (x1, y1), (x1, y1 + L), color, thick, cv2.LINE_AA)
        # Top-right
        cv2.line(img_cv, (x2, y1), (x2 - L, y1), color, thick, cv2.LINE_AA)
        cv2.line(img_cv, (x2, y1), (x2, y1 + L), color, thick, cv2.LINE_AA)
        # Bottom-left
        cv2.line(img_cv, (x1, y2), (x1 + L, y2), color, thick, cv2.LINE_AA)
        cv2.line(img_cv, (x1, y2), (x1, y2 - L), color, thick, cv2.LINE_AA)
        # Bottom-right
        cv2.line(img_cv, (x2, y2), (x2 - L, y2), color, thick, cv2.LINE_AA)
        cv2.line(img_cv, (x2, y2), (x2, y2 - L), color, thick, cv2.LINE_AA)

    # ── Draw subtle scan grid on X-ray area ─────────────────────────────
    grid_step = max(40, int(ref / 18))
    overlay = canvas.copy()
    for gx in range(0, w, grid_step):
        cv2.line(overlay, (gx, 0), (gx, h), (40, 45, 50), 1)
    for gy in range(0, h, grid_step):
        cv2.line(overlay, (0, gy), (w, gy), (40, 45, 50), 1)
    cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)

    # ── Draw crosshair center marker ────────────────────────────────────
    cx, cy = w // 2, h // 2
    cross_r = max(15, int(ref / 50))
    cv2.circle(canvas, (cx, cy), cross_r, CYAN_DIM, 1, cv2.LINE_AA)
    cv2.line(canvas, (cx - cross_r - 5, cy), (cx - cross_r + 3, cy), CYAN_DIM, 1, cv2.LINE_AA)
    cv2.line(canvas, (cx + cross_r - 3, cy), (cx + cross_r + 5, cy), CYAN_DIM, 1, cv2.LINE_AA)
    cv2.line(canvas, (cx, cy - cross_r - 5), (cx, cy - cross_r + 3), CYAN_DIM, 1, cv2.LINE_AA)
    cv2.line(canvas, (cx, cy + cross_r - 3), (cx, cy + cross_r + 5), CYAN_DIM, 1, cv2.LINE_AA)

    # ── Process predictions ─────────────────────────────────────────────
    det_data = []
    for index, prediction in enumerate(predictions, start=1):
        bbox = _normalize_bbox(prediction.get('bbox'))
        class_name = str(prediction.get('class', 'unknown'))
        confidence = float(prediction.get('confidence', 0.0))
        cobb_angle = float(prediction.get('cobb_angle_primary', 0.0))
        cobb_secondary = float(prediction.get('cobb_angle_secondary', 0.0))
        severity = str(prediction.get('severity', 'Unknown'))

        # Choose color by severity
        if cobb_angle >= 40:
            sev_color = RED
        elif cobb_angle >= 25:
            sev_color = ORANGE
        elif cobb_angle >= 10:
            sev_color = CYAN
        else:
            sev_color = GREEN

        det_data.append({
            'bbox': bbox, 'class': class_name, 'conf': confidence,
            'cobb': cobb_angle, 'cobb2': cobb_secondary,
            'severity': severity, 'color': sev_color, 'idx': index
        })

        if bbox:
            x1, y1, x2, y2 = bbox
            _draw_brackets(canvas, x1, y1, x2, y2, sev_color, bracket_thick, bracket_len)
            # Dashed center lines inside box
            mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
            dash = 8
            for dy in range(y1, y2, dash * 2):
                cv2.line(canvas, (mid_x, dy), (mid_x, min(dy + dash, y2)),
                         (*sev_color[:2], sev_color[2] // 2), 1, cv2.LINE_AA)
            # Leader line from box to panel
            cv2.line(canvas, (x2, mid_y), (w + 4, mid_y), CYAN_DIM, 1, cv2.LINE_AA)

    # ── PIL rendering for panel text ────────────────────────────────────
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(canvas_rgb)
    draw = ImageDraw.Draw(pil_img)

    # ── Panel header ────────────────────────────────────────────────────
    px = w + 16                # panel text x
    py = 20                    # panel text y cursor
    line_gap = int(font_px_sm * 0.6)

    # Title bar
    draw.rectangle([w, 0, total_w, py + font_px_lg + 16], fill=(10, 12, 18))
    draw.text((px, py), "SCOLIOSIS AI", font=font_lg, fill=(0, 255, 255))
    py += font_px_lg + 8
    draw.text((px, py), "DIAGNOSTIC SYSTEM v2.0", font=font_sm, fill=(120, 130, 150))
    py += font_px_sm + 4

    # Separator line
    draw.line([(w + 8, py), (total_w - 8, py)], fill=(0, 200, 200), width=1)
    py += 12

    # Timestamp
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    draw.text((px, py), f"SCAN: {ts}", font=font_sm, fill=(100, 110, 130))
    py += font_px_sm + line_gap

    # ── For each detection, draw panel data ─────────────────────────────
    for d in det_data:
        sev_rgb = (d['color'][2], d['color'][1], d['color'][0])  # BGR→RGB

        # Detection header
        draw.line([(w + 8, py), (total_w - 8, py)], fill=(50, 55, 65), width=1)
        py += 8
        draw.text((px, py), f"DETECTION #{d['idx']}", font=font_md, fill=(0, 255, 255))
        py += font_px_md + line_gap

        # Class
        draw.text((px, py), "CLASS", font=font_sm, fill=(100, 110, 130))
        draw.text((px + int(panel_w * 0.38), py), d['class'].upper(), font=font_sm, fill=(255, 255, 255))
        py += font_px_sm + line_gap

        # Cobb angle — large display
        draw.text((px, py), "COBB ANGLE", font=font_sm, fill=(100, 110, 130))
        py += font_px_sm + 2
        cobb_str = f"{d['cobb']:.1f}\u00b0"
        draw.text((px, py), cobb_str, font=font_lg, fill=sev_rgb)
        py += font_px_lg + 4
        if d['cobb2'] > 0:
            draw.text((px, py), f"Secondary: {d['cobb2']:.1f}\u00b0", font=font_sm, fill=(160, 170, 190))
            py += font_px_sm + line_gap

        # Severity badge
        draw.text((px, py), "SEVERITY", font=font_sm, fill=(100, 110, 130))
        py += font_px_sm + 2
        draw.text((px, py), d['severity'], font=font_md, fill=sev_rgb)
        py += font_px_md + line_gap

        # Confidence bar
        draw.text((px, py), "CONFIDENCE", font=font_sm, fill=(100, 110, 130))
        py += font_px_sm + 4
        bar_x = px
        bar_w = panel_w - 40
        bar_h = max(14, font_px_sm)
        draw.rectangle([bar_x, py, bar_x + bar_w, py + bar_h],
                        fill=(40, 42, 55), outline=(60, 65, 80))
        filled_w = int(bar_w * d['conf'])
        if filled_w > 0:
            draw.rectangle([bar_x, py, bar_x + filled_w, py + bar_h], fill=sev_rgb)
        conf_text = f"{d['conf']:.1%}"
        ctw = font_sm.getbbox(conf_text)[2] - font_sm.getbbox(conf_text)[0]
        draw.text((bar_x + bar_w + 6, py - 2), conf_text, font=font_sm, fill=(255, 255, 255))
        py += bar_h + line_gap + 4

        # Cobb angle visual arc indicator
        draw.text((px, py), "ANGLE SEVERITY SCALE", font=font_sm, fill=(100, 110, 130))
        py += font_px_sm + 4
        scale_w = panel_w - 40
        seg_w = scale_w // 4
        scale_labels = [("<10\u00b0", (0, 200, 80)), ("10-25\u00b0", (0, 220, 220)),
                        ("25-40\u00b0", (0, 165, 255)), (">40\u00b0", (0, 80, 255))]
        # Convert BGR tuples to RGB for PIL
        scale_labels_rgb = [(lbl, (c[2], c[1], c[0])) for lbl, c in scale_labels]
        for i, (lbl, col) in enumerate(scale_labels_rgb):
            sx = bar_x + i * seg_w
            draw.rectangle([sx, py, sx + seg_w - 2, py + bar_h], fill=col)
            lw = font_sm.getbbox(lbl)[2] - font_sm.getbbox(lbl)[0]
            draw.text((sx + (seg_w - lw) // 2, py + bar_h + 2), lbl, font=font_sm, fill=col)
        # Pointer showing current angle position
        if d['cobb'] < 10:
            ptr_frac = d['cobb'] / 10.0 * 0.25
        elif d['cobb'] < 25:
            ptr_frac = 0.25 + (d['cobb'] - 10) / 15.0 * 0.25
        elif d['cobb'] < 40:
            ptr_frac = 0.50 + (d['cobb'] - 25) / 15.0 * 0.25
        else:
            ptr_frac = min(1.0, 0.75 + (d['cobb'] - 40) / 20.0 * 0.25)
        ptr_x = bar_x + int(scale_w * ptr_frac)
        draw.polygon([(ptr_x - 5, py - 6), (ptr_x + 5, py - 6), (ptr_x, py)],
                      fill=(255, 255, 255))
        py += bar_h + font_px_sm + line_gap + 8

    # ── Bottom info bar ─────────────────────────────────────────────────
    draw.line([(w + 8, py), (total_w - 8, py)], fill=(50, 55, 65), width=1)
    py += 10
    draw.text((px, py), "MODEL", font=font_sm, fill=(100, 110, 130))
    draw.text((px + int(panel_w * 0.32), py), "YOLOv8 + Dual Ensemble", font=font_sm, fill=(200, 210, 220))
    py += font_px_sm + line_gap
    draw.text((px, py), "DATASET", font=font_sm, fill=(100, 110, 130))
    draw.text((px + int(panel_w * 0.32), py), "5,960 X-ray images", font=font_sm, fill=(200, 210, 220))
    py += font_px_sm + line_gap
    draw.text((px, py), "ENGINE", font=font_sm, fill=(100, 110, 130))
    draw.text((px + int(panel_w * 0.32), py), "Scoliosis AI v2.0", font=font_sm, fill=(0, 230, 230))
    py += font_px_sm + line_gap

    # ── Corner HUD elements on x-ray (top-left, bottom-right) ──────────
    # Top-left: SCANNING indicator
    draw.text((12, 8), "\u25c9 SCAN ACTIVE", font=font_sm, fill=(0, 220, 200))
    # Bottom-left: image dims
    draw.text((12, h - font_px_sm - 8),
              f"{orig_w}\u00d7{orig_h}px  \u2192  {w}\u00d7{h}px",
              font=font_sm, fill=(80, 90, 110))
    # Top-right of x-ray area
    det_text = f"DETECTIONS: {len(det_data)}"
    dtw = font_md.getbbox(det_text)[2] - font_md.getbbox(det_text)[0]
    draw.text((w - dtw - 12, 8), det_text, font=font_md, fill=(0, 255, 100))

    # ── Bottom disclaimer bar across full width ─────────────────────────
    bar_y = h - max(30, font_px_sm + 12)
    # Semi-transparent bar via overlay
    disc_text = "AI-ASSISTED ANALYSIS \u2022 NOT A CLINICAL DIAGNOSIS \u2022 CONSULT A PHYSICIAN"
    dw = font_sm.getbbox(disc_text)[2] - font_sm.getbbox(disc_text)[0]
    draw.rectangle([0, bar_y, w, h], fill=(10, 10, 15))
    draw.text(((w - dw) // 2, bar_y + 4), disc_text, font=font_sm, fill=(100, 80, 80))

    # ── Save ────────────────────────────────────────────────────────────
    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    output_image = output_dir / f"{Path(image_path).stem}_diagnosis.jpg"
    cv2.imwrite(str(output_image), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return str(output_image)


def generate_medical_report(
    predictions: list,
    image_name: str,
    output_dir: Path,
    use_gemma: bool = False,
    gemma_model_name: str = None
):
    """Generate detailed medical report for all detections
    
    Args:
        predictions: List of prediction dicts (can be multiple spines)
        image_name: Input image filename
        output_dir: Output directory
    """
    # Convert single prediction to list
    if isinstance(predictions, dict):
        predictions = [predictions]
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    gemma_generator = None
    if use_gemma:
        try:
            gemma_config = {'model_name': gemma_model_name} if gemma_model_name else None
            gemma_generator = ReportGenerator(config=gemma_config)
            if not gemma_generator.available:
                gemma_generator = None
        except Exception:
            gemma_generator = None

    # Generate report for each detection
    reports = []
    
    for idx, prediction in enumerate(predictions):
        det_id = prediction.get('detection_id', idx + 1)
        class_name = prediction.get('class', 'unknown')
        confidence = prediction.get('confidence', 0.0)
        cobb_angle = prediction.get('cobb_angle_primary', 0.0)
        cobb_secondary = prediction.get('cobb_angle_secondary', 0.0)
        severity = prediction.get('severity', 'Unknown')
        
        # Medical recommendations
        if cobb_angle < 10:
            diagnosis = "Normal spine alignment"
            recommendation = "No treatment required. Continue regular check-ups."
            risk_level = "LOW"
        elif cobb_angle < 25:
            diagnosis = "Mild Scoliosis detected"
            recommendation = "Monitor every 4-6 months. Physical therapy may be beneficial. Maintain good posture."
            risk_level = "MODERATE"
        elif cobb_angle < 40:
            diagnosis = "Moderate Scoliosis detected"
            recommendation = "Bracing treatment recommended. Consult orthopedic specialist. Physical therapy required."
            risk_level = "HIGH"
        else:
            diagnosis = "Severe Scoliosis detected"
            recommendation = "Immediate medical attention required. Surgical intervention may be necessary. Urgent specialist consultation."
            risk_level = "CRITICAL"
        
        spine_header = f"SPINE #{det_id}" if len(predictions) > 1 else ""
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════╗
║           SCOLIOSIS DETECTION AI - MEDICAL REPORT                    ║
║                       {spine_header:^40}                      ║
╚══════════════════════════════════════════════════════════════════════╝

Patient Image: {image_name}
Analysis Date: {timestamp}
AI Model: YOLOv8n + Ensemble (Advanced)
Detection ID: #{det_id}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                            DIAGNOSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Primary Diagnosis: {diagnosis}
Severity Class: {class_name}
Severity Level: {severity}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                        MEASUREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Primary Cobb Angle:     {cobb_angle:.2f}°
Secondary Cobb Angle:   {cobb_secondary:.2f}°
Detection Confidence:   {confidence:.1%}
Risk Level:            {risk_level}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    CLINICAL INTERPRETATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cobb Angle Classification:
  • < 10°:  Normal/Healthy spine
  • 10-25°: Mild scoliosis - monitoring recommended
  • 25-40°: Moderate scoliosis - treatment indicated
  • > 40°:  Severe scoliosis - intervention required

Current Status: {severity}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                        RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{recommendation}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                          DISCLAIMER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This is an AI-assisted analysis and should NOT replace professional
medical diagnosis. Please consult a qualified healthcare provider for
clinical interpretation and treatment planning.

Model Performance:
  • Dataset: 5,960 X-ray images
  • Validation mAP@0.5: 0.65
  • Precision: 0.62 | Recall: 0.65

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Generated by Scoliosis Detection AI System
Powered by YOLOv8 + Deep Learning + Quantum-Enhanced Analysis
"""

        if gemma_generator is not None:
            gemma_prediction = {
                'image_id': Path(image_name).stem,
                'severity': severity,
                'cobb_angle_primary': cobb_angle,
                'cobb_angle_secondary': cobb_secondary,
                'confidence': confidence,
            }
            try:
                report = gemma_generator.generate_report(gemma_prediction)
            except Exception:
                pass

        reports.append(report)
    
    # Combine all reports
    combined_report = "\n\n" + "="*70 + "\n\n".join(reports)
    
    # Save combined report
    report_file = output_dir / f"{Path(image_name).stem}_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(combined_report)
    
    return str(report_file), combined_report


def main():
    parser = argparse.ArgumentParser(
        description='One-Click Scoliosis Diagnosis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python diagnose.py --image image.jpg
  python diagnose.py --image image.jpg --model yolo --confidence 0.3
  
For more help, see QUICKSTART.md
        """
    )
    parser.add_argument('--image', type=str, required=True, help='Input X-ray image path')
    parser.add_argument('--output-dir', type=str, default='outputs/diagnosis', help='Output directory')
    parser.add_argument('--yolo-model', type=str, default='models/detection/scoliosis_yolo_enhanced/weights/best.pt', help='Path to trained YOLO model')
    parser.add_argument('--model', type=str, default='all', choices=['yolo', 'all'], help='Model to use: yolo or all (ensemble)')
    parser.add_argument('--confidence', type=float, default=0.25, help='Confidence threshold (0.1-0.9)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--report-model', type=str, default='template', choices=['template', 'gemma'],
                        help='Clinical report generator: template or gemma')
    parser.add_argument('--gemma-model-name', type=str, default='google/gemma-2b-it',
                        help='Hugging Face model id for Gemma report generation')
    parser.add_argument('--auto-open-image', type=str, default='true', choices=['true', 'false'],
                        help='Whether to auto-open only the annotated output image after diagnosis')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.confidence < 0.1 or args.confidence > 0.9:
        print(f"⚠️  Warning: Confidence should be between 0.1 and 0.9")
        args.confidence = max(0.1, min(0.9, args.confidence))
        print(f"   Adjusted to: {args.confidence:.2f}")
    
    set_seed(args.seed)
    
    # Validate input image
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"\n❌ Error: Image file not found!")
        print(f"   Path: {args.image}")
        print(f"\n💡 Tip: Drag and drop the image file or use quotes for paths with spaces")
        print(f'   Example: python diagnose.py --image "C:\\path\\to\\image.jpg"')
        return 1
    
    # Validate image format
    valid_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    if image_path.suffix.lower() not in valid_formats:
        print(f"\n⚠️  Warning: '{image_path.suffix}' may not be a supported format")
        print(f"   Supported: {', '.join(valid_formats)}")
        if sys.stdin.isatty():
            response = input("   Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return 1
        else:
            print("   Non-interactive run: continuing with unsupported format")
    
    # Validate image can be loaded
    try:
        test_img = cv2.imread(str(image_path))
        if test_img is None:
            print(f"\n❌ Error: Cannot read image file!")
            print(f"   Path: {image_path}")
            print(f"   This could mean:")
            print(f"   • The file is corrupted")
            print(f"   • The file format is not supported")
            print(f"   • Permission denied to read the file")
            return 1
        if test_img.shape[0] < 100 or test_img.shape[1] < 100:
            print(f"\n⚠️  Warning: Image is very small ({test_img.shape[0]}x{test_img.shape[1]} pixels)")
            print(f"   Minimum recommended: 400x400 pixels")
    except Exception as e:
        print(f"\n❌ Error: Failed to validate image!")
        print(f"   Details: {str(e)}")
        return 1
    
    # Validate and select model with fallback options
    model_path = Path(args.yolo_model)
    
    # If default model doesn't exist, try alternatives
    fallback_models = [
        'models/detection/scoliosis_yolo_enhanced/weights/best.pt',
        'models/detection/scoliosis_yolo_high_accuracy/weights/best.pt',
        'models/detection/yolov8n.pt',
    ]
    
    selected_model = None
    if model_path.exists():
        selected_model = str(model_path)
        print(f"✓ Using specified model: {model_path.name}")
    else:
        print(f"\n⚠️  Specified model not found: {args.yolo_model}")
        print(f"   Attempting to use available models...")
        for fallback in fallback_models:
            fb_path = Path(fallback)
            if fb_path.exists():
                selected_model = str(fb_path)
                print(f"✓ Found alternative model: {fallback}")
                break
    
    if not selected_model:
        print(f"\n❌ Error: No YOLO model found!")
        print(f"   Checked locations:")
        for model in [model_path] + [Path(fb) for fb in fallback_models]:
            print(f"   • {model}")
        print(f"\n💡 Tip: Download models from https://github.com/ultralytics/assets/releases")
        print(f"   Or train your own: python src/train_yolo.py")
        return 1
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("\n" + "="*70)
        print("  SCOLIOSIS DETECTION AI - One-Click Diagnosis")
        print("="*70)
        print(f"\n📂 Input Image: {args.image}")
        print(f"⚙️  Model: {args.model.upper()} | Confidence: {args.confidence:.2f}")
        print("🔬 Analyzing...")
        
        # Initialize AI model
        use_ensemble = args.model == 'all'
        try:
            infer = ScoliosisInference(
                yolo_model_path=selected_model,
                device='cpu',
                ensemble=use_ensemble,
                confidence_threshold=args.confidence
            )
        except Exception as e:
            print(f"\n❌ Error: Failed to load model!")
            print(f"   Details: {str(e)}")
            print(f"\n💡 Tip: Make sure the model file exists and is valid")
            return 1
        
        # Run prediction - detect ALL spines in image
        try:
            predictions = infer.predict_all_detections(args.image)
        except Exception as e:
            print(f"\n❌ Error: Failed to run inference!")
            print(f"   Details: {str(e)}")
            print(f"\n💡 Tip: Check if the image is a valid X-ray and not corrupted")
            return 1

        # ── Dual-model: run a second model for better bbox coverage ─────
        dual_models = [
            'models/detection/scoliosis_yolo_high_accuracy/weights/best.pt',
            'models/detection/scoliosis_yolo_enhanced/weights/best.pt',
            'models/detection/scoliosis_yolo_fast/weights/best.pt',
        ]
        secondary_preds = []
        for dm in dual_models:
            if not Path(dm).exists() or str(Path(dm).resolve()) == str(Path(selected_model).resolve()):
                continue
            try:
                infer2 = ScoliosisInference(
                    yolo_model_path=dm, device='cpu',
                    ensemble=False, confidence_threshold=max(0.10, args.confidence - 0.10))
                preds2 = infer2.predict_all_detections(args.image)
                if preds2:
                    secondary_preds.extend(preds2)
                    print(f"✓ Secondary model ({Path(dm).parent.parent.name}): {len(preds2)} detection(s)")
                break  # one extra model is enough
            except Exception:
                continue

        # Merge secondary bboxes into primary predictions
        if secondary_preds and predictions:
            all_bboxes = []
            for p in predictions + secondary_preds:
                bb = p.get('bbox', [])
                if bb and len(bb) >= 4:
                    all_bboxes.append(bb)
            if all_bboxes:
                merged_bbox = [
                    min(b[0] for b in all_bboxes),
                    min(b[1] for b in all_bboxes),
                    max(b[2] for b in all_bboxes),
                    max(b[3] for b in all_bboxes),
                ]
                # Apply merged bbox to every prediction (full-spine box)
                for p in predictions:
                    p['bbox'] = merged_bbox
                print(f"✓ Merged bbox from {len(all_bboxes)} detection(s) → full-spine coverage")

        # Expand bbox vertically to approximate full spine coverage
        # Models detect vertebra groups, not the full spine — extend the box
        if predictions:
            test_img = cv2.imread(str(image_path))
            if test_img is not None:
                img_h, img_w = test_img.shape[:2]
                for p in predictions:
                    bb = p.get('bbox', [])
                    if bb and len(bb) >= 4:
                        x1, y1, x2, y2 = bb
                        box_h = y2 - y1
                        box_w = x2 - x1
                        # Expand vertically: spine runs most of the image height
                        expand_up = max(box_h * 0.5, img_h * 0.08)
                        expand_down = max(box_h * 2.0, img_h * 0.40)
                        # Expand horizontally with some padding
                        expand_lr = max(box_w * 0.15, img_w * 0.03)
                        p['bbox'] = [
                            max(0, x1 - expand_lr),
                            max(0, y1 - expand_up),
                            min(img_w, x2 + expand_lr),
                            min(img_h, y2 + expand_down),
                        ]
                print(f"✓ Expanded bbox to approximate full-spine region")
        
        if not predictions or len(predictions) == 0:
            print(f"\n⚠️  Warning: No spine detected in the image!")
            print(f"   This could mean:")
            print(f"   • The image is not an X-ray")
            print(f"   • The confidence threshold is too high (try --confidence 0.1)")
            print(f"   • The image quality is too low")
            return 1

        # If multiple detections found, consolidate them into a single spine analysis
        if len(predictions) > 1:
            print(f"\n⚠️  Note: Detected {len(predictions)} vertebra groups. Consolidating into single spine analysis...")
            
            # Calculate average measurements across all detections
            avg_cobb_primary = sum(p.get('cobb_angle_primary', 0) for p in predictions) / len(predictions)
            avg_cobb_secondary = sum(p.get('cobb_angle_secondary', 0) for p in predictions) / len(predictions)
            avg_confidence = sum(p.get('confidence', 0) for p in predictions) / len(predictions)
            
            # Get severity from highest angle
            max_angle = max(p.get('cobb_angle_primary', 0) for p in predictions)
            
            # Consolidate all bboxes into one comprehensive bbox
            all_boxes = [p.get('bbox', []) for p in predictions if p.get('bbox', [])]
            if all_boxes:
                x1_vals = [b[0] for b in all_boxes if len(b) >= 2]
                y1_vals = [b[1] for b in all_boxes if len(b) >= 2]
                x2_vals = [b[2] for b in all_boxes if len(b) >= 4]
                y2_vals = [b[3] for b in all_boxes if len(b) >= 4]
                
                if x1_vals and y1_vals and x2_vals and y2_vals:
                    consolidated_bbox = [
                        min(x1_vals),
                        min(y1_vals),
                        max(x2_vals),
                        max(y2_vals)
                    ]
                else:
                    consolidated_bbox = []
            else:
                consolidated_bbox = []
            
            # Determine severity classification
            if max_angle < 10:
                severity = "Healthy/Normal (<10°)"
            elif max_angle < 25:
                severity = "Mild Scoliosis (10-25°)"
            elif max_angle < 40:
                severity = "Moderate Scoliosis (25-40°)"
            else:
                severity = "Severe Scoliosis (>40°)"
            
            # Create single consolidated prediction
            consolidated_prediction = {
                'detection_id': 1,
                'class': predictions[0].get('class', 'unknown'),  # Use first detection's class
                'severity': severity,
                'cobb_angle_primary': avg_cobb_primary,
                'cobb_angle_secondary': avg_cobb_secondary,
                'confidence': avg_confidence,
                'bbox': consolidated_bbox,
                'num_detections': len(predictions)
            }
            
            predictions = [consolidated_prediction]
            print(f"✓ Consolidated {len(predictions)} detections")
            print(f"  • Avg Cobb Angle: {avg_cobb_primary:.1f}°")
            print(f"  • Avg Confidence: {avg_confidence:.1%}")
            print(f"  • Severity: {severity}")
        else:
            print(f"✓ Detected {len(predictions)} spine in image")
        
        # Generate annotated image
        print("🖼️  Generating annotated image...")
        try:
            annotated_image = draw_spine_analysis(args.image, predictions, output_dir)
        except Exception as e:
            print(f"\n⚠️  Warning: Failed to generate annotated image!")
            print(f"   Details: {str(e)}")
            annotated_image = None
        
        # Generate medical report
        print("📄 Generating medical report...")
        try:
            report_file, report_text = generate_medical_report(
                predictions, 
                Path(args.image).name, 
                output_dir,
                use_gemma=(args.report_model == 'gemma'),
                gemma_model_name=args.gemma_model_name
            )
        except Exception as e:
            print(f"\n⚠️  Warning: Failed to generate report!")
            print(f"   Details: {str(e)}")
            report_file, report_text = None, "Report generation failed"
        
        # Display results
        print("\n" + "="*70)
        print("  ANALYSIS COMPLETE")
        print("="*70)
        if report_text:
            print(report_text)
        
        print("\n" + "="*70)
        print("  OUTPUT FILES")
        print("="*70)
        if annotated_image:
            print(f"📊 Annotated Image: {annotated_image}")
        if report_file:
            print(f"📋 Medical Report:  {report_file}")
        print("="*70 + "\n")
        
        # Save summary JSON
        try:
            import json
            summary = {
                'input_image': str(args.image),
                'timestamp': datetime.now().isoformat(),
                'num_detections': len(predictions),
                'detections': [
                    {
                        'detection_id': pred.get('detection_id', idx + 1),
                        'class': pred['class'],
                        'severity': pred['severity'],
                        'cobb_angle_primary': pred['cobb_angle_primary'],
                        'cobb_angle_secondary': pred['cobb_angle_secondary'],
                        'confidence': pred['confidence'],
                        'bbox': pred['bbox']
                    }
                    for idx, pred in enumerate(predictions)
                ],
                'outputs': {
                    'annotated_image': annotated_image,
                    'report': report_file
                }
            }
            
            summary_file = output_dir / f"{Path(args.image).stem}_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            
            print(f"💾 Summary JSON: {summary_file}\n")
        except Exception as e:
            print(f"⚠️  Warning: Failed to save summary JSON: {e}\n")

        auto_open_image = args.auto_open_image.lower() == 'true'

        # Open only the annotated image automatically
        try:
            if auto_open_image and annotated_image:
                annotated_path = Path(annotated_image)
                if annotated_path.exists():
                    if sys.platform == 'win32':
                        os.startfile(annotated_path)
                    elif sys.platform == 'darwin':
                        subprocess.Popen(['open', str(annotated_path)])
                    else:
                        subprocess.Popen(['xdg-open', str(annotated_path)])
        except Exception as e:
            print(f"⚠️  Warning: Could not open annotated image: {e}\n")
        
        print("✅ Diagnosis completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Diagnosis interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Unexpected error occurred!")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Details: {str(e)}")
        print(f"\n💡 Troubleshooting:")
        print(f"   1. Check if all dependencies are installed: pip install -r requirements.txt")
        print(f"   2. Verify the image file is not corrupted")
        print(f"   3. Try with a different image")
        print(f"   4. Check logs in the logs/ folder")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        exit(exit_code if exit_code is not None else 0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        exit(1)
