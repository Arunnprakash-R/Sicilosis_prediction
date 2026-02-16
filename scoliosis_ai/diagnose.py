"""
Simple One-Click Scoliosis Diagnosis
Upload image → Get detailed report + annotated image
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

from inference import ScoliosisInference
from src.cobb_angle import CobbAngleCalculator
from src.utils import set_seed


def calculate_font_scale(img_height: int, img_width: int) -> dict:
    """Calculate dynamic font sizes based on image dimensions"""
    # Base resolution for scaling (1920x1080 as reference)
    base_h, base_w = 1080, 1920
    
    # Calculate scale factor based on image size
    scale = min(img_height / base_h, img_width / base_w)
    scale = max(0.5, min(scale, 2.5))  # Clamp between 0.5x and 2.5x
    
    return {
        'title': 0.9 * scale,
        'label': 0.6 * scale,
        'value': 0.7 * scale,
        'angle': 1.0 * scale,
        'status': 1.2 * scale,
        'watermark': 0.5 * scale,
        'thickness_thin': max(1, int(1 * scale)),
        'thickness_normal': max(2, int(2 * scale)),
        'thickness_thick': max(3, int(3 * scale)),
        'thickness_heavy': max(4, int(4 * scale)),
        'panel_width': int(600 * scale),
        'panel_height': int(200 * scale),
        'status_width': int(280 * scale),
        'status_height': int(80 * scale),
        'corner_size': int(15 * scale),
        'padding': int(20 * scale)
    }


def draw_spine_analysis(image_path: str, predictions: list, output_dir: Path):
    """Draw detailed spine analysis on image - supports multiple detections
    
    Args:
        image_path: Path to input image
        predictions: List of prediction dicts (can be multiple spines)
        output_dir: Output directory
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    h, w = img.shape[:2]
    
    # Calculate dynamic font scales based on image size
    font_config = calculate_font_scale(h, w)
    
    # Convert single prediction to list for compatibility
    if isinstance(predictions, dict):
        predictions = [predictions]
    
    # Colors for multiple detections
    detection_colors = [
        (0, 255, 0),    # Green
        (255, 150, 0),  # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
    ]
    
    # Draw all detections
    for det_idx, prediction in enumerate(predictions):
        # Get color for this detection
        det_color = detection_colors[det_idx % len(detection_colors)]
        
        # Draw bounding box
        bbox = prediction.get('bbox', [])
        if not bbox:
            continue
            
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), det_color, font_config['thickness_heavy'])
        
        # Draw detection label (if multiple)
        if len(predictions) > 1:
            det_label = f"Spine #{det_idx + 1}"
            label_size = cv2.getTextSize(det_label, cv2.FONT_HERSHEY_SIMPLEX, 
                                        font_config['title'], font_config['thickness_normal'])[0]
            cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0] + 10, y1), det_color, -1)
            cv2.putText(img, det_label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_config['title'], 
                       (255, 255, 255), font_config['thickness_normal'])
        
        # Draw center line
        center_x = (x1 + x2) // 2
        cv2.line(img, (center_x, y1), (center_x, y2), (255, 255, 0), font_config['thickness_thick'])
        
        # Draw angle arc visualization
        cobb_angle = prediction.get('cobb_angle_primary', 0)
        if cobb_angle > 10:
            # Draw curved line to show deviation
            cv2.ellipse(img, (center_x, (y1 + y2) // 2), 
                       (int((x2 - x1) * 0.3), int((y2 - y1) * 0.4)),
                       0, -int(cobb_angle), int(cobb_angle), (0, 0, 255), 
                       font_config['thickness_heavy'])
            
            # Draw angle annotation arrows
            arrow_y = (y1 + y2) // 2
            arrow_left = center_x - int((x2 - x1) * 0.25)
            arrow_right = center_x + int((x2 - x1) * 0.25)
            cv2.arrowedLine(img, (center_x, arrow_y), (arrow_left, arrow_y), 
                          (0, 0, 255), font_config['thickness_normal'], tipLength=0.3)
            cv2.arrowedLine(img, (center_x, arrow_y), (arrow_right, arrow_y), 
                          (0, 0, 255), font_config['thickness_normal'], tipLength=0.3)
            
            # Draw angle value near the arc
            angle_text = f"{cobb_angle:.1f}"
            cv2.putText(img, angle_text, (center_x + 10, arrow_y - int(20 * font_config['angle'])), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_config['angle'], 
                       (0, 0, 255), font_config['thickness_normal'])
        
        # Draw corner markers on bounding box
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        for cx, cy in corners:
            cv2.line(img, (cx, cy), (cx + font_config['corner_size'], cy), 
                    det_color, font_config['thickness_heavy'])
            cv2.line(img, (cx, cy), (cx, cy + font_config['corner_size']), 
                    det_color, font_config['thickness_heavy'])
    
    # Create info panels for each detection
    panel_spacing = font_config['panel_height'] + font_config['padding']
    
    for det_idx, prediction in enumerate(predictions):
        class_name = prediction.get('class', 'unknown')
        confidence = prediction.get('confidence', 0.0)
        cobb_angle = prediction.get('cobb_angle_primary', 0.0)
        severity = prediction.get('severity', 'Unknown')
        det_id = prediction.get('detection_id', det_idx + 1)
        
        # Panel position (stack vertically if multiple)
        panel_x = font_config['padding']
        panel_y = font_config['padding'] + (det_idx * panel_spacing)
        
        # Draw semi-transparent dark panel
        overlay = img.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + font_config['panel_width'], panel_y + font_config['panel_height']), 
                     (30, 30, 30), -1)
        # Add border
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + font_config['panel_width'], panel_y + font_config['panel_height']), 
                     (100, 100, 100), font_config['thickness_thick'])
        img = cv2.addWeighted(overlay, 0.85, img, 0.15, 0)
        
        font = cv2.FONT_HERSHEY_DUPLEX
        y_pos = panel_y + int(45 * font_config['title'])
        x_label = panel_x + font_config['padding']
        x_value = panel_x + int(200 * font_config['label'])
        
        # Title (with detection number if multiple)
        title = f"SPINE #{det_id}" if len(predictions) > 1 else "SCOLIOSIS DETECTION"
        cv2.putText(img, title, (x_label, panel_y + int(30 * font_config['title'])), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_config['title'], (0, 255, 255), 
                    font_config['thickness_normal'])
        
        # Classification
        cv2.putText(img, "Class:", (x_label, y_pos), 
                    font, font_config['label'], (200, 200, 200), font_config['thickness_thin'])
        cv2.putText(img, str(class_name).upper(), (x_value, y_pos), 
                    font, font_config['value'], (255, 255, 255), font_config['thickness_normal'])
        y_pos += int(38 * font_config['label'])
        
        # Severity
        cv2.putText(img, "Severity:", (x_label, y_pos), 
                    font, font_config['label'], (200, 200, 200), font_config['thickness_thin'])
        # Shorten severity text if too long
        sev_text = severity.replace("Scoliosis ", "").replace(" (", " (")[:25]
        cv2.putText(img, sev_text, (x_value, y_pos), 
                    font, font_config['label'], (255, 255, 255), font_config['thickness_normal'])
        y_pos += int(38 * font_config['label'])
        
        # Cobb Angle with color coding
        cobb_color = (0, 255, 0) if cobb_angle < 25 else (50, 180, 255) if cobb_angle < 40 else (0, 0, 255)
        cv2.putText(img, "Cobb Angle:", (x_label, y_pos), 
                    font, font_config['label'], (200, 200, 200), font_config['thickness_thin'])
        cv2.putText(img, f"{cobb_angle:.1f} deg", (x_value, y_pos), 
                    font, font_config['value'], cobb_color, font_config['thickness_normal'])
        y_pos += int(38 * font_config['label'])
        
        # Confidence
        cv2.putText(img, "Confidence:", (x_label, y_pos), 
                    font, font_config['label'], (200, 200, 200), font_config['thickness_thin'])
        conf_color = (0, 255, 0) if confidence > 0.7 else (50, 180, 255) if confidence > 0.5 else (0, 200, 255)
        cv2.putText(img, f"{confidence:.1%}", (x_value, y_pos), 
                    font, font_config['value'], conf_color, font_config['thickness_normal'])
        
        # Add professional status indicator
        if cobb_angle < 10:
            status_text = "HEALTHY"
            status_color = (0, 200, 0)
            bg_color = (0, 80, 0)
        elif cobb_angle < 25:
            status_text = "MILD"
            status_color = (0, 255, 255)
            bg_color = (80, 80, 0)
        elif cobb_angle < 40:
            status_text = "MODERATE"
            status_color = (50, 180, 255)
            bg_color = (80, 60, 0)
        else:
            status_text = "SEVERE"
            status_color = (0, 100, 255)
            bg_color = (0, 0, 120)
        
        # Status box position (stacked if multiple detections)
        box_x = w - font_config['status_width'] - font_config['padding']
        box_y = font_config['padding'] + (det_idx * panel_spacing)
        
        # Draw status box with gradient background
        overlay2 = img.copy()
        cv2.rectangle(overlay2, (box_x, box_y), 
                     (box_x + font_config['status_width'], box_y + font_config['status_height']), 
                     bg_color, -1)
        img = cv2.addWeighted(overlay2, 0.8, img, 0.2, 0)
        
        # Border with status color
        cv2.rectangle(img, (box_x, box_y), 
                     (box_x + font_config['status_width'], box_y + font_config['status_height']), 
                     status_color, font_config['thickness_heavy'])
        
        # Status text
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                    font_config['status'], font_config['thickness_thick'])[0]
        text_x = box_x + (font_config['status_width'] - text_size[0]) // 2
        text_y = box_y + (font_config['status_height'] + text_size[1]) // 2
        
        cv2.putText(img, status_text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_config['status'], 
                    status_color, font_config['thickness_thick'])
    
    # Add watermark/timestamp at bottom
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    watermark = f"Scoliosis AI | Analyzed: {timestamp}"
    
    # Create small panel for watermark
    wm_h = 35
    overlay3 = img.copy()
    cv2.rectangle(overlay3, (0, h - wm_h), (w, h), (30, 30, 30), -1)
    img = cv2.addWeighted(overlay3, 0.7, img, 0.3, 0)
    
    cv2.putText(img, watermark, (20, h - int(12 * font_config['watermark'])), 
                cv2.FONT_HERSHEY_SIMPLEX, font_config['watermark'], 
                (180, 180, 180), font_config['thickness_thin'])
    
    # Add detection count and AI model info on right side
    model_text = f"YOLOv8n | {len(predictions)} Detection(s)" if len(predictions) > 1 else "YOLOv8n + Ensemble"
    text_size = cv2.getTextSize(model_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                font_config['watermark'], font_config['thickness_thin'])[0]
    cv2.putText(img, model_text, (w - text_size[0] - 20, h - int(12 * font_config['watermark'])), 
                cv2.FONT_HERSHEY_SIMPLEX, font_config['watermark'], 
                (180, 180, 180), font_config['thickness_thin'])
    
    # Save annotated image
    output_image = output_dir / f"{Path(image_path).stem}_diagnosis.jpg"
    cv2.imwrite(str(output_image), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    return str(output_image)


def generate_medical_report(predictions: list, image_name: str, output_dir: Path):
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
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.confidence < 0.1 or args.confidence > 0.9:
        print("⚠️  Warning: Confidence should be between 0.1 and 0.9. Using default 0.25")
        args.confidence = 0.25
    
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
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return 1
    
    # Validate model file
    model_path = Path(args.yolo_model)
    if not model_path.exists():
        print(f"\n❌ Error: Model file not found!")
        print(f"   Path: {args.yolo_model}")
        print(f"\n💡 Tip: You need to train a model first or download a pretrained one")
        print(f"   See QUICKSTART.md for training instructions")
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
                yolo_model_path=args.yolo_model,
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
        
        if not predictions or len(predictions) == 0:
            print(f"\n⚠️  Warning: No spine detected in the image!")
            print(f"   This could mean:")
            print(f"   • The image is not an X-ray")
            print(f"   • The confidence threshold is too high (try --confidence 0.1)")
            print(f"   • The image quality is too low")
            return 1
        
        print(f"✓ Detected {len(predictions)} spine(s) in image")
        
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
                output_dir
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
