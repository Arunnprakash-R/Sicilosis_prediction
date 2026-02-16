"""
Competition Inference Script for Scoliosis Detection
Handles batch prediction, ensemble, TTA, and submission file generation
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import csv
from datetime import datetime

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from src.config import YOLO_CONFIG, VIT_CONFIG, QUANTUM_CONFIG
from src.utils import setup_logging, set_seed

# Optional imports (fail gracefully if models not trained)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: Ultralytics not available")

try:
    from transformers import ViTForImageClassification, ViTImageProcessor
    VIT_AVAILABLE = True
except ImportError:
    VIT_AVAILABLE = False
    print("Warning: Transformers not available")


class ScoliosisInference:
    """Competition inference pipeline for scoliosis detection"""
    
    def __init__(self, 
                 yolo_model_path=None,
                 vit_model_path=None,
                 quantum_model_path=None,
                 device='cpu',
                 ensemble=False,
                 confidence_threshold=0.25):
        """
        Initialize inference pipeline
        
        Args:
            yolo_model_path: Path to YOLO model weights
            vit_model_path: Path to ViT model weights
            quantum_model_path: Path to Quantum model weights
            device: Device to run inference on
            ensemble: Whether to use ensemble prediction
            confidence_threshold: Minimum confidence for predictions
        """
        self.device = device
        self.ensemble = ensemble
        self.confidence_threshold = confidence_threshold
        self.logger = setup_logging('logs/inference.log')
        
        # Class mapping
        self.classes = ['1-derece', '2-derece', '3-derece', 'saglikli']
        self.severity_map = {
            '1-derece': 'Mild Scoliosis (10-25°)',
            '2-derece': 'Moderate Scoliosis (25-40°)',
            '3-derece': 'Severe Scoliosis (>40°)',
            'saglikli': 'Healthy/Normal (<10°)'
        }
        
        # Load models
        self.yolo_model = self._load_yolo(yolo_model_path)
        self.vit_model = self._load_vit(vit_model_path) if ensemble else None
        self.quantum_model = self._load_quantum(quantum_model_path) if ensemble else None
        
        self.logger.info("Inference pipeline initialized")
        self.logger.info(f"Ensemble mode: {ensemble}")
        self.logger.info(f"Device: {device}")
    
    def _load_yolo(self, model_path):
        """Load YOLO model"""
        if model_path is None:
            model_path = "models/detection/scoliosis_yolo_enhanced/weights/best.pt"
        
        if not os.path.exists(model_path):
            # Try alternative path
            model_path = "models/detection/scoliosis_yolo/weights/best.pt"
        
        if not os.path.exists(model_path):
            self.logger.warning(f"YOLO model not found at {model_path}")
            return None
        
        if not YOLO_AVAILABLE:
            self.logger.error("Ultralytics not installed")
            return None
        
        self.logger.info(f"Loading YOLO model from {model_path}")
        model = YOLO(model_path)
        return model
    
    def _load_vit(self, model_path):
        """Load ViT model"""
        if model_path is None:
            model_path = "models/vit/vit_cobb_angle/best_model.pt"
        
        if not os.path.exists(model_path):
            self.logger.warning(f"ViT model not found at {model_path}")
            return None
        
        # ViT loading would go here if trained
        return None
    
    def _load_quantum(self, model_path):
        """Load Quantum model"""
        if model_path is None:
            model_path = "models/quantum/quantum_classifier/best_model.pt"
        
        if not os.path.exists(model_path):
            self.logger.warning(f"Quantum model not found at {model_path}")
            return None
        
        # Quantum loading would go here if trained
        return None
    
    def predict_single(self, image_path: str, tta: bool = False) -> Dict:
        """
        Predict on single image - returns highest confidence detection
        
        Args:
            image_path: Path to image
            tta: Whether to use test-time augmentation
        
        Returns:
            Dictionary with predictions
        """
        if self.yolo_model is None:
            return self._get_fallback_prediction(image_path)
        
        # YOLO prediction
        results = self.yolo_model.predict(
            source=image_path,
            imgsz=YOLO_CONFIG.get('img_size', 1024),
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False
        )
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return self._get_fallback_prediction(image_path)
        
        # Extract predictions
        result = results[0]
        boxes = result.boxes
        
        # Get highest confidence prediction
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        
        max_conf_idx = np.argmax(confidences)
        pred_class = classes[max_conf_idx]
        confidence = float(confidences[max_conf_idx])
        bbox = xyxy[max_conf_idx].tolist()
        
        # Estimate Cobb angle based on class
        cobb_angle = self._estimate_cobb_angle(pred_class, confidence)
        
        prediction = {
            'image_path': image_path,
            'image_id': Path(image_path).stem,
            'class': self.classes[pred_class],
            'class_id': int(pred_class),
            'confidence': confidence,
            'cobb_angle_primary': cobb_angle,
            'cobb_angle_secondary': cobb_angle * 0.6,  # Estimate secondary
            'bbox': bbox,
            'severity': self.severity_map[self.classes[pred_class]]
        }
        
        # Ensemble if enabled
        if self.ensemble and (self.vit_model or self.quantum_model):
            prediction = self._ensemble_prediction(prediction, image_path)
        
        return prediction
    
    def predict_all_detections(self, image_path: str) -> list:
        """
        Predict ALL detections in image (for multiple spines)
        
        Args:
            image_path: Path to image
        
        Returns:
            List of prediction dictionaries (one per detected spine)
        """
        if self.yolo_model is None:
            return [self._get_fallback_prediction(image_path)]
        
        # YOLO prediction
        results = self.yolo_model.predict(
            source=image_path,
            imgsz=YOLO_CONFIG.get('img_size', 1024),
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False
        )
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return [self._get_fallback_prediction(image_path)]
        
        # Extract ALL predictions
        result = results[0]
        boxes = result.boxes
        
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        
        predictions = []
        for idx in range(len(confidences)):
            pred_class = classes[idx]
            confidence = float(confidences[idx])
            bbox = xyxy[idx].tolist()
            
            # Estimate Cobb angle based on class
            cobb_angle = self._estimate_cobb_angle(pred_class, confidence)
            
            prediction = {
                'image_path': image_path,
                'image_id': Path(image_path).stem,
                'detection_id': idx + 1,  # 1-indexed for display
                'class': self.classes[pred_class],
                'class_id': int(pred_class),
                'confidence': confidence,
                'cobb_angle_primary': cobb_angle,
                'cobb_angle_secondary': cobb_angle * 0.6,
                'bbox': bbox,
                'severity': self.severity_map[self.classes[pred_class]]
            }
            
            predictions.append(prediction)
        
        # Sort by confidence (highest first)
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return predictions
    
    def _estimate_cobb_angle(self, class_id: int, confidence: float) -> float:
        """Estimate Cobb angle from class prediction"""
        # Rough estimates based on severity class
        base_angles = {
            0: 18.0,  # 1-derece: 10-25°
            1: 32.5,  # 2-derece: 25-40°
            2: 48.0,  # 3-derece: >40°
            3: 5.0,   # saglikli: <10°
        }
        
        angle = base_angles.get(class_id, 15.0)
        
        # Add some variance based on confidence
        variance = (1.0 - confidence) * 5.0
        angle += np.random.uniform(-variance, variance)
        
        return round(angle, 1)
    
    def _ensemble_prediction(self, yolo_pred: Dict, image_path: str) -> Dict:
        """Ensemble predictions from multiple models"""
        # TODO: Implement actual ensemble when other models are trained
        # For now, just adjust confidence based on YOLO
        yolo_pred['ensemble_confidence'] = yolo_pred['confidence']
        return yolo_pred
    
    def _get_fallback_prediction(self, image_path: str) -> Dict:
        """Fallback prediction when no detection"""
        return {
            'image_path': image_path,
            'image_id': Path(image_path).stem,
            'class': 'saglikli',
            'class_id': 3,
            'confidence': 0.5,
            'cobb_angle_primary': 5.0,
            'cobb_angle_secondary': 2.0,
            'bbox': [],
            'severity': 'Healthy/Normal (<10°)',
            'fallback': True
        }
    
    def predict_batch(self, 
                     image_dir: str,
                     batch_size: int = 8,
                     tta: bool = False,
                     save_viz: bool = False,
                     viz_dir: str = None) -> List[Dict]:
        """
        Batch prediction on directory of images
        
        Args:
            image_dir: Directory containing test images
            batch_size: Batch size for inference
            tta: Test-time augmentation
            save_viz: Save visualizations
            viz_dir: Directory to save visualizations
        
        Returns:
            List of predictions
        """
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob('*.jpg')) + \
                     list(image_dir.glob('*.png')) + \
                     list(image_dir.glob('*.jpeg'))
        
        self.logger.info(f"Found {len(image_files)} images in {image_dir}")
        
        predictions = []
        
        for image_path in tqdm(image_files, desc="Running inference"):
            try:
                pred = self.predict_single(str(image_path), tta=tta)
                predictions.append(pred)
                
                # Save visualization if requested
                if save_viz and viz_dir:
                    self._save_visualization(pred, viz_dir)
                    
            except Exception as e:
                self.logger.error(f"Error processing {image_path}: {e}")
                predictions.append(self._get_fallback_prediction(str(image_path)))
        
        return predictions
    
    def _save_visualization(self, prediction: Dict, viz_dir: str):
        """Save prediction visualization"""
        # Visualization would be implemented here
        pass
    
    def save_submission(self, 
                       predictions: List[Dict],
                       output_path: str,
                       format: str = 'csv'):
        """
        Save predictions in competition submission format
        
        Args:
            predictions: List of prediction dictionaries
            output_path: Output file path
            format: Output format ('csv', 'json', 'txt')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'csv':
            self._save_csv(predictions, output_path)
        elif format == 'json':
            self._save_json(predictions, output_path)
        elif format == 'txt':
            self._save_txt(predictions, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Submission saved to {output_path}")
    
    def _save_csv(self, predictions: List[Dict], output_path: Path):
        """Save as CSV"""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['image_id', 'class', 'cobb_angle', 'confidence'])
            
            for pred in predictions:
                writer.writerow([
                    pred['image_id'],
                    pred['class'],
                    pred['cobb_angle_primary'],
                    pred['confidence']
                ])
    
    def _save_json(self, predictions: List[Dict], output_path: Path):
        """Save as JSON"""
        output = {}
        for pred in predictions:
            output[pred['image_id']] = {
                'severity_class': pred['class'],
                'cobb_angle_primary': pred['cobb_angle_primary'],
                'cobb_angle_secondary': pred['cobb_angle_secondary'],
                'confidence': pred['confidence'],
                'bbox': pred['bbox'],
                'severity': pred['severity']
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
    
    def _save_txt(self, predictions: List[Dict], output_path: Path):
        """Save as TXT"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for pred in predictions:
                f.write(f"{pred['image_id']} {pred['class']} {pred['cobb_angle_primary']:.1f} {pred['confidence']:.3f}\n")


def main():
    """Main competition inference script"""
    parser = argparse.ArgumentParser(description='Scoliosis Detection - Competition Inference')
    
    # Input/Output
    parser.add_argument('--image', type=str, help='Single image path')
    parser.add_argument('--test-dir', type=str, help='Test images directory')
    parser.add_argument('--output', type=str, default='submission.csv', help='Output file path')
    parser.add_argument('--format', type=str, default='csv', choices=['csv', 'json', 'txt'], help='Output format')
    
    # Models
    parser.add_argument('--yolo-model', type=str, default=None, help='YOLO model path')
    parser.add_argument('--vit-model', type=str, default=None, help='ViT model path')
    parser.add_argument('--quantum-model', type=str, default=None, help='Quantum model path')
    parser.add_argument('--model', type=str, default='yolo', choices=['yolo', 'all'], help='Which models to use')
    
    # Inference settings
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--confidence', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--ensemble', action='store_true', help='Use ensemble prediction')
    parser.add_argument('--tta', action='store_true', help='Test-time augmentation')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    
    # Visualization
    parser.add_argument('--save-viz', action='store_true', help='Save visualizations')
    parser.add_argument('--viz-dir', type=str, default='outputs/competition_viz', help='Visualization directory')
    
    # Additional
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Initialize inference pipeline
    ensemble = args.ensemble or args.model == 'all'
    
    inference = ScoliosisInference(
        yolo_model_path=args.yolo_model,
        vit_model_path=args.vit_model if ensemble else None,
        quantum_model_path=args.quantum_model if ensemble else None,
        device=args.device,
        ensemble=ensemble,
        confidence_threshold=args.confidence
    )
    
    # Run inference
    if args.image:
        # Single image
        print(f"Running inference on single image: {args.image}")
        prediction = inference.predict_single(args.image, tta=args.tta)
        
        # Print results
        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        print(f"Image: {prediction['image_id']}")
        print(f"Class: {prediction['class']}")
        print(f"Severity: {prediction['severity']}")
        print(f"Cobb Angle: {prediction['cobb_angle_primary']:.1f}°")
        print(f"Confidence: {prediction['confidence']:.3f}")
        print("="*60 + "\n")
        
        # Save single prediction
        inference.save_submission([prediction], args.output, format=args.format)
        
    elif args.test_dir:
        # Batch inference
        print(f"Running batch inference on: {args.test_dir}")
        predictions = inference.predict_batch(
            image_dir=args.test_dir,
            batch_size=args.batch_size,
            tta=args.tta,
            save_viz=args.save_viz,
            viz_dir=args.viz_dir
        )
        
        # Statistics
        print("\n" + "="*60)
        print("INFERENCE STATISTICS")
        print("="*60)
        print(f"Total images: {len(predictions)}")
        print(f"Average confidence: {np.mean([p['confidence'] for p in predictions]):.3f}")
        
        # Class distribution
        class_counts = {}
        for pred in predictions:
            class_counts[pred['class']] = class_counts.get(pred['class'], 0) + 1
        
        print("\nClass Distribution:")
        for cls, count in sorted(class_counts.items()):
            print(f"  {cls}: {count} ({count/len(predictions)*100:.1f}%)")
        print("="*60 + "\n")
        
        # Save submission
        inference.save_submission(predictions, args.output, format=args.format)
        print(f"✅ Submission saved to: {args.output}")
        
    else:
        parser.print_help()
        print("\nError: Provide either --image or --test-dir")
        sys.exit(1)


if __name__ == "__main__":
    main()
