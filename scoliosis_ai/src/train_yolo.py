"""
YOLOv8 Training Module for Scoliosis Detection
Trains object detection model to identify vertebrae and classify scoliosis severity
"""

import os
import sys
import logging
from pathlib import Path
from ultralytics import YOLO
import yaml

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from src.config import YOLO_CONFIG, DATASET_CONFIG, TRAINING_CONFIG, LOGGING_CONFIG
from src.utils import set_seed, setup_logging


class YOLOTrainer:
    """YOLOv8 trainer for scoliosis detection"""
    
    def __init__(self, config=None):
        """Initialize YOLO trainer
        
        Args:
            config: Dictionary with training configuration (optional)
        """
        self.config = config or YOLO_CONFIG
        self.logger = setup_logging(LOGGING_CONFIG['log_file'])
        set_seed(TRAINING_CONFIG['seed'])
        
        # Create save directories
        os.makedirs(self.config['save_dir'], exist_ok=True)
        
        self.logger.info("YOLOv8 Trainer initialized")
        self.logger.info(f"Model: {self.config['model_name']}")
        self.logger.info(f"Epochs: {self.config['epochs']}")
        self.logger.info(f"Batch size: {self.config['batch_size']}")
    
    def train(self, data_yaml=None, resume=False):
        """Train YOLOv8 model
        
        Args:
            data_yaml: Path to data.yaml file
            resume: Whether to resume from last checkpoint
        
        Returns:
            Training results
        """
        # Use dataset config if not provided
        if data_yaml is None:
            data_yaml = DATASET_CONFIG['yaml_path']
        
        self.logger.info(f"Training with data config: {data_yaml}")
        
        # Load pre-trained YOLOv8 model
        model = YOLO(self.config['model_name'])
        
        # Training arguments with advanced features
        train_args = {
            # Basic training
            'data': data_yaml,
            'epochs': self.config['epochs'],
            'batch': self.config['batch_size'],
            'imgsz': self.config['img_size'],
            'patience': self.config['patience'],
            'device': 'cpu',  # Force CPU
            'workers': TRAINING_CONFIG['num_workers'],
            
            # Optimizer settings
            'optimizer': self.config['optimizer'],
            'lr0': self.config['lr0'],
            'lrf': self.config.get('lrf', 0.01),
            'momentum': self.config.get('momentum', 0.937),
            'weight_decay': self.config['weight_decay'],
            'cos_lr': self.config.get('cos_lr', True),
            
            # Warmup
            'warmup_epochs': self.config.get('warmup_epochs', 3.0),
            'warmup_momentum': self.config.get('warmup_momentum', 0.8),
            'warmup_bias_lr': self.config.get('warmup_bias_lr', 0.1),
            
            # Augmentation - HSV
            'hsv_h': self.config.get('hsv_h', 0.015),
            'hsv_s': self.config.get('hsv_s', 0.7),
            'hsv_v': self.config.get('hsv_v', 0.4),
            
            # Augmentation - Geometric
            'degrees': self.config.get('degrees', 0.0),
            'translate': self.config.get('translate', 0.1),
            'scale': self.config.get('scale', 0.5),
            'shear': self.config.get('shear', 0.0),
            'perspective': self.config.get('perspective', 0.0),
            'flipud': self.config.get('flipud', 0.0),
            'fliplr': self.config.get('fliplr', 0.5),
            
            # Augmentation - Advanced
            'mosaic': self.config.get('mosaic', 1.0),
            'mixup': self.config.get('mixup', 0.0),
            'copy_paste': self.config.get('copy_paste', 0.0),
            'auto_augment': self.config.get('auto_augment', None),
            'erasing': self.config.get('erasing', 0.4),
            'crop_fraction': self.config.get('crop_fraction', 1.0),
            
            # Multi-scale
            'rect': self.config.get('rect', False),
            'close_mosaic': self.config.get('close_mosaic', 10),
            
            # Model settings
            'pretrained': True,
            'freeze': self.config.get('freeze', None),
            
            # Saving
            'project': str(self.config['save_dir']),
            'name': 'scoliosis_yolo_enhanced',
            'exist_ok': True,
            'save': True,
            'save_period': TRAINING_CONFIG['save_frequency'],
            
            # Visualization & Logging
            'plots': True,
            'verbose': True,
            'resume': resume,
            'amp': False,  # AMP not supported on CPU
            'fraction': self.config.get('fraction', 1.0),
            'profile': self.config.get('profile', False),
            
            # Validation
            'val': True,
        }
        
        self.logger.info("Starting YOLOv8 training...")
        self.logger.info(f"Training arguments: {train_args}")
        
        try:
            # Train model
            results = model.train(**train_args)
            
            self.logger.info("Training completed successfully!")
            self.logger.info(f"Best model saved to: {self.config['save_dir']}/scoliosis_yolo_enhanced/weights/best.pt")
            
            # Log final metrics
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
                self.logger.info("\n" + "="*60)
                self.logger.info("FINAL TRAINING METRICS")
                self.logger.info("="*60)
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.logger.info(f"{key}: {value:.4f}")
                self.logger.info("="*60 + "\n")
            
            return results
        
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
    
    def validate(self, model_path=None, data_yaml=None):
        """Validate trained model
        
        Args:
            model_path: Path to trained model weights
            data_yaml: Path to data.yaml file
        
        Returns:
            Validation metrics
        """
        if model_path is None:
            model_path = self.config['save_dir'] / 'scoliosis_yolo_enhanced' / 'weights' / 'best.pt'
        
        if data_yaml is None:
            data_yaml = DATASET_CONFIG['yaml_path']
        
        self.logger.info(f"Validating model: {model_path}")
        
        # Load trained model
        model = YOLO(str(model_path))
        
        # Validate
        metrics = model.val(
            data=data_yaml,
            imgsz=self.config['img_size'],
            batch=self.config['batch_size'],
            device='cpu',
            plots=True,
            save_json=True,
        )
        
        # Log metrics
        self.logger.info("Validation Results:")
        self.logger.info(f"mAP@0.5: {metrics.box.map50:.4f}")
        self.logger.info(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
        self.logger.info(f"Precision: {metrics.box.mp:.4f}")
        self.logger.info(f"Recall: {metrics.box.mr:.4f}")
        
        return metrics
    
    def predict(self, image_path, model_path=None, save=True):
        """Run inference on single image
        
        Args:
            image_path: Path to input image
            model_path: Path to trained model weights
            save: Whether to save results
        
        Returns:
            Prediction results
        """
        if model_path is None:
            model_path = self.config['save_dir'] / 'scoliosis_yolo_enhanced' / 'weights' / 'best.pt'
        
        self.logger.info(f"Running inference on: {image_path}")
        
        # Load model
        model = YOLO(str(model_path))
        
        # Predict
        results = model.predict(
            source=image_path,
            imgsz=self.config['img_size'],
            conf=self.config['conf_threshold'],
            iou=self.config['iou_threshold'],
            device='cpu',
            save=save,
            project=str(Path(self.config['save_dir']).parent.parent / 'outputs'),
            name='predictions',
        )
        
        return results
    
    def export_model(self, model_path=None, format='onnx'):
        """Export model to different formats
        
        Args:
            model_path: Path to trained model weights
            format: Export format (onnx, torchscript, etc.)
        
        Returns:
            Export path
        """
        if model_path is None:
            model_path = self.config['save_dir'] / 'scoliosis_yolo_enhanced' / 'weights' / 'best.pt'
        
        self.logger.info(f"Exporting model to {format} format...")
        
        model = YOLO(str(model_path))
        export_path = model.export(format=format)
        
        self.logger.info(f"Model exported to: {export_path}")
        return export_path


def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLOv8 for Scoliosis Detection')
    parser.add_argument('--data', type=str, default=None, help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    parser.add_argument('--validate', action='store_true', help='Run validation only')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config = YOLO_CONFIG.copy()
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.imgsz:
        config['img_size'] = args.imgsz
    
    # Initialize trainer
    trainer = YOLOTrainer(config)
    
    if args.validate:
        # Run validation only
        trainer.validate(data_yaml=args.data)
    else:
        # Train model
        trainer.train(data_yaml=args.data, resume=args.resume)
        
        # Validate best model
        trainer.validate(data_yaml=args.data)


if __name__ == "__main__":
    main()
