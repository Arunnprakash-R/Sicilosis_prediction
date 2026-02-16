"""
Utility functions for the Scoliosis Detection System
"""

import os
import random
import numpy as np
import torch
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from datetime import datetime


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging(log_file=None, log_level=logging.INFO):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


def get_device():
    """Get the best available device (CPU in this case)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logging.info("Running on CPU (CUDA not available)")
    else:
        logging.info(f"Running on GPU: {torch.cuda.get_device_name(0)}")
    return device


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    logging.info(f"Checkpoint saved: {path}")


def load_checkpoint(model, optimizer, path):
    """Load model checkpoint"""
    if not os.path.exists(path):
        logging.warning(f"Checkpoint not found: {path}")
        return 0, float('inf')
    
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    
    logging.info(f"Checkpoint loaded: {path} (epoch {epoch})")
    return epoch, loss


def plot_training_history(history, save_path=None):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot metric (e.g., accuracy or MAE)
    if 'train_metric' in history and 'val_metric' in history:
        axes[1].plot(history['train_metric'], label='Train Metric')
        axes[1].plot(history['val_metric'], label='Validation Metric')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Metric')
        axes[1].set_title('Training and Validation Metric')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"Training history plot saved: {save_path}")
    
    plt.close()


def visualize_predictions(image, predictions, ground_truth=None, save_path=None):
    """Visualize detection predictions on image"""
    img = image.copy()
    h, w = img.shape[:2]
    
    # Draw predictions
    for pred in predictions:
        x1, y1, x2, y2 = map(int, pred['bbox'])
        class_name = pred['class']
        confidence = pred['confidence']
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw ground truth if available
    if ground_truth:
        for gt in ground_truth:
            x1, y1, x2, y2 = map(int, gt['bbox'])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img)
        logging.info(f"Prediction visualization saved: {save_path}")
    
    return img


def calculate_metrics(predictions, ground_truth, metric_type='detection'):
    """Calculate evaluation metrics"""
    if metric_type == 'detection':
        # Calculate precision, recall, F1
        tp = fp = fn = 0
        for pred, gt in zip(predictions, ground_truth):
            # Simple IoU-based matching
            if iou(pred['bbox'], gt['bbox']) > 0.5:
                tp += 1
            else:
                fp += 1
        fn = len(ground_truth) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {'precision': precision, 'recall': recall, 'f1_score': f1}
    
    elif metric_type == 'regression':
        # Calculate MAE, RMSE for Cobb angle
        errors = np.array(predictions) - np.array(ground_truth)
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))
        
        return {'mae': mae, 'rmse': rmse}


def iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection area
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)
    
    if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
        return 0.0
    
    inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
    
    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def preprocess_image(image_path, target_size=(640, 640)):
    """Preprocess image for model input"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    return img


def postprocess_yolo_output(results, conf_threshold=0.25):
    """Convert YOLO results to standardized format"""
    predictions = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue
            
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            class_id = int(box.cls[0])
            
            predictions.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': conf,
                'class_id': class_id,
                'class': result.names[class_id]
            })
    
    return predictions


def create_submission_file(predictions, output_path):
    """Create submission file in required format"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("image_id,class,confidence,bbox\n")
        for pred in predictions:
            image_id = pred['image_id']
            class_name = pred['class']
            confidence = pred['confidence']
            bbox = ",".join(map(str, pred['bbox']))
            f.write(f"{image_id},{class_name},{confidence},{bbox}\n")
    
    logging.info(f"Submission file created: {output_path}")


if __name__ == "__main__":
    # Test utilities
    set_seed(42)
    logger = setup_logging()
    device = get_device()
    logger.info(f"Device: {device}")
