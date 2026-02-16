# Dataset Validation Script
# Validates YOLO format dataset and provides statistics

import os
import sys
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import cv2

sys.path.append(str(Path(__file__).parent.parent))
from src.config import DATASET_CONFIG


def validate_dataset():
    """Validate YOLO format dataset"""
    
    print("="*80)
    print("DATASET VALIDATION")
    print("="*80)
    
    dataset_root = Path(DATASET_CONFIG['root'])
    yaml_path = Path(DATASET_CONFIG['yaml_path'])
    
    print(f"\nDataset root: {dataset_root}")
    print(f"YAML config: {yaml_path}")
    
    # Load data.yaml
    if not yaml_path.exists():
        print(f"ERROR: data.yaml not found at {yaml_path}")
        return
    
    with open(yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"\nClasses: {data_config.get('names', [])}")
    print(f"Number of classes: {data_config.get('nc', 0)}")
    
    # Validate splits
    splits = ['train', 'valid', 'test']
    stats = {}
    
    for split in splits:
        print(f"\n{'-'*80}")
        print(f"Validating {split.upper()} split")
        print(f"{'-'*80}")
        
        images_dir = dataset_root / split / 'images'
        labels_dir = dataset_root / split / 'labels'
        
        if not images_dir.exists():
            print(f"⚠ {split}/images directory not found")
            continue
        
        if not labels_dir.exists():
            print(f"⚠ {split}/labels directory not found")
            continue
        
        # Count images and labels
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        label_files = list(labels_dir.glob('*.txt'))
        
        print(f"Images: {len(image_files)}")
        print(f"Labels: {len(label_files)}")
        
        # Check image-label correspondence
        mismatched = []
        for img_file in image_files:
            label_file = labels_dir / f"{img_file.stem}.txt"
            if not label_file.exists():
                mismatched.append(img_file.name)
        
        if mismatched:
            print(f"⚠ {len(mismatched)} images without labels:")
            for name in mismatched[:5]:
                print(f"  - {name}")
            if len(mismatched) > 5:
                print(f"  ... and {len(mismatched) - 5} more")
        else:
            print("✓ All images have corresponding labels")
        
        # Analyze class distribution
        class_counts = Counter()
        bbox_counts = []
        
        for label_file in label_files:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                bbox_counts.append(len(lines))
                
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1
        
        print(f"\nClass distribution:")
        for class_id, count in sorted(class_counts.items()):
            class_name = data_config['names'][class_id] if class_id < len(data_config['names']) else 'Unknown'
            print(f"  Class {class_id} ({class_name}): {count} instances")
        
        print(f"\nBounding boxes per image:")
        print(f"  Mean: {np.mean(bbox_counts):.2f}")
        print(f"  Median: {np.median(bbox_counts):.0f}")
        print(f"  Min: {min(bbox_counts) if bbox_counts else 0}")
        print(f"  Max: {max(bbox_counts) if bbox_counts else 0}")
        
        # Sample images
        if image_files:
            sample_img = cv2.imread(str(image_files[0]))
            if sample_img is not None:
                h, w = sample_img.shape[:2]
                print(f"\nSample image dimensions: {w}x{h}")
        
        stats[split] = {
            'num_images': len(image_files),
            'num_labels': len(label_files),
            'class_counts': dict(class_counts),
            'avg_boxes_per_image': np.mean(bbox_counts) if bbox_counts else 0
        }
    
    # Summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")
    
    total_images = sum(s['num_images'] for s in stats.values())
    total_labels = sum(s['num_labels'] for s in stats.values())
    
    print(f"\nTotal images: {total_images}")
    print(f"Total labels: {total_labels}")
    print(f"Match: {total_images == total_labels}")
    
    for split, split_stats in stats.items():
        print(f"\n{split.capitalize()}:")
        print(f"  Images: {split_stats['num_images']}")
        print(f"  Avg boxes/image: {split_stats['avg_boxes_per_image']:.2f}")
    
    print(f"\n{'='*80}")
    print("✓ Dataset validation complete!")
    print(f"{'='*80}")
    
    return stats


def visualize_samples(num_samples=5):
    """Visualize sample images with annotations"""
    
    print("\n" + "="*80)
    print("VISUALIZING SAMPLES")
    print("="*80)
    
    dataset_root = Path(DATASET_CONFIG['root'])
    yaml_path = Path(DATASET_CONFIG['yaml_path'])
    
    with open(yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    class_names = data_config['names']
    
    # Get sample images from train set
    images_dir = dataset_root / 'train' / 'images'
    labels_dir = dataset_root / 'train' / 'labels'
    
    image_files = list(images_dir.glob('*.jpg'))[:num_samples]
    
    fig, axes = plt.subplots(1, min(num_samples, 5), figsize=(20, 5))
    if num_samples == 1:
        axes = [axes]
    
    for idx, img_file in enumerate(image_files):
        # Load image
        img = cv2.imread(str(img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Load labels
        label_file = labels_dir / f"{img_file.stem}.txt"
        
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        
                        # Convert YOLO format to pixel coordinates
                        x1 = int((x_center - width/2) * w)
                        y1 = int((y_center - height/2) * h)
                        x2 = int((x_center + width/2) * w)
                        y2 = int((y_center + height/2) * h)
                        
                        # Draw bounding box
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label
                        label = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
                        cv2.putText(img, label, (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display
        axes[idx].imshow(img)
        axes[idx].axis('off')
        axes[idx].set_title(img_file.name, fontsize=8)
    
    plt.tight_layout()
    
    # Save visualization
    output_dir = Path(__file__).parent.parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'dataset_samples.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    # Run validation
    stats = validate_dataset()
    
    # Visualize samples
    try:
        visualize_samples(num_samples=5)
    except Exception as e:
        print(f"\n⚠ Visualization failed: {e}")
