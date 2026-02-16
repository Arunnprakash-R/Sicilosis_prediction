"""
Vision Transformer Training Module for Cobb Angle Regression
Fine-tunes pretrained ViT for direct Cobb angle prediction
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import logging

sys.path.append(str(Path(__file__).parent.parent))
from src.config import VIT_CONFIG, TRAINING_CONFIG, LOGGING_CONFIG
from src.utils import set_seed, setup_logging, save_checkpoint, load_checkpoint


class ScoliosisDataset(Dataset):
    """Dataset for Cobb angle regression"""
    
    def __init__(self, image_dir, labels_dict, processor, augment=False):
        """
        Args:
            image_dir: Directory with spine X-ray images
            labels_dict: Dictionary mapping image_id to Cobb angle
            processor: ViT image processor
            augment: Whether to apply data augmentation
        """
        self.image_dir = Path(image_dir)
        self.labels_dict = labels_dict
        self.processor = processor
        self.augment = augment
        
        # Get list of valid images
        self.image_files = [
            f for f in self.image_dir.glob('*.jpg')
            if f.stem in labels_dict
        ]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply augmentation if enabled
        if self.augment:
            image = self._augment(image)
        
        # Process image for ViT
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        # Get Cobb angle label
        image_id = img_path.stem
        cobb_angle = self.labels_dict[image_id]
        
        return {
            'pixel_values': pixel_values,
            'labels': torch.tensor(cobb_angle, dtype=torch.float32),
            'image_id': image_id
        }
    
    def _augment(self, image):
        """Apply random augmentation"""
        # Random horizontal flip
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
        
        # Random brightness
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)
            image = cv2.convertScaleAbs(image, alpha=alpha)
        
        return image


class ViTCobbRegressor(nn.Module):
    """Vision Transformer with regression head for Cobb angle prediction"""
    
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        
        # Load pretrained ViT
        self.vit = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=1,  # Regression task
            ignore_mismatched_sizes=True
        )
        
        # Modify classifier for regression
        hidden_size = self.vit.config.hidden_size
        self.vit.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)  # Single output for Cobb angle
        )
    
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values)
        return outputs.logits.squeeze(-1)


class ViTTrainer:
    """Trainer for Vision Transformer Cobb angle regression"""
    
    def __init__(self, config=None):
        self.config = config or VIT_CONFIG
        self.logger = setup_logging(LOGGING_CONFIG['log_file'])
        set_seed(TRAINING_CONFIG['seed'])
        
        self.device = torch.device('cpu')  # CPU only
        
        # Create save directory
        os.makedirs(self.config['save_dir'], exist_ok=True)
        
        self.logger.info("ViT Trainer initialized")
        self.logger.info(f"Model: {self.config['model_name']}")
        self.logger.info(f"Device: {self.device}")
    
    def prepare_data(self, train_dir, val_dir, labels_dict):
        """Prepare data loaders
        
        Args:
            train_dir: Training images directory
            val_dir: Validation images directory
            labels_dict: Dictionary mapping image_id to Cobb angle
        
        Returns:
            train_loader, val_loader
        """
        # Load image processor
        processor = ViTImageProcessor.from_pretrained(self.config['model_name'])
        
        # Create datasets
        train_dataset = ScoliosisDataset(train_dir, labels_dict, processor, augment=True)
        val_dataset = ScoliosisDataset(val_dir, labels_dict, processor, augment=False)
        
        self.logger.info(f"Train dataset size: {len(train_dataset)}")
        self.logger.info(f"Val dataset size: {len(val_dataset)}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=TRAINING_CONFIG['num_workers'],
            pin_memory=TRAINING_CONFIG['pin_memory']
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=TRAINING_CONFIG['num_workers'],
            pin_memory=TRAINING_CONFIG['pin_memory']
        )
        
        return train_loader, val_loader
    
    def train(self, train_loader, val_loader, resume=False):
        """Train ViT model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            resume: Whether to resume from checkpoint
        
        Returns:
            Training history
        """
        # Initialize model
        model = ViTCobbRegressor(
            self.config['model_name'],
            dropout=self.config['dropout']
        ).to(self.device)
        
        # Loss function (MAE for regression)
        criterion = nn.L1Loss()
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Load checkpoint if resuming
        start_epoch = 0
        best_val_loss = float('inf')
        if resume:
            checkpoint_path = self.config['save_dir'] / 'checkpoint.pt'
            start_epoch, best_val_loss = load_checkpoint(model, optimizer, checkpoint_path)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }
        
        self.logger.info("Starting ViT training...")
        
        # Training loop
        for epoch in range(start_epoch, self.config['epochs']):
            # Train epoch
            train_loss, train_mae = self._train_epoch(model, train_loader, criterion, optimizer)
            
            # Validate epoch
            val_loss, val_mae = self._validate_epoch(model, val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_mae'].append(train_mae)
            history['val_mae'].append(val_mae)
            
            # Log progress
            self.logger.info(
                f"Epoch {epoch+1}/{self.config['epochs']} - "
                f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = self.config['save_dir'] / 'best.pt'
                save_checkpoint(model, optimizer, epoch, val_loss, best_path)
                self.logger.info(f"Best model saved (Val MAE: {val_mae:.4f}°)")
            
            # Save periodic checkpoint
            if (epoch + 1) % TRAINING_CONFIG['save_frequency'] == 0:
                checkpoint_path = self.config['save_dir'] / 'checkpoint.pt'
                save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
        
        self.logger.info("Training completed!")
        return history
    
    def _train_epoch(self, model, loader, criterion, optimizer):
        """Train for one epoch"""
        model.train()
        total_loss = 0
        total_mae = 0
        
        for batch in tqdm(loader, desc="Training"):
            pixel_values = batch['pixel_values'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(pixel_values)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                self.config['max_grad_norm']
            )
            
            optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_mae += torch.abs(outputs - labels).mean().item()
        
        avg_loss = total_loss / len(loader)
        avg_mae = total_mae / len(loader)
        
        return avg_loss, avg_mae
    
    def _validate_epoch(self, model, loader, criterion):
        """Validate for one epoch"""
        model.eval()
        total_loss = 0
        total_mae = 0
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Validation"):
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = model(pixel_values)
                loss = criterion(outputs, labels)
                
                # Accumulate metrics
                total_loss += loss.item()
                total_mae += torch.abs(outputs - labels).mean().item()
        
        avg_loss = total_loss / len(loader)
        avg_mae = total_mae / len(loader)
        
        return avg_loss, avg_mae
    
    def predict(self, image_path, model_path=None):
        """Predict Cobb angle for single image
        
        Args:
            image_path: Path to spine X-ray image
            model_path: Path to trained model weights
        
        Returns:
            Predicted Cobb angle
        """
        if model_path is None:
            model_path = self.config['save_dir'] / 'best.pt'
        
        # Load model
        model = ViTCobbRegressor(self.config['model_name']).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Load and process image
        processor = ViTImageProcessor.from_pretrained(self.config['model_name'])
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)
        
        # Predict
        with torch.no_grad():
            output = model(pixel_values)
            cobb_angle = output.item()
        
        return cobb_angle


def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ViT for Cobb Angle Regression')
    parser.add_argument('--train-dir', type=str, required=True, help='Training images directory')
    parser.add_argument('--val-dir', type=str, required=True, help='Validation images directory')
    parser.add_argument('--labels', type=str, required=True, help='Labels file (CSV or JSON)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    
    args = parser.parse_args()
    
    # Load labels (implement based on your label format)
    # labels_dict = load_labels(args.labels)
    labels_dict = {}  # Placeholder
    
    # Initialize trainer
    trainer = ViTTrainer()
    
    # Prepare data
    train_loader, val_loader = trainer.prepare_data(
        args.train_dir,
        args.val_dir,
        labels_dict
    )
    
    # Train
    history = trainer.train(train_loader, val_loader, resume=args.resume)


if __name__ == "__main__":
    main()
