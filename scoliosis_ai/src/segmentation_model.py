"""
Spine Segmentation Model using U-Net
This is a PhD-level implementation for precise spine segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DoubleConv(nn.Module):
    """Double Convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """
    U-Net Architecture for Spine Segmentation
    
    Paper: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    Modified for single-class spine segmentation
    """
    
    def __init__(
        self, 
        in_channels: int = 1,  # Grayscale X-ray
        out_channels: int = 1,  # Binary mask (spine vs background)
        features: list = [64, 128, 256, 512]
    ):
        super().__init__()
        
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder (downsampling)
        for feature in features:
            self.encoder_blocks.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # Decoder (upsampling)
        for feature in reversed(features):
            self.decoder_blocks.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder_blocks.append(DoubleConv(feature * 2, feature))
        
        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder path
        skip_connections = []
        
        for encoder in self.encoder_blocks:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path
        skip_connections = skip_connections[::-1]  # Reverse
        
        for idx in range(0, len(self.decoder_blocks), 2):
            x = self.decoder_blocks[idx](x)  # Upsample
            skip = skip_connections[idx // 2]
            
            # Handle size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            
            x = torch.cat([skip, x], dim=1)  # Concatenate skip connection
            x = self.decoder_blocks[idx + 1](x)  # Double conv
        
        return torch.sigmoid(self.final_conv(x))


class AttentionUNet(nn.Module):
    """
    U-Net with Attention Gates
    
    Novel Contribution: Attention-guided spine segmentation
    Paper: "Attention U-Net: Learning Where to Look for the Pancreas"
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        # TODO: Implement attention gates for PhD-level contribution
        # This focuses the model on spine regions
        pass


class SegmentationLoss(nn.Module):
    """
    Combined loss for better segmentation
    
    Loss = Dice Loss + Binary Cross Entropy
    """
    
    def __init__(self, weight_dice: float = 0.5, weight_bce: float = 0.5):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.bce = nn.BCELoss()
    
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6):
        """
        Dice Loss = 1 - Dice Coefficient
        
        Dice = 2 * |X ∩ Y| / (|X| + |Y|)
        """
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        
        return 1 - dice
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        dice = self.dice_loss(pred, target)
        bce = self.bce(pred, target)
        
        return self.weight_dice * dice + self.weight_bce * bce


class SpineSegmentationMetrics:
    """
    Evaluation metrics for segmentation
    
    PhD requirement: Report multiple metrics
    """
    
    @staticmethod
    def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
        """Dice Similarity Coefficient"""
        pred_binary = (pred > threshold).float()
        target_binary = target.float()
        
        intersection = (pred_binary * target_binary).sum()
        dice = (2. * intersection) / (pred_binary.sum() + target_binary.sum() + 1e-6)
        
        return dice.item()
    
    @staticmethod
    def iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
        """Intersection over Union (IoU)"""
        pred_binary = (pred > threshold).float()
        target_binary = target.float()
        
        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum() - intersection
        
        return (intersection / (union + 1e-6)).item()
    
    @staticmethod
    def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
        """Pixel-wise accuracy"""
        pred_binary = (pred > threshold).float()
        target_binary = target.float()
        
        correct = (pred_binary == target_binary).float().sum()
        total = target_binary.numel()
        
        return (correct / total).item()
    
    @staticmethod
    def sensitivity(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
        """Sensitivity (Recall, True Positive Rate)"""
        pred_binary = (pred > threshold).float()
        target_binary = target.float()
        
        true_positive = (pred_binary * target_binary).sum()
        actual_positive = target_binary.sum()
        
        return (true_positive / (actual_positive + 1e-6)).item()
    
    @staticmethod
    def specificity(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
        """Specificity (True Negative Rate)"""
        pred_binary = (pred > threshold).float()
        target_binary = target.float()
        
        true_negative = ((1 - pred_binary) * (1 - target_binary)).sum()
        actual_negative = (1 - target_binary).sum()
        
        return (true_negative / (actual_negative + 1e-6)).item()


def test_unet():
    """Test U-Net architecture"""
    model = UNet(in_channels=1, out_channels=1)
    
    # Test forward pass
    x = torch.randn(2, 1, 512, 512)  # Batch of 2 X-ray images
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test loss
    target = torch.randint(0, 2, (2, 1, 512, 512)).float()
    loss_fn = SegmentationLoss()
    loss = loss_fn(output, target)
    print(f"Loss: {loss.item():.4f}")
    
    # Test metrics
    metrics = SpineSegmentationMetrics()
    dice = metrics.dice_coefficient(output, target)
    iou = metrics.iou(output, target)
    
    print(f"Dice: {dice:.4f}")
    print(f"IoU: {iou:.4f}")


if __name__ == "__main__":
    print("="*70)
    print("Spine Segmentation U-Net - PhD Implementation")
    print("="*70)
    test_unet()
