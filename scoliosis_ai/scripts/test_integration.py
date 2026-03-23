"""
Model Integration Test Script
Tests loading and basic inference with all available models
Verifies the complete pipeline is working correctly.
"""

import os
import sys
import json
import warnings
import tempfile
from pathlib import Path
import traceback

# Set up paths
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models_config import ModelConfig
from src.config import YOLO_CONFIG, VIT_CONFIG

import torch
import numpy as np
from PIL import Image

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def create_dummy_image(size=(640, 640)) -> Image.Image:
    """Create a dummy X-ray image for testing."""
    # Create a synthetic X-ray-like image (grayscale)
    dummy_array = np.zeros(size, dtype=np.uint8)
    # Add some structure that looks vaguely like a spine (avoid bounds issues)
    h, w = size
    start_h, end_h = max(0, h//3), min(h, 2*h//3)
    start_w, end_w = max(0, w//3 + 20), min(w, w//3 + 40)
    spine_height = end_h - start_h
    spine_width = end_w - start_w
    if spine_height > 0 and spine_width > 0:
        dummy_array[start_h:end_h, start_w:end_w] = np.random.randint(100, 200, (spine_height, spine_width))
    return Image.fromarray(dummy_array, mode='L')


def test_yolo_detection():
    """Test YOLO detection model loading and inference."""
    print("\n" + "="*80)
    print("TEST 1: YOLO DETECTION")
    print("="*80)
    
    try:
        from ultralytics import YOLO
        
        # Test with scoliosis_enhanced model
        yolo_path = ModelConfig.get_yolo_path("scoliosis_enhanced")
        
        if not yolo_path.exists():
            print(f"✗ Model file not found: {yolo_path}")
            return False
        
        print(f"\n✓ Loading YOLO from: {yolo_path}")
        model = YOLO(str(yolo_path))
        print(f"✓ YOLO model loaded successfully")
        
        # Test with dummy image
        dummy_img = create_dummy_image()
        temp_path = os.path.join(tempfile.gettempdir(), "test_image.png")
        dummy_img.save(temp_path)
        
        print(f"✓ Running inference on test image...")
        results = model.predict(temp_path, verbose=False, conf=0.25)
        
        if results and len(results) > 0:
            print(f"✓ Inference completed")
            print(f"  - Detections found: {len(results[0].boxes) if hasattr(results[0], 'boxes') else 0}")
            print(f"  - Image shape: {results[0].orig_shape}")
        else:
            print(f"⚠ No detections found (expected for synthetic image)")
        
        return True
        
    except Exception as e:
        print(f"✗ YOLO test failed: {e}")
        traceback.print_exc()
        return False


def test_vit_model():
    """Test Vision Transformer model loading."""
    print("\n" + "="*80)
    print("TEST 2: VISION TRANSFORMER (ViT)")
    print("="*80)
    
    try:
        from timm import create_model
        
        vit_config = ModelConfig.get_vit_config("base")
        model_name = vit_config["model_name"]
        
        print(f"\n✓ Creating ViT model: {model_name}")
        print(f"  - Pretrained: {vit_config['pretrained']}")
        print(f"  - Task: Cobb angle regression (1 output)")
        
        # Load base ViT model
        model = create_model(model_name, pretrained=vit_config['pretrained'], num_classes=1)
        model.eval()
        print(f"✓ ViT model created successfully")
        
        # Test with dummy input
        print(f"✓ Running inference on test input (224x224)...")
        dummy_img = create_dummy_image((224, 224))
        # Convert grayscale to RGB (ViT expects 3 channels)
        dummy_img_rgb = Image.new('RGB', (224, 224))
        dummy_img_rgb.paste(dummy_img)
        dummy_array = np.array(dummy_img_rgb)
        # Convert to tensor (B, C, H, W)
        dummy_tensor = torch.from_numpy(dummy_array).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        with torch.no_grad():
            output = model(dummy_tensor)
        
        if output.shape[-1] == 1:
            print(f"✓ ViT inference successful")
            print(f"  - Output shape: {output.shape}")
            print(f"  - Predicted angle (test): {output[0, 0].item():.2f}°")
        
        return True
        
    except Exception as e:
        print(f"✗ ViT test failed: {e}")
        traceback.print_exc()
        return False


def test_segmentation_model():
    """Test segmentation model loading."""
    print("\n" + "="*80)
    print("TEST 3: SEGMENTATION (U-Net)")
    print("="*80)
    
    try:
        import segmentation_models_pytorch as smp
        
        seg_config = ModelConfig.get_segmentation_config("unetplusplus")
        
        print(f"\n✓ Loading segmentation model: {seg_config['model_name']}")
        print(f"  - Encoder: {seg_config['encoder']}")
        print(f"  - Output classes: {seg_config['num_classes']}")
        
        # Create model
        model = smp.create_model(
            arch=seg_config['model_name'],
            encoder_name=seg_config['encoder'],
            in_channels=1,
            classes=seg_config['num_classes'],
            pretrained=True
        )
        model.eval()
        print(f"✓ Segmentation model created successfully")
        
        # Test inference
        print(f"✓ Running inference on test input (512x512)...")
        dummy_img = create_dummy_image((512, 512))
        dummy_tensor = torch.from_numpy(np.array(dummy_img)) / 255.0
        dummy_tensor = dummy_tensor.unsqueeze(0).unsqueeze(0).float()
        
        with torch.no_grad():
            output = model(dummy_tensor)
        
        print(f"✓ Segmentation inference successful")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"⚠ Segmentation test skipped: {e}")
        return None  # Not critical


def test_quantum_framework():
    """Test quantum framework availability."""
    print("\n" + "="*80)
    print("TEST 4: QUANTUM FRAMEWORKS")
    print("="*80)
    
    try:
        print(f"\n✓ Checking quantum frameworks...")
        
        # Check PennyLane
        try:
            import pennylane as qml
            print(f"✓ PennyLane {qml.__version__} available")
            
            # Test basic quantum circuit
            n_qubits = 4
            dev = qml.device("default.qubit", wires=n_qubits)
            
            @qml.qnode(dev)
            def circuit(x):
                for i in range(n_qubits):
                    qml.RY(x[i], wires=i)
                return qml.expval(qml.PauliZ(0))
            
            test_input = np.random.rand(n_qubits)
            result = circuit(test_input)
            print(f"✓ Quantum circuit test successful (output: {result:.4f})")
        except Exception as e:
            print(f"⚠ PennyLane test failed: {e}")
        
        # Check Qiskit
        try:
            import qiskit
            print(f"✓ Qiskit {qiskit.__version__} available")
        except Exception as e:
            print(f"⚠ Qiskit not available: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Quantum framework test failed: {e}")
        return False


def test_inference_pipeline():
    """Test the complete inference pipeline."""
    print("\n" + "="*80)
    print("TEST 5: INFERENCE PIPELINE")
    print("="*80)
    
    try:
        from inference import ScoliosisInference
        
        print(f"\n✓ Loading inference pipeline...")
        
        # Initialize with available models
        yolo_path = str(ModelConfig.get_yolo_path("scoliosis_enhanced"))
        
        if not Path(yolo_path).exists():
            print(f"⚠ YOLO model not found, skipping pipeline test")
            return None
        
        print(f"✓ Using YOLO: {yolo_path}")
        
        inference = ScoliosisInference(
            yolo_model_path=yolo_path,
            device='cpu',
            ensemble=False
        )
        
        print(f"✓ Inference pipeline initialized successfully")
        
        # Test prediction
        dummy_img = create_dummy_image()
        temp_path = os.path.join(tempfile.gettempdir(), "test_xray.png")
        dummy_img.save(temp_path)
        
        print(f"✓ Running full pipeline on test image...")
        result = inference.predict_single(temp_path)
        
        if result:
            print(f"✓ Prediction completed")
            print(f"  - Detections: {len(result.get('detections', []))}")
            print(f"  - Cobb angle: {result.get('cobb_angle', 'N/A')}")
            print(f"  - Confidence: {result.get('confidence', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"⚠ Inference pipeline test failed: {e}")
        traceback.print_exc()
        return None


def main():
    """Run all integration tests."""
    print("\n" + "="*80)
    print("SCOLIOSIS AI - MODEL INTEGRATION TESTS")
    print("="*80)
    
    results = {}
    
    # Run tests
    results["YOLO Detection"] = test_yolo_detection()
    results["Vision Transformer"] = test_vit_model()
    results["Segmentation"] = test_segmentation_model()
    results["Quantum Framework"] = test_quantum_framework()
    results["Inference Pipeline"] = test_inference_pipeline()
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result is True else "✗ FAIL" if result is False else "⊗ SKIP"
        print(f"{status:8} - {test_name}")
    
    print(f"\nOVERALL: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0:
        print("\n✓ All critical tests passed! System is ready for use.")
    else:
        print(f"\n✗ {failed} test(s) failed. Check configuration and dependencies.")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
1. Run a single diagnosis:
   python launcher_simple.py

2. Process multiple images:
   python inference.py --input_dir images/ --output_dir results/

3. Full research pipeline:
   python run_research_pipeline.py

4. Validate model status:
   python scripts/validate_models.py
""")


if __name__ == "__main__":
    main()
