"""
Clinical Report Generation using Gemma LLM
Generates patient-ready diagnostic reports from scoliosis predictions
"""

import os
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
import logging

sys.path.append(str(Path(__file__).parent.parent))
from src.config import GEMMA_CONFIG, REPORT_TEMPLATE
from src.utils import setup_logging


class ReportGenerator:
    """Generate clinical reports using Gemma LLM"""
    
    def __init__(self, config=None):
        """Initialize report generator
        
        Args:
            config: Dictionary with model configuration
        """
        self.config = config or GEMMA_CONFIG
        self.logger = setup_logging()
        
        self.logger.info(f"Loading Gemma model: {self.config['model_name']}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model_name'],
            trust_remote_code=True
        )
        
        # Load model with 8-bit quantization for CPU efficiency
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model_name'],
            device_map='cpu',
            torch_dtype=torch.float32,  # CPU doesn't support bfloat16
            load_in_8bit=self.config.get('load_in_8bit', False),
            trust_remote_code=True
        )
        
        self.logger.info(f"Model loaded on {self.config['device']}")
    
    def generate_report(self, predictions):
        """Generate clinical report from predictions
        
        Args:
            predictions: Dictionary with prediction results
                - image_id: str
                - severity_class: str
                - confidence: float
                - primary_cobb: float
                - secondary_cobb: float
                - detections: list of detected vertebrae
        
        Returns:
            Generated clinical report as string
        """
        # Create prompt for Gemma
        prompt = self._create_prompt(predictions)
        
        self.logger.info("Generating clinical report...")
        
        # Generate report
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config['device'])
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.config['max_length'],
                temperature=self.config['temperature'],
                top_p=self.config['top_p'],
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract report (remove prompt)
        report = generated_text[len(prompt):].strip()
        
        # Format final report
        final_report = self._format_report(predictions, report)
        
        return final_report
    
    def _create_prompt(self, predictions):
        """Create prompt for LLM
        
        Args:
            predictions: Prediction results dictionary
        
        Returns:
            Formatted prompt string
        """
        severity_class = predictions.get('severity_class', 'Unknown')
        primary_cobb = predictions.get('primary_cobb', 0)
        secondary_cobb = predictions.get('secondary_cobb', 0)
        confidence = predictions.get('confidence', 0)
        
        prompt = f"""You are a medical AI assistant specializing in spine radiology. 
Generate a professional clinical report for the following scoliosis analysis:

Patient X-ray Analysis Results:
- Severity Classification: {severity_class}
- Primary Cobb Angle: {primary_cobb:.1f} degrees
- Secondary Cobb Angle: {secondary_cobb:.1f} degrees
- Detection Confidence: {confidence:.1f}%

Please provide:
1. Clinical findings summary
2. Interpretation of the measurements
3. Severity assessment
4. Recommended next steps

Keep the report professional, concise, and clinically relevant.

Clinical Report:
"""
        return prompt
    
    def _format_report(self, predictions, generated_text):
        """Format final clinical report
        
        Args:
            predictions: Prediction results
            generated_text: LLM generated text
        
        Returns:
            Formatted report string
        """
        # Extract values
        image_id = predictions.get('image_id', 'Unknown')
        severity_class = predictions.get('severity_class', 'Unknown')
        primary_cobb = predictions.get('primary_cobb', 0)
        secondary_cobb = predictions.get('secondary_cobb', 0)
        confidence = predictions.get('confidence', 0)
        
        # Get current date
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Determine severity interpretation
        if primary_cobb < 10:
            interpretation = "Normal spine alignment with no significant scoliosis."
            recommendation = "No treatment required. Routine follow-up recommended."
        elif 10 <= primary_cobb < 25:
            interpretation = "Mild scoliosis detected. Patient should be monitored."
            recommendation = "Observation with periodic X-rays every 6 months. Physical therapy may be beneficial."
        elif 25 <= primary_cobb < 40:
            interpretation = "Moderate scoliosis present. Active intervention may be needed."
            recommendation = "Consider bracing if patient is skeletally immature. Consultation with orthopedic specialist recommended."
        else:
            interpretation = "Severe scoliosis detected. Surgical evaluation indicated."
            recommendation = "Refer to spine surgeon for evaluation. Discuss surgical options including spinal fusion."
        
        # Build report
        report = f"""
{'='*80}
SPINE X-RAY AUTOMATED ANALYSIS REPORT
{'='*80}

Patient Information:
  Image ID: {image_id}
  Analysis Date: {current_date}
  Analysis Type: Automated Scoliosis Detection

{'='*80}
MEASUREMENTS:
{'='*80}

  Primary Cobb Angle:      {primary_cobb:6.1f}°
  Secondary Cobb Angle:    {secondary_cobb:6.1f}°
  Severity Classification: {severity_class}
  Detection Confidence:    {confidence:6.1f}%

{'='*80}
FINDINGS:
{'='*80}

{generated_text if generated_text else 'Automated analysis completed successfully.'}

{'='*80}
INTERPRETATION:
{'='*80}

{interpretation}

{'='*80}
RECOMMENDATIONS:
{'='*80}

{recommendation}

{'='*80}
DISCLAIMER:
{'='*80}

This report is generated by an AI-powered automated analysis system (Scoliosis AI v1.0).
All findings and measurements should be reviewed and validated by a qualified radiologist
or orthopedic specialist before clinical decision-making.

This automated analysis is intended to assist medical professionals and should not replace
professional medical judgment.

{'='*80}
"""
        return report
    
    def save_report(self, report, output_path):
        """Save report to file
        
        Args:
            report: Report text
            output_path: Path to save report
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"Report saved to: {output_path}")
    
    def generate_simplified_report(self, predictions):
        """Generate simplified report without LLM (fallback method)
        
        Args:
            predictions: Prediction results dictionary
        
        Returns:
            Simplified clinical report
        """
        self.logger.info("Generating simplified report (no LLM)...")
        
        # Use template-based generation
        report = self._format_report(predictions, "")
        
        return report


def main():
    """Test report generation"""
    # Example predictions
    predictions = {
        'image_id': 'patient_001_spine_xray.jpg',
        'severity_class': 'Moderate Scoliosis (25-40°)',
        'primary_cobb': 32.5,
        'secondary_cobb': 18.3,
        'confidence': 94.2,
        'detections': [
            {'class': '2-derece', 'confidence': 0.95},
            {'class': '2-derece', 'confidence': 0.93}
        ]
    }
    
    # Initialize generator
    try:
        generator = ReportGenerator()
        
        # Generate report
        report = generator.generate_report(predictions)
        
        print(report)
        
        # Save report
        output_path = Path('outputs/sample_report.txt')
        generator.save_report(report, output_path)
        
    except Exception as e:
        print(f"Error loading Gemma model: {e}")
        print("Generating simplified report instead...")
        
        # Fallback to simplified report
        generator = ReportGenerator.__new__(ReportGenerator)
        generator.logger = setup_logging()
        report = generator.generate_simplified_report(predictions)
        print(report)


if __name__ == "__main__":
    main()
