"""Clinical report generation powered by Gemma 2B with safe fallback."""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import GEMMA_CONFIG
from src.utils import setup_logging


class ReportGenerator:
    """Generate clinical reports using a Gemma instruction-tuned model."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or GEMMA_CONFIG
        self.logger = setup_logging('logs/report_generation.log')
        self.model_name = self.config.get('model_name', 'google/gemma-2b-it')
        self.device = self.config.get('device', 'cpu')

        self.tokenizer = None
        self.model = None
        self.available = False

        self._load_model()

    def _load_model(self):
        """Load tokenizer/model and keep system usable if loading fails."""
        try:
            self.logger.info(f"Loading report LLM: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            model_kwargs = {
                'torch_dtype': torch.float32,
                'trust_remote_code': True,
            }

            if self.device == 'cpu':
                model_kwargs['device_map'] = 'cpu'

            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)

            if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.available = True
            self.logger.info("Gemma report model loaded successfully")
        except Exception as exc:
            self.available = False
            self.logger.warning(f"Gemma unavailable, using template report. Reason: {exc}")

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clinical_recommendation(primary_cobb: float):
        if primary_cobb < 10:
            return (
                "Normal spinal alignment.",
                "Routine follow-up as clinically indicated.",
                "LOW"
            )
        if primary_cobb < 25:
            return (
                "Mild scoliosis range.",
                "Periodic follow-up and conservative management may be considered.",
                "MODERATE"
            )
        if primary_cobb < 40:
            return (
                "Moderate scoliosis range.",
                "Orthopedic consultation and bracing evaluation may be appropriate.",
                "HIGH"
            )
        return (
            "Severe scoliosis range.",
            "Urgent specialist evaluation for advanced intervention is recommended.",
            "CRITICAL"
        )

    def _build_prompt(self, prediction: Dict[str, Any]) -> str:
        image_id = prediction.get('image_id', 'unknown')
        severity = prediction.get('severity', 'Unknown')
        primary_cobb = self._safe_float(prediction.get('cobb_angle_primary', 0.0))
        secondary_cobb = self._safe_float(prediction.get('cobb_angle_secondary', 0.0))
        confidence = self._safe_float(prediction.get('confidence', 0.0)) * 100.0

        return (
            "You are an assistant writing a concise scoliosis imaging report for clinicians. "
            "Do not provide diagnosis certainty claims. Keep it factual and brief.\n\n"
            f"Image ID: {image_id}\n"
            f"Severity label: {severity}\n"
            f"Primary Cobb angle: {primary_cobb:.1f} degrees\n"
            f"Secondary Cobb angle: {secondary_cobb:.1f} degrees\n"
            f"Detection confidence: {confidence:.1f}%\n\n"
            "Write 3 short sections with headings:\n"
            "1) Findings\n"
            "2) Interpretation\n"
            "3) Suggested next step\n"
        )

    def _generate_llm_findings(self, prediction: Dict[str, Any]) -> str:
        if not self.available:
            return ""

        prompt = self._build_prompt(prediction)
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=int(self.config.get('max_length', 512)),
                do_sample=True,
                temperature=float(self.config.get('temperature', 0.7)),
                top_p=float(self.config.get('top_p', 0.9)),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if generated.startswith(prompt):
            generated = generated[len(prompt):]
        return generated.strip()

    def generate_report(self, prediction: Dict[str, Any]) -> str:
        """Generate a complete text report for one prediction."""
        image_id = prediction.get('image_id', 'Unknown')
        severity = prediction.get('severity', 'Unknown')
        primary_cobb = self._safe_float(prediction.get('cobb_angle_primary', 0.0))
        secondary_cobb = self._safe_float(prediction.get('cobb_angle_secondary', 0.0))
        confidence = self._safe_float(prediction.get('confidence', 0.0)) * 100.0
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        interpretation, recommendation, risk_level = self._clinical_recommendation(primary_cobb)
        llm_section = self._generate_llm_findings(prediction)

        if not llm_section:
            llm_section = (
                f"Findings: Detected curvature pattern is consistent with {severity}.\n"
                f"Interpretation: Primary Cobb angle is {primary_cobb:.1f}°.\n"
                f"Suggested next step: {recommendation}"
            )

        return f"""
{'='*80}
SPINE X-RAY AUTOMATED ANALYSIS REPORT
{'='*80}

Patient Information:
  Image ID: {image_id}
  Analysis Date: {timestamp}
  Analysis Type: Automated Scoliosis Detection

{'='*80}
MEASUREMENTS:
{'='*80}

  Primary Cobb Angle:      {primary_cobb:6.1f}°
  Secondary Cobb Angle:    {secondary_cobb:6.1f}°
  Severity Classification: {severity}
  Detection Confidence:    {confidence:6.1f}%
  Risk Level:              {risk_level}

{'='*80}
CLINICAL SUMMARY:
{'='*80}

{llm_section}

{'='*80}
RULE-BASED REFERENCE:
{'='*80}

Interpretation: {interpretation}
Recommendation: {recommendation}

{'='*80}
DISCLAIMER:
{'='*80}

This report is AI-assisted and must be reviewed by a qualified clinician.
It is not a standalone medical diagnosis.

{'='*80}
"""

