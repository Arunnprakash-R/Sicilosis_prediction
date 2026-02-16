"""
Cobb Angle Calculation Module
Automated measurement of spine curvature angles from vertebra detections
"""

import numpy as np
import cv2
import logging
from typing import List, Dict, Tuple
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.utils import setup_logging


class CobbAngleCalculator:
    """Calculate Cobb angles from vertebra detections"""
    
    def __init__(self, min_angle_threshold=10):
        """Initialize Cobb angle calculator
        
        Args:
            min_angle_threshold: Minimum angle (degrees) to consider significant
        """
        self.min_angle_threshold = min_angle_threshold
        self.logger = setup_logging()
    
    def calculate_from_detections(self, detections: List[Dict]) -> Dict:
        """Calculate Cobb angle from YOLO detections
        
        Args:
            detections: List of detection dictionaries with bbox and class info
        
        Returns:
            Dictionary with primary and secondary Cobb angles
        """
        if len(detections) < 3:
            self.logger.warning("Need at least 3 vertebrae for Cobb angle calculation")
            return {'primary_angle': 0, 'secondary_angle': 0, 'error': 'Insufficient detections'}
        
        # Extract vertebra centers and sort by y-coordinate (top to bottom)
        vertebrae = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            vertebrae.append({
                'center': (center_x, center_y),
                'bbox': det['bbox'],
                'class': det.get('class', ''),
            })
        
        # Sort vertebrae from top to bottom
        vertebrae.sort(key=lambda v: v['center'][1])
        
        # Calculate angles using landmark-based method
        primary_angle = self._calculate_cobb_angle_landmark(vertebrae)
        
        # Calculate secondary curve if enough vertebrae
        secondary_angle = 0
        if len(vertebrae) >= 6:
            secondary_angle = self._calculate_secondary_curve(vertebrae)
        
        result = {
            'primary_angle': round(primary_angle, 2),
            'secondary_angle': round(secondary_angle, 2),
            'num_vertebrae': len(vertebrae),
            'method': 'landmark_based',
        }
        
        self.logger.info(f"Cobb angle calculated: Primary={primary_angle:.2f}°, Secondary={secondary_angle:.2f}°")
        
        return result
    
    def _calculate_cobb_angle_landmark(self, vertebrae: List[Dict]) -> float:
        """Calculate Cobb angle using landmark-based method
        
        Args:
            vertebrae: List of vertebra dictionaries sorted top to bottom
        
        Returns:
            Cobb angle in degrees
        """
        if len(vertebrae) < 3:
            return 0
        
        # Get top, apex (middle), and bottom vertebrae
        top_vertebra = vertebrae[0]
        bottom_vertebra = vertebrae[-1]
        
        # Find apex (vertebra with maximum lateral deviation)
        apex_idx = self._find_apex_vertebra(vertebrae)
        apex_vertebra = vertebrae[apex_idx]
        
        # Calculate angles of top and bottom endplates
        top_angle = self._calculate_endplate_angle(top_vertebra)
        bottom_angle = self._calculate_endplate_angle(bottom_vertebra)
        
        # Cobb angle is the angle between the two endplates
        cobb_angle = abs(top_angle - bottom_angle)
        
        # Ensure angle is in valid range [0, 180]
        if cobb_angle > 180:
            cobb_angle = 360 - cobb_angle
        
        return cobb_angle
    
    def _find_apex_vertebra(self, vertebrae: List[Dict]) -> int:
        """Find the apex vertebra (maximum lateral deviation)
        
        Args:
            vertebrae: List of vertebra dictionaries
        
        Returns:
            Index of apex vertebra
        """
        # Calculate centerline (average x-coordinate)
        centerline_x = np.mean([v['center'][0] for v in vertebrae])
        
        # Find vertebra with maximum deviation from centerline
        max_deviation = 0
        apex_idx = len(vertebrae) // 2  # Default to middle
        
        for idx, vertebra in enumerate(vertebrae):
            deviation = abs(vertebra['center'][0] - centerline_x)
            if deviation > max_deviation:
                max_deviation = deviation
                apex_idx = idx
        
        return apex_idx
    
    def _calculate_endplate_angle(self, vertebra: Dict) -> float:
        """Calculate endplate angle of a vertebra
        
        Args:
            vertebra: Vertebra dictionary with bbox
        
        Returns:
            Angle in degrees
        """
        x1, y1, x2, y2 = vertebra['bbox']
        
        # Use bottom edge of top bounding box as endplate
        dx = x2 - x1
        dy = 0  # Assume horizontal endplate
        
        # Calculate angle (in degrees)
        angle = np.degrees(np.arctan2(dy, dx))
        
        return angle
    
    def _calculate_secondary_curve(self, vertebrae: List[Dict]) -> float:
        """Calculate secondary curve angle (compensatory curve)
        
        Args:
            vertebrae: List of vertebra dictionaries
        
        Returns:
            Secondary Cobb angle in degrees
        """
        # Split vertebrae into upper and lower sections
        mid_idx = len(vertebrae) // 2
        upper_vertebrae = vertebrae[:mid_idx]
        lower_vertebrae = vertebrae[mid_idx:]
        
        # Calculate curve for each section
        if len(upper_vertebrae) >= 3:
            upper_angle = self._calculate_cobb_angle_landmark(upper_vertebrae)
        else:
            upper_angle = 0
        
        if len(lower_vertebrae) >= 3:
            lower_angle = self._calculate_cobb_angle_landmark(lower_vertebrae)
        else:
            lower_angle = 0
        
        # Return the smaller angle (compensatory curve)
        return min(upper_angle, lower_angle)
    
    def calculate_from_image(self, image_path: str, detections: List[Dict]) -> Tuple[Dict, np.ndarray]:
        """Calculate Cobb angle and visualize on image
        
        Args:
            image_path: Path to spine X-ray image
            detections: List of vertebra detections
        
        Returns:
            Tuple of (results dict, annotated image)
        """
        # Calculate angles
        results = self.calculate_from_detections(detections)
        
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            self.logger.error(f"Failed to load image: {image_path}")
            return results, None
        
        # Visualize vertebrae and angles
        annotated_img = self._visualize_cobb_angle(img, detections, results)
        
        return results, annotated_img
    
    def _visualize_cobb_angle(self, image: np.ndarray, detections: List[Dict], results: Dict) -> np.ndarray:
        """Visualize Cobb angle measurement on image
        
        Args:
            image: Input image
            detections: List of vertebra detections
            results: Cobb angle results
        
        Returns:
            Annotated image
        """
        img = image.copy()
        
        # Extract vertebra centers
        vertebrae = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            vertebrae.append((center_x, center_y))
            
            # Draw bounding box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(img, (center_x, center_y), 5, (255, 0, 0), -1)
        
        # Draw spine curve
        if len(vertebrae) >= 2:
            vertebrae.sort(key=lambda v: v[1])  # Sort by y-coordinate
            for i in range(len(vertebrae) - 1):
                cv2.line(img, vertebrae[i], vertebrae[i+1], (255, 255, 0), 2)
        
        # Draw Cobb angle text
        primary_angle = results['primary_angle']
        secondary_angle = results['secondary_angle']
        
        cv2.putText(img, f"Primary Cobb: {primary_angle:.1f} deg", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(img, f"Secondary Cobb: {secondary_angle:.1f} deg", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        return img
    
    def classify_severity(self, cobb_angle: float) -> str:
        """Classify scoliosis severity based on Cobb angle
        
        Args:
            cobb_angle: Primary Cobb angle in degrees
        
        Returns:
            Severity classification string
        """
        if cobb_angle < 10:
            return "Normal (< 10°)"
        elif 10 <= cobb_angle < 25:
            return "Mild (10-25°)"
        elif 25 <= cobb_angle < 40:
            return "Moderate (25-40°)"
        else:
            return "Severe (≥ 40°)"


def main():
    """Test Cobb angle calculation"""
    # Example detections (simulated)
    detections = [
        {'bbox': [100, 50, 150, 80], 'class': 'vertebra'},
        {'bbox': [110, 100, 160, 130], 'class': 'vertebra'},
        {'bbox': [130, 150, 180, 180], 'class': 'vertebra'},
        {'bbox': [140, 200, 190, 230], 'class': 'vertebra'},
        {'bbox': [135, 250, 185, 280], 'class': 'vertebra'},
    ]
    
    calculator = CobbAngleCalculator()
    results = calculator.calculate_from_detections(detections)
    
    print("Cobb Angle Results:")
    print(f"Primary: {results['primary_angle']}°")
    print(f"Secondary: {results['secondary_angle']}°")
    print(f"Severity: {calculator.classify_severity(results['primary_angle'])}")


if __name__ == "__main__":
    main()
