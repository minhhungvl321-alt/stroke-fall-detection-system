import numpy as np


class MotionAnalyzer:
    """Analyze motion from bounding box and center tracking"""
    
    def __init__(self):
        pass
    
    @staticmethod
    def calculate_velocity(center_prev, center_curr):
        """Calculate movement velocity between two frames"""
        if center_prev is None or center_curr is None:
            return 0.0
        
        return float(np.linalg.norm(
            np.array(center_curr, dtype=float) - np.array(center_prev, dtype=float)
        ))
    
    @staticmethod
    def calculate_area_change(area_prev, area_curr):
        """Calculate relative area change"""
        if area_prev is None or area_curr is None or area_prev == 0:
            return 0.0
        
        return float((area_curr - area_prev) / area_prev)
