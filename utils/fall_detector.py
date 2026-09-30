from collections import deque
from config import (
    FALL_MIN_FRAMES,
    FALL_STATIC_FRAMES,
    FALL_STATIC_VELOCITY,
    FACE_ASYMMETRY_THRESHOLD,
    ARM_ASYMMETRY_THRESHOLD,
    WALKING_UNSTABLE_THRESHOLD,
    FALL_ASPECT_THRESHOLD,
    FALL_BODY_TILT_THRESHOLD,
)


class FallDetector:
    """
    Detect fall events using:
    1. Pose-based features (body aspect ratio, tilt, head position)
    2. Motion-based features (center velocity, area change)
    3. Temporal patterns (sustained positions)
    
    Logic:
        Stage 1: FALLING - Detect sudden change in posture
        Stage 2: FALLEN - Confirm person is static on ground for 3-5 seconds
        Stage 3: ALERT - Issue alert if person remains fallen
    """
    
    def __init__(self):
        self.falling_state = False
        self.frames_in_fall = 0
        self.min_frames_to_confirm = FALL_MIN_FRAMES
        self.min_static_frames_after_fall = FALL_STATIC_FRAMES
        self.fall_position_history = deque(maxlen=10)
    
    def detect_fall(self, features, motion):
        """
        Detect fall with staged logic:
        
        Returns:
            dict: {status, alert_level, risk_score}
        """
        
        if not features.get("valid", False):
            self.falling_state = False
            self.frames_in_fall = 0
            return {
                "status": "NO PERSON",
                "alert_level": "normal",
                "risk_score": 0.0,
            }
        
        body_aspect = features.get("body_aspect_ratio", 1.0)
        body_tilt = features.get("body_tilt", 0.0)
        fall_score = features.get("fall_score", 0.0)
        face_asym = features.get("face_asymmetry", 0.0)
        arm_asym = features.get("arm_asymmetry", 0.0)
        walking_unstable = features.get("walking_unstable", 0.0)
        
        center_velocity = motion.get("center_velocity", 0.0)
        area_change = motion.get("area_change_rate", 0.0)
        
        # ========== STAGE 1: Detect FALLING ==========
        falling_indicators = 0
        
        # Indicator 1: Low aspect ratio (person lying down)
        if body_aspect < FALL_ASPECT_THRESHOLD:
            falling_indicators += 1
        
        # Indicator 2: High body tilt
        if body_tilt > FALL_BODY_TILT_THRESHOLD:
            falling_indicators += 1
        
        # Indicator 3: Large area change (sudden change)
        if abs(area_change) > 0.3:
            falling_indicators += 1
        
        # Indicator 4: High fall score from pose
        if fall_score > 0.5:
            falling_indicators += 1
        
        # Determine if in falling state (need 2+ indicators)
        if falling_indicators >= 2:
            self.falling_state = True
            self.frames_in_fall += 1
        else:
            self.falling_state = False
            self.frames_in_fall = 0
        
        # ========== STAGE 2: Detect FALLEN (static on ground) ==========
        if self.falling_state and self.frames_in_fall > self.min_frames_to_confirm:
            if center_velocity < FALL_STATIC_VELOCITY:
                # Person is static on ground for confirmed frames
                risk_score = 0.95
                return {
                    "status": "FALLEN - ALERT",
                    "alert_level": "high",
                    "risk_score": risk_score,
                }
            else:
                # Person is falling but still moving (may recover)
                risk_score = fall_score + 0.2
                return {
                    "status": "FALLING",
                    "alert_level": "medium",
                    "risk_score": min(1.0, risk_score),
                }
        
        # ========== Check for ABNORMAL MOVEMENT (without fall) ==========
        abnormal_score = 0.0
        abnormal_reasons = []
        
        if face_asym > FACE_ASYMMETRY_THRESHOLD:
            abnormal_score += 0.3
            abnormal_reasons.append(f"Face asymmetry: {face_asym:.2f}")
        
        if arm_asym > ARM_ASYMMETRY_THRESHOLD:
            abnormal_score += 0.3
            abnormal_reasons.append(f"Arm imbalance: {arm_asym:.2f}")
        
        if walking_unstable > WALKING_UNSTABLE_THRESHOLD:
            abnormal_score += 0.25
            abnormal_reasons.append(f"Unstable gait: {walking_unstable:.2f}")
        
        if abnormal_score > 0.25:
            risk_score = min(1.0, abnormal_score)
            status = "ABNORMAL MOVEMENT - " + ", ".join(abnormal_reasons)
            return {
                "status": status,
                "alert_level": "medium" if risk_score > 0.5 else "low",
                "risk_score": risk_score,
            }
        
        # ========== NORMAL STATE ==========
        return {
            "status": "NORMAL",
            "alert_level": "normal",
            "risk_score": 0.0,
        }
