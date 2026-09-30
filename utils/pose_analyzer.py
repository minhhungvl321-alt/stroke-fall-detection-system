import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

from config import (
    POSE_CONFIDENCE,
    POSE_TRACKING_CONFIDENCE,
    ARM_ASYMMETRY_THRESHOLD,
    FACE_ASYMMETRY_THRESHOLD,
    FALL_ASPECT_THRESHOLD,
    FALL_BODY_TILT_THRESHOLD,
    WALKING_UNSTABLE_THRESHOLD,
)


class PoseAnalyzer:
    """Analyze pose using MediaPipe Pose"""
    
    def __init__(self):
        self.mp_pose = solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=POSE_CONFIDENCE,
            min_tracking_confidence=POSE_TRACKING_CONFIDENCE,
        )
    
    def extract_landmarks(self, frame):
        """Extract pose landmarks from frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None
        
        h, w, _ = frame.shape
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            z = lm.z
            landmarks.append((x, y, z))
        
        return landmarks
    
    def draw_landmarks(self, frame, landmarks):
        """Draw skeleton on frame"""
        if not landmarks or len(landmarks) < 33:
            return frame
        
        # Define connections between joints
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10), (11, 12), (11, 13), (13, 15),
            (12, 14), (14, 16), (11, 23), (12, 24),
            (23, 24), (23, 25), (25, 27), (24, 26),
            (27, 28), (28, 29), (29, 30), (26, 31),
            (31, 32), (32, 33), (27, 31), (24, 32)
        ]
        
        # Draw connections (lines)
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_pos = landmarks[start_idx]
                end_pos = landmarks[end_idx]
                cv2.line(frame, (start_pos[0], start_pos[1]), (end_pos[0], end_pos[1]), (0, 255, 0), 2)
        
        # Draw keypoints (circles)
        for i, (x, y, z) in enumerate(landmarks):
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        
        return frame
    
    def calculate_features(self, landmarks):
        """Calculate pose features for fall and abnormal movement detection"""
        if not landmarks or len(landmarks) < 33:
            return {
                "valid": False,
                "face_asymmetry": 0.0,
                "arm_asymmetry": 0.0,
                "body_tilt": 0.0,
                "walking_unstable": 0.0,
                "body_aspect_ratio": 1.0,
                "fall_score": 0.0,
                "risk_score": 0.0,
            }
        
        # Extract key landmarks
        nose = landmarks[0]
        left_eye = landmarks[1]
        right_eye = landmarks[2]
        left_ear = landmarks[7]
        right_ear = landmarks[8]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        
        def dist(p1, p2):
            """Calculate Euclidean distance between two points"""
            return np.linalg.norm(np.array(p1[:2], dtype=float) - np.array(p2[:2], dtype=float))
        
        # ========== Face Asymmetry ==========
        face_width = dist(left_ear, right_ear)
        face_center_x = (left_ear[0] + right_ear[0]) / 2.0
        left_eye_offset = abs(left_eye[0] - face_center_x)
        right_eye_offset = abs(right_eye[0] - face_center_x)
        face_asymmetry = abs(left_eye_offset - right_eye_offset) / max(face_width, 1.0)
        
        # ========== Arm Asymmetry ==========
        left_arm_len = dist(left_shoulder, left_wrist)
        right_arm_len = dist(right_shoulder, right_wrist)
        total_arm_len = left_arm_len + right_arm_len
        arm_asymmetry = abs(left_arm_len - right_arm_len) / max(total_arm_len, 1.0)
        
        # ========== Body Tilt ==========
        shoulder_mid_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
        hip_mid_y = (left_hip[1] + right_hip[1]) / 2.0
        body_tilt = abs(shoulder_mid_y - hip_mid_y)
        
        # ========== Body Aspect Ratio (width / height) ==========
        body_height = dist(np.array([0, hip_mid_y]), np.array([0, nose[1]]))
        body_width = dist(left_shoulder, right_shoulder)
        body_aspect_ratio = body_width / max(body_height, 1.0)
        
        # ========== Walking Stability ==========
        left_leg_len = dist(left_hip, left_ankle)
        right_leg_len = dist(right_hip, right_ankle)
        total_leg_len = left_leg_len + right_leg_len
        walking_unstable = abs(left_leg_len - right_leg_len) / max(total_leg_len, 1.0)
        
        # ========== Fall Score ==========
        fall_score = 0.0
        
        # Low aspect ratio = person lying down
        if body_aspect_ratio < FALL_ASPECT_THRESHOLD:
            fall_score += 0.45
        
        # Large body tilt = person tilted/fallen
        if body_tilt > FALL_BODY_TILT_THRESHOLD:
            fall_score += 0.25
        
        # Head too low = person on ground
        if abs(nose[1] - hip_mid_y) > 60:
            fall_score += 0.15
        
        # Extreme tilt with low aspect = fallen
        if body_tilt > 50 and body_aspect_ratio < 0.8:
            fall_score += 0.15
        
        # ========== Overall Risk Score ==========
        risk_score = min(1.0, fall_score + 0.4 * face_asymmetry + 0.4 * arm_asymmetry + 0.3 * walking_unstable)
        
        return {
            "valid": True,
            "face_asymmetry": float(face_asymmetry),
            "arm_asymmetry": float(arm_asymmetry),
            "body_tilt": float(body_tilt),
            "walking_unstable": float(walking_unstable),
            "body_aspect_ratio": float(body_aspect_ratio),
            "fall_score": float(fall_score),
            "risk_score": float(risk_score),
        }
