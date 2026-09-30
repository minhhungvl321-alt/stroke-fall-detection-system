import argparse
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    CAMERA_INDEX,
    CONFIDENCE_THRESHOLD,
    DEBUG_MODE,
    PERSON_CLASS_ID,
    VIDEO_PATH,
    YOLO_MODEL,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    ALERT_COOLDOWN_SECONDS,
)
from utils.pose_analyzer import PoseAnalyzer
from utils.motion_analyzer import MotionAnalyzer
from utils.fall_detector import FallDetector
from utils.alert_system import AlertSystem


class FallDetectionSystem:
    """
    Integrated system for fall and abnormal movement detection
    
    Architecture:
        Camera -> YOLO Detection -> MediaPipe Pose -> Motion Analysis -> Fall Detection -> Alert
    """
    
    def __init__(self, video_source=None):
        print("[*] Initializing Fall Detection System...")
        
        # Load models
        print("[*] Loading YOLO model...")
        self.model = YOLO(YOLO_MODEL)
        
        # Initialize analyzers
        self.pose_analyzer = PoseAnalyzer()
        self.motion_analyzer = MotionAnalyzer()
        self.fall_detector = FallDetector()
        self.alert_system = AlertSystem()
        
        # Video source
        self.video_source = video_source
        self.cap = None
        
        # History tracking
        self.center_history = deque(maxlen=30)
        self.area_history = deque(maxlen=30)
        self.velocity_history = deque(maxlen=10)
        
        print("[*] System initialized successfully!")
    
    def start_capture(self):
        """Initialize video capture"""
        source = self.video_source if self.video_source else CAMERA_INDEX
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        print(f"[+] Video capture started from source: {source}")
    
    def detect_persons(self, frame):
        """
        Detect persons using YOLO
        
        Returns:
            list: List of detected persons with bbox and center
        """
        results = self.model.track(frame, persist=True, verbose=False)
        persons = []
        
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                if cls_id != PERSON_CLASS_ID:
                    continue
                
                conf = float(box.conf[0].item())
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                area = (x2 - x1) * (y2 - y1)
                
                persons.append({
                    "bbox": (x1, y1, x2, y2),
                    "center": center,
                    "area": area,
                    "confidence": conf,
                })
        
        return persons
    
    def update_motion_history(self, center, area):
        """
        Update center and area history, calculate velocity
        """
        motion = {"center_velocity": 0.0, "area_change_rate": 0.0}
        
        # Calculate center velocity
        if len(self.center_history) > 0:
            prev_center = self.center_history[-1]
            motion["center_velocity"] = self.motion_analyzer.calculate_velocity(
                prev_center, center
            )
        
        # Calculate area change rate
        if len(self.area_history) > 0:
            prev_area = self.area_history[-1]
            motion["area_change_rate"] = self.motion_analyzer.calculate_area_change(
                prev_area, area
            )
        
        # Update history
        self.center_history.append(center)
        self.area_history.append(area)
        self.velocity_history.append(motion["center_velocity"])
        
        return motion
    
    def classify_alert_level(self, status):
        """
        Classify alert level based on status
        """
        if "FALLEN" in status or "ALERT" in status:
            return "high"
        elif "FALLING" in status or "ABNORMAL" in status:
            return "medium"
        elif "NORMAL" in status:
            return "normal"
        else:
            return "low"
    
    def draw_visualization(self, frame, persons, features, motion, status, alert_level):
        """
        Draw bounding box, skeleton, metrics on frame
        """
        if not persons:
            cv2.putText(
                frame,
                "NO PERSON DETECTED",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            return frame
        
        # Get main person (first detected)
        person = persons[0]
        x1, y1, x2, y2 = person["bbox"]
        
        # Draw bounding box with color based on alert level
        if alert_level == "high":
            color = (0, 0, 255)  # Red
        elif alert_level == "medium":
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 255, 0)  # Green
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Draw skeleton if landmarks available
        landmarks = self.pose_analyzer.extract_landmarks(frame)
        if landmarks:
            frame = self.pose_analyzer.draw_landmarks(frame, landmarks)
        
        # Draw status
        status_color = (0, 0, 255) if alert_level == "high" else (255, 255, 255)
        cv2.putText(
            frame,
            f"Status: {status}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            status_color,
            2,
        )
        
        # Draw metrics
        y_offset = 80
        metrics = [
            f"Face asymmetry: {features.get('face_asymmetry', 0.0):.2f}",
            f"Arm asymmetry: {features.get('arm_asymmetry', 0.0):.2f}",
            f"Body tilt: {features.get('body_tilt', 0.0):.1f}px",
            f"Unstable gait: {features.get('walking_unstable', 0.0):.2f}",
            f"Center velocity: {motion.get('center_velocity', 0.0):.1f}px",
            f"Area change: {motion.get('area_change_rate', 0.0):.2f}",
            f"Risk Score: {features.get('risk_score', 0.0):.2f}",
        ]
        
        for metric in metrics:
            cv2.putText(
                frame,
                metric,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )
            y_offset += 28
        
        return frame
    
    def run(self):
        """
        Main detection loop
        """
        self.start_capture()
        print("[*] Starting detection loop... (Press 'q' to quit)")
        
        frame_count = 0
        fps_start = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[!] End of video stream")
                break
            
            frame_count += 1
            
            # Step 1: YOLO Detection
            persons = self.detect_persons(frame)
            
            if not persons:
                cv2.putText(
                    frame,
                    "NO PERSON DETECTED",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("Fall & Abnormal Movement Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            # Step 2: Extract pose landmarks
            landmarks = self.pose_analyzer.extract_landmarks(frame)
            features = (
                self.pose_analyzer.calculate_features(landmarks)
                if landmarks
                else {
                    "valid": False,
                    "face_asymmetry": 0.0,
                    "arm_asymmetry": 0.0,
                    "body_tilt": 0.0,
                    "walking_unstable": 0.0,
                    "body_aspect_ratio": 1.0,
                    "fall_score": 0.0,
                    "risk_score": 0.0,
                }
            )
            
            # Step 3: Motion analysis
            person = persons[0]
            motion = self.update_motion_history(person["center"], person["area"])
            
            # Step 4: Fall detection
            result = self.fall_detector.detect_fall(features, motion)
            status = result["status"]
            alert_level = result["alert_level"]
            risk_score = result["risk_score"]
            
            # Step 5: Alert triggering
            if alert_level in ["high", "medium"]:
                if self.alert_system.should_alert(alert_level, ALERT_COOLDOWN_SECONDS):
                    self.alert_system.log_alert(status, risk_score)
            
            # Step 6: Visualization
            frame = self.draw_visualization(
                frame, persons, features, motion, status, alert_level
            )
            
            # Calculate and display FPS
            if frame_count % 30 == 0:
                elapsed = time.time() - fps_start
                fps = 30 / elapsed
                print(f"[*] FPS: {fps:.2f}")
                fps_start = time.time()
            
            cv2.imshow("Fall & Abnormal Movement Detection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[*] Exiting...")
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("[+] Detection system stopped")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fall and Abnormal Movement Detection System"
    )
    parser.add_argument(
        "--video", type=str, default=None, help="Path to video file (optional)"
    )
    parser.add_argument(
        "--camera", type=int, default=CAMERA_INDEX, help="Camera index"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    try:
        video_source = args.video if args.video else args.camera
        system = FallDetectionSystem(video_source=video_source)
        system.run()
    except KeyboardInterrupt:
        print("\n[*] Interrupted by user")
    except Exception as e:
        print(f"[!] Error: {e}")
        raise


if __name__ == "__main__":
    main()
