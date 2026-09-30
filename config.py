import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
LOG_DIR = BASE_DIR / "logs"
DATASET_DIR = BASE_DIR / "dataset"

for directory in [MODEL_DIR, OUTPUT_DIR, LOG_DIR, DATASET_DIR]:
    directory.mkdir(exist_ok=True)

# ========== YOLO Configuration ==========
YOLO_MODEL = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.5
PERSON_CLASS_ID = 0

# ========== MediaPipe Pose Configuration ==========
POSE_CONFIDENCE = 0.7
POSE_TRACKING_CONFIDENCE = 0.5

# ========== Fall Detection Thresholds ==========
FALL_ASPECT_THRESHOLD = 0.60  # width/height ratio
FALL_BODY_TILT_THRESHOLD = 40  # degrees
FALL_MIN_FRAMES = 5  # frames to confirm fall
FALL_STATIC_FRAMES = 30  # frames (approx 1 second at 30fps) to alert
FALL_STATIC_VELOCITY = 5.0  # pixels/frame

# ========== Abnormal Movement Thresholds ==========
FACE_ASYMMETRY_THRESHOLD = 0.15
ARM_ASYMMETRY_THRESHOLD = 0.25
WALKING_UNSTABLE_THRESHOLD = 0.15
BODY_TILT_THRESHOLD = 25

# ========== Alert Configuration ==========
RISK_ALERT_THRESHOLD = 0.70
ALERT_COOLDOWN_SECONDS = 2.5

# ========== Video Configuration ==========
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_INDEX = 0
VIDEO_PATH = None

# ========== Debug Mode ==========
DEBUG_MODE = True
SAVE_LOGS = True
