import argparse
import os
import time
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

from config import (
    YOLO_MODEL,
    CONFIDENCE_THRESHOLD,
    PERSON_CLASS_ID,
    POSE_CONFIDENCE,
    POSE_TRACKING_CONFIDENCE,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    CAMERA_INDEX,
    VIDEO_PATH,
    DEBUG_MODE,
    SAVE_LOGS,
)


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=POSE_CONFIDENCE,
    min_tracking_confidence=POSE_TRACKING_CONFIDENCE,
)

def extract_pose_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    if results.pose_landmarks is None:
        return None

    h, w, _ = frame.shape
    landmarks = []
    for lm in results.pose_landmarks.landmark:
        x, y, z = lm.x, lm.y, lm.z
        px = int(x * w)
        py = int(y * h)
        landmarks.append((px, py, z))
    return landmarks


def draw_landmarks(frame, landmarks):
    if not landmarks:
        return frame

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10), (11, 12), (11, 13), (13, 15),
        (12, 14), (14, 16), (11, 23), (12, 24),
        (23, 24), (23, 25), (25, 27), (24, 26),
        (27, 28), (28, 29), (29, 30), (26, 31),
        (31, 32), (32, 33), (27, 31), (24, 32)
    ]
    for p1, p2 in connections:
        if p1 < len(landmarks) and p2 < len(landmarks):
            x1, y1, _ = landmarks[p1]
            x2, y2, _ = landmarks[p2]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for idx, (x, y, _) in enumerate(landmarks):
        if idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]:
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
    return frame


def compute_pose_features(landmarks):
    if len(landmarks) < 33:
        return {
            "valid": False,
            "face_asymmetry": 0.0,
            "arm_asymmetry": 0.0,
            "body_tilt": 0.0,
            "walking_unstable": 0.0,
            "fall_score": 0.0,
            "risk_score": 0.0,
        }

    nose = landmarks[0]
    left_ear = landmarks[7]
    right_ear = landmarks[8]
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_hip = landmarks[23]
    right_hip = landmarks[24]
    left_knee = landmarks[25]
    right_knee = landmarks[26]
    left_ankle = landmarks[27]
    right_ankle = landmarks[28]
    left_wrist = landmarks[15]
    right_wrist = landmarks[16]

    def euclidean(a, b):
        return np.linalg.norm(np.array(a[:2]) - np.array(b[:2]))

    face_width = euclidean(left_ear, right_ear)
    face_mid = ((left_ear[0] + right_ear[0]) / 2, (left_ear[1] + right_ear[1]) / 2)
    left_eye = landmarks[1]
    right_eye = landmarks[2]
    eye_left_dist = abs(left_eye[0] - face_mid[0])
    eye_right_dist = abs(right_eye[0] - face_mid[0])
    face_asymmetry = abs(eye_left_dist - eye_right_dist) / max(face_width, 1)

    left_arm = euclidean(left_shoulder, left_wrist)
    right_arm = euclidean(right_shoulder, right_wrist)
    arm_asymmetry = abs(left_arm - right_arm) / max(left_arm + right_arm, 1)

    body_height = euclidean(left_hip, nose)
    body_width = euclidean(left_shoulder, right_shoulder)
    aspect_ratio = body_width / max(body_height, 1)

    left_hip_y = left_hip[1]
    right_hip_y = right_hip[1]
    shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
    body_tilt = abs(shoulder_y - ((left_hip_y + right_hip_y) / 2))

    left_leg = euclidean(left_hip, left_ankle)
    right_leg = euclidean(right_hip, right_ankle)
    gait = abs(left_leg - right_leg) / max(left_leg + right_leg, 1)

    fall_score = 0.0
    if aspect_ratio < 0.6:
        fall_score += 0.45
    if body_tilt > 40:
        fall_score += 0.25
    if abs(left_hip_y - right_hip_y) > 20:
        fall_score += 0.15
    if abs(nose[1] - ((left_hip_y + right_hip_y) / 2)) > 60:
        fall_score += 0.15

    risk_score = min(1.0, fall_score + 0.4 * face_asymmetry + 0.4 * arm_asymmetry + 0.3 * gait)

    return {
        "valid": True,
        "face_asymmetry": float(face_asymmetry),
        "arm_asymmetry": float(arm_asymmetry),
        "body_tilt": float(body_tilt),
        "walking_unstable": float(gait),
        "fall_score": float(fall_score),
        "risk_score": float(risk_score),
    }


def detect_person(frame, model):
    results = model(frame, verbose=False)
    boxes = results[0].boxes
    detections = []
    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0].item())
            if cls_id != PERSON_CLASS_ID:
                continue
            conf = float(box.conf[0].item())
            if conf < CONFIDENCE_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": conf,
            })
    return detections


def annotate_frame(frame, person_detection, features, status_text, alert_level):
    if person_detection:
        x1, y1, x2, y2 = person_detection[0]["bbox"]
        color = (0, 255, 0)
        if alert_level == "high":
            color = (0, 0, 255)
        elif alert_level == "medium":
            color = (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"Status: {status_text}"
    cv2.putText(frame, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if features["valid"]:
        cv2.putText(frame, f"Face asym: {features['face_asymmetry']:.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Arm asym: {features['arm_asymmetry']:.2f}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Risk: {features['risk_score']:.2f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame


def main():
    parser = argparse.ArgumentParser(description="Fall and abnormal movement detection")
    parser.add_argument("--video", type=str, default=VIDEO_PATH, help="Optional path to a video file")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="Camera index")
    parser.add_argument("--debug", action="store_true", default=DEBUG_MODE)
    args = parser.parse_args()

    model = YOLO(YOLO_MODEL)

    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print("Error: cannot open camera/video.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    frame_counter = 0
    alert_state = "normal"
    last_alert_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        if frame_counter % 2 != 0:
            pass

        person_detections = detect_person(frame, model)
        landmarks = extract_pose_landmarks(frame)
        features = compute_pose_features(landmarks) if landmarks else {"valid": False, "face_asymmetry": 0.0, "arm_asymmetry": 0.0, "body_tilt": 0.0, "walking_unstable": 0.0, "fall_score": 0.0, "risk_score": 0.0}

        if landmarks:
            frame = draw_landmarks(frame, landmarks)

        status = "normal"
        if features["valid"]:
            if features["risk_score"] > 0.7:
                status = "alert"
            elif features["fall_score"] > 0.35:
                status = "warning"
            elif features["face_asymmetry"] > 0.15 or features["arm_asymmetry"] > 0.25:
                status = "abnormal"
            elif features["walking_unstable"] > 0.15:
                status = "unstable"

        if status == "alert":
            alert_state = "high"
        elif status in ["warning", "abnormal", "unstable"]:
            alert_state = "medium"
        else:
            alert_state = "normal"

        frame = annotate_frame(frame, person_detections, features, status, alert_state)

        if status != "normal" and time.time() - last_alert_time > 2.5:
            print(f"ALERT: {status} risk={features['risk_score']:.2f}")
            last_alert_time = time.time()

        cv2.imshow("Fall & Abnormal Movement Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
