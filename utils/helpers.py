import cv2
import numpy as np


def draw_status(frame, text, color=(0, 255, 0), position=(20, 30)):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def draw_box(frame, x1, y1, x2, y2, color=(0, 255, 0), thickness=2):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))
