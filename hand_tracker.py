import cv2
import mediapipe as mp
import numpy as np
import time

# TIPAGEM ========================
confidence = float
webcam_image = np.ndarray
rgb_tuple = tuple[int, int, int]

# CLASSE ========================
class Detector:
    def __init__(self,
                 mode: bool = False,
                 number_hands: int = 2,
                 model_complexity: int = 1,
                 min_detection_confidence: confidence = 0.5,
                 min_tracking_confidence: confidence = 0.5):
        self.mode = mode
        self.max_number_hands = number_hands
        self.complexity = model_complexity
        self.detection_con = min_detection_confidence
        self.tracking_con = min_tracking_confidence
