import mediapipe as mp
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult
import os
from functools import partial
from event_manager import Event_Manager
import cv2
import numpy as np
import threading
import common

class Hand_Detector:
    MP_KEY_POINTS = {
        "THUMB_TIP": 4,
        "THUMB_IP": 3,
        "THUMB_MCP": 2,
        "INDEX_FINGER_TIP": 8,
        "INDEX_FINGER_DIP": 7,
        "INDEX_FINGER_PIP": 6,
        "INDEX_FINGER_MCP": 5,
        "MIDDLE_FINGER_TIP": 12,
        "MIDDLE_FINGER_DIP": 11,
        "MIDDLE_FINGER_PIP": 10,
        "MIDDLE_FINGER_MCP": 9,
        "RING_FINGER_TIP": 16,
        "RING_FINGER_DIP": 15,
        "RING_FINGER_PIP": 14,
        "RING_FINGER_MCP": 13,
        "PINKY_TIP": 20,
        "PINKY_DIP": 19,
        "PINKY_PIP": 18,
        "PINKY_MCP": 17,
        "WRIST": 0,
    }

    def __init__(
        self,
        event_manager: Event_Manager,
        detection_confidence=0.5,
        num_hands=1,
        model_path="$YARVISPATH/models/hand_landmarker.task",
    ):
        self._lock = threading.Lock()
        self.event_manager = event_manager
        event_manager.push_event("hand_detected", False)
        self.uncalibrated_detection_result = None
        self.landmarker = None
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self._callback = partial(self._handle_result)

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=os.path.expandvars(model_path)),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self._callback,
            min_hand_detection_confidence=detection_confidence,
            num_hands=num_hands,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

        self.detection_result = []

        self.calibration_matrix = None
        try:
            self.calibration_matrix = np.load("calibration.npy")
            print("[DEBUG] calibration found")
        except Exception as e:
            print(f"[DEBUG] no calibration found: {e}")

    def _handle_result(
        self, result: HandLandmarkerResult, output_image: mp.Image, time_stamp: int
    ) -> None:
        """Handle the results from the hand landmarker and update event manager"""
        with self._lock:
            self.uncalibrated_detection_result = result
            self.detection_result = []
            if result.hand_landmarks:
                for hand in result.hand_landmarks:
                    current_result = {}
                    for key_point in Hand_Detector.MP_KEY_POINTS:
                        landmark = hand[Hand_Detector.MP_KEY_POINTS[key_point]]
                        raw_x = landmark.x * output_image.width
                        raw_y = landmark.y * output_image.height
                        landmark_pos = np.array([[(raw_x, raw_y)]], dtype=np.float32)

                        if self.calibration_matrix is not None:
                            landmark_pos = cv2.perspectiveTransform(
                                landmark_pos, self.calibration_matrix
                            )
                        current_result[key_point] = landmark_pos[0][0].astype(dtype=int)
                        self.detection_result.append(current_result)

            self.event_manager.push_event("uncalibrated_hand_result", bool(result.hand_landmarks))
            self.event_manager.push_event("hand_result", bool(self.detection_result))

    def detect_async(self, image: mp.Image, time_stamp: int):
        if self.landmarker:
            self.landmarker.detect_async(image, time_stamp)

    def get_uncalibrated_result(self):
        with self._lock:
            return self.uncalibrated_detection_result

    def get_calibrated_result(self):
        with self._lock:
            return self.detection_result

    def get_uncalibrated_landmark_pos(self, hand: int, landmark_key):
        landmark = self.get_uncalibrated_result().hand_landmarks[hand][landmark_key]
        x = landmark.x * common.IR_WIDTH
        y = landmark.y * common.IR_HEIGHT
        return (x, y)

    def get_landmark_pos(self, hand, landmark_key):
        return self.get_calibrated_result()[hand][landmark_key]

    def close(self):
        if self.landmarker:
            self.landmarker.close()
            self.landmarker = None
