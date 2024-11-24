import mediapipe as mp
import os
from functools import partial
from event_manager import Event_Manager


class Hand_Detector:
    KEY_POINTS = {
        'THUMB_TIP': 4,
        'THUMB_IP': 3,
        'THUMB_MCP': 2,
        
        'INDEX_FINGER_TIP': 8,
        'INDEX_FINGER_DIP': 7,
        'INDEX_FINGER_PIP': 6,
        'INDEX_FINGER_MCP': 5,
        
        'MIDDLE_FINGER_TIP': 12,
        'MIDDLE_FINGER_DIP': 11,
        'MIDDLE_FINGER_PIP': 10,
        'MIDDLE_FINGER_MCP': 9,
        
        'RING_FINGER_TIP': 16,
        'RING_FINGER_DIP': 15,
        'RING_FINGER_PIP': 14,
        'RING_FINGER_MCP': 13,
        
        'PINKY_TIP': 20,
        'PINKY_DIP': 19,
        'PINKY_PIP': 18,
        'PINKY_MCP': 17,
        
        'WRIST': 0,
    }
    
    def __init__(self, event_manager: Event_Manager, detection_confidence=0.5, num_hands=1, model_path='$YARVISPATH/models/hand_landmarker.task'):
        self.event_manager = event_manager
        event_manager.push_event('hand_detected', False)
        self.detection_result = None
        self.landmarker = None
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        self._callback = partial(self._handle_result)
        
        options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=os.path.expandvars(model_path)
            ),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self._callback,
            min_hand_detection_confidence=detection_confidence,
            num_hands=num_hands
        )
        self.landmarker = HandLandmarker.create_from_options(options)
    

    def _handle_result(self, result, output_image: mp.Image, time_stamp: int):
        """Handle the results from the hand landmarker and update event manager"""
        self.detection_result = result
        self.event_manager.push_event('hand_detected', bool(result.hand_landmarks))
    

    def detect_async(self, image, time_stamp):
        if self.landmarker:
            self.landmarker.detect_async(image, time_stamp)
    

    def get_latest_result(self):
        return self.detection_result
    

    def close(self):
        if self.landmarker:
            self.landmarker.close()
            self.landmarker = None