import mediapipe as mp
import os
from functools import partial
from event_manager import Event_Manager

KEY_POINTS = {
    'thumb_tip': 4,
    'thumb_ip': 3,
    'thumb_mcp': 2,
    
    'index_tip': 8,
    'index_dip': 7,
    'index_pip': 6,
    'index_mcp': 5,
    
    'middle_tip': 12,
    'middle_dip': 11,
    'middle_pip': 10,
    'middle_mcp': 9,
    
    'ring_tip': 16,
    'ring_dip': 15,
    'ring_pip': 14,
    'ring_mcp': 13,
    
    'pinky_tip': 20,
    'pinky_dip': 19,
    'pinky_pip': 18,
    'pinky_mcp': 17,
    
    'wrist': 0,
}

class HandLandmarker:
    def __init__(self, event_manager: Event_Manager):
        self.event_manager = event_manager
        event_manager.write_event('hand_detected', False)

        self.detection_result = None
        self.landmarker = None
        
        self.BaseOptions = mp.tasks.BaseOptions
        self.HandLandmarker = mp.tasks.vision.HandLandmarker
        self.HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        self.HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        
        self._callback = partial(self._handle_result)
        
        options = self.HandLandmarkerOptions(
            base_options=self.BaseOptions(
                model_asset_path=os.path.expandvars('$YARVISPATH/models/hand_landmarker.task')
            ),
            running_mode=self.VisionRunningMode.LIVE_STREAM,
            result_callback=self._callback,
            min_hand_detection_confidence=0.1
        )
        self.landmarker = self.HandLandmarker.create_from_options(options)
    

    def _handle_result(self, result, output_image: mp.Image, time_stamp: int):
        """Handle the results from the hand landmarker and update event manager"""
        self.detection_result = result
        
        self.event_manager.write_event('hand_detected', bool(result and result.hand_landmarks))
        
        if result and result.hand_landmarks:
            landmarks = result.hand_landmarks[0]  # get first hand
            
            index_tip = landmarks[KEY_POINTS['index_tip']]
            x = int(index_tip.x * output_image.width)
            y = int(index_tip.y * output_image.height)
            
            self.event_manager.write_event('index_tip_x', x)
            self.event_manager.write_event('index_tip_y', y)

        else:
            self.event_manager.write_event('hand_detected', False)
    

    def detect_async(self, image, time_stamp):
        if self.landmarker:
            self.landmarker.detect_async(image, time_stamp)
    

    def get_latest_result(self):
        return self.detection_result
    

    def close(self):
        if self.landmarker:
            self.landmarker.close()
            self.landmarker = None