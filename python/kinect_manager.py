import cv2
import numpy as np
from kinect_bridge import KinectBridge 

class Kinect:
    def __init__(self, event_manager) -> None:
        try:
            print("Initializing Kinect...")
            self.kinect = KinectBridge()
            print("Kinect initialized successfully")

        except:
              print("Kinect failed")
        self.event_manager = event_manager

    def get_frames(self):
                # Get frames from Kinect
                try:
                    frames = self.kinect.get_frames()
                    # bgr_frame = frames['bgr']
                    depth_frame = frames['depth']
                    ir_frame = frames['ir']              
                    ir_frame = ir_frame / 256.0
                    ir_frame = ir_frame.astype(np.uint8)
                    ir_frame = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)

                    return depth_frame, ir_frame
                except RuntimeError as e:
                    print(e)