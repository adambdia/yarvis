import cv2
import time
import numpy as np

class Kinect:
    def __init__(self, kinect, start_time) -> None:
        self.kinect = kinect
        self.start_time = start_time

    def get_frames(self) -> tuple[any ,any ,int]:
        # Get frames from Kinect
                frames = self.kinect.get_frames()
                time_stamp = (time.time_ns() - self.start_time) * 1000
                # bgr_frame = frames['bgr']
                depth_frame = frames['depth']
                ir_frame = frames['ir']              
                ir_frame = ir_frame / 256.0
                ir_frame = ir_frame.astype(np.uint8)
                ir_frame = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)

                return depth_frame, ir_frame, time_stamp