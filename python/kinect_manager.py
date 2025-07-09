import cv2
import numpy as np
from kinect_bridge import KinectBridge
import asyncio
import queue


class Kinect:
    def __init__(self, event_manager) -> None:
        try:
            print("Initializing Kinect...")
            self.kinect = KinectBridge()
            print("Kinect initialized successfully")

        except Exception as e:
            print("Kinect failed: %s", e)
        self.event_manager = event_manager
        self.depth_frame = None
        self.ir_frame = None
        self.rgb_frame = None

    def update_frames(self) -> None:
        try:
            frames = self.kinect.get_frames()
            # bgr_frame = frames['bgr']
            rgb_frame = frames["bgr"]
            depth_frame = frames["depth"]
            ir_frame = frames["ir"]
            ir_frame = ir_frame / 256.0
            ir_frame = ir_frame.astype(np.uint8)
            ir_frame = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
            ir_frame = cv2.flip(ir_frame, 0)
            self.depth_frame = depth_frame
            self.ir_frame = ir_frame
            self.rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
        except RuntimeError as e:
            print(e)

    def get_ir_frame(self):
        return self.ir_frame

    def get_depth_frame(self):
        return self.depth_frame

    def get_rgb_frame(self):
        return self.rgb_frame
