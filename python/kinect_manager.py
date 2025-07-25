import cv2
import numpy as np
from kinect_bridge import KinectBridge


class Kinect:
    def __init__(self, event_manager) -> None:
        try:
            print("[DEBUG] Initializing Kinect...")
            self.kinect = KinectBridge()
            print("[DEBUG] Kinect initialized successfully")

        except Exception as e:
            print("[DEBUG] Kinect failed: %s", e)
        self.event_manager = event_manager
        self.depth_frame = None
        self.ir_frame = None
        self.rgb_frame = None
        self.registered = None

    def update_frames(self) -> None:
        try:
            frames = self.kinect.get_frames()
            # bgr_frame = frames['bgr']
            registered_frame = frames["registered"]
            rgb_frame = frames["bgr"]
            depth_frame = frames["depth"]
            ir_frame = frames["ir"]
            ir_frame = ir_frame / 256.0
            ir_frame = ir_frame.astype(np.uint8)
            ir_frame = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
            ir_frame = cv2.flip(ir_frame, 0)
            self.depth_frame = depth_frame
            self.ir_frame = ir_frame
            try:
                self.rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(e)
            try:
                self.registered = cv2.cvtColor(registered_frame, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(e)
        except RuntimeError as e:
            print(e)

    def get_ir_frame(self):
        return self.ir_frame

    def get_depth_frame(self):
        return self.depth_frame

    def get_rgb_frame(self):
        return self.rgb_frame

    def get_registered_frame(self):
        return self.registered

    def get_point_xyz(self, x: int, y: int):
        return self.kinect.get_point_xyz(x, y)

    def get_point_xyzrgb(self, x: int, y: int):
        return self.kinect.get_point_xyzrgb(x, y)
