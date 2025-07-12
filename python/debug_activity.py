from activity import Activity
import pygame
import numpy as np
from hand_detector import Hand_Detector
import common

class debug_activy(Activity):
    def __init__(self, screen, event_manager, clock, kinect_manager, hand_detector):
        super().__init__(screen, event_manager, clock)
        self.kinect = kinect_manager
        self.hand_detector = hand_detector
        self.font = pygame.font.Font(None, 100)

    def events(self, event):
        #for the calibration stuff eventually
        pass

    def update(self, ir_frame):

        ir_frame = np.rot90(ir_frame) 
        ir_frame = pygame.surfarray.make_surface(ir_frame)
        self.screen.blit(ir_frame, (0, 0))

        # rgb_frame = np.rot90(rgb_frame)
        # rgb_frame = pygame.surfarray.make_surface(rgb_frame)
        # self.screen.blit(rgb_frame, (0, 0))

        # registered_frame = np.rot90(registered_frame)
        # registered_frame = pygame.surfarray.make_surface(registered_frame)
        # self.screen.blit(registered_frame, (0, 0))

        if self.event_manager.poll_event("hand_result"):
            hand_detected = self.font.render("hand detected", True, common.BLUE)
            self.screen.blit(hand_detected, (100, 900))
            detection_result = self.hand_detector.get_calibrated_result()
            for key_point in Hand_Detector.MP_KEY_POINTS:
                pos = detection_result[key_point]
                pygame.draw.circle(self.screen, common.BLUE, pos, 10)
                if key_point == "INDEX_FINGER_TIP":
                    x, y, z = self.kinect.get_point_xyz(pos[0], pos[1])

        fps = self.clock.get_fps()
        fps = self.font.render("{:.1f}".format(fps), True, common.WHITE)
        self.screen.blit(fps, (100, 1000))

        pass