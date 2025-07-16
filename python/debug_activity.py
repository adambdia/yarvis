from activity import Activity
import pygame
import numpy as np
from hand_detector import Hand_Detector
import common
import cv2

CALIB_POINT_RADIUS = 10

class debug_activy(Activity):
    def __init__(self, screen, event_manager, clock, kinect_manager, hand_detector):
        super().__init__(screen, event_manager, clock)
        self.kinect = kinect_manager
        self.hand_detector = hand_detector
        self.font = pygame.font.Font(None, 100)
        
        self.display_ir = False
        self.display_rgb = False
        self.display_landmarks = True
        self.do_calibrate = False

        self.calib_points = [(200, 90), (150, 950), (1750, 120), (1600, 800)]
        self.calib_attempts = []
        self.save_point_flag = False
        self.calib_matrix = None

    def events(self, event, key):
        if key == pygame.K_i: 
            self.display_ir = ~self.display_ir
                
        if key == pygame.K_r:
            self.display_rgb = ~self.display_rgb

        if key == pygame.K_c:
            self.do_calibrate = ~self.do_calibrate

        if key == pygame.K_l:
            self.display_landmarks = ~self.display_landmarks
        
        if self.do_calibrate:
            if key == pygame.K_a:
                self.save_point_flag = True
            if key == pygame.K_d:
                self.calib_attempts = []

    def calibration_routine(self, ir_frame):
        for i, calib_point in enumerate(self.calib_points):
            pygame.draw.circle(self.screen, common.RED, calib_point, CALIB_POINT_RADIUS)
            common.draw_text(str(i+1), calib_point, common.WHITE, self.font, self.screen)
        
        common.draw_text("c", 
                        (300, 1000), 
                        common.GREEN, 
                        self.font, 
                        self.screen)
        common.draw_text(str(len(self.calib_attempts)), 
                        (350, 1000), 
                        common.GREEN, 
                        self.font, 
                        self.screen)
        
        if self.save_point_flag and self.event_manager.poll_event("uncalibrated_hand_result"):
            result =  self.hand_detector.get_uncalibrated_result()
            index_pos = result.hand_landmarks[0][Hand_Detector.MP_KEY_POINTS["INDEX_FINGER_TIP"]]
            index_x = index_pos.x * ir_frame.shape[1] # mediapipe landmark coordinates go from 0 to 1, we need to scale it up to the frame size
            index_y = index_pos.y * ir_frame.shape[0]
            index_pos = (index_x, index_y) 
            self.calib_attempts.append(index_pos)
            self.save_point_flag = False
        
        if len(self.calib_attempts) == len(self.calib_points):
            calib_points_np = np.array(self.calib_points, dtype=np.float32)
            calib_attempts_np = np.array(self.calib_attempts, dtype=np.float32)
            self.calib_matrix, _ = cv2.findHomography(calib_attempts_np, calib_points_np)
            np.save('calibration.npy', self.calib_matrix)
            self.do_calibrate = False
            self.event_manager.push_event("update_matrix", True)
            self.calib_attempts = []

    def draw_landmarks(self, ir_frame):
        if self.event_manager.poll_event("hand_result"):
            hand_detected = self.font.render("hand detected", True, common.BLUE)
            self.screen.blit(hand_detected, (100, 900))
            detection_result = self.hand_detector.get_calibrated_result()
            for hand in detection_result:    
                for key_point in Hand_Detector.MP_KEY_POINTS:
                    pos = hand[key_point]
                    pygame.draw.circle(self.screen, common.BLUE, pos, 10)
                    # if key_point == "INDEX_FINGER_TIP":
                    #     x, y, z = self.kinect.get_point_xyz(pos[0], pos[1])
        if self.event_manager.poll_event("uncalibrated_hand_result"):
            detection_result = self.hand_detector.get_uncalibrated_result()
            uncalib_index_landmark = detection_result.hand_landmarks[0][Hand_Detector.MP_KEY_POINTS["INDEX_FINGER_TIP"]]
            uncalib_index_x = uncalib_index_landmark.x * ir_frame.shape[0]
            uncalib_index_y = uncalib_index_landmark.y * ir_frame.shape[1]
            uncalib_index_pos = (uncalib_index_x, uncalib_index_y)
            pygame.draw.circle(self.screen, common.RED, uncalib_index_pos, 5)

    def update(self):
        ir_frame = self.kinect.get_ir_frame()

        if self.display_ir:
            ir_frame_pg = common.cv_to_pygame(ir_frame)
            self.screen.blit(ir_frame_pg, (0, 0))

        if self.display_rgb:
            rgb_frame = self.kinect.get_rgb_frame()
            rgb_frame = common.cv_to_pygame(rgb_frame)
            self.screen.blit(rgb_frame, (0, 0))

        if self.do_calibrate:
            self.calibration_routine(ir_frame)

        if self.display_landmarks:
            self.draw_landmarks(ir_frame)

        # registered_frame = np.rot90(registered_frame)
        # registered_frame = pygame.surfarray.make_surface(registered_frame)
        # self.screen.blit(registered_frame, (0, 0))

        fps = self.clock.get_fps()
        common.draw_text("{:.1f}".format(fps), (100, 1000), common.WHITE, self.font, self.screen)