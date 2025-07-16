#!/usr/bin/env python

import mediapipe as mp
from event_manager import Event_Manager
from hand_detector import Hand_Detector
from kinect_manager import Kinect
from debug_activity import debug_activy
import pygame
import common

class App:
    def __init__(self):
        self.event_manager = Event_Manager()
        self.kinect = Kinect(self.event_manager)
        self.hand_detector = Hand_Detector(self.event_manager, detection_confidence=common.HAND_DETECTION_CONFIDENCE, num_hands=common.NUM_HANDS)

        pygame.init()
        self.screen = pygame.display.set_mode((common.WINDOW_WIDTH, common.WINDOW_HEIGHT), pygame.FULLSCREEN)
        pygame.display.set_caption(common.WINDOW_NAME)
        self.clock = pygame.time.Clock()
        self.running = True
        self.current_activity = debug_activy(self.screen, self.event_manager, self.clock, self.kinect, self.hand_detector)

    def quit(self):
        self.kinect.kinect.stop()
        self.hand_detector.close()
        pygame.quit()
        self.running = False

    def events(self):
        if self.event_manager.poll_event("update_matrix"):
            self.hand_detector.calibration_matrix = self.current_activity.calib_matrix
            self.event_manager.push_event("update_matrix", False)

        for event in pygame.event.get():
            key = None
            if event.type == pygame.KEYDOWN:
                key = event.key

            if key == pygame.K_ESCAPE or event.type == pygame.QUIT:
                self.quit()
            
            self.current_activity.events(event, key)

    def run(self):
        counter = 0
        while self.running:
            self.events()
            self.clock.tick()
            time_stamp = pygame.time.get_ticks()
            self.screen.fill(common.BLACK)

            self.kinect.update_frames()
            ir_frame = self.kinect.get_ir_frame()
            # rgb_frame = self.kinect.get_rgb_frame()
            # registered_frame = self.kinect.get_registered_frame()

            if counter == 4: # send frame to model every four frames, runs better
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=ir_frame)
                self.hand_detector.detect_async(mp_image, time_stamp)
                counter = 0

            self.current_activity.update()

            
            counter += 1
            pygame.display.flip()  # update the screen


if __name__ == "__main__":
    yarvis = App()
    try:
        yarvis.run()
        # main()
    finally:
        if yarvis.running:
            yarvis.quit()
