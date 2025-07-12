#!/usr/bin/env python

import mediapipe as mp
from event_manager import Event_Manager
from hand_detector import Hand_Detector
from kinect_manager import Kinect
from debug_activity import debug_activy
import pygame
import common


WINDOW_HEIGHT = 1080
WINDOW_WIDTH = 1920
WINDOW_NAME = "yarvis"

class App:
    def __init__(self):
        self.event_manager = Event_Manager()
        self.kinect = Kinect(self.event_manager)
        self.hand_detector = Hand_Detector(self.event_manager)

        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.FULLSCREEN)
        pygame.display.set_caption(WINDOW_NAME)
        self.clock = pygame.time.Clock()
        self.running = True
        self.current_activity = debug_activy(self.screen, self.event_manager, self.clock, self.kinect, self.hand_detector)

    def quit(self):
        self.kinect.kinect.stop()
        self.hand_detector.close()
        pygame.quit()
        self.running = False

    def events(self):
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

            if counter == 4:
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
