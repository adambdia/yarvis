#!/usr/bin/env python
import cv2
import traceback
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import time
from event_manager import Event_Manager
from hand_detector import Hand_Detector
from kinect_manager import Kinect
import pygame


WINDOW_HEIGHT = 1080
WINDOW_WIDTH = 1920
WINDOW_NAME = "yarvis"

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)


def main():
    start_time = time.time_ns()  #
    event_manager = Event_Manager()  #
    hand_detector = Hand_Detector(event_manager, num_hands=2)  #
    kinect_manager = Kinect(event_manager)  #
    window = cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)  #
    cv2.setWindowProperty(
        WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
    )  #

    calib_attempts = []
    calib_points = [(320, 110), (670, 500), (1180, 725), (1670, 325)]
    do_calibrate = False

    calibration_matrix = None

    try:
        calibration_matrix = np.load("calibration.npy")
        print("calibration found")
    except:
        print("no calibration found")

    while True:
        key = cv2.waitKey(1)
        time_stamp = (time.time_ns() - start_time) * 1000
        kinect_manager.update_frames()
        depth_frame = kinect_manager.get_depth_frame()
        ir_frame = kinect_manager.get_ir_frame()
        window_frame = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3))

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=ir_frame)
        hand_detector.detect_async(mp_image, time_stamp)

        if event_manager.poll_event("hand_detected"):
            cv2.putText(
                window_frame,
                "hand detected",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                6,
            )
            detection_result = hand_detector.get_uncalibrated_result()
            # window_frame = draw_landmarks_on_image(window_frame, detection_result)

            landmarks = detection_result.hand_landmarks[0]
            index_tip = landmarks[Hand_Detector.MP_KEY_POINTS["INDEX_FINGER_TIP"]]

            raw_x = int(index_tip.x * ir_frame.shape[1])
            raw_y = int(index_tip.y * ir_frame.shape[0])
            raw_z = depth_frame[raw_y][raw_x]

            if calibration_matrix is not None:
                index_pos = np.array([[(raw_x, raw_y)]], dtype=np.float32)
                index_pos = cv2.perspectiveTransform(index_pos, calibration_matrix)
                index_pos = index_pos[0][0].astype(dtype=int)
                cv2.circle(
                    window_frame,
                    center=index_pos,
                    radius=15,
                    color=(255, 0, 0),
                    thickness=-1,
                )
            cv2.circle(
                window_frame,
                center=(raw_x, raw_y),
                radius=15,
                color=(0, 0, 255),
                thickness=-1,
            )

        if do_calibrate:
            cv2.putText(
                window_frame,
                "points: {}".format(str(len(calib_attempts))),
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                6,
            )
            for i, calib_point in enumerate(calib_points):
                cv2.circle(
                    window_frame,
                    center=calib_point,
                    radius=15,
                    color=(0, 0, 255),
                    thickness=-1,
                )
                text_pos = (calib_point[0] + 20, calib_point[1])
                cv2.putText(
                    window_frame,
                    str(i + 1),
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    6,
                )

            if key == ord("a") and event_manager.poll_event("hand_detected"):
                calib_attempts.append((raw_x, raw_y))

            if len(calib_attempts) == len(calib_points):
                calib_points = np.array(calib_points, dtype=np.float32)
                calib_attempts = np.array(calib_attempts, dtype=np.float32)
                calibration_matrix, _ = cv2.findHomography(calib_attempts, calib_points)
                np.save("calibration.npy", calibration_matrix)
                print("saving calibration")
                do_calibrate = False

        # cv2.imshow('Depth', depth_frame / 4500.0 )  # Normalize depth for visualization
        cv2.imshow(WINDOW_NAME, window_frame)
        # event_manager.view_event()

        event_manager.push_event("key_pressed", key)
        if key == 27:  # escape key
            break
        elif key == ord("c"):
            do_calibrate = not do_calibrate

    hand_detector.close()
    print("Closing")
    cv2.destroyAllWindows()


class App:
    def __init__(self):
        self.event_manager = Event_Manager()
        self.kinect = Kinect(self.event_manager)
        self.hand_detector = Hand_Detector(self.event_manager)

        pygame.init()
        self.screen = pygame.display.set_mode(
            (WINDOW_WIDTH, WINDOW_HEIGHT), pygame.FULLSCREEN
        )
        pygame.display.set_caption(WINDOW_NAME)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 100)
        self.running = True

        self.do_calibrate = False

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

    def run(self):
        counter = 0
        while self.running:
            self.events()  # check inputs
            self.clock.tick()
            time_stamp = pygame.time.get_ticks()
            self.screen.fill(BLACK)

            self.kinect.update_frames()
            ir_frame = self.kinect.get_ir_frame()
            rgb_frame = self.kinect.get_rgb_frame()
            registered_frame = self.kinect.get_registered_frame()
            #cv2.resize(ir_frame, (0,0), fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)
            if counter == 4:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=ir_frame)
                self.hand_detector.detect_async(mp_image, time_stamp)
                counter = 0

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
                hand_detected = self.font.render("hand detected", True, BLUE)
                self.screen.blit(hand_detected, (100, 900))
                detection_result = self.hand_detector.get_calibrated_result()
                for key_point in Hand_Detector.MP_KEY_POINTS:
                    pos = detection_result[key_point]
                    pygame.draw.circle(self.screen, BLUE, pos, 10)

            fps = self.clock.get_fps()
            fps = self.font.render("{:.1f}".format(fps), True, WHITE)
            self.screen.blit(fps, (100, 1000))

            pygame.display.flip()  # update the screen
            counter += 1


if __name__ == "__main__":
    yarvis = App()
    try:
        yarvis.run()
        # main()
    finally:
        if yarvis.running:
            yarvis.quit()
