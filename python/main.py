#!/usr/bin/env python
import cv2
from kinect_bridge import KinectBridge 
import traceback
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import time
from event_manager import Event_Manager
from hand_detector import Hand_Detector
from kinect_manager import Kinect


def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)
  
  MARGIN = 10  # pixels
  FONT_SIZE = 1
  FONT_THICKNESS = 1
  HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green


  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image


def main():
    start_time = time.time_ns()
    event_manager = Event_Manager()
    hand_detector = Hand_Detector(event_manager)
    

    try:
        print("Initializing Kinect...")
        kinect = KinectBridge()
        print("Kinect initialized successfully")
        kinect_manager = Kinect(kinect,start_time)
        while True:
            try:
                # Get frames from Kinect
                time_stamp, depth_frame, ir_frame = kinect_manager.get_frames()

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=ir_frame)
                hand_detector.detect_async(mp_image, time_stamp)
                
                if event_manager.poll_event('hand_detected'): 
                    detection_result = hand_detector.get_latest_result()
                    ir_frame = draw_landmarks_on_image(ir_frame, detection_result)

                    landmarks = detection_result.hand_landmarks[0]
                    index_tip = landmarks[Hand_Detector.KEY_POINTS['INDEX_FINGER_TIP']]

                    x = int(index_tip.x * ir_frame.shape[1])
                    y = int(index_tip.y * ir_frame.shape[0])
                    z = depth_frame[y][x]

                    #cv2.circle(ir_frame, (x,y), 10, (0,0,255), -1)
                
                #cv2.imshow('Depth', depth_frame / 4500.0 )  # Normalize depth for visualization
                cv2.imshow('IR', ir_frame)
                event_manager.view_event()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except RuntimeError as e:
                print(f"Frame capture error: {e}")
                print("Stack trace:")
                traceback.print_exc()
                continue
    
    finally:
        hand_detector.close()
        cv2.destroyAllWindows()
        print("Cleaning up...")


if __name__ == "__main__":
    main()