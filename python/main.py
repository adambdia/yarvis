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


WINDOW_HEIGHT = 1080
WINDOW_WIDTH = 1920
WINDOW_NAME = 'yarvis'

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
    hand_detector = Hand_Detector(event_manager, num_hands=2)
    kinect_manager = Kinect(event_manager)
    window = cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);

    calib_attempts= []
    calib_points= [(320, 110), (670, 500), (1180, 725), (1670, 325)]
    do_calibrate = False

    calibration_matrix = None
    
    try:
      calibration_matrix = np.load('calibration.npy')
      print('calibration found')
    except:
        print('no calibration found')
    
    while True:
        time_stamp = time_stamp = (time.time_ns() - start_time) * 1000
        kinect_manager.update_frames()
        depth_frame = kinect_manager.get_depth_frame() 
        ir_frame = kinect_manager.get_ir_frame()
        window_frame = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3))

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=ir_frame)
        hand_detector.detect_async(mp_image, time_stamp)

        
        if event_manager.poll_event('hand_detected'):
            cv2.putText(window_frame, 'hand detected', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 6)
            detection_result = hand_detector.get_latest_result()
            #window_frame = draw_landmarks_on_image(window_frame, detection_result)

            landmarks = detection_result.hand_landmarks[0]
            index_tip = landmarks[Hand_Detector.KEY_POINTS['INDEX_FINGER_TIP']]

            raw_x = int(index_tip.x * ir_frame.shape[1])
            raw_y = int(index_tip.y * ir_frame.shape[0])
            raw_z = depth_frame[raw_y][raw_x]
            
            if calibration_matrix is not None:
              index_pos = np.array([[(raw_x, raw_y)]], dtype=np.float32)
              index_pos = cv2.perspectiveTransform(index_pos, calibration_matrix)
              index_pos = index_pos[0][0].astype(dtype=int)
              cv2.circle(window_frame, center = index_pos, radius = 15, color= (255,0,0), thickness = -1)
              
          
        if do_calibrate: 
          cv2.putText(window_frame, 'points: {}'.format(str(len(calib_attempts))), (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 6)
          for i, calib_point in enumerate(calib_points):
            cv2.circle(window_frame, center=calib_point, radius = 15, color =(0,0,255), thickness=-1)
            text_pos = (calib_point[0] + 20, calib_point[1])
            cv2.putText(window_frame, str(i+1), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 6)
          
          if event_manager.poll_event('key_pressed') == ord('a') and event_manager.poll_event('hand_detected'):
            event_manager.push_event('key_pressed', -1)
            calib_attempts.append((raw_x, raw_y))
          
          if len(calib_attempts) == 4:
            calib_points = np.array(calib_points, dtype=np.float32)
            calib_attempts = np.array(calib_attempts, dtype=np.float32)
            calibration_matrix, _ = cv2.findHomography(calib_attempts, calib_points)
            np.save('calibration.npy', calibration_matrix)
            print('saving calibration')
            do_calibrate = False
        
        #cv2.imshow('Depth', depth_frame / 4500.0 )  # Normalize depth for visualization
        cv2.imshow(WINDOW_NAME, window_frame)
        #event_manager.view_event()
        key = cv2.waitKey(1)
        event_manager.push_event('key_pressed', key)
        if key == 27: # escape key
            break
        elif key == ord('c'):
          do_calibrate = not do_calibrate

    hand_detector.close()
    print("Closing")
    cv2.destroyAllWindows()
    


if __name__ == "__main__":
    main()
