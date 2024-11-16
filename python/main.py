import cv2
from kinect_bridge import KinectBridge
import traceback
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import time


DETECTION_RESULT = None

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

def display_result(result, output_image, timestamp_ms):
  global DETECTION_RESULT
  DETECTION_RESULT = result
  


def main():
    start_time = time.time_ns()
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=os.path.expandvars('$MODELPATH/hand_landmarker.task')),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=display_result)
    

    try:
        print("Initializing Kinect...")
        kinect = KinectBridge()
        print("Kinect initialized successfully")
        with HandLandmarker.create_from_options(options) as landmarker:
          while True:
              try:
                  # Get frames from Kinect
                  frames = kinect.get_frames()
                  time_stamp = (time.time_ns() - start_time) * 1000
                  # bgr_frame = frames['bgr']
                  depth_frame = frames['depth']
                  ir_frame = frames['ir']
                  
                  ir_frame = ir_frame / 256.0
                  ir_frame = ir_frame.astype(np.uint8)
                  ir_frame = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
                  mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=ir_frame)
                  landmarker.detect_async(mp_image, time_stamp)
                  
                  if DETECTION_RESULT:
                    if DETECTION_RESULT.hand_landmarks:
                       index_tip = DETECTION_RESULT.hand_landmarks[0][8]
                       x, y = int(index_tip.x*mp_image.width), int(index_tip.y*mp_image.height)
                       print(depth_frame[y][x])
                       #print(x,y)
                    #   cv2.circle(ir_frame, (x,y), 10, (0,0,255), -1)
                    
                    ir_frame = draw_landmarks_on_image(ir_frame, DETECTION_RESULT)
                  
                  cv2.imshow('Depth', depth_frame / 4500.0 )  # Normalize depth for visualization
                  cv2.imshow('IR', ir_frame)


                  if cv2.waitKey(1) & 0xFF == ord('q'):
                      break
                      
              except RuntimeError as e:
                  print(f"Frame capture error: {e}")
                  print("Stack trace:")
                  traceback.print_exc()
                  continue
                
    except Exception as e:
        print(f"Error: {e}")
        print("Stack trace:")
        traceback.print_exc()
        
    finally:
        cv2.destroyAllWindows()
        print("Cleaning up...")

if __name__ == "__main__":
    main()