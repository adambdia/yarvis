import cv2
from kinect_bridge import KinectBridge
import sys
import traceback
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, 
                    model_complexity=1,
                    min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils



def main():
    try:
        print("Initializing Kinect...")
        kinect = KinectBridge()
        print("Kinect initialized successfully")
        
        while True:
            try:
                # Get frames from Kinect
                frames = kinect.get_frames()
                # bgr_frame = frames['bgr']
                depth_frame = frames['depth']
                ir_frame = frames['ir']
                
                # Display the frames
                #cv2.imshow('RGB', bgr_frame)
                #cv2.imshow('Depth', depth_frame / 4500.0)  # Normalize depth for visualization
                #cv2.imshow('ir', ir_frame / 65536.0)
                ir_frame = ir_frame / 65536.0
                ir_frame = ir_frame.astype(np.uint8)

                results = pose.process(cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2RGB))

                if results.pose_landmarks:
                    mp_draw.draw_landmarks(
                    ir_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                    )

                cv2.imshow('Kinect MediaPipe', ir_frame)


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