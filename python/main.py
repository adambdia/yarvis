import cv2
from kinect_bridge import KinectBridge


# Initialize Kinect
kinect = KinectBridge()

try:
    while True:
        # Get frames from Kinect
        frames = kinect.get_frames()
        bgr_frame = frames['bgr']  # This is compatible with OpenCV/MediaPipe
        depth_frame = frames['depth']
              
        # Display the frame
        cv2.imshow('Kinect MediaPipe', bgr_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cv2.destroyAllWindows()
    del kinect  # Clean up Kinect resources