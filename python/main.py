import cv2
from kinect_bridge import KinectBridge

def main():
    try:
        # Initialize Kinect
        kinect = KinectBridge()
        
        while True:
            try:
                # Get frames from Kinect
                frames = kinect.get_frames()
                bgr_frame = frames['bgr']
                depth_frame = frames['depth']
                
                # Display the frames
                cv2.imshow('RGB', bgr_frame)
                cv2.imshow('Depth', depth_frame / 4500.0)  # Normalize depth for visualization
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except RuntimeError as e:
                print(f"Frame capture error: {e}")
                continue
                
    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()