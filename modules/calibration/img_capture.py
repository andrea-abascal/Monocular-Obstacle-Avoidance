# This program captures images from video stream. Images are used in calibration.
import cv2
from djitellopy import tello

# Initialize and connect to Tello
tellos = tello.Tello()
tellos.connect()
print(f"Battery: {tellos.get_battery()}%")

# Start video stream
tellos.streamon()
num = 0
while True:
    # Obtain frame from videostream
    frame = tellos.get_frame_read().frame
    if frame is not None and frame.any():
        cv2.imshow("Tello Stream", frame)
    
    key = cv2.waitKey(1)
    
    if key & 0xFF == ord('q'):
        break
    
    # Capture frame
    elif key & 0xFF == ord('s'): # wait for 's' key to save
        
        cv2.imwrite('images/img'+str(num)+'.jpg',frame)
    
        print(f"image {num} saved!")
        
        num+=1

cv2.destroyAllWindows()
tellos.end()