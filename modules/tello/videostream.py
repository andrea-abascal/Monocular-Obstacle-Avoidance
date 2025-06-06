import cv2
from djitellopy import Tello

# Initialize and connect to Tello
tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")

# Start video stream
tello.streamon()

while True:
    frame = tello.get_frame_read().frame
    if frame is not None and frame.any():
        cv2.imshow("Tello Stream", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
tello.end()
