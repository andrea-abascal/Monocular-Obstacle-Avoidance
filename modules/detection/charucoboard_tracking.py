import cv2
import time
from djitellopy import Tello     
from charucoboard import CharucoBoardDetector
from pathlib import Path


class FPSMeter:
    """Thread-/process-local fps meter:  call .tick() each event."""
    def __init__(self, tag, every=50):
        self.tag, self.every = tag, every
        self.t0 = time.perf_counter()
        self.n  = 0
    def tick(self):
        self.n += 1
        if self.n == self.every:
            now  = time.perf_counter()
            fps  = self.every / (now - self.t0)
            self.t0, self.n = now, 0
            print(f"[{self.tag}] {fps:6.1f} Hz", flush=True)
            return fps        
        return None
    
WEBCAM = False

if __name__ == "__main__":
    
    if not WEBCAM:
        # Initialize and connect to Tello
        tello = Tello()
        tello.connect()
        print(f"Battery: {tello.get_battery()}%")

        # Start video stream
        tello.streamon() 
         
    else:
        cap = cv2.VideoCapture(0)
                            
    calib_path = next(Path.cwd().rglob("modules/calibration/data/calibrationParameters.xml"),None)
    
    if calib_path is None:
        raise FileNotFoundError("Could not find calibrationParameters.xml under any parent folders.")

    fs = cv2.FileStorage(str(calib_path), cv2.FILE_STORAGE_READ)


    K  = fs.getNode("cameraMatrix").mat()
    D  = fs.getNode("dist").mat()

    meter = FPSMeter("DETECT",every=30)
    last_hz = 0.0 

    detector = CharucoBoardDetector(camera_matrix=K, dist_coeffs=D, view=True)
   
    while True:
        if not WEBCAM:
            raw = tello.get_frame_read().frame          
            if raw is None or raw.size == 0:
                continue                                   
            
        else:
            ok, raw = cap.read()
            if not ok:
                print("Failed to read frame from webcam.")
                break
        frame = raw.copy()
        hz_val = meter.tick()
        if hz_val is not None:
            last_hz = hz_val

        _ , center, yaw_deg, _ = detector.estimate_pose(frame) #rvec, center, yaw, distance
        cv2.putText(frame, f"{last_hz:5.1f} hz", (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
 

        cv2.imshow("ChArUco detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    if not WEBCAM:
        tello.streamoff()
        print(f"Battery: {tello.get_battery()} %")
        tello.end()