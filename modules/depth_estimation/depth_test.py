import os
import cv2
import time 
import numpy as np
from depth_estimator import DepthEstimator
from collections import deque
from djitellopy import Tello  # Not needed for path logic

# ── Determine save directory & filename ─────────────────────────────────────
# This makes SAVE_PATH = <folder-of-this-script>/depth_map.png
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(SCRIPT_DIR, "depth_map.png")

# ── FPS Meter ───────────────────────────────────────────────────────────────
class FPSMeter:
    """Thread-/process-local fps meter: call .tick() each event."""
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

# ── Smoothing Utilities ──────────────────────────────────────────────────────
def patch_depth(depth_map, u, v, k=1):
    """Spatial average over a (2k+1)x(2k+1) neighborhood around (u,v)."""
    h, w = depth_map.shape
    u0 = int(np.clip(u, k, w - k - 1))
    v0 = int(np.clip(v, k, h - k - 1))
    patch = depth_map[v0-k:v0+k+1, u0-k:u0+k+1]
    return float(np.nanmean(patch))

class DepthEMA:
    """Exponential Moving Average filter for depth values."""
    def __init__(self, alpha=0.7):
        self.alpha = alpha
        self.ema = None

    def update(self, z):
        if self.ema is None:
            self.ema = z
        else:
            self.ema = self.alpha * z + (1 - self.alpha) * self.ema
        return self.ema

# ── Configurations & Depth Estimator ────────────────────────────────────────
TARGET_WH = (380, 260)
metric = True  

if metric:
    estimator = DepthEstimator(engine_path='monocular_obstacle_avoidance/modules/depth_estimation/engines/depth_anything_v2_metric_hypersim_vits_350x350.engine')
    #estimator = DepthEstimator(encoder='vits',device='cuda', metric=True) # NO TensorRT
else:
    estimator = DepthEstimator(engine_path='monocular_obstacle_avoidance/modules/depth_estimation/engines/depth_anything_v2_vits_350x350.engine')
     #estimator = DepthEstimator(encoder='vits',device='cuda', metric=False) # NO TensorRT 

# ── Capture Initialization ──────────────────────────────────────────────────
WEBCAM = False
if WEBCAM:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam could not be opened.")
else:
    tello = Tello()
    tello.connect()
    print(f"[Capture] Battery {tello.get_battery():d}%")
    tello.streamon()
    stream = tello.get_frame_read()

meter      = FPSMeter("DEP", every=50)
hist       = deque(maxlen=10)
ema_filter = DepthEMA(alpha=0.7)

# ── Main Loop ────────────────────────────────────────────────────────────────
while True:
    if WEBCAM:
        ret, frame = cap.read()
        if not ret:
            print("Frame not received from webcam.")
            break
    else:
        frame = stream.frame.copy()
    frame = cv2.resize(frame, TARGET_WH, interpolation=cv2.INTER_AREA)
    H, W = frame.shape[:2]
    cx, cy = W // 2, H // 2

    # Depth estimation in centimeters
    depth = estimator.predict_depth(frame) * 100
    meter.tick()

    # Spatial + temporal smoothing
    #raw = patch_depth(depth, cy, cx, k=3)
    #hist.append(raw)
    #ema = ema_filter.update(raw)
    #smooth_depth = ema

    # Build visualization
    vis = frame.copy()
    cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
    center_depth = depth[H//2, W//2]
    cv2.putText(vis,
                f"{center_depth:.2f} cm",
                (cx - 50, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA)

    norm_depth    = (depth - depth.min()) / (depth.max() - depth.min())
    depth_8bit    = (norm_depth * 255).astype(np.uint8)
    color_mapped  = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_CIVIDIS)
    vis = cv2.hconcat([vis,
                       np.ones((H, 10, 3), np.uint8) * 255,
                       color_mapped])

    cv2.imshow('Depth Anything', vis)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        # save to SCRIPT_DIR/depth_map.png
        cv2.imwrite(SAVE_PATH, vis)
        print(f"[INFO] Frame saved to {SAVE_PATH}")
    elif key == ord("q"):
        break

if WEBCAM:
    cap.release()
else:
    tello.streamoff()
    tello.end()
cv2.destroyAllWindows()
