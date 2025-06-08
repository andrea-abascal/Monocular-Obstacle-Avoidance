import cv2
from collections import deque

#1. Temporal Filtering (Smoothing Over Time)
#1.1 Exponential Moving Average (EMA)
class DepthSmoother:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.depth_smooth = None  # Store previous depth

    def update(self, new_depth):
        if self.depth_smooth is None:
            self.depth_smooth = new_depth  # Initialize with first value
        else:
            self.depth_smooth = self.alpha * new_depth + (1 - self.alpha) * self.depth_smooth
        return self.depth_smooth

#1.2 Simple Moving Average (SMA)
class MovingAverageFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def update(self, new_depth):
        self.buffer.append(new_depth)
        return sum(self.buffer) / len(self.buffer)

#2. Spatial Filtering (Smoothing Over Pixels)
class SpatialFiltering:
    
        
    def Gaussian(depth_map):
        smoothed_depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0)
        return smoothed_depth_map
    def Bilateral(depth_map):
        smoothed_depth_map = cv2.bilateralFilter(depth_map, d=9, sigmaColor=75, sigmaSpace=75)
        return smoothed_depth_map
    def medianblur(depth_map):
        smoothed_depth_map = cv2.medianBlur(depth_map, 5)
        return smoothed_depth_map


