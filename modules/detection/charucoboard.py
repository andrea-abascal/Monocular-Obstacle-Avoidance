import cv2
import numpy as np
from collections import deque

class CharucoBoardDetector:
    def __init__(self,
                 square_length=0.034,#0.018,
                 marker_length=0.025,#0.014,
                 squares_x=6, squares_y=6,
                 dictionary_id=cv2.aruco.DICT_4X4_50,
                 camera_matrix=None,
                 dist_coeffs=None,
                 view=False,
                 clahe_period=5):
        
        # camera intrinsics
        if camera_matrix is None:
            camera_matrix = np.array([[800, 0, 320],
                                      [0, 800, 240],
                                      [0,   0,   1]], dtype=np.float32)
        if dist_coeffs is None:
            dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        self.camera_matrix    = camera_matrix
        self.dist_coeffs      = dist_coeffs
        self.view             = view
        self.clahe_period     = max(1, clahe_period)
        self._frame_counter   = 0
        self.square_length    = square_length
        self.marker_length    = marker_length
        self.squares_x        = squares_x   
        self.squares_y        = squares_y

        # ArUco / Charuco setup
        self.aruco_dict       = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.board            = cv2.aruco.CharucoBoard(
            (squares_x, squares_y), square_length, marker_length, self.aruco_dict
        )
        self.total_charuco_corners = (squares_x - 1) * (squares_y - 1)

        # tuned detector params
        dp = cv2.aruco.DetectorParameters()
        dp.adaptiveThreshWinSizeMin      = 3
        dp.adaptiveThreshWinSizeMax      = 67 #23
        dp.adaptiveThreshWinSizeStep     = 10
        dp.adaptiveThreshConstant        = 7
        dp.cornerRefinementMethod        = cv2.aruco.CORNER_REFINE_SUBPIX
        dp.cornerRefinementWinSize       = 5
        dp.cornerRefinementMaxIterations = 30
        self.detector_params = dp

    @staticmethod
    def _to_gray(img_bgr):
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _clahe(gray):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    @staticmethod
    def visible_center(ch_corners):
        if ch_corners is None or len(ch_corners) == 0:
            return None
        pts = ch_corners.reshape(-1, 2)
        return pts.mean(axis=0)
   

    def detect(self, frame_bgr ):
        """Returns center, ch_corners, ch_ids, mk_corners, mk_ids."""
        
        self._frame_counter += 1

        gray = self._to_gray(frame_bgr)
        if (self._frame_counter % self.clahe_period) == 0:
            gray = self._clahe(gray)
        # detect markers

        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.detector_params
        )
        if ids is None or len(ids) == 0:
            return None, None, None, corners, ids               # no markers

        cv2.aruco.refineDetectedMarkers(
            gray, self.board, corners, ids, rejected,
            cameraMatrix=self.camera_matrix, distCoeffs=self.dist_coeffs
        )

        _, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners, markerIds=ids,
            image=gray, board=self.board,
            cameraMatrix=self.camera_matrix, distCoeffs=self.dist_coeffs
        )
        
        center = self.visible_center(ch_corners)

        if self.view:
            if center is not None:
                cx, cy = map(int, center)
                cv2.circle(frame_bgr, (cx, cy), 5, (255, 0, 0), -1)
                cv2.putText(frame_bgr, "center", (cx+8, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            cv2.waitKey(1)

        return center, ch_corners, ch_ids, corners, ids

    def estimate_pose(self, frame_bgr, min_points=6):
        """Returns (success, center, yaw_deg)."""
        center, ch_corners, ch_ids, mk_corners, mk_ids = self.detect(frame_bgr)
        if ch_ids is None or len(ch_ids) < min_points:
            return False, center, 0.0, 0.0

        # pose solver
        try:
            retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                ch_corners, ch_ids, self.board,
                self.camera_matrix, self.dist_coeffs, None, None
            )
        except cv2.error:
            # legacy signature
            rvec = np.zeros((3,1), np.float32)
            tvec = np.zeros((3,1), np.float32)
            retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                ch_corners, ch_ids, self.board,
                self.camera_matrix, self.dist_coeffs, rvec, tvec
            )

        if not retval:
            return False, center, 0.0, 0.0

        R, _ = cv2.Rodrigues(rvec)
        yaw_rad = -np.arctan2(R[0, 2], R[2, 2])
        yaw_deg = float(np.degrees(yaw_rad))
        distance_scale = float(tvec[2]) * 100.0 

        if self.view:
            cv2.drawFrameAxes(frame_bgr, self.camera_matrix, self.dist_coeffs,
                              rvec, tvec, 0.1)
            cv2.putText(frame_bgr, f"Yaw: {yaw_deg:5.1f}°", (10,120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        

        return True, center, yaw_deg, distance_scale

    