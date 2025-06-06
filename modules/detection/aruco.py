import cv2
import math
class ArucoDetector:
   
    def __init__(self, detector, cam_matrix, dist_coeffs, obj_points, marker_length=0.1, view=True):
        self.detector      = detector         # (aruco_dict, parameters)
        self.cam_matrix    = cam_matrix
        self.dist_coeffs   = dist_coeffs
        self.obj_points    = obj_points
        self.marker_length = marker_length
        self.view          = view

    def compute_yaw_angle(self, rvec):
        """
        Compute yaw angle (in degrees) from translation vector tvec.
        """
        R, _ = cv2.Rodrigues(rvec)

        yaw_rad = math.atan2(R[0, 2], -R[2, 2])
        yaw_deg = math.degrees(yaw_rad)
            
        return float(yaw_deg) 

    def detect(self, image, estimate_pose=True):

        output_image = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Unpack ArUco detector tuple
        aruco_dict, parameters = self.detector

        # Detect markers
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is None:
            # No markers found
            return False, output_image, (0, 0), 0.0

        detected = False
        last_center = (0, 0)
        last_angle = 0.0

        if estimate_pose:
            for i in range(len(ids)):
                marker_corners = corners[i]  # shape: (1, 4, 2)
                # Compute marker center (using two diagonal corners)
                c0 = marker_corners[0][0]
                c2 = marker_corners[0][2]
                center_y = int((c0[1] + c2[1]) / 2)
                center_x = int((c0[0] + c2[0]) / 2)
                last_center = (center_x, center_y)

                # SolvePnP to get rotation (rvec) and translation (tvec)
                # obj_points should be shape (4, 3), corners[i] is (1,4,2)
                success, rvec, tvec = cv2.solvePnP(self.obj_points, marker_corners, self.cam_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

                if not success:
                    continue

                # Compute yaw
                angle = self.compute_yaw_angle(rvec)
                last_angle = angle

                if self.view:
                    # Draw axes on marker
                    cv2.drawFrameAxes(output_image, self.cam_matrix, self.dist_coeffs, rvec, tvec, self.marker_length * 1.5, 2)

                    # Overlay angle text at top-left corner of marker
                    text = f"Angle = {angle:.2f}"
                    text_pos = (int(marker_corners[0][0][0]), int(marker_corners[0][0][1]))
                    cv2.putText(output_image, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                detected = True

        else:
            # If not estimating pose, we can still return True (markers found) but skip drawing
            detected = True

        return detected, output_image, last_center, last_angle