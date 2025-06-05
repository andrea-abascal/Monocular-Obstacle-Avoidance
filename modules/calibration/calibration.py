# Camera Calibration Script It saves the calibration parameters to an XML file for later use.
import numpy as np
import cv2
import glob

## FIND CHESSBOARD CORNERS - OBJECT POINTS (3D COORD) AND IMAGE POINTS (2D COORD) ##########

chessboardSize = (7,5) # Inner corners
h,w,_ = cv2.imread("images/image0.png").shape
frameSize = (w,h)  # Frame resolution
print(frameSize)

# Default from documentation when calibration should be ended
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 30
objp = objp * size_of_chessboard_squares_mm  # know the exact real distance in the object points

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.



images = sorted(glob.glob('images/image*.png'))
for image in images:

    img = cv2.imread(image)
    # pre process image to make code more faster by working with only one channel instead of three and find corners more easily
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, chessboardSize,  None)

    # If found, add object points, image points (after refining them)
    if ret:

        objpoints.append(objp)
        
        #get a more accurate estimate of corners

        corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        #cv2.drawChessboardCorners(img, chessboardSize, corners, ret)
        #cv2.imshow('img', img)
        #cv2.waitKey(1000)

      

cv2.destroyAllWindows()

### CALIBRATION #############################################

ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
print("Camera matrix : \n")
print(cameraMatrix)


### Re-projection Error  ###################################
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(objpoints)) )

### UNDISTORTION ###########################################
height, width, _ = img.shape
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (width, height), 1, (width, height))

#Undistort
#dst = cv2.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

######################


cv_file = cv2.FileStorage('data/calibrationParameters.xml', cv2.FILE_STORAGE_WRITE)

cv_file.write('shape',gray.shape)
cv_file.write('cameraMatrix',cameraMatrix)
cv_file.write('newCameraMatrix',newCameraMatrix)
cv_file.write('dist',dist)

cv_file.release()
