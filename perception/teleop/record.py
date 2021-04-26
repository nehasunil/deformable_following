import datetime
import glob
import os

import cv2
import numpy as np

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

cap = cv2.VideoCapture(0)
dir_output = "images/03062021/"
os.makedirs(dir_output, exist_ok=True)
# images = glob.glob("*.jpg")

# for fname in images:
while True:
    ret, img = cap.read()
    raw_img = img.copy()

    print(img.shape)
    # img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
    cv2.imshow("img", img)
    c = cv2.waitKey(1)
    if c == ord("q"):
        break
    elif c == ord("l"):
        fn_output = datetime.datetime.now().strftime("%Y%M%d_%H%m%S%f")
        cv2.imwrite(os.path.join(dir_output, fn_output) + ".png", raw_img)

cv2.destroyAllWindows()
