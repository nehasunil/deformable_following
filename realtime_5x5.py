import cv2
import pyk4a
from helpers import colorize
from pyk4a import Config, PyK4A
import numpy as np

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((3*3,3), np.float32)
objp[:,:2] = np.mgrid[0:3,0:3].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

def main():
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
        )
    )
    k4a.start()

    exp_dict = {-11: 500, -10: 1250, -9: 2500, -8: 8330, -7: 16670, -6: 33330}
    exp_val = -7  # to be changed when running
    k4a.exposure = exp_dict[exp_val]

    id = 0

    while True:
        capture = k4a.get_capture()

        if capture.color is not None:
            color = capture.color
            cv2.imshow("Color", color)

        if capture.transformed_depth is not None:
            depth_transformed = capture.transformed_depth
            cv2.imshow(
                "Transformed Depth", colorize(depth_transformed, (600, 1100))
            )

        gray = cv.cvtColor(color, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (3,3), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (3,3), (-1,-1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv.drawChessboardCorners(img, (3,3), corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(1)

        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite(f"data/intrinsic_test/color_{id}.png", color)
            id += 1
        elif key == ord('q'):
            cv2.destroyAllWindows()
            break

    k4a.stop()


if __name__ == "__main__":
    main()
