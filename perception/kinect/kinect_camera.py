import cv2
import pyk4a
from pyk4a import Config, PyK4A
import numpy as np

class Kinect():
    def __init__(self, cam_intrinsics, dist):
        k4a = PyK4A(
            Config(
                color_resolution=pyk4a.ColorResolution.RES_1080P,
                depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            )
        )
        k4a.start()

        self.mtx = cam_intrinsics
        self.dist = dist

        exp_dict = {-11: 500, -10: 1250, -9: 2500, -8: 8330, -7: 16670, -6: 33330}
        exp_val = -6  # to be changed when running
        k4a.exposure = exp_dict[exp_val]

        self.k4a = k4a

    def undistort_kinect(self, img):
        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), 1, (w,h))

        # undistort
        dst = cv2.undistort(img, self.mtx, self.dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        return dst

    def get_image(self):
        capture = self.k4a.get_capture()
        color, depth_transformed = None, None
        if capture.color is not None:
            color = capture.color
            color_undist = self.undistort_kinect(color)

        if capture.transformed_depth is not None:
            depth_transformed = capture.transformed_depth
            depth_undist = self.undistort_kinect(depth_transformed)

        return color_undist, depth_undist

if __name__ == "__main__":
    camera = Kinect()
    while True:
        color, depth = camera.get_image()
        if color is None or depth is None:
            continue
        cv2.imshow("color", color)
        cv2.imshow("depth", depth / 1100)
        c = cv2.waitKey(1)
        if c == ord('q'):
            break
