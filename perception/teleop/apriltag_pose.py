import time

import cv2
import numpy as np
import _thread
import yaml
from dt_apriltags import Detector
import urllib
import urllib.request
from threading import Thread
from scipy.spatial.transform import Rotation as R
import os


class Streaming(object):
    def __init__(self, url):
        self.image = None
        self.url = url

        self.streaming = False

        self.start_stream()

    def __del__(self):
        self.stop_stream()

    def start_stream(self):
        self.streaming = True
        self.stream = urllib.request.urlopen(self.url)

    def stop_stream(self):
        if self.streaming == True:
            self.stream.close()
        self.streaming = False

    def load_stream(self):
        stream = self.stream
        bytess = b""

        while True:
            if self.streaming == False:
                time.sleep(0.01)
                continue

            bytess += stream.read(32767)

            a = bytess.find(b"\xff\xd8")  # JPEG start
            b = bytess.find(b"\xff\xd9")  # JPEG end

            if a != -1 and b != -1:
                jpg = bytess[a : b + 2]  # actual image
                bytess = bytess[b + 2 :]  # other informations

                self.image = cv2.imdecode(
                    np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR
                )

class AprilTagPose(Thread):
    def __init__(self, url):
        Thread.__init__(self)

        self.img_undist = None

        self.stream = Streaming(url)
        _thread.start_new_thread(self.stream.load_stream, ())

    def run(self):
        at_detector = Detector(
            families="tag36h11",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

        tag_size = 40  ## unit: mm

        # cap = cv2.VideoCapture(0)
        dir_abs = os.path.dirname(os.path.realpath(__file__))
        print(os.path.join(dir_abs, "camera_parameters.yaml"))
        with open(os.path.join(dir_abs, "camera_parameters.yaml"), "r") as stream:
            parameters = yaml.load(stream)

        mtx_origin = np.array(parameters["K"]).reshape([3, 3])
        dist = np.array(parameters["dist"])


        H, W = parameters["H"], parameters["W"]
        camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx_origin, dist, (W, H), 1, (W, H))

        camera_params = (
            camera_matrix[0, 0],
            camera_matrix[1, 1],
            camera_matrix[0, 2],
            camera_matrix[1, 2],
        )

        axis = (
            np.float32([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, -1]]).reshape(-1, 3)
            * tag_size
            / 2
        )


        def draw(img, imgpts):
            corner = tuple(imgpts[0].ravel())
            img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (255, 0, 0), 5)
            img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 255, 0), 5)
            img = cv2.line(img, corner, tuple(imgpts[3].ravel()), (0, 0, 255), 5)
            return img



        while True:
            img = self.stream.image

            if img is None:
                print("Waiting for streaming...")
                time.sleep(0.1)
                continue

            # undistort
            img_undist = cv2.undistort(img, mtx_origin, dist, None, camera_matrix)

            # crop the image
            x, y, w, h = roi
            img_undist = img_undist[y : y + h, x : x + w]

            img_undist_gray = np.mean(img, axis=-1).astype(np.uint8)
            st = time.time()
            # detect apriltag
            tags = at_detector.detect(img_undist_gray, True, camera_params, tag_size)

            if len(tags) > 0:

                # project 3D points to image plane
                xy, jac = cv2.projectPoints(
                    axis, tags[0].pose_R, tags[0].pose_t, camera_matrix, (0, 0, 0, 0, 0)
                )
                draw(img_undist, np.round(xy).astype(np.int))
                self.pose = (tags[0].pose_t, tags[0].pose_R)
            else:
                self.pose = None
            # print(time.time() - st)

            self.img_undist = img_undist.copy()

        cv2.destroyAllWindows()

if __name__ == "__main__":
    url = "http://rpigelsight2.local:8080/?action=stream"
    april_tag_pose = AprilTagPose(url)
    april_tag_pose.start()
    while True:
        if april_tag_pose.img_undist is None:
            continue

        cv2.imshow("frame", cv2.resize(april_tag_pose.img_undist, (0, 0), fx=0.5, fy=0.5))
        c = cv2.waitKey(1)

        if c == ord("q"):
            break
    april_tag_pose.join()
