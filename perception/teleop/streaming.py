import urllib.request
import urllib
import cv2
import numpy as np
import time
import _thread
import datetime
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


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.


if __name__ == "__main__":
    url = "http://rpigelsight2.local:8080/?action=stream"
    stream = Streaming(url)
    _thread.start_new_thread(stream.load_stream, ())

    # dirname = "data/chessboard_2/"
    # dirname = "data/screw/"
    dirname = "data/chessboard_3rd/"
    os.makedirs(dirname, exist_ok=True)

    while True:
        img = stream.image

        if img is None:
            print("Waiting for streaming...")
            time.sleep(0.1)
            continue

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

        # show the image
        cv2.imshow("frame", cv2.resize(img, (0, 0), fx=0.25, fy=0.25))

        c = cv2.waitKey(1)
        if c == ord("s"):
            # Save
            fn = dirname + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
            cv2.imwrite(fn, img)
            pass
        elif c == ord("q"):
            break
