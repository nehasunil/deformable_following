import csv
import datetime
import time
import urllib
import urllib.request

import _thread
import cv2
import numpy as np
import util
from util.processing import warp_perspective
from util.reconstruction import Class_3D
from util.streaming import Streaming
from util.Vis3D import ClassVis3D

import deepdish as dd

sensor_id = "Fabric0"

stream = urllib.request.urlopen("http://rpigelsightfabric.local:8080/?action=stream")

stream = Streaming(stream)
_thread.start_new_thread(stream.load_stream, ())

# n, m = 300, 400
n, m = 150, 200
# Vis3D = ClassVis3D(m, n)

# fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
# col = 400
# row = 300
# out = cv2.VideoWriter(
#     "data/fabric.mov", fourcc, 30.0, (col * 1, row * 1)
# )  # The fps depends on CPU


def read_csv(filename=f"config/config_{sensor_id}.csv"):
    rows = []

    with open(filename, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        for row in csvreader:
            rows.append((int(row[1]), int(row[2])))

    return rows


TOPLEFT, TOPRIGHT, BOTTOMLEFT, BOTTOMRIGHT = tuple(read_csv())

cnt = 0
# n, m = 48, 48
# n, m = 300, 300
diff_thresh = 5

diff_max = 0
image_peak = None
flag_recording = False
dirname = "cali_data_1/"

last_frame = None

while True:
    frame = stream.image
    if frame == "":
        print("waiting for streaming")
        continue

    cnt += 1

    im = warp_perspective(frame, TOPLEFT, TOPRIGHT, BOTTOMLEFT, BOTTOMRIGHT)  # left
    # im[:, :, 0] = 0.0

    im_raw = cv2.resize(im, (400, 300))
    # im_raw = cv2.resize(im, (200, 150))
    im = cv2.resize(im, (m, n))

    if cnt == 1:
        frame0 = im.copy()

    cv2.imshow("im_raw", im_raw)

    diff = ((im * 1.0 - frame0)) / 255.0 + 0.5
    cv2.imshow("warped", cv2.resize((diff - 0.5) * 4 + 0.5, (m, n)))


    # if last_frame is not None and np.sum(last_frame - im_raw) != 0:
    #     print("new frame")
    # last_frame = im_raw.copy()

    c = cv2.waitKey(1)
    if c == ord("q"):
        break

cv2.destroyAllWindows()
