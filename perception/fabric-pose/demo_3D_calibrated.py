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

sensor_id = "W00"

stream = urllib.request.urlopen("http://rpigelsight.local:8080/?action=stream")

stream = Streaming(stream)
_thread.start_new_thread(stream.load_stream, ())

# n, m = 300, 400
n, m = 150, 200
Vis3D = ClassVis3D(m, n)


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
last_tm = 0
# n, m = 48, 48
# n, m = 300, 300
diff_thresh = 5

diff_max = 0
image_peak = None
flag_recording = False
dirname = "cali_data_1/"

model_id = "RGB"
model_fn = f"models/LUT_{sensor_id}_{model_id}.pkl"
c3d = Class_3D(model_fn=model_fn, features_type=model_id)

while True:
    frame = stream.image
    if frame == "":
        print("waiting for streaming")
        continue

    cnt += 1

    im = warp_perspective(frame, TOPLEFT, TOPRIGHT, BOTTOMLEFT, BOTTOMRIGHT)  # left
    # im[:, :, 0] = 0.0

    im_raw = cv2.resize(im, (400, 300))
    im = cv2.resize(im, (m, n))
    # w = im.shape[0]
    # h = im.shape[1]
    # print("W", w, "H", h)
    # im = im[w//2-n:w//2+n, h//2-m:h//2+m]
    # im = im[50:-50:,100:-100]
    # im = cv2.resize(im, (n, m))

    if cnt == 1:
        frame0 = im.copy()

    diff = ((im * 1.0 - frame0)) / 255.0 + 0.5
    cv2.imshow("warped", cv2.resize((diff - 0.5) * 4 + 0.5, (m, n)))

    depth, gx, gy = c3d.infer(diff * 255.0)
    Vis3D.update(depth[::-1])

    depth = depth - 0.2
    depth[depth < 0] = 0
    cv2.imshow("depth", depth / 5)
    print("total {:.4f} s".format(time.time() - last_tm))
    last_tm = time.time()

    cv2.imshow("im_raw", im_raw)

    c = cv2.waitKey(1)
    if c == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
