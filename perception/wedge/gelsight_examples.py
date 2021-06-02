#! /usr/bin/env python
# -*- coding: utf-8

import time
import cv2
import numpy as np
from math import pi, sin, cos, asin, acos
import csv
from gelsight.util.Vis3D import ClassVis3D

from gelsight.gelsight_driver import GelSight

# sensor_id = "W03"
sensor_id = "fabric_0"

IP = "http://rpigelsightfabric.local"

#                   N   M  fps x0  y0  dx  dy
tracking_setting = (10, 14, 5, 16, 41, 27, 27)


n, m = 150, 200
# Vis3D = ClassVis3D(m, n)


def read_csv(filename=f"config_{sensor_id}.csv"):
    rows = []

    with open(filename, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        for row in csvreader:
            rows.append((int(row[1]), int(row[2])))

    return rows


corners = tuple(read_csv())
# corners=((252, 137), (429, 135), (197, 374), (500, 380))


gs = GelSight(
    IP=IP,
    corners=corners,
    tracking_setting=tracking_setting,
    output_sz=(400, 300),
    id="right",
)
gs.start()


def test_combined():

    while True:
        img = gs.stream.image

        # get pose image
        pose_img = gs.pc.pose_img
        # pose_img = gs.pc.frame_large
        if pose_img is None:
            continue

        # get tracking image
        tracking_img = gs.tc.tracking_img
        if tracking_img is None:
            continue

        pose = gs.pc.pose
        # if pose is not None:
        #     v_max, v_min, w_max, w_min = pose
        #     theta = acos(v_max[0] / (np.sum(v_max ** 2) ** 0.5))
        #     print(theta / pi * 180)

        # slip_index_realtime = gs.tc.slip_index_realtime
        # print("slip_index_realtime", slip_index_realtime)


        cv2.imshow("pose", pose_img)
        cv2.imshow("marker", tracking_img)
        cv2.imshow("mask", gs.tc.mask*1.0)
        cv2.imshow("diff", gs.tc.diff_raw / 255)

        c = cv2.waitKey(1) & 0xFF
        if c == ord("q"):
            break


if __name__ == "__main__":
    try:
        test_combined()
    finally:
        del gs
