#! /usr/bin/env python
# -*- coding: utf-8

import time
import cv2
import numpy as np
from math import pi, sin, cos, asin, acos
import csv
from gelsight.util.Vis3D import ClassVis3D

from gelsight.gelsight_driver import GelSight

IP = "http://rpigelsight.local"

n, m = 150, 200
Vis3D = ClassVis3D(m, n)

#                   N   M  fps x0  y0  dx  dy  
tracking_setting = (10, 14, 5, 16, 41, 27, 27)

def read_csv(filename="config.csv"):
    rows = [] 

    with open(filename, 'r') as csvfile: 
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        for row in csvreader: 
            rows.append((int(row[1]), int(row[2])))

    return rows

corners = tuple(read_csv())
# corners=((252, 137), (429, 135), (197, 374), (500, 380))


gs = GelSight(IP=IP, corners=corners, tracking_setting=None, output_sz=(400, 300), id='right')
gs.start()

def test_combined():

    while True:
        img = gs.stream.image

        # get pose image
        pose_img = gs.pc.pose_img

        if pose_img is None: continue

        pose = gs.pc.pose
        if pose is not None:
            v_max, v_min, w_max, w_min = pose
            theta = acos(v_max[0]/(np.sum(v_max**2)**0.5))
            print(theta/pi*180)

        depth = gs.pc.depth / 255.
        Vis3D.update(depth / 2)

        cv2.imshow('pose', pose_img)
        
        # depth[depth<0] = 0
        # cv2.imshow('depth', depth*8)

        c = cv2.waitKey(1) & 0xFF
        if c == ord('q'):
            break

if __name__ == "__main__":
    try:
        test_combined()
    finally:
        del gs
