#! /usr/bin/env python
# -*- coding: utf-8

import time
import cv2
import numpy as np
from math import pi, sin, cos, asin, acos
import csv
from perception.wedge.gelsight.util.Vis3D import ClassVis3D

from perception.wedge.gelsight.gelsight_driver import GelSight
from controller.gripper.gripper_control import Gripper_Controller
from controller.ur5.ur_controller import UR_Controller
from controller.mini_robot_arm.RX150_driver import RX150_Driver

import collections

urc = UR_Controller()
grc = Gripper_Controller()

urc.start()
grc.start()

# pose0 = np.array([-0.51, 0.376, 0.409, -1.416, -1.480, -1.031])
# pose0 = np.array([-0.539, 0.312, 0.29, -1.787, -1.604, -0.691])
pose0 = np.array([-0.505, -0.219, 0.235, -1.129, -1.226, 1.326])
grc.gripper_helper.set_gripper_current_limit(0.6)


rx150 = RX150_Driver(port="/dev/ttyACM0", baudrate=1000000)
rx150.torque(enable=1)
print(rx150.readpos())

def rx_move(g_open):
    values = [2048, 2549, 1110, 1400, 3072, g_open]
    x = 320
    y = 90
    end_angle = -30. / 180. * np.pi
    rx150.gogo(values, x, y, end_angle, 320, 90, end_angle, 3072, timestamp=30)

# rx_move(2000)

# sensor_id = "W03"
sensor_id = "Fabric0"

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


    # grc.follow_gripper_pos = 0.8
    # grc.follow_gripper_pos = 1
    a = 0.15
    v = 0.08
    urc.movel_wait(pose0, a=a, v=v)
    rx_move(1200)
    c = input()

    rx_move(810)
    grc.follow_gripper_pos = 1
    time.sleep(0.5)

    depth_queue = []

    cnt = 0
    dt = 0.05
    pos_x = 0.5
    # dt = 0.5
    # dy = 0.002
    # dz = 0.004
    # th = np.arctan(dy/dz)\

    vel = [0.00, 0.008, 0, 0, 0, 0]

    while True:
        img = gs.stream.image


        # get pose image
        pose_img = gs.pc.pose_img
        # pose_img = gs.pc.frame_large
        if pose_img is None:
            continue

        # depth_current = gs.pc.depth.max()
        # depth_queue.append(depth_current)
        #
        # if len(depth_queue) > 2:
        #     depth_queue = depth_queue[1:]
        #
        # if depth_current == np.max(depth_queue):
        pose = gs.pc.pose
        cv2.imshow("pose", pose_img)

        if gs.pc.inContact:

            # if cnt % 4 < 2:
            #     # grc.follow_gripper_pos = 1
            #     rx_move(810)
            # else:
            a = 0.02
            v = 0.02
            kp = .03
            # kp_rot = .2

            # pos_x = (2*pose[0] + (1 - pose[1])*np.tan(pose[2]))/2
            pos_x = (pose[0] + (1 - pose[1])*np.tan(pose[2]))
            # pos_x = pose[0]
            # e = (pos_x-0.5)*kp


            # vel = [0, (pos_x-0.3)*kp, -0.008, 0, 0, 0]
            # vel = [0, (pos_x-0.6)*kp, -0.008, kp_rot*gs.pc.pose[2], 0, 0]
            # vel = [0, e*np.cos(th) - dy, -e*np.sin(th) - dz, kp_rot*gs.pc.pose[2], 0, 0]
            vel = [(pos_x-0.2)*kp, 0.008, -(pos_x-0.2)*kp*.2, 0, 0, 0]
            vel = np.array(vel)

            # grc.follow_gripper_pos = .885
            # grc.follow_gripper_pos = .88
            # rx_move(830)
            # urc.speedl([(pose[0]-0.2)*kp, 0.008, 0, 0, 0, 0], a=a, t=dt*2)

            ur_pose = urc.getl_rt()
            if ur_pose[0] < -0.7:
                vel[0] = max(vel[0], 0.)
            if ur_pose[0] > -0.3:
                vel[0] = min(vel[0], 0.)
            if ur_pose[2] < .08:
                vel[2] = 0.
            if ur_pose[1] > .3:
                vel[0] = min(vel[0], 0.)
                vel[1] = 0.


            print("sliding vel ", vel[0], "posx ", pos_x)

            vel = np.array(vel)
            urc.speedl(vel, a=a, t=dt*2)

            time.sleep(dt)

        # # get tracking image
        # tracking_img = gs.tc.tracking_img
        # if tracking_img is None:
        #     continue


        # slip_index_realtime = gs.tc.slip_index_realtime
        # print("slip_index_realtime", slip_index_realtime)


        # cv2.imshow("marker", tracking_img[:, ::-1])
        # cv2.imshow("diff", gs.tc.diff_raw[:, ::-1] / 255)

        # if urc.getl_rt()[0] < -.45:
        #     break



        # cnt += 1

        c = cv2.waitKey(1) & 0xFF
        if c == ord("q"):
            break


if __name__ == "__main__":
    try:
        test_combined()
    finally:
        del gs
