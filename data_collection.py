#! /usr/bin/env python
# -*- coding: utf-8

import time
import cv2
import numpy as np
from math import pi, sin, cos, asin, acos
import csv
from perception.wedge.gelsight.util.Vis3D import ClassVis3D

from perception.wedge.gelsight.gelsight_driver import GelSight
from control.gripper.gripper_control import Gripper_Controller
from control.ur5.ur_controller import UR_Controller
from control.mini_robot_arm.RX150_driver import RX150_Driver

import collections

import keyboard
from queue import Queue
from logger_class import Logger

urc = UR_Controller()
grc = Gripper_Controller()

urc.start()
grc.start()

# pose0 = np.array([-0.51, 0.376, 0.409, -1.416, -1.480, -1.031])
# pose0 = np.array([-0.539, 0.312, 0.29, -1.787, -1.604, -0.691])
pose0 = np.array([-0.463, -0.198, 0.189, -1.152, -1.234, 1.300])
grc.gripper_helper.set_gripper_current_limit(0.6)


rx150 = RX150_Driver(port="/dev/ttyACM1", baudrate=1000000)
rx150.torque(enable=1)
print(rx150.readpos())

def rx_move(g_open):
    values = [2048, 2549, 1110, 1400, 3072, g_open]
    x = 350
    y = 60
    end_angle = -20. / 180. * np.pi
    rx150.gogo(values, x, y, end_angle, 350, 60, end_angle, 3072, timestamp=30)

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


    grc.follow_gripper_pos = 0.7
    a = 0.15
    v = 0.08
    urc.movel_wait(pose0, a=a, v=v)
    rx_move(1200)
    c = input()

    rx_move(820)
    grc.follow_gripper_pos = 1
    time.sleep(0.5)

    dt = 0.05
    pos_x = 0.5

    tm_key = time.time()

    logger = Logger()
    flag_record = False
    tm = 0

    start_tm = time.time()

    vel = [0.00, 0.008, 0, 0, 0, 0]

    while True:
        img = gs.stream.image

        # get pose image
        pose_img = gs.pc.pose_img
        # pose_img = gs.pc.frame_large
        if pose_img is None:
            continue
        pose = gs.pc.pose
        cv2.imshow("pose", pose_img)

        if gs.pc.inContact:
            a = 0.02
            v = 0.02
            kp = .03

            pos_x = (pose[0] + (1 - pose[1])*np.tan(pose[2]))

            vel = [(pos_x-0.2)*kp, 0.008, -(pos_x-0.2)*kp*.2, 0, 0, 0]

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

        vel = np.array(vel)
        urc.speedl(vel*2, a=a, t=dt*2)

        time.sleep(dt)

        cnt += 1

        c = cv2.waitKey(1) & 0xFF
        if c == ord("q"):
            break

        ##################################################################
        # Record data
            # 'gelsight_url'  : self.gelsight_url,
            # 'fabric_pose'   : self.fabric_pose,
            # 'ur_velocity'   : self.ur_velocity,
            # 'ur_pose'       : self.ur_pose,
            # 'slip_index'    : self.slip_index,
            # 'x'             : self.x,
            # 'y'             : self.x,
            # 'theta'         : self.theta,
            # 'phi'           : self.phi,
            # 'dt'            : self.dt

            if gs.pc.inContact:
                logger.gelsight = gs.pc.diff
                logger.fabric_pose = pose
                logger.ur_velocity = urc.dp
                logger.ur_pose = urc.pose.copy()
                logger.slip_index = gs.tc.slip_index_realtime

                v = np.array([logger.ur_velocity[0], logger.ur_velocity[1]])
                alpha = asin(v[1] / np.sum(v**2)**0.5)

                logger.x = pose[0]
                logger.y = pose[1]
                logger.theta = pose[2]
                logger.phi = alpha - logger.theta

                logger.dt = time.time() - tm
                tm = time.time()

                logger.add()


            ##################################################################



if __name__ == "__main__":
    try:
        test_combined()
    finally:
        del gs
