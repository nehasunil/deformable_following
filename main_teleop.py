#! /usr/bin/env python
# -*- coding: utf-8

import time
import cv2
import numpy as np
from math import pi, sin, cos, asin, acos
import csv
from perception.wedge.gelsight.util.Vis3D import ClassVis3D

from perception.wedge.gelsight.gelsight_driver import GelSight
from perception.teleop.apriltag_pose import AprilTagPose
from control.gripper.gripper_control import Gripper_Controller
from control.ur5.ur_controller import UR_Controller
from control.mini_robot_arm.RX150_driver import RX150_Driver
from scipy.spatial.transform import Rotation as R

import collections

urc = UR_Controller()
grc = Gripper_Controller()

urc.start()
grc.start()


# pose0 = np.array([-0.51, 0.376, 0.409, -1.416, -1.480, -1.031])
# pose0 = np.array([-0.539, 0.312, 0.29, -1.787, -1.604, -0.691])
pose0 = np.array([-0.520, 0, 0.235, -1.129, -1.226, 1.326])
grc.gripper_helper.set_gripper_current_limit(0.6)


rx150 = RX150_Driver(port="/dev/ttyACM0", baudrate=1000000)
rx150.torque(enable=1)
grc.follow_gripper_pos = 0.7
print(rx150.readpos())

def rx_move(g_open):
    values = [2048, 2549, 1110, 1400, 3072, g_open]
    x = 320
    y = 90
    end_angle = -30. / 180. * np.pi
    rx150.gogo(values, x, y, end_angle, 320, 90, end_angle, 3072, timestamp=30)

rx_move(820)
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


april_url = "http://rpigelsight2.local:8080/?action=stream"
april_tag_pose = AprilTagPose(april_url)
april_tag_pose.start()

def test_combined():

    grc.follow_gripper_pos = 0.5
    a = 0.15
    v = 0.08
    urc.movel_wait(pose0, a=a, v=v)

    depth_queue = []
    dt = 0.01

    cnt = 0
    flag_start = False
    pose_t0, pose_R0 = None, None
    pose_t_last, pose_R_last = None, None

    while True:
        img = gs.stream.image


        # get pose image
        pose_img = gs.pc.pose_img
        # pose_img = gs.pc.frame_large
        if pose_img is None:
            continue

        # waiting for apriltag tracking
        if april_tag_pose.img_undist is None:
            continue


        depth_current = gs.pc.depth.max()
        depth_queue.append(depth_current)

        if len(depth_queue) > 4:
            depth_queue = depth_queue[1:]

        if depth_current == np.max(depth_queue):
            pose = gs.pc.pose
            # cv2.imshow("pose", pose_img)

        # cv2.imshow("apriltag", april_tag_pose.img_undist)
        cv2.imshow("frame", cv2.resize(april_tag_pose.img_undist, (0, 0), fx=0.5, fy=0.5))
        c = cv2.waitKey(1)


        # if ur_pose[0] < -0.7:
        #     vel[0] = max(vel[0], 0.)
        # if ur_pose[0] > -0.3:
        #     vel[0] = min(vel[0], 0.)
        # if ur_pose[2] < .08:
        #     vel[2] = 0.
        # if ur_pose[1] > .3:
        #     vel[0] = min(vel[0], 0.)
        #     vel[1] = 0.


        if flag_start is True and april_tag_pose.pose is not None:
            pose_t, pose_R = april_tag_pose.pose

            if pose_t_last is not None:
                distance = np.sum((pose_t_last - pose_t)**2)**0.5
            else:
                distance = 0
            if distance > 50:
                break

            pose_t_last, pose_R_last = pose_t.copy(), pose_R.copy()

            if pose_t0 is None:
                pose_t0, pose_R0 = pose_t.copy(), pose_R.copy()

            pose_a_t = pose_t - pose_t0

            # apriltag to UR5
            alpha = 0.002
            uXa = alpha * np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]], dtype=np.float32)
            pose_u_t = (uXa @ pose_a_t).T[0] + pose0[:3]


            # move to goal pose
            vel = np.array([0., 0., 0., 0., 0., 0.])
            ur_pose = urc.getl_rt()

            kp = 2
            vel[:3] = kp*(pose_u_t - ur_pose[:3])

            urc.speedl(vel, a=a, t=dt*4)

            # r = R.from_matrix(pose_R)
            # print(r.as_euler('xyz', degrees=True))
            # print(pose_u_t)


        time.sleep(dt)
        cnt += 1

        c = cv2.waitKey(1) & 0xFF
        if c == ord("q"):
            break
        elif c == ord("s"):
            flag_start = True


if __name__ == "__main__":
    try:
        test_combined()
    finally:
        del gs
