#! /usr/bin/env python
# -*- coding: utf-8

import time
import cv2
import numpy as np
from math import pi, sin, cos, asin, acos
import csv
import random
from perception.wedge.gelsight.util.Vis3D import ClassVis3D

from perception.wedge.gelsight.gelsight_driver import GelSight
from controller.gripper.gripper_control import Gripper_Controller
from controller.ur5.ur_controller import UR_Controller
from controller.mini_robot_arm.RX150_driver import RX150_Driver

import keyboard
from queue import Queue
from logger_class import Logger

import collections

urc = UR_Controller()
grc = Gripper_Controller()

urc.start()
grc.start()

# pose0 = np.array([-0.51, 0.376, 0.409, -1.416, -1.480, -1.031])
# pose0 = np.array([-0.539, 0.312, 0.29, -1.787, -1.604, -0.691])
# pose0 = np.array([-0.520, -0.219, 0.235, -1.129, -1.226, 1.326])
# pose0 = np.array([-0.382, -0.246, 0.372, -1.129, -1.226, 1.326]) # vertical
pose0 = np.array([-0.539, -0.226, 0.092, -1.129, -1.226, 1.326]) # downward
pose_prep = np.array([-0.435, -0.187, 0.155, -1.609, -1.661, 0.995])
grc.gripper_helper.set_gripper_current_limit(0.6)


rx150 = RX150_Driver(port="/dev/ttyACM0", baudrate=1000000)
rx150.torque(enable=1)
print(rx150.readpos())

def rx_move(g_open):
    values = [2048, 2549, 1110, 1400, 3072, g_open]
    x = 360
    y = 30
    end_angle = 85. / 180. * np.pi
    rx150.gogo(values, x, y, end_angle, 360, 30, end_angle, 3072, timestamp=300)

# rx_move(2000)

# sensor_id = "W03"
# sensor_id = "Fabric0"
sensor_id = "cable_0"

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


grc.follow_gripper_pos = 0.7
time.sleep(0.5)

gs = GelSight(
    IP=IP,
    corners=corners,
    tracking_setting=tracking_setting,
    output_sz=(400, 300),
    id="right",
)
gs.start()


def test_combined():

    # grc.follow_gripper_pos = 1
    a = 0.15
    v = 0.08
    # urc.movel_wait(pose_prep, a=a, v=v)
    # time.sleep(0.5)
    urc.movel_wait(pose0, a=a, v=v)
    rx_move(790)
    c = input()

    rx_move(790)
    grc.follow_gripper_pos = 0.965
    time.sleep(0.5)

    depth_queue = []

    cnt = 0
    dt = 0.05

    tm_key = time.time()
    logger = Logger()
    noise_acc = 0.
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

        # depth_current = gs.pc.depth.max()
        # depth_queue.append(depth_current)
        #
        # if len(depth_queue) > 2:
        #     depth_queue = depth_queue[1:]
        #
        # if depth_current == np.max(depth_queue):
        pose = gs.pc.pose
        cv2.imshow("pose", pose_img)
        cv2.waitKey(1)


        if gs.pc.inContact:

            if pose is not None:
                v_max, v_min, w_max, w_min, m = pose
                # theta = acos(v_max[0]/(np.sum(v_max**2)**0.5))
                theta = asin(-v_max[0]/(np.sum(v_max**2)**0.5))
                if theta > pi / 2:
                    theta -= pi
                if theta > pi / 3 or theta < -pi / 3 or w_max < w_min * 1.5:
                    theta = 0.
                cable_xy = -gs.pc.mv_relative
                # print("xy: ", cable_xy, "theta: ", theta *180/np.pi)

                fixpoint_x = pose0[0] + 0.006
                fixpoint_y = pose0[1] - 0.039
                # pixel_size = 0.2e-3
                pixel_size = 0.2e-3
                ur_pose = urc.getl_rt()
                ur_xy = ur_pose[:2]
                cable_real_xy = np.array(ur_xy) + np.array([0., -0.039]) + cable_xy*pixel_size
                alpha = np.arctan((cable_real_xy[0] - fixpoint_x)/(cable_real_xy[1] - fixpoint_y))

                # K = np.array([-778.8, 1.8188, 4.656]) # linear regression Q = 1, 1, 0.1
                # K = np.array([-775.1, 1.7560, 4.644]) # linear regression Q = 1, 0.8, 0.1
                # K = np.array([-769.4, 1.659, 4.626]) # linear regression Q = 1, 0.8, 0.1
                # K = np.array([-180.57, 10.44, 1.758]) # GP linearized at origin Q = 1, 0.8, 0.1, 200 inducing
                K = np.array([-7.946, -0.0625, 0.3853])
                state = np.array([[cable_xy[0]*pixel_size], [theta], [alpha]])
                phi = -K.dot(state)
                target_ur_dir = phi + alpha
                print("STATE", state)
                print("TARGET UR DIR", target_ur_dir/pi*180)
                limit_phi = pi / 3
                target_ur_dir = max(-limit_phi, min(target_ur_dir, limit_phi))
                v_norm = 0.04
                vel = np.array([v_norm * sin(target_ur_dir), v_norm * cos(target_ur_dir), 0, 0, 0, 0])
                # if grc.follow_gripper_pos > 0.965:
                #     grc.follow_gripper_pos -= 0.001

            else:
                gs.pc.inContact = False
                print("no pose estimate")
                print("distance followed: ", ((cable_real_xy[0] - fixpoint_x)**2 + (cable_real_xy[1] - fixpoint_y)**2)**0.5)
                # grc.follow_gripper_pos += 0.002
                # print("log saved: ", logger.save_logs())
                continue

            a = 0.02
            v = 0.02
            kp = .0002

            # noise = random.random() * 0.03 - 0.015
            # a = 0.8
            # noise_acc = a * noise_acc + (1-a) * noise
            # vel = [cable_xy[0]*kp+noise_acc, 0.01, 0, 0, 0, 0]
            # vel = np.array(vel)

            # Workspace Bounds
            # ur_pose = urc.getl_rt()
            if vel[1] < 0:
                print("going the wrong way!")
                vel[1] = max(vel[1], 0.)
            if ur_pose[0] < -0.7:
                vel[0] = max(vel[0], 0.)
                print("hit workspace limit")
            if ur_pose[0] > -0.3:
                vel[0] = min(vel[0], 0.)
                print("hit workspace limit")
            if ur_pose[2] < .08:
                vel[2] = 0.
                print("hit workspace limit")
            if ur_pose[1] > .45:
                print("end of workspace")
                # print("log saved: ", logger.save_logs())
                gs.pc.inContact = False
                vel[0] = min(vel[0], 0.)
                vel[1] = 0.

            # vel = np.array(vel)
            urc.speedl(vel, a=a, t=dt*2)
            print(vel)

            time.sleep(dt)
            # cnt += 1

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
            # logger.cable_pose = pose
            logger.ur_velocity = vel
            logger.ur_pose = urc.getl_rt()

            v = np.array([logger.ur_velocity[0], logger.ur_velocity[1]])
            alpha = asin(v[1] / np.sum(v**2)**0.5)

            logger.x = cable_xy
            logger.theta = theta
            # logger.phi = alpha - logger.theta

            logger.dt = time.time() - tm
            tm = time.time()

            logger.add()
        ##################################################################


if __name__ == "__main__":
    try:
        test_combined()
    finally:
        del gs
