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

pose0 = np.array([-0.505-.1693, -0.219, 0.235, -1.129, -1.226, 1.326])
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

def rx_regrasp():
    rx_move(890)
    time.sleep(0.5)
    values = [2048, 2549, 1110, 1400, 3072, 890]
    x = 335
    y = 95
    end_angle = -30. / 180. * np.pi
    rx150.gogo(values, x, y, end_angle, x, y, end_angle, 3072, timestamp=30)
    time.sleep(0.5)
    values[-1] = 760
    rx150.gogo(values, x, y, end_angle, x, y, end_angle, 3072, timestamp=30)
    time.sleep(0.5)
    x = 320
    y = 90
    rx150.gogo(values, x, y, end_angle, x, y, end_angle, 3072, timestamp=30)
    time.sleep(0.5)


# rx_move(2000)

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





def test_combined():

    a = 0.15
    v = 0.08
    urc.movel_wait(pose0, a=a, v=v)
    rx_move(1200)
    time.sleep(2)

    gs = GelSight(
        IP=IP,
        corners=corners,
        tracking_setting=tracking_setting,
        output_sz=(400, 300),
        id="right",
    )
    gs.start()
    c = input()

    rx_move(760)
    grc.follow_gripper_pos = 1
    time.sleep(0.5)



    depth_queue = []

    cnt = 0
    dt = 0.05
    pos_x = 0.5

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

        if gs.pc.inContact:

            # if cnt % 4 < 2:
            #     # grc.follow_gripper_pos = 1
            #     rx_move(810)
            # else:
            a = 0.02
            v = 0.02

            fixpoint_x = pose0[0]
            fixpoint_y = pose0[1] - 0.08
            pixel_size = 0.2e-3
            ur_pose = urc.getl_rt()
            ur_xy = np.array(ur_pose[:2])

            x = 0.05 - pose[0] + 0.5 * (1 - 2*pose[1])*np.tan(pose[2])
            alpha = np.arctan(ur_xy[0] - fixpoint_x)/(ur_xy[1] - fixpoint_y) * np.cos(np.pi * 30 / 180)

            print("x: ", x, "; input: ", x*pixel_size)

            # K = np.array([6528.5, 0.79235, 2.18017]) #10 degrees
            # K = np.array([7012, 8.865, 6.435]) #30 degrees
            K = np.array([1383, 3.682, 3.417])

            state = np.array([[x*pixel_size],[pose[2]],[alpha]])
            phi = -K.dot(state)

            # noise = random.random() * 0.07 - 0.02
            # a = 0.8
            # noise_acc = a * noise_acc + (1 - a) * noise
            # phi += noise_acc

            target_ur_dir = phi + alpha
            limit_phi = np.pi/3
            target_ur_dir = max(-limit_phi, min(target_ur_dir, limit_phi))
            if abs(target_ur_dir) == limit_phi:
                print("reached phi limit")
            v_norm = 0.01
            vel = np.array([v_norm * sin(target_ur_dir)*cos(np.pi * 30 / 180), v_norm * cos(target_ur_dir), v_norm * sin(target_ur_dir)*sin(np.pi * -30 / 180), 0, 0, 0])

            # if x < -0.2:
            #     print("regrasp")
            #     rx_regrasp()

            if ur_pose[0] < -0.7-.1693:
                vel[0] = max(vel[0], 0.)
                print("reached x limit")
            if ur_pose[0] > -0.3-.1693:
                vel[0] = min(vel[0], 0.)
                print("reached x limit")
            if ur_pose[2] < .08:
                vel[2] = 0.
                print("reached z limit")
            if ur_pose[1] > .34:
                print("end of workspace")
                print("log saved: ", logger.save_logs())
                gs.pc.inContact = False
                vel[0] = min(vel[0], 0.)
                vel[1] = 0.


            # print("sliding vel ", vel[0], "posx ", pos_x)

            vel = np.array(vel)
            urc.speedl(vel, a=a, t=dt*2)

            time.sleep(dt)

        else:
            print("no pose estimate")
            print("log saved: ", logger.save_logs())
            break

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
            # print("LOGGING")
            # logger.gelsight = gs.pc.diff
            # logger.cable_pose = pose
            logger.ur_velocity = vel
            logger.ur_pose = urc.getl_rt()

            v = np.array([logger.ur_velocity[0], logger.ur_velocity[1]])
            alpha = asin(v[1] / np.sum(v ** 2) ** 0.5)

            logger.x = pose[0]
            logger.y = pose[1]
            logger.theta = pose[2]
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
