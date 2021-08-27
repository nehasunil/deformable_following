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

import GPy
from load_data import loadall
import control as ctrl
import slycot

import keyboard
from queue import Queue
from logger_class import Logger

import collections


X, Y = loadall()

N = X.shape[0]
print(N)
idx = list(range(N))
random.seed(0)
random.shuffle(idx)

train_idx = idx[:int(N * 0.8)]
test_idx = idx[int(N * 0.8):]
X_train, Y_train = X[train_idx], Y[train_idx]

# kernel1 = GPy.kern.Linear(input_dim=4,ARD=True,initialize=False)
# m1 = GPy.models.SparseGPRegression(X_train, Y_train[:, 0].reshape(Y_train.shape[0], 1), kernel1, num_inducing=500, initialize=False)
# m1.update_model(False)
# m1.initialize_parameter()
# m1[:] = np.load('./controller/GP/m1_lin_80_500i.npy')
# m1.update_model(True)

kernel2 = GPy.kern.Exponential(input_dim=4, ARD=True, initialize=False)
m2 = GPy.models.SparseGPRegression(X_train, Y_train[:, 1].reshape(Y_train.shape[0], 1), kernel2, num_inducing=500, initialize=False)
m2.update_model(False)
m2.initialize_parameter()
m2[:] = np.load('./controller/GP/m2_exp_80_500i.npy')
m2.update_model(True)

kernel3 = GPy.kern.Exponential(input_dim=4, ARD=True, initialize=False)
m3 = GPy.models.SparseGPRegression(X_train, Y_train[:, 2].reshape(Y_train.shape[0], 1), kernel3, num_inducing=500, initialize=False)
m3.update_model(False)
m3.initialize_parameter()
m3[:] = np.load('./controller/GP/m3_exp_80_500i.npy')
m3.update_model(True)

# def tv_linA(x):
#     m = 3
#     model = [m1, m2, m3]
#     A = np.zeros((m, m))
#     for i in range(m):
#         grad = model[i].predictive_gradients(np.array([x]))
#         for j in range(m):
#             A[i][j] = grad[0][0][j]
#     return A
#
# def tv_linB(x):
#     m = 3
#     model = [m1, m2, m3]
#     B = np.zeros((m, 1))
#     for i in range(m):
#         grad = model[i].predictive_gradients(np.array([x]))
#         B[i, 0] = grad[0][0][3]
#     return B

def tv_linA(x):
    m = 3
    model = [m2, m2, m3]
    A = np.zeros((m, m))
    for i in range(1, m):
        grad = model[i].predictive_gradients(np.array([x]))
        for j in range(m):
            A[i][j] = grad[0][0][j]
    A[0][0] = 9.0954e-02
    A[0][1] = 4.2307e-06
    A[0][2] = 4.77888e-06
    return A

def tv_linB(x):
    m = 3
    model = [m2, m2, m3]
    B = np.zeros((m, 1))
    for i in range(1, m):
        grad = model[i].predictive_gradients(np.array([x]))
        B[i, 0] = grad[0][0][3]
    B[0][0] = -4.700e-07
    return B


urc = UR_Controller()
grc = Gripper_Controller()

urc.start()
grc.start()

# pose0 = np.array([-0.505-.1693, -0.219, 0.235, -1.129, -1.226, 1.326])
pose0 = np.array([-0.667, -0.196, 0.228, 1.146, -1.237, -1.227])
grc.gripper_helper.set_gripper_current_limit(0.6)


rx150 = RX150_Driver(port="/dev/ttyACM0", baudrate=1000000)
rx150.torque(enable=1)
print(rx150.readpos())

def rx_move(g_open):
    values = [2048, 2549, 1110, 1400, 3072, g_open]
    x = 320
    y = 90
    end_angle = -30. / 180. * np.pi
    rx150.gogo(values, x, y, end_angle, 3072, timestamp=30)

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
            fixpoint_y = pose0[1] - 0.133
            pixel_size = 0.2e-3
            ur_pose = urc.getl_rt()
            ur_xy = np.array(ur_pose[:2])

            x = 0.1 - pose[0] - 0.5 * (1 - 2*pose[1])*np.tan(pose[2])
            alpha = np.arctan(ur_xy[0] - fixpoint_x)/(ur_xy[1] - fixpoint_y) * np.cos(np.pi * 30 / 180)

            # print("x: ", x, "; input: ", x*pixel_size)
            # print("theta: ", pose[2] * 180/np.pi)


            state = np.array([[x*pixel_size],[pose[2]],[alpha]])
            # phi = -K.dot(state)

            Q = np.array([[100000.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.1]])
            R = [[0.1]]
            x0 = np.zeros(4)
            x0[0], x0[1], x0[2] = state[0, 0], state[1, 0], state[2, 0]
            A = tv_linA(x0)
            B = tv_linB(x0)
            try:
                K,S,E = ctrl.lqr(A, B, Q, R)
                print(K)
            except:
                print("LQR ERROR!!!!")
                K = np.array([862689, 42.704, 37.518])
            # K = np.array([862689, 42.704, 37.518])
            phi = -K.dot(state)
            print(phi)

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

            if ur_pose[0] < -0.7:
                vel[0] = max(vel[0], 0.)
                print("reached x limit")
            if ur_pose[0] > -0.3:
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
            urc.speedl(vel, a=a, t=dt*7)

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
