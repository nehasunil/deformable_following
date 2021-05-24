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

import GPy
<<<<<<< HEAD
import control as ctrl
import slycot
=======
>>>>>>> 54e5c9699fa13c5a5cf18d19ef43c1ffe9fcf782
from load_data import loadall
from typing import Tuple
import torch
from torch import Tensor
from constrained_cem_mpc import ConstrainedCemMpc, ActionConstraint, box2torchpoly, TerminalConstraint, StateConstraint, \
    DynamicsFunc

urc = UR_Controller()
grc = Gripper_Controller()

X, Y = loadall()

N = X.shape[0]
print(N)
idx = list(range(N))
random.seed(0)
random.shuffle(idx)

<<<<<<< HEAD
train_idx = idx[:int(N * 0.4)]
test_idx = idx[int(N * 0.4):]
=======
train_idx = idx[:int(N * 0.5)]
test_idx = idx[int(N * 0.5):]
>>>>>>> 54e5c9699fa13c5a5cf18d19ef43c1ffe9fcf782

X_train, Y_train = X[train_idx], Y[train_idx]

kernel1 = GPy.kern.Matern32(input_dim=4,ARD=True,initialize=False)
<<<<<<< HEAD
m1 = GPy.models.SparseGPRegression(X_train, Y_train[:, 0].reshape(Y_train.shape[0], 1), kernel1, num_inducing=400, initialize=False)
m1.update_model(False)
m1.initialize_parameter()
m1[:] = np.load('./controller/GP/m1_m32_a_400i_40.npy')
=======
m1 = GPy.models.SparseGPRegression(X_train, Y_train[:, 0].reshape(Y_train.shape[0], 1), kernel1, num_inducing=1000, initialize=False)
m1.update_model(False)
m1.initialize_parameter()
m1[:] = np.load('./controller/GP/m1_m32_a_1000i_50.npy')
>>>>>>> 54e5c9699fa13c5a5cf18d19ef43c1ffe9fcf782
m1.update_model(True)
# m.initialize_parameter()
# mu,var = m.predict(X_test)

kernel2 = GPy.kern.Exponential(input_dim=4, ARD=True, initialize=False)
<<<<<<< HEAD
m2 = GPy.models.SparseGPRegression(X_train, Y_train[:, 1].reshape(Y_train.shape[0], 1), kernel2, num_inducing=400, initialize=False)
m2.update_model(False)
m2.initialize_parameter()
m2[:] = np.load('./controller/GP/m2_exp_a_400i_40.npy')
m2.update_model(True)

kernel3 = GPy.kern.Exponential(input_dim=4, ARD=True, initialize=False)
m3 = GPy.models.SparseGPRegression(X_train, Y_train[:, 2].reshape(Y_train.shape[0], 1), kernel3, num_inducing=400, initialize=False)
m3.update_model(False)
m3.initialize_parameter()
m3[:] = np.load('./controller/GP/m3_exp_a_400i_40.npy')
m3.update_model(True)

def tv_linA(x):
    m = 3
    model = [m1, m2, m3]
    A = np.zeros((m, m))
    for i in range(m):
        grad = model[i].predictive_gradients(np.array([x]))
        for j in range(m):
            A[i][j] = grad[0][0][j]
    return A

def tv_linB(x):
    m = 3
    model = [m1, m2, m3]
    B = np.zeros((m, 1))
    for i in range(m):
        grad = model[i].predictive_gradients(np.array([x]))
        B[i, 0] = grad[0][0][3]
    return B

=======
m2 = GPy.models.SparseGPRegression(X_train, Y_train[:, 1].reshape(Y_train.shape[0], 1), kernel2, num_inducing=1000, initialize=False)
m2.update_model(False)
m2.initialize_parameter()
m2[:] = np.load('./controller/GP/m2_exp_a_1000i_50.npy')
m2.update_model(True)

kernel3 = GPy.kern.Exponential(input_dim=4, ARD=True, initialize=False)
m3 = GPy.models.SparseGPRegression(X_train, Y_train[:, 2].reshape(Y_train.shape[0], 1), kernel3, num_inducing=1000, initialize=False)
m3.update_model(False)
m3.initialize_parameter()
m3[:] = np.load('./controller/GP/m3_exp_a_1000i_50.npy')
m3.update_model(True)

>>>>>>> 54e5c9699fa13c5a5cf18d19ef43c1ffe9fcf782
class Dynamics(DynamicsFunc):
    """
    From GP Dynamics model
    """

    def __call__(self, states: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
<<<<<<< HEAD
        dt = .29
=======
        dt = .05
>>>>>>> 54e5c9699fa13c5a5cf18d19ef43c1ffe9fcf782
        x = states
        x_np = states.numpy()
        n = x_np.shape[0]
        x_in = np.array([x_np[:, 0].reshape(n,), x_np[:,1].reshape(n,), x_np[:, 2].reshape(n,), actions.numpy().reshape(n,)]).T
<<<<<<< HEAD
        pred_xdot = m1.predict(x_in)
        xdot =  torch.tensor(np.array([pred_xdot[0], m2.predict(x_in)[0], m3.predict(x_in)[0]]).reshape(n, 3))
        newx = x + xdot * dt

        # objective_cost = torch.zeros_like(x[:, 0])
        objective_cost = torch.tensor(np.array([10.0*(x_np[i,0]/.013)**2 + 1.0*(x_np[i,1]/(pi/3))**2 + 0.1*(x_np[i,2]/(pi/2))**2 + 0.1*(actions.numpy()[i]/(pi/3))**2 + 100*pred_xdot[1][i][0]**2 for i in range(n)]).reshape(n, ))

=======
        xdot =  torch.tensor(np.array([m1.predict(x_in)[0], m2.predict(x_in)[0], m3.predict(x_in)[0]]).T.reshape(n, 3))
        newx = x + xdot * dt

        objective_cost = torch.zeros_like(x[:, 0])
>>>>>>> 54e5c9699fa13c5a5cf18d19ef43c1ffe9fcf782

        return newx, objective_cost


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

    torch.set_default_dtype(torch.double)
    # constraints = [ActionConstraint(box2torchpoly([[-pi / 3, pi / 3]])), TerminalConstraint(box2torchpoly([[-0.002, 0.002], [-pi/5, pi/5], [-pi/4, pi/4]])),  #
    #    StateConstraint(box2torchpoly([[-0.02, 0.015], [-pi / 2, pi / 2], [-pi / 2, pi/2]]))]
<<<<<<< HEAD
    constraints = [ActionConstraint(box2torchpoly([[-pi / 3, pi / 3]])),   #
       StateConstraint(box2torchpoly([[-0.015, 0.001], [-pi / 3, pi / 3], [-pi / 3, pi/3]]))]
    mpc = ConstrainedCemMpc(dynamics_func=Dynamics(), constraints=constraints, state_dimen=3, action_dimen=1,
                time_horizon=5, num_rollouts=30, num_elites=5, num_iterations=4)
=======
    constraints = [ActionConstraint(box2torchpoly([[-pi / 3, pi / 3]])), TerminalConstraint(box2torchpoly([[-0.0025, 0.0025], [-pi/4, pi/4], [-pi/4, pi/4]])),  #
       StateConstraint(box2torchpoly([[-0.02, 0.015], [-pi / 2, pi / 2], [-pi / 2, pi/2]]))]
    mpc = ConstrainedCemMpc(dynamics_func=Dynamics(), constraints=constraints, state_dimen=3, action_dimen=1,
                time_horizon=30, num_rollouts=100, num_elites=10, num_iterations=10)
>>>>>>> 54e5c9699fa13c5a5cf18d19ef43c1ffe9fcf782
# -0.017, 0.01
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

<<<<<<< HEAD
                K = np.array([-372.25, 8.62, -1.984]) # linear regression
=======
                # K = np.array([-372.25, 8.62, -1.984]) # linear regression
>>>>>>> 54e5c9699fa13c5a5cf18d19ef43c1ffe9fcf782
                # K = np.array([-923.3, 22.1, -19.65]) # GP regression linearized about origin
                # state = np.array([[cable_xy[0]*pixel_size], [theta], [alpha]])
                state = torch.tensor([cable_xy[0]*pixel_size, theta, alpha], dtype=torch.double)
                actions, _ = mpc.get_actions(state)

                # Sometimes the optimisation process may fail to find a safe action sequence, in which case we do nothing.
                if actions is None:
<<<<<<< HEAD
                    # phi = torch.tensor([0])
                    # phi = -K.dot(state)
                    st = np.array([[cable_xy[0]*pixel_size], [theta], [alpha]])
                    # Q = np.array([[10000.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.1]])
                    # R = [[0.1]]
                    # x0 = np.zeros(4)
                    # x0[0], x0[1], x0[2] = st[0, 0], st[1, 0], st[2, 0]
                    # # print(x0)
                    # A = tv_linA(x0)
                    # B = tv_linB(x0)
                    # # A = tv_linA([0, 0, 0, 0])
                    # # B = tv_linB([0, 0, 0, 0])
                    # K,S,E = ctrl.lqr(A, B, Q, R)
                    K = np.array([-762.47, 1.791, 4.681])
                    phi = -K.dot(st)
                    print('taking default action: ', phi*180/pi)
                else:
                    phi = actions[0][0].numpy()
                    print('taking mpc action: ', phi*180/pi)
                    cnt += 1
=======
                    phi = torch.tensor([0])
                    print('taking default action: ', phi*180/pi)
                else:
                    phi = actions[0].numpy()
                    print('taking mpc action: ', phi*180/pi)
>>>>>>> 54e5c9699fa13c5a5cf18d19ef43c1ffe9fcf782

                # phi = -K.dot(state)
                target_ur_dir = phi + alpha
                print("STATE", state)
                print("TARGET UR DIR", target_ur_dir/pi*180)
                limit_phi = pi / 3
                target_ur_dir = max(-limit_phi, min(target_ur_dir, limit_phi))
<<<<<<< HEAD
                v_norm = 0.02
                vel = np.array([v_norm * sin(target_ur_dir), v_norm * cos(target_ur_dir), 0, 0, 0, 0])
                # if grc.follow_gripper_pos > 0.965:
                #     grc.follow_gripper_pos -= 0.001
                x_np = state.numpy()
                x_in = np.array([[x_np[0], x_np[1], x_np[2], phi]])
                # print("x: ", x_np.shape)
                xdot =  np.array([m1.predict(x_in)[0], m2.predict(x_in)[0], m3.predict(x_in)[0]]).reshape(3, )
                # print(xdot.shape)
                newx = x_np + xdot * dt*12
                print("PREDICTED STATE", newx)
=======
                v_norm = 0.01
                vel = np.array([v_norm * sin(target_ur_dir), v_norm * cos(target_ur_dir), 0, 0, 0, 0])
                # if grc.follow_gripper_pos > 0.965:
                #     grc.follow_gripper_pos -= 0.001
>>>>>>> 54e5c9699fa13c5a5cf18d19ef43c1ffe9fcf782

            else:
                gs.pc.inContact = False
                print("no pose estimate")
<<<<<<< HEAD
                print("distance followed: ", ((cable_real_xy[0] - fixpoint_x)**2 + (cable_real_xy[1] - fixpoint_y)**2)**0.5)
                # grc.follow_gripper_pos += 0.002
                print("log saved: ", logger.save_logs())
                print("number of MPC actions: ", cnt)
                continue

            a = 0.02
=======
                # grc.follow_gripper_pos += 0.002
                print("log saved: ", logger.save_logs())
                continue

            a = 0.02
            v = 0.02
>>>>>>> 54e5c9699fa13c5a5cf18d19ef43c1ffe9fcf782
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
            if ur_pose[0] > -0.3:
                vel[0] = min(vel[0], 0.)
            if ur_pose[2] < .08:
                vel[2] = 0.
            if ur_pose[1] > .34:
                print("end of workspace")
<<<<<<< HEAD
                print("number of MPC actions: ", cnt)
=======
>>>>>>> 54e5c9699fa13c5a5cf18d19ef43c1ffe9fcf782
                print("log saved: ", logger.save_logs())
                gs.pc.inContact = False
                vel[0] = min(vel[0], 0.)
                vel[1] = 0.

            vel = np.array(vel)
<<<<<<< HEAD
            urc.speedl(vel/2, a=a, t=dt*15)
            print(vel)

            # time.sleep(dt)
=======
            urc.speedl(vel, a=a, t=dt*2)
            print(vel)

            time.sleep(dt)
>>>>>>> 54e5c9699fa13c5a5cf18d19ef43c1ffe9fcf782
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
<<<<<<< HEAD
            print("dt: ", logger.dt)
=======
>>>>>>> 54e5c9699fa13c5a5cf18d19ef43c1ffe9fcf782
            tm = time.time()

            logger.add()
        ##################################################################


if __name__ == "__main__":
    try:
        test_combined()
    finally:
        del gs
