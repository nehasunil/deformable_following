
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
from controller.ur5.ur_controller import UR_Controller
from controller.mini_robot_arm.RX150_driver import RX150_Driver
from perception.kinect.kinect_camera import Kinect
from scipy.spatial.transform import Rotation as R
# from controller.gripper.gripper_control import Gripper_Controller
from controller.gripper.gripper_control_v2 import Gripper_Controller_V2
from scipy.interpolate import griddata
from perception.fabric.util.segmentation import segmentation
from perception.fabric.run import Run
from PIL import Image
import skimage.measure
import torchvision.transforms as T
import os
from perception.kinect.Vis3D import ClassVis3D
from perception.fabric.grasp_selector import select_grasp


# cam_intrinsics_origin = np.array([
# [917.3927285,    0.,         957.21294894],
# [  0.,         918.96234057, 555.32910487],
# [  0.,           0.,           1.        ]])
# cam_intrinsics = np.array([
# [965.24853516,   0.,         950.50838964],
#  [  0.,         939.67144775, 554.55567298],
#  [  0.,           0.,           1.        ]]) # New Camera Intrinsic Matrix
# dist = np.array([[ 0.0990126,  -0.10306044,  0.00024658, -0.00268176,  0.05763196]])
# camera = Kinect(cam_intrinsics_origin, dist)
#
# cam_pose = np.loadtxt('real/camera_pose.txt', delimiter=' ')
# cam_depth_scale = np.loadtxt('real/camera_depth_scale.txt', delimiter=' ')
#
# # User options (change me)
# # --------------- Setup options ---------------
# urc = UR_Controller()
# urc.start()
# a = 0.2
# v = 0.2
#
rx150 = RX150_Driver(port="/dev/ttyACM0", baudrate=1000000)
rx150.torque(enable=1)
print(rx150.readpos())

def rx_move(g_open, x_pos, end_angle=0, timestamp=100):
    values = [1024, 2549, 1110, 1400, 0, g_open]
    x = x_pos
    y = 120
    end_angle = end_angle
    rx150.gogo(values, x, y, end_angle, 0, timestamp=timestamp)

# ---------------------------------------------
# initialize rx150
# rx_move(1600, 270, timestamp=200)
# rx_move(760, 270, timestamp=200)
rx_move(760, 370, end_angle=0.8, timestamp=200)
# rx_move(1600, 370, end_angle=0.8, timestamp=200)
# for i in range(100):
#     print(rx150.readpos())
#     time.sleep(0.01)
# time.sleep(0.5)
# # ---------------------------------------------
#
# # workspace_limits = np.asarray([[0.3, 0.748], [-0.224, 0.224], [-0.255, -0.1]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
# workspace_limits = np.asarray([[-0.845, -0.605], [-0.14, 0.2], [0, 0.2]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
#
# # tool_orientation = [2.22,-2.22,0]
# # tool_orientation = [0., -np.pi/2, 0.] # [0,-2.22,2.22] # [2.22,2.22,0]
# from scipy.spatial.transform import Rotation as R
# tool_orientation_euler = [180, 0, 90]
# # tool_orientation_euler = [180, 0, 0]
# # tool_orientation_euler = [180, 0, 180]
# tool_orientation = R.from_euler('xyz', tool_orientation_euler, degrees=True).as_rotvec()
# # tool_orientation = [0., -np.pi, 0.] # [0,-2.22,2.22] # [2.22,2.22,0]
#
# # pose0 = np.array([-0.511, 0.294, 0.237, -0.032, -1.666, 0.138])
# pose0 = np.hstack([[-0.505, 0.06, 0.2], tool_orientation])
# pose_up = pose0.copy()
# pose_transfer = np.array([-0.431, 0.092, 0.232, -2.230, -2.194, -0.019])
#
# urc.movel_wait(pose0, a=a, v=v)
# # ---------------------------------------------

# Start gripper
grc = Gripper_Controller_V2()
grc.follow_gripper_pos = 0
grc.follow_dc = [0, 0]
grc.gripper_helper.set_gripper_current_limit(0.3)
grc.start()

time.sleep(1)
k = -1
sign = 1

pos_list = [
# [-20, 20, 0.4],
# [-10, 10, 0.7],
# [0, 30, 0.32],
# [0, 30, 0.5],
[0, 30, 0.4],
# [0, 20, 0.51],
# [-5, 5, 0.8],
# [0, 0, 1.4],
[0, 30, 0.5],
[0, 0, 0.]
# [-5, 5, 0.8],
# [-10, 10, 0.7],
# [-20, 20, 0.4],
]
# grc.follow_gripper_pos = 0.
for i in range(10000):
    # grc.follow_dc = [-20, 20]
    # grc.follow_gripper_pos = 0.3
    # grc.follow_dc = [0, 30]
    # grc.follow_gripper_pos = 0.5
    # time.sleep(1)
    # grc.follow_dc = [0, 0]
    # grc.follow_gripper_pos = 0
    # time.sleep(1)

    # k = k + 0.1 * sign
    # print(k)
    # if k > 20 or k < -5:
    #     sign *= -1

    # grc.follow_dc = [-k, k]
    grc.follow_dc = [pos_list[k][0], pos_list[k][1]]
    grc.follow_gripper_pos = pos_list[k][2]
    # k = (k + 1) % len(pos_list)
    c = input()
    print(c)
    if c == "o":
        k = -1
    else:
        k = (k+1) % (len(pos_list) - 1)




    # grc.follow_dc = [-5, -5]
    # grc.follow_gripper_pos = 1.1
    # time.sleep(1)
    # grc.follow_dc = [0, 0]
    # grc.follow_gripper_pos = 0
    # time.sleep(1)

    # grc.follow_dc = [-30, 0]
    # grc.follow_gripper_pos = 0.5
    # time.sleep(1)
    # grc.follow_dc = [0, 0]
    # grc.follow_gripper_pos = 0
    # time.sleep(1)
