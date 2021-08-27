#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
from controller.ur5.ur_controller import UR_Controller
from controller.mini_robot_arm.RX150_driver import RX150_Driver, RX150_Driver_Thread
from perception.kinect.kinect_camera import Kinect
from scipy.spatial.transform import Rotation as R
from controller.gripper.gripper_control_v2 import Gripper_Controller_V2
from scipy.interpolate import griddata
from perception.fabric.util.segmentation import segmentation
from perception.fabric.run import Run
from PIL import Image
import skimage.measure
import torchvision.transforms as T
import os




################################### initialize ur5
urc = UR_Controller()
urc.start()

################################### initialize rx150
rx150_thread = RX150_Driver_Thread(port="/dev/ttyACM0", baudrate=1000000)
rx150_thread.rx150.torque(enable=1)
rx150_thread.start()


################################### initialize gripper
# Start gripper
# grc = Gripper_Controller_V2()
# grc.follow_gripper_pos = 1.2
# grc.follow_dc = [0, 0]
# grc.gripper_helper.set_gripper_current_limit(0.3)
# grc.start()


################################### 0 degrees
g_open = 800
values = [1024, 2549, 1110, 1400, 0, g_open]
x = 300
y = 90
end_angle = 0 / 180. * np.pi # in pi
# rx150_thread.rx150.gogo(values, x, y, end_angle, 0, timestamp=100)




################################### main loop



rx150_thread.gogo(values, x, y, end_angle, 0, timestamp=100)
pose0 = np.array([-0.431, 0.108, 0.24, -2.23, -2.194, -0.019])
# urc.movel_wait(pose0)
urc.pose_following = pose0

time.sleep(2)
pose = pose0.copy()
for i in range(240):
	# print(urc.getl_rt())

    if i < 60:
        dz = (1 - np.cos(i / 30 * np.pi)) * 0.03
        dx = 0
    elif i < 120:
        dz = 0
        dx = np.sin(i / 30 * np.pi) * 0.03
    elif i < 240:
        dz = (1 - np.cos(i / 30 * np.pi)) * 0.03
        dx = np.sin(i / 30 * np.pi) * 0.03



    # dx = np.sin(i / 60 * np.pi) * 0.02

    # rx150 move
    values = [1024, 2549, 1110, 1400, 0, g_open]
    rx150_thread.gogo(values, x + dx * 1000, y + dz * 1000, end_angle, 0, timestamp=20)

    # ur5 move
    pose[1] = pose0[1] + dx
    pose[2] = pose0[2] + dz
    urc.pose_following = pose


    time.sleep(0.05)
urc.flag_terminate = True
urc.join()


# for i in (list(range(30)) + list(range(30, -1, -1)))*1:
#     values = [1024, 2549, 1110, 1400, 0, g_open]
#     rx150_thread.gogo(values, x, y+i*2, end_angle, 0, timestamp=10)
#     time.sleep(0.05)


rx150_thread.running = False
rx150_thread.join()




#
#
#
#
#
# def rx_move(g_open, x_pos):
#     values = [1024, 2549, 1110, 1400, 0, g_open]
#     x = x_pos
#     y = 90
#     end_angle = 0
#     rx150.gogo(values, x, 90, end_angle, 0, timestamp=100)
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
# pose_start = np.array([-0.431, 0.05, 0.21, -2.230, -2.194, -0.019])
#
# urc.movel_wait(pose_start, a=a, v=v)
#
# g_open = 1200
# values = [1024, 2549, 1110, 1400, 0, g_open]
# x = 420
# y = 120
# end_angle = 0*np.pi/180
# rx150.gogo(values, x, y, end_angle, 0, timestamp=100)
#
#
# urc.movel_wait(pose_start+[0,0,0.12,0,0,0], a=a, v=v)
# rx150.gogo(values, x, y+120, end_angle, 0, timestamp=100)
#
#
# urc.movel_wait(pose_start+[0,0,0,0,0,0], a=a, v=v)
# rx150.gogo(values, x, y, end_angle, 0, timestamp=100)
#
# rx150.gogo(values, x-120, y, end_angle, 0, timestamp=100)
# urc.movel_wait(pose_start+[0,-0.12,0,0,0,0], a=a, v=v)
#
# urc.movel_wait(pose_start+[0,0,0,0,0,0], a=a, v=v)
# rx150.gogo(values, x, y, end_angle, 0, timestamp=100)
#
# # inc = 10
# # for i in range(50):
# #     y = y + inc
# #     if y > 240 or y < 0:
# #         inc *= -1
# #         time.sleep(0.2)
# #     rx150.gogo(values, x, y, end_angle, 0, timestamp=10)
# #     pose_start[2] += inc / 1000.
# #     urc.movel_nowait(pose_start, a=a, v=v)
# #     time.sleep(0.01)
