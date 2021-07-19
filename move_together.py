#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
from controller.ur5.ur_controller import UR_Controller
from controller.mini_robot_arm.RX150_driver import RX150_Driver
from perception.kinect.kinect_camera import Kinect
from scipy.spatial.transform import Rotation as R
from controller.gripper.gripper_control import Gripper_Controller
from scipy.interpolate import griddata
from perception.fabric.util.segmentation import segmentation
from perception.fabric.run import Run
from PIL import Image
import skimage.measure
import torchvision.transforms as T
import os

urc = UR_Controller()
urc.start()
a = 0.2
v = 0.2

rx150 = RX150_Driver(port="/dev/ttyACM0", baudrate=1000000)
rx150.torque(enable=1)
print(rx150.readpos())

def rx_move(g_open, x_pos):
    values = [1024, 2549, 1110, 1400, 0, g_open]
    x = x_pos
    y = 90
    end_angle = 0
    rx150.gogo(values, x, 90, end_angle, 0, timestamp=100)

# workspace_limits = np.asarray([[0.3, 0.748], [-0.224, 0.224], [-0.255, -0.1]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
workspace_limits = np.asarray([[-0.845, -0.605], [-0.14, 0.2], [0, 0.2]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)

# tool_orientation = [2.22,-2.22,0]
# tool_orientation = [0., -np.pi/2, 0.] # [0,-2.22,2.22] # [2.22,2.22,0]
from scipy.spatial.transform import Rotation as R
tool_orientation_euler = [180, 0, 90]
# tool_orientation_euler = [180, 0, 0]
# tool_orientation_euler = [180, 0, 180]
tool_orientation = R.from_euler('xyz', tool_orientation_euler, degrees=True).as_rotvec()
# tool_orientation = [0., -np.pi, 0.] # [0,-2.22,2.22] # [2.22,2.22,0]

# pose0 = np.array([-0.511, 0.294, 0.237, -0.032, -1.666, 0.138])
pose0 = np.hstack([[-0.505, 0.06, 0.2], tool_orientation])
pose_up = pose0.copy()
pose_start = np.array([-0.431, 0.05, 0.21, -2.230, -2.194, -0.019])

urc.movel_wait(pose_start, a=a, v=v)

g_open = 1200
values = [1024, 2549, 1110, 1400, 0, g_open]
x = 420
y = 120
end_angle = 0*np.pi/180
rx150.gogo(values, x, y, end_angle, 0, timestamp=100)


urc.movel_wait(pose_start+[0,0,0.12,0,0,0], a=a, v=v)
rx150.gogo(values, x, y+120, end_angle, 0, timestamp=100)


urc.movel_wait(pose_start+[0,0,0,0,0,0], a=a, v=v)
rx150.gogo(values, x, y, end_angle, 0, timestamp=100)


rx150.gogo(values, x-120, y, end_angle, 0, timestamp=100)
urc.movel_wait(pose_start+[0,-0.12,0,0,0,0], a=a, v=v)

urc.movel_wait(pose_start+[0,0,0,0,0,0], a=a, v=v)
rx150.gogo(values, x, y, end_angle, 0, timestamp=100)

# inc = 10
# for i in range(50):
#     y = y + inc
#     if y > 240 or y < 0:
#         inc *= -1
#         time.sleep(0.2)
#     rx150.gogo(values, x, y, end_angle, 0, timestamp=10)
#     pose_start[2] += inc / 1000.
#     urc.movel_nowait(pose_start, a=a, v=v)
#     time.sleep(0.01)
