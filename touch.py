#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
from controller.ur5.ur_controller import UR_Controller
from perception.kinect.kinect_camera import Kinect
from scipy.spatial.transform import Rotation as R

cam_intrinsics_origin = np.array([
[917.3927285,    0.,         957.21294894],
[  0.,         918.96234057, 555.32910487],
[  0.,           0.,           1.        ]])
cam_intrinsics = np.array([
[965.24853516,   0.,         950.50838964],
 [  0.,         939.67144775, 554.55567298],
 [  0.,           0.,           1.        ]]) # New Camera Intrinsic Matrix
dist = np.array([[ 0.0990126,  -0.10306044,  0.00024658, -0.00268176,  0.05763196]])
camera = Kinect(cam_intrinsics_origin, dist)

cam_pose = np.loadtxt('real/camera_pose.txt', delimiter=' ')
cam_depth_scale = np.loadtxt('real/camera_depth_scale.txt', delimiter=' ')

# User options (change me)
# --------------- Setup options ---------------
urc = UR_Controller()
urc.start()
a = 0.05
v = 0.05

# workspace_limits = np.asarray([[0.3, 0.748], [-0.224, 0.224], [-0.255, -0.1]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
workspace_limits = np.asarray([[-0.845, -0.605], [-0.14, 0.2], [0, 0.2]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)

# tool_orientation = [2.22,-2.22,0]
# tool_orientation = [0., -np.pi/2, 0.] # [0,-2.22,2.22] # [2.22,2.22,0]
from scipy.spatial.transform import Rotation as R
# tool_orientation_euler = [180, 0, 90]
# tool_orientation_euler = [180, 0, 0]
tool_orientation_euler = [180, 0, 180]
tool_orientation = R.from_euler('xyz', tool_orientation_euler, degrees=True).as_rotvec()


# tool_orientation = [0., -np.pi/2, 0.] # [0,-2.22,2.22] # [2.22,2.22,0]
# tool_orientation = [0., -np.pi/2, 0.] # [0,-2.22,2.22] # [2.22,2.22,0]

# pose0 = np.array([-0.511, 0.294, 0.237, -0.032, -1.666, 0.138])
pose0 = np.hstack([[-0.505, 0.2, 0.2], tool_orientation])
urc.movel_wait(pose0, a=a, v=v)
# ---------------------------------------------


# Move robot to home pose
# robot = Robot(False, None, None, workspace_limits,
#               tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
#               False, None, None)
# robot.open_gripper()

# Slow down robot
# robot.joint_acc = 1.4
# robot.joint_vel = 1.05

# Callback function for clicking on OpenCV window
click_point_pix = ()
# camera_color_img, camera_depth_img = robot.get_camera_data()
camera_color_img, camera_depth_img = camera.get_image()
def mouseclick_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global camera, robot, click_point_pix
        click_point_pix = (x,y)

        # Get click point in camera coordinates
        click_z = camera_depth_img[y][x] * cam_depth_scale
        click_x = np.multiply(x-cam_intrinsics[0][2],click_z/cam_intrinsics[0][0])
        click_y = np.multiply(y-cam_intrinsics[1][2],click_z/cam_intrinsics[1][1])
        if click_z == 0:
            return
        click_point = np.asarray([click_x,click_y,click_z])
        click_point.shape = (3,1)

        # Convert camera to robot coordinates
        # camera2robot = np.linalg.inv(robot.cam_pose)
        camera2robot = cam_pose
        target_position = np.dot(camera2robot[0:3,0:3],click_point) + camera2robot[0:3,3:]

        target_position = target_position[0:3,0]
        print(target_position)
        # robot.move_to(target_position, tool_orientation)
        pose = np.hstack([target_position, tool_orientation])
        urc.movel_wait(pose, a=a, v=v)


# Show color and depth frames
cv2.namedWindow('color')
cv2.setMouseCallback('color', mouseclick_callback)
cv2.namedWindow('depth')

while True:
    # camera_color_img, camera_depth_img = robot.get_camera_data()
    camera_color_img, camera_depth_img = camera.get_image()
    bgr_data = camera_color_img# cv2.cvtColor(camera_color_img, cv2.COLOR_RGB2BGR)
    if len(click_point_pix) != 0:
        bgr_data = cv2.circle(bgr_data, click_point_pix, 7, (0,0,255), 2)
    cv2.imshow('color', bgr_data)
    cv2.imshow('depth', camera_depth_img / 1100.)

    if cv2.waitKey(1) == ord('c'):
        break

cv2.destroyAllWindows()
