#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
from controller.ur5.ur_controller import UR_Controller
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
a = 0.2
v = 0.2

# workspace_limits = np.asarray([[0.3, 0.748], [-0.224, 0.224], [-0.255, -0.1]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
workspace_limits = np.asarray([[-0.845, -0.605], [-0.14, 0.2], [0, 0.2]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)

# tool_orientation = [2.22,-2.22,0]
# tool_orientation = [0., -np.pi/2, 0.] # [0,-2.22,2.22] # [2.22,2.22,0]
from scipy.spatial.transform import Rotation as R
# tool_orientation_euler = [180, 0, 90]
# tool_orientation_euler = [180, 0, 0]
tool_orientation_euler = [180, 0, 180]
tool_orientation = R.from_euler('xyz', tool_orientation_euler, degrees=True).as_rotvec()
# tool_orientation = [0., -np.pi, 0.] # [0,-2.22,2.22] # [2.22,2.22,0]

# pose0 = np.array([-0.511, 0.294, 0.237, -0.032, -1.666, 0.138])
pose0 = np.hstack([[-0.505, 0.06, 0.2], tool_orientation])
pose_up = pose0.copy()

urc.movel_wait(pose0, a=a, v=v)
# ---------------------------------------------

# Start gripper
grc = Gripper_Controller()
grc.gripper_helper.set_gripper_current_limit(0.3)
grc.start()

grc.follow_gripper_pos = 0.
time.sleep(0.5)

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
        print(x, y)
        # target_position[2] -= 0.03
        if target_position[2] < -0.118:
            print("WARNNING: Reach Z Limit, set to minimal Z")
            target_position[2] = -0.118
        # robot.move_to(target_position, tool_orientation)
        pose = np.hstack([target_position, tool_orientation])
        urc.movel_wait(pose, a=a, v=v)


# Show color and depth frames
cv2.namedWindow('color')
cv2.setMouseCallback('color', mouseclick_callback)
cv2.namedWindow('depth')

seq_id = 0
sequence = [ord('p'), ord('g'), ord('h'), ord('r'), ord('g')]


###################################################
# Record kinect images
###################################################

# dir_data = "data/imgs/0625_10k"
dir_data = "data/imgs/test"
os.makedirs(dir_data, exist_ok=True)
count = 0

sz = 650
topleft = [300, 255]
bottomright = [topleft[0] + sz, topleft[1] + sz]

def record_kinect():
    global count
    # get kinect images
    color, depth_transformed = camera.get_image()

    # Crop images
    color_small = cv2.resize(color, (0, 0), fx=1, fy=1)
    color_small = color_small[
        topleft[1] : bottomright[1], topleft[0] : bottomright[0]
    ]

    depth_transformed_small = cv2.resize(depth_transformed, (0, 0), fx=1, fy=1)
    depth_transformed_small = depth_transformed_small[
        topleft[1] : bottomright[1], topleft[0] : bottomright[0]
    ]

    # Normalize depth images
    depth_min = 400.0
    depth_max = 900.0
    depth_transformed_small_img = (
        (depth_transformed_small - depth_min) / (depth_max - depth_min) * 255
    )
    depth_transformed_small_img = depth_transformed_small_img.clip(0, 255)

    mask = segmentation(color_small)
    cv2.imshow("mask", mask)
    cv2.imshow("color_nn", color_small)
    cv2.imshow("depth_nn", depth_transformed_small_img / 255.)
    cv2.waitKey(1)


    cv2.imwrite(f"{dir_data}/color_{count}.jpg", color_small)
    cv2.imwrite(
        f"{dir_data}/depth_{count}.jpg",
        depth_transformed_small_img.astype(np.uint8),
    )
    np.save(f"{dir_data}/depth_{count}.npy", depth_transformed_small)

    count += 1

###################################################
# Demo segmentation
###################################################

model_id = 29
epoch = 200
pretrained_model = "/home/gelsight/Code/Fabric/models/%d/chkpnts/%d_epoch%d" % (
    model_id,
    model_id,
    epoch,
)

def seg_output(depth, model=None):
    max_d = np.nanmax(depth)
    depth[np.isnan(depth)] = max_d
    # depth_min, depth_max = 400.0, 1100.0
    # depth = (depth - depth_min) / (depth_max - depth_min)
    # depth = depth.clip(0.0, 1.0)

    img_depth = Image.fromarray(depth)
    transform = T.Compose([T.ToTensor()])
    img_depth = transform(img_depth)
    img_depth = np.array(img_depth[0])

    out = model.evaluate(img_depth).squeeze()
    seg_pred = out[:, :, :3]

    #         prob_pred *= mask
    # seg_pred_th = deepcopy(seg_pred)
    # seg_pred_th[seg_pred_th < 0.8] = 0.0

    return seg_pred

t1 = Run(model_path=pretrained_model, n_features=3)

def demo_segmentation():
    # get kinect images
    color, depth_transformed = camera.get_image()

    # Crop images
    color_small = cv2.resize(color, (0, 0), fx=1, fy=1)
    color_small = color_small[
        topleft[1] : bottomright[1], topleft[0] : bottomright[0]
    ]

    depth_transformed_small = cv2.resize(depth_transformed, (0, 0), fx=1, fy=1)
    depth_transformed_small = depth_transformed_small[
        topleft[1] : bottomright[1], topleft[0] : bottomright[0]
    ]

    depth_min = 400.0
    depth_max = 1100.0
    depth_transformed_small_img = (depth_transformed_small - depth_min) / (
        depth_max - depth_min
    )
    depth_transformed_small_img = depth_transformed_small_img.clip(0, 1)

    cv2.imshow("color_crop", color_small)
    cv2.imshow("depth_crop", depth_transformed_small_img)

    depth_transformed_100x100 = skimage.measure.block_reduce(
        depth_transformed_small_img, (4, 4), np.mean
    )
    seg_pred = seg_output(depth_transformed_100x100, model=t1)
    mask = seg_pred.copy()
    H, W = mask.shape[0], mask.shape[1]
    mask = (
        mask.reshape((H * W, 3))
        @ np.array([[0, 0, 1.0], [0, 1.0, 1.0], [0, 1.0, 0]])
    ).reshape((H, W, 3))
    # mask = 1.0 / (1 + np.exp(-5 * (mask - 0.8)))

    mask = cv2.resize(mask, (sz, sz))
    cv2.imshow("prediction", mask)
    cv2.waitKey(1)


def move_record(pose, a, v):
    urc.movel_nowait(pose, a=a, v=v)
    while True:
        # record_kinect()
        demo_segmentation()
        if urc.check_stopped():
            break
        time.sleep(0.05)

while True:
    if seq_id % len(sequence) == 0:
        time.sleep(0.5)
    # camera_color_img, camera_depth_img = robot.get_camera_data()
    camera_color_img, camera_depth_img = camera.get_image()
    bgr_data = camera_color_img# cv2.cvtColor(camera_color_img, cv2.COLOR_RGB2BGR)
    if len(click_point_pix) != 0:
        bgr_data = cv2.circle(bgr_data, click_point_pix, 7, (0,0,255), 2)
    cv2.imshow('color', bgr_data)
    cv2.imshow('depth', camera_depth_img / 1100.)

    crop_x = [421, 960]
    crop_y = [505, 897]
    camera_depth_img_crop = camera_depth_img[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]]
    camera_color_img_crop = camera_color_img[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]]
    x = np.arange(camera_depth_img_crop.shape[0]) + crop_y[0]
    y = np.arange(camera_depth_img_crop.shape[1]) + crop_x[0]
    yy, xx = np.meshgrid(x, y)
    yy = yy.reshape([-1])
    xx = xx.reshape([-1])

    # cv2.imshow("camera_color_img_crop", camera_color_img_crop)
    cv2.imshow("camera_depth_img_crop", camera_depth_img_crop/1300.)

    zz = (camera_depth_img_crop.T).reshape([-1])

    # Validation
    # x = 525
    # y = 424
    # id = x * camera_depth_img.shape[0] + y
    # print("Desired", x,y,camera_depth_img[y][x])
    # print("Flattened", xx[id], yy[id], zz[id])

    zz = zz * cam_depth_scale
    zz[zz==0] = -1000
    xx = np.multiply(xx-cam_intrinsics[0][2],zz/cam_intrinsics[0][0])
    yy = np.multiply(yy-cam_intrinsics[1][2],zz/cam_intrinsics[1][1])
    xyz_kinect = np.vstack([xx,yy,zz])
    # shape: (3, W*H)

    camera2robot = cam_pose
    xyz_robot = np.dot(camera2robot[0:3,0:3],xyz_kinect) + camera2robot[0:3,3:]

    # X, Y
    #[-0.64, 0.2]
    #[-0.36, -0.072]

    stepsize=0.01
    xlim = [-0.64, -0.36]
    ylim = [-0.07, 0.20]

    scale = 100
    x_ws = np.arange(int(xlim[0] * scale), int(xlim[1] * scale))
    y_ws = np.arange(int(ylim[0] * scale), int(ylim[1] * scale))
    xx_ws, yy_ws = np.meshgrid(x_ws, y_ws)
    points = xyz_robot[:2].T
    # shape: (W*H, 2)
    values = xyz_robot[2]
    # shape: (W*H)
    z_ws = griddata(points*scale, values, (xx_ws, yy_ws), method='nearest')

    z_ws_blur = cv2.GaussianBlur(z_ws, (5, 5), 0)
    # ind_highest = np.unravel_index(np.argmax(z_ws_blur, axis=None), z_ws_blur.shape)
    # xyz_robot_highest = [ind_highest[1] / scale + xlim[0], ind_highest[0] / scale + ylim[0], z_ws[ind_highest]]

    x_high, y_high = np.where(z_ws_blur >= (z_ws_blur.min() + (z_ws_blur.max() - z_ws_blur.min()) *0.6))
    ind = np.random.randint(len(x_high))
    grasp_point = (x_high[ind], y_high[ind])


    xyz_robot_highest = [grasp_point[1] / scale + xlim[0], grasp_point[0] / scale + ylim[0], z_ws[grasp_point]]
    print("XYZ_HIGHEST", xyz_robot_highest)

    # scale for visualization
    z_min = -0.14
    z_max = 0.0
    z_ws_scaled = (z_ws_blur - z_min) / (z_max - z_min + 1e-6)
    z_ws_scaled = np.dstack([z_ws_scaled, z_ws_scaled, z_ws_scaled])
    z_ws_scaled[grasp_point[0], grasp_point[1], :] = [0, 0, 1]
    z_ws_scaled = np.fliplr(np.rot90(z_ws_scaled, k=1))

    cv2.imshow("z workspace", z_ws_scaled)

    c = sequence[seq_id % len(sequence)]
    seq_id += 1
    cv2.waitKey(1)
    if c == ord('c'):
        break
    elif c == ord('g'):
        grc.follow_gripper_pos = 1. - grc.follow_gripper_pos
        time.sleep(0.5)
        print("GRASP"*20)
    elif c == ord('h'):
        tool_orientation_euler = [180, 0, 180]
        tool_orientation_euler[2] = np.random.randint(180)+90
        tool_orientation = R.from_euler('xyz', tool_orientation_euler, degrees=True).as_rotvec()
        random_perturb_rng = 0.05
        pose_up = np.hstack([[-0.505+np.random.random()*random_perturb_rng - random_perturb_rng / 2, 0.06 + np.random.random()*random_perturb_rng - random_perturb_rng / 2, 0.2], tool_orientation])
        move_record(pose_up, a=a, v=v)
    elif c == ord('p'):
        xyz_robot_highest[2] -= 0.02
        if xyz_robot_highest[2] < -0.118:
            print("WARNNING: Reach Z Limit, set to minimal Z")
            xyz_robot_highest[2] = -0.118
        pose = np.hstack([xyz_robot_highest, tool_orientation])
        move_record(pose, a=a, v=v)
    elif c == ord('r'):
        tool_orientation_euler = [180, 0, 180]
        tool_orientation_euler[2] = np.random.randint(180)+90
        tool_orientation = R.from_euler('xyz', tool_orientation_euler, degrees=True).as_rotvec()
        pose_up = np.hstack([pose_up[:3], tool_orientation])
        move_record(pose_up, a=a, v=v)

    record_kinect()
    time.sleep(0.05)

    if count > 10000:
        break







cv2.destroyAllWindows()
