import csv
import datetime
import time
import urllib
import urllib.request

import _thread
import cv2
import numpy as np
import util
from util.processing import warp_perspective
from util.reconstruction import Class_3D
from util.streaming import Streaming
from util.Vis3D import ClassVis3D
from controller.mini_robot_arm.RX150_driver import RX150_Driver

import deepdish as dd
import os

#######################################################################################

rx150 = RX150_Driver(port="/dev/ttyACM0", baudrate=1000000)
print(rx150.readpos())

rx150.torque(enable=1)
# g_open = 780
g_open = 1200

goal_rot = 3072+4096
# values = [1024, 2549, 1110, 1400, 0, g_open]
values = [1300, 2549, 1110, 1400, 0, g_open]
# x = 420
# x = 380
x = 280
y = 120
end_angle = 0
# 270 - 420
rx150.gogo(values, x, y, end_angle, goal_rot, timestamp=100)
time.sleep(2)

#######################################################################################

sensor_id = "fabric_0"

stream = urllib.request.urlopen("http://rpigelsightfabric.local:8080/?action=stream")

stream = Streaming(stream)
_thread.start_new_thread(stream.load_stream, ())

# n, m = 300, 400
n, m = 150, 200
# Vis3D = ClassVis3D(m, n)

fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")

dir_data = "data/touch/08272021/corner/"
os.makedirs(dir_data, exist_ok=True)

def save_video(frame_list, vid, gripper_width_list):
    fn_video = dir_data + "F{:03d}.mov".format(vid)
    fn_gripper = dir_data + "F{:03d}.npy".format(vid)
    np.save(fn_gripper, gripper_width_list)

    col = 200
    row = 150
    out = cv2.VideoWriter(
        fn_video, fourcc, 20.0, (col * 1, row * 1)
    )  # The fps depends on CPU
    for frame in frame_list:
        out.write(frame)
    out.release()



def read_csv(filename=f"config/config_{sensor_id}.csv"):
    rows = []

    with open(filename, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        for row in csvreader:
            rows.append((int(row[1]), int(row[2])))

    return rows


TOPLEFT, TOPRIGHT, BOTTOMLEFT, BOTTOMRIGHT = tuple(read_csv())

cnt = 0
last_tm = 0
# n, m = 48, 48
# n, m = 300, 300
diff_thresh = 5

diff_max = 0
image_peak = None
flag_recording = False

frames = []
last_frame = None
gripper_width_list = []
gripper_width_command_list = []

gripper_width = 1000
gripper_width_inc = -10

st = time.time()
flag_recording = True

vid = 0
cnt_close = 0 # compensate for camera latency

while True:
    frame = stream.image
    if frame == "":
        print("waiting for streaming")
        continue


    cnt += 1

    im = warp_perspective(frame, TOPLEFT, TOPRIGHT, BOTTOMLEFT, BOTTOMRIGHT)  # left
    # im[:, :, 0] = 0.0

    im_raw = cv2.resize(im, (400, 300))
    # im_raw = cv2.resize(im, (200, 150))
    im = cv2.resize(im, (m, n))



    if cnt == 1:
        frame0 = im.copy()
        frame0 = cv2.GaussianBlur(frame0, (31, 31), 0)

    cv2.imshow("im_raw", im_raw)

    diff = ((im * 1.0 - frame0)) / 255.0 + 0.5
    cv2.imshow("warped", cv2.resize((diff - 0.5) * 4 + 0.5, (m, n)))

    last_tm = time.time()




    if flag_recording:

        values[-1] = max(gripper_width, 750)
        rx150.gogo(values, x, y, end_angle, goal_rot, timestamp=1)

    time.sleep(0.02)

    if last_frame is not None and np.sum(last_frame - im_raw) != 0 and flag_recording:
        im_small = cv2.resize(im_raw, (0, 0), fx=0.5, fy=0.5)
        # frames.append(im_raw)
        frames.append(im_small)

        # save the command for the gripper width
        gripper_width_command_list.append(gripper_width)

        # save the actual reading of the gripper width
        encoder = rx150.readpos_float()
        gripper_width_list.append(encoder[-1])
        print("frame", len(frames), "command:", gripper_width_command_list[-1], "actual:", gripper_width_list[-1])


    gripper_width_inc_multiplier = 1
    # gripper_width_inc_multiplier = 1 + (900 - gripper_width) * 0.01

    gripper_width += gripper_width_inc
    if  gripper_width < 650 and flag_recording:
        # save data
        save_video(frames, vid, gripper_width_list)
        frames, gripper_width_list, gripper_width_command_list = [], [], []
        vid += 1

        if vid > 110:
            break

        gripper_width = 1050
        values[-1] = gripper_width
        rx150.gogo(values, x, y, end_angle, goal_rot, timestamp=1)
        time.sleep(2)

    last_frame = im_raw.copy()

    c = cv2.waitKey(1)
    if c == ord("q"):
        break
    if c == ord("g"):
        if flag_recording:
            values[-1] = 1200
            rx150.gogo(values, x, y, end_angle, goal_rot, timestamp=100)
            flag_recording = False
        else:
            values[-1] = 750
            rx150.gogo(values, x, y, end_angle, goal_rot, timestamp=100)
            time.sleep(1)
            values[-1] = 1000
            rx150.gogo(values, x, y, end_angle, goal_rot, timestamp=100)
            time.sleep(1)
            flag_recording = True



print(len(frames))
# data = {"frames":frames, "gripper_width_command_list":gripper_width_command_list, "gripper_width_list":gripper_width_list}
# dd.io.save("data/touch/fabric_test.h5", frames)
# dd.io.save("data/touch/08192021/fabric_all_fabric.h5", data)
# dd.io.save("data/touch/08192021/fabric_edge.h5", data)
# dd.io.save("data/touch/08192021/fabric_edge_skinny.h5", data)
# dd.io.save("data/touch/08192021/fabric_test.h5", data)
# dd.io.save("data/touch/08242021/fabric_edge_close.h5", data)
# dd.io.save("data/touch/08242021/fabric_fold_close.h5", data)
# dd.io.save("data/touch/08242021/fabric_edge_close_2.h5", data)
# dd.io.save("data/touch/08192021/fabric_left_edge2.h5", data)
cv2.destroyAllWindows()
