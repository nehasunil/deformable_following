#! /usr/bin/env python
# -*- coding: utf-8
import math
import numpy as np
import socket
import time
from math import pi, sin, cos, asin
from .util.streaming import Streaming
import cv2
import _thread
from threading import Thread
from .util.processing import ini_frame, warp_perspective
from .util.fast_poisson import poisson_reconstruct
from numpy import linalg as LA
from scipy import interpolate
import torch
import torch.nn as nn
import torch.nn.functional as F
from .util.reconstruction import Class_3D
from .util.reconstruction import demark
from .util import helper
import os

dir_abs = os.path.dirname(os.path.realpath(__file__))

sensor_id = "Fabric0"
model_id = "RGB"
model_fn = os.path.join(dir_abs, f"models/LUT_{sensor_id}_{model_id}.pkl")
c3d = Class_3D(model_fn=model_fn, features_type=model_id)

def trim(img):
    img[img<0] = 0
    img[img>255] = 255


def sigmoid(x):
    return (np.exp(x) / (1+np.exp(x)))

def draw_line(img, theta, x0, y0):
    theta = theta / 180. * np.pi
    img = img.copy()
    rows, cols = img.shape[:2]

    center = np.array([x0 * cols, y0 * rows])

    d = 1100

    start_point = center + (d * np.sin(theta), d * np.cos(theta))
    end_point = center - (d * np.sin(theta), d * np.cos(theta))

    start_point = tuple(start_point.astype(np.int))
    end_point = tuple(end_point.astype(np.int))

    color = (0, 0, 1)
    thickness = 4

    img = cv2.line(img, start_point, end_point, color, thickness)
    return img

class Pose(Thread):
    def __init__(self, stream, corners, output_sz=(100,130), id='right'):
        Thread.__init__(self)
        self.stream = stream

        self.corners = corners
        self.output_sz = output_sz
        self.id = id

        self.running = False
        self.pose_img = None
        self.frame_large = None

        self.pose = None
        self.mv = None
        self.inContact = True

    def __del__(self):
        pass


    def get_pose(self):

        self.running = True

        device = torch.device("cuda:0")
        net = Net().to(device)
        fn_nn = os.path.join(dir_abs, "models/combined_mse_0328.pth")
        # fn_nn = os.path.join(dir_abs, "models/combined_0421.pth")
        # fn_nn = os.path.join(dir_abs, "models/edge_markers_depth.pth")
        net.load_state_dict(torch.load(fn_nn, map_location=device))

        cnt = 0
        while self.running:
            img = self.stream.image.copy()
            if img is None: continue

            # Warp frame
            frame = warp_perspective(img, self.corners, self.output_sz)

            # Store first frame
            cnt += 1
            if cnt == 1:
                frame0 = frame.copy()
                # frame0 = cv2.GaussianBlur(frame0,(21,21),0)
                frame0_blur = cv2.GaussianBlur(frame0, (45, 45), 0)

                x = np.arange(frame0.shape[1])
                y = np.arange(frame0.shape[0])
                xx, yy = np.meshgrid(x, y)

            raw = frame.copy()

            diff = (frame * 1.0 - frame0_blur) / 255. + 0.5
            diff_magnified = (frame * 1.0 - frame0_blur) / 255. * 4 + 0.5
            diff_small = cv2.resize(diff, (120, 90))
            depth, gx, gy = c3d.infer(diff_small * 255.0, demark=demark, display=False)
            depth3 = np.dstack([depth] * 3)

            self.depth = depth.copy()
            # print("max_depth", depth.max())

            Y_pred = net.predict([depth3 * 400/120/10], device)
            y = Y_pred[0]
            theta = np.arctan2(y[2], y[3])

            class_probs = torch.softmax(torch.from_numpy(y[5:]), dim=0)
            classification = class_probs.argmax(dim=0)

            rendered_img_line = diff_magnified

            # diff_small = cv2.resize(diff, (120, 90))
            if classification == 0:
                rendered_img_line = cv2.putText(rendered_img_line, "No Fabric", (30,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA, False)
                self.pose = (1, 0, 0)
                self.inContact = False
            elif classification == 2:
                rendered_img_line = cv2.putText(rendered_img_line, "All Fabric", (30,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA, False)
                self.pose = (0, 0.5, 0)
                self.inContact = True
            else:
                rendered_img_line = draw_line(diff_magnified, theta/np.pi*180., y[0], y[1])
                self.pose = (y[0], y[1], theta)
                self.inContact = True

            self.pose_img = rendered_img_line
            # self.pose = (y[0], y[1], theta)

    def run(self):
        print("Run pose estimation")
        self.get_pose()
        pass

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
#         self.conv4 = nn.Conv2d(32, 128, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 11, 512)
#         self.fc1 = nn.Linear(128 * 1 * 3, 120)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 8)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = torch.tanh(self.conv1(x))
        x = self.pool(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool(x)
        x = torch.tanh(self.conv3(x))
        x = self.pool(x)
#         x = self.pool(torch.tanh(self.conv4(x)))
#         x = x.permute(0, 2, 3, 1)
#         print(x.size())
        x = x.reshape(-1, 32 * 7 * 11)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def predict(self, X_test, device):
        with torch.no_grad():
            return self.forward(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
