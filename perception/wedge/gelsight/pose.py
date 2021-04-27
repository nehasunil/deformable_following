#! /usr/bin/env python
# -*- coding: utf-8
import math
import numpy as np
import socket
import time
from math import pi, sin, cos, asin, acos
from .util.streaming import Streaming
import cv2
import _thread
from threading import Thread
from .util.processing import ini_frame, warp_perspective
from .util.fast_poisson import poisson_reconstruct
from numpy import linalg as LA
from scipy import interpolate

def trim(img):
    img[img<0] = 0
    img[img>255] = 255


def sigmoid(x):
    return (np.exp(x) / (1+np.exp(x)))

def PCA(pts):
    pts = pts.reshape(-1, 2).astype(np.float64)
    mv = np.mean(pts, 0).reshape(2,1)
    pts -= mv.T
    w,v = LA.eig( np.dot(pts.T, pts) )
    w_max = np.max(w)
    w_min = np.min(w)

    col = np.where(w == w_max)[0]
    if len(col) > 1:
        col = col[-1]
    V_max = v[:,col]

    col_min = np.where(w == w_min)[0]
    if len(col_min) > 1:
        col_min = col_min[-1]
    V_min = v[:,col_min]

    return V_max, V_min, w_max, w_min

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


    def img2grad(self, frame0, frame, bias = 1.):

        # blur = cv2.GaussianBlur(frame,(1,1),0)
        blur = frame
        diff = blur * 1.0 - frame0
        diff = diff * bias


        if self.id == 'left':
            dx = (diff[:,:,1] - (diff[:,:,0]  + diff[:,:,2]) * 0) / 255.
            dy = (diff[:,:,0]  - diff[:,:,2]) / 255.
        else:
            dx = (diff[:,:,1] - (diff[:,:,0]  + diff[:,:,2]) * 0) / 255.
            dy = (diff[:,:,0]  - diff[:,:,2]) / 255.


        dx = dx / (1 - dx ** 2) ** 0.5 / 32
        dy = dy / (1 - dy ** 2) ** 0.5 / 32

        # dx = (diff[:,:,2] * cos(pi/6) - diff[:,:,0] * cos(pi/6)) / 255.
        # dy = (diff[:,:,0] * sin(pi/6) + diff[:,:,2] * sin(pi/6) - diff[:,:,1]) / 255.

        return dx, dy


    def img2depth(self, frame0, frame, bias = 1.):
        dx, dy = self.img2grad(frame0, frame)

        zeros = np.zeros_like(dx)
        return poisson_reconstruct(dy, dx, zeros)

    def draw_ellipse(self, frame, pose):
        v_max, v_min, w_max, w_min, m = pose
        lineThickness = 2
        K = 2

        w_max = w_max ** 0.5 / 20 * K
        w_min = w_min ** 0.5 / 30 * K

        v_max = v_max.reshape(-1) * w_max
        v_min = v_min.reshape(-1) * w_min


        m1 = m - v_min
        mv = m + v_min
        cv2.line(frame, (int(m1[0]), int(m1[1])), (int(mv[0]), int(mv[1])),
                 (0,255,0), lineThickness)

        m1 = m - v_max
        # mv = m
        mv = m + v_max

        cv2.line(frame, (int(m1[0]), int(m1[1])), (int(mv[0]), int(mv[1])),
                 (0,0,255), lineThickness)

        theta = acos(v_max[0]/(v_max[0]**2+v_max[1]**2)**0.5)
        length_max = w_max
        length_min = w_min
        axis = (int(length_max), int(length_min))

        cv2.ellipse(frame, (int(m[0]), int(m[1])), axis, -theta/pi*180, 0, 360, (255, 255, 255), lineThickness)

    def get_pose(self):

        self.running = True

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
                frame0 = cv2.GaussianBlur(frame0,(13,13),0)

                x = np.arange(frame0.shape[1])
                y = np.arange(frame0.shape[0])
                xx, yy = np.meshgrid(x, y)

            raw = frame.copy()

            # frame = cv2.GaussianBlur(frame,(21,21),0)
            # diff = (frame * 1.0 - frame0) * 0.02
            # diff = (diff + 0.5) * 255


            # diff[diff<0] = 0
            # diff[diff>255] = 255

            # blur = cv2.GaussianBlur(diff,(35,35),0)
            # diff = diff - np.mean(np.mean(diff, axis=0), axis=0)
            # diff = cv2.GaussianBlur(diff,(35,35),0)


            # # Compensate for illumination
            bias = (4 - yy * 3. / frame0.shape[0])
            bias = np.dstack([bias]*3)
            bias = 1

            diff = (frame * 1.0 - frame0) * 4 + 127
            trim(diff)

            self.diff_raw = diff.copy()

            # diff = diff * bias + 127
            # # diff = diff + 127

            # frame = demark(frame, K=2)

            depth = self.img2depth(frame0, frame, bias) * 255
            # depth[depth < 0] = 0
            if self.id == 'right':
                # thresh = 80
                thresh = max(4, depth.max() / 2)
            else:
                thresh = 4



            mask = depth > thresh

            mask_intensity = sigmoid((depth - thresh)/5)
            for _ in range(3):
                frame[:,:,_] = raw[:,:,_] / (3 - 2*mask_intensity)

            # print("MAX DEPTH", depth.max())

            # Display the resulting frame
            coors = np.where(mask == 1)
            X = coors[1].reshape(-1,1)
            y = coors[0].reshape(-1,1)


            # frame_large = warp_perspective(img, self.corners, (400, 520))

            # K = 4
            # if cnt == 1:
            #     frame0_large = frame_large.copy()
            #     frame0_large = cv2.GaussianBlur(frame0_large,(141,141),0)
            # diff_large = (cv2.GaussianBlur(frame_large,(141,141),0) * 1.0 - frame0_large) / 255. + 0.5

            self.area = 0
            if len(X) > 1:
                pts = np.concatenate([X, y], axis=1)
                v_max, v_min, w_max, w_min = PCA(pts)
                if v_max[0] > 0 and v_max[1] > 0:
                    v_max *= -1
                m = np.mean(pts, 0).reshape(-1)


                # if v_max[0] > 0 and v_max[1] > 0:

                # print(w_max, w_min)

                # Record pose estimation
                self.pose = (v_max, v_min, w_max, w_min, m)
                self.area = len(X)

                # for manipulation
                # self.mv = np.mean(pts, 0)
                self.mv = np.mean(pts, 0)
                self.mv_relative = self.mv - [frame0.shape[1] / 3, frame0.shape[0] *2]
                self.vr = self.mv - [frame0.shape[1], frame0.shape[0]*2/4]
                self.theta_to_middle_right = asin(self.vr[1] / np.sum(self.vr**2)**0.5)
                # print("theta_to_middle_right", self.theta_to_middle_right/pi*180, )

                if (v_max[0] == 0):
                    cable_out = [0., self.mv[1]]
                else:
                    cable_out = [0., self.mv[1] - (self.mv[0]) / v_max[0,0] * v_max[1,0]]

                cable_out = np.array(cable_out)

                v_out = np.array([frame0.shape[1] / 2, frame0.shape[0] *2 / 3]) - cable_out
                v_out[1] *= -1
                v_out = v_out / (np.sum(v_out**2)**0.5)

                self.v_out = v_out

                self.draw_ellipse(diff, self.pose)

                # cv2.line(frame_large, (int(m1[0]*K), int(m1[1]*K)), (int(mv[0]*K), int(mv[1]*K)),
                #          (0,0,255), lineThickness)
            else:
                # No contact
                self.pose = None


            self.depth = depth
            self.pose_img = diff / 255.
            # self.pose_img = frame[::-1, ::-1]
            self.diff = diff
            self.bias = bias
            # self.frame_large = frame_large

    def run(self):
        print("Run pose estimation")
        self.get_pose()
        pass
