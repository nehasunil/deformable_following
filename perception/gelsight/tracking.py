#! /usr/bin/env python
# -*- coding: utf-8
import socket
import time

import cv2
import numpy as np
from math import pi, sin, cos
import math3d as m3d

from threading import Thread
import _thread

from .util.streaming import Streaming
from .util.find_marker import Matching
from .util.processing import warp_perspective
from numpy import linalg as LA


def trim(img):
    img[img<0] = 0
    img[img>255] = 255


class Tracking(Thread):
    def __init__(self, stream, tracking_setting, corners, output_sz, id='right'):
        Thread.__init__(self)
        self.stream = stream

        self.tracking_setting = tracking_setting
        self.m = Matching(*self.tracking_setting)

        self.corners = corners
        self.output_sz = output_sz

        self.running = False
        self.tracking_img = None
        self.tracking_img_large = None

        self.slip_index_realtime = 0.

        self.flow = None

        self.id = id

    def __del__(self):
        pass

    def find_marker(self, frame, RESCALE=4):
        # RESCALE = setting.RESCALE
        frame_small = frame

        # Blur image to remove noise
        blur = cv2.GaussianBlur(frame_small, (int(127/RESCALE), int(127/RESCALE)), 0)

        # Subtract the surrounding pixels to magnify difference between markers and background
        diff = blur - frame_small.astype(np.float32)
        
        diff *= 16.0
        diff[diff<0.] = 0.
        diff[diff>255.] = 255.
        diff = cv2.GaussianBlur(diff, (int(63/RESCALE), int(63/RESCALE)), 0)

        if self.id == 'left':
            mask = (diff[:,:,0] > 20) & (diff[:,:,2] > 20) & (diff[:,:,1] > 20)
        else:
            # Note: This is good for the right camera
            # mask = (diff[:,:,0] > -10) & (diff[:,:,2] > 40) & (diff[:,:,1] > 40)
            mask = (diff[:,:,0] > 20) & (diff[:,:,2] > 40) & (diff[:,:,1] > 40)

        mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]))

        return mask


    def marker_center(self, mask, frame, RESCALE=4):

        K = 8//RESCALE
        
        areaThresh1=10*K**2
        areaThresh2=200*K**2

        MarkerCenter = []

        contours=cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours[0])<15:  # if too little markers, then give up
            print("Too less markers detected: ", len(contours))
            return MarkerCenter

        for contour in contours[0]:
            x,y,w,h = cv2.boundingRect(contour)
            AreaCount=cv2.contourArea(contour)
            # print(AreaCount)
            if AreaCount>areaThresh1 and AreaCount<areaThresh2:
                t=cv2.moments(contour)
                # print("moments", t)
                mc = [t['m10']/t['m00'], t['m01']/t['m00']]
                MarkerCenter.append(mc)
                # print(mc)
                # cv2.circle(frame, (int(mc[0]), int(mc[1])), 10, ( 0, 0, 255 ), 2, 6);

        # 0:x 1:y
        return MarkerCenter


    def draw_flow(self, frame, flow, s=1):
        Ox, Oy, Cx, Cy, Occupied = flow

        K = 2
        for i in range(len(Ox)):
            for j in range(len(Ox[i])):
                pt1 = (int(Ox[i][j]*s), int(Oy[i][j]*s))
                pt2 = (int(Cx[i][j]*s + K * (Cx[i][j]*s - Ox[i][j]*s)), int(Cy[i][j]*s + K * (Cy[i][j]*s - Oy[i][j]*s)))
                # color = (0, 0, 255)
                color = (0, 255, 255)
                # if Occupied[i][j] <= -1:
                    # color = (127, 127, 255)
                cv2.arrowedLine(frame, pt1, pt2, color, 2,  tipLength=0.3)

    def tracking(self):
        m = self.m
        frame0 = None
        frame0_large = None

        self.running = True

        cnt = 0
        while self.running:
            img = self.stream.image.copy()
            if img is None: continue

            # Warp frame
            im = warp_perspective(img, corners=self.corners, output_sz=self.output_sz)

            ############################################################
            # # find marker masks
            mask = self.find_marker(im)
            self.mask = mask

            # # # # find marker centers
            mc = self.marker_center(mask, im)

            m.init(mc)

            m.run()

            flow = m.get_flow()
            ############################################################

            if frame0 is None:
                frame0 = im.copy()
                frame0 = cv2.GaussianBlur(frame0, (int(63), int(63)), 0)

            diff = (im * 1.0 - frame0) * 2 + 127
            trim(diff)


            (Ox, Oy, Cx, Cy, Occupied) = flow
            Ox, Oy, Cx, Cy = np.array(Ox), np.array(Oy), np.array(Cx), np.array(Cy)

            self.flow = flow
            # draw flow
            self.draw_flow(diff, flow)

            # print(time.time()-tm)
            tm = time.time()

                            
            # # Motor reaction based on the sliding information    
            self.slip_index_realtime = float(np.mean(((Cx-Ox)**2 + (Cy-Oy)**2) ** 0.5))
            # # slip_index.put(slip_index_realtime)
            # print("ArrowMean CurveRight:", self.slip_index_realtime, end =" ")



            # Large image
            # im_large = warp_perspective(img, corners=self.corners, output_sz=[self.output_sz[0]*2, self.output_sz[1]*2])

            # if frame0_large is None:
            #     frame0_large = im_large.copy()
            #     frame0_large = cv2.GaussianBlur(frame0_large, (int(41), int(41)), 0)

            # diff_large = (im_large * 1.0 - frame0_large) * 2 + 127
            # trim(diff_large)
            # self.draw_flow(diff_large, flow, s=2)
            # self.tracking_img_large = diff_large / 255.


            # self.tracking_img = (mask*1.0)
            self.tracking_img = diff / 255.


    def run(self):
        print("Run tracking algorithm")
        self.tracking()
        pass
