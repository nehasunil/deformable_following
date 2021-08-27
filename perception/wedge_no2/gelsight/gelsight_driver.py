#! /usr/bin/env python
# -*- coding: utf-8
import math
import numpy as np
import socket
import time
from math import pi, sin, cos
from .util.streaming import Streaming
import cv2
import _thread
from threading import Thread
from .pose import Pose
# from .tracking import Tracking
try:
    from .tracking import Tracking
except:
    print("warning: tracking is not installed")

class GelSight(Thread):
    def __init__(self, IP, corners, tracking_setting=None, output_sz=(210, 270), id='right', pose_enable=True, tracking_enable=True):
        Thread.__init__(self)

        url = "{}:8080/?action=stream".format(IP)
        self.stream = Streaming(url)
        _thread.start_new_thread(self.stream.load_stream, ())

        self.id = id
        self.corners = corners
        self.output_sz = output_sz
        self.tracking_setting = tracking_setting
        self.pose_enable = pose_enable
        self.tracking_enable = tracking_enable

        K_pose = 2
        self.output_sz_pose = (output_sz[0]//K_pose, output_sz[1]//K_pose)

        # Wait for video streaming
        self.wait_for_stream()

        # Start thread for calculating pose
        self.start_pose()

        # Start thread for tracking markers
        self.start_tracking()



    def __del__(self):
        self.pc.running = False
        self.tc.running = False
        self.stream.stop_stream()
        print("stop_stream")

    def start_pose(self):
        self.pc = Pose(self.stream, self.corners, self.output_sz_pose, id=self.id) # Pose class
        self.pc.start()

    def start_tracking(self):
        if self.tracking_setting is not None:
            self.tc = Tracking(self.stream, self.tracking_setting, self.corners, self.output_sz, id=self.id) # Tracking class
            self.tc.start()

    def wait_for_stream(self):
        while True:
            img = self.stream.image
            if img is None: 
                continue
            else:
                break
        print("GelSight {} image found".format(self.id))

    def run(self):
        print("Run GelSight driver")
        pass
