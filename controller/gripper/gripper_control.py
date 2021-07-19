import os
import sys, tty, termios
from simple_pid import PID

import cv2
import numpy as np
from threading import Thread
import time

from dynamixel_sdk import *


class Gripper_Controller(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.gripper_helper = GripperHelper(DXL_ID=1, min_position=938, max_position=2600)
        self.follow_gripper_pos = 0.
        self.flag_terminate = False

    def follow(self):
        # Set the position to self.follow_gripper_pos
        self.gripper_helper.set_gripper_pos(self.follow_gripper_pos)

    def run(self):
        while not self.flag_terminate:
            self.follow()
            time.sleep(0.01)

class GripperHelper(object):
        def __init__(self, DXL_ID, min_position, max_position, DEVICENAME='/dev/ttyUSB0'):
            self.DXL_ID = DXL_ID
            self.min_position = min_position
            self.max_position = max_position
            self.DEVICENAME = DEVICENAME

            self.init()

        def set_gripper_pos(self, pos):
            # Set gripper position, 0-1
            ADDR_MX_GOAL_POSITION  = 116
            CurrentPosition = int(self.min_position + pos * (self.max_position - self.min_position))
            dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID, ADDR_MX_GOAL_POSITION, CurrentPosition)

        def set_gripper_current_limit(self, current_limit):
            # Set gripper current limit, 0-1
            ADDR_GOAL_CURRENT = 102
            CURRENT_LIMIT_UPBOUND = 1193
            CurrentTorque = int(CURRENT_LIMIT_UPBOUND * current_limit)
            dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, self.DXL_ID, ADDR_GOAL_CURRENT, CurrentTorque)


        def init(self):
            ################################################################################################################
            #setup for the motor
            ADDR_MX_TORQUE_ENABLE      = 64               # Control table address is different in Dynamixel model
            ADDR_MX_PRESENT_POSITION   = 132

            # Protocol version
            PROTOCOL_VERSION            = 2.0               # See which protocol version is used in the Dynamixel

            # Default setting

            BAUDRATE                    = 57600            # Dynamixel default baudrate : 57600
            # DEVICENAME                  = '/dev/tty.usbserial-FT2N061F'    # Check which port is being used on your controller
                                                            # ex) Windows: "COM1"   Linux: "/dev/ttyUSB0" Mac: "/dev/tty.usbserial-*"
            DEVICENAME                  = self.DEVICENAME    # Check which port is being used on your controller
                                                            # ex)



            TORQUE_ENABLE               = 1                 # Value for enabling the torque
            TORQUE_DISABLE              = 0                 # Value for disabling the torque
            DXL_MINIMUM_POSITION_VALUE  = self.min_position           # Dynamixel will rotate between this value
            DXL_MAXIMUM_POSITION_VALUE  = self.max_position            # and this value (note that the Dynamixel would not move when the position value is out of movable range. Check e-manual about the range of the Dynamixel you use.)
            DXL_MOVING_STATUS_THRESHOLD = 5


            DXL_ID = self.DXL_ID


            # Initialize PortHandler instance
            # Set the port path
            # Get methods and members of PortHandlerLinux or PortHandlerWindows
            portHandler = PortHandler(DEVICENAME)
            self.portHandler = portHandler

            # Initialize PacketHandler instance
            # Set the protocol version
            # Get methods and members of Protocol1PacketHandler or Protocol2PacketHandler
            packetHandler = PacketHandler(PROTOCOL_VERSION)
            self.packetHandler = packetHandler

            # Open port
            if portHandler.openPort():
                print("Succeeded to open the port")
            else:
                print("Failed to open the port")



            # Set port baudrate
            if portHandler.setBaudRate(BAUDRATE):
                print("Succeeded to change the baudrate")
            else:
                print("Failed to change the baudrate")

            # Enable Dynamixel Torque
            dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE)
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % packetHandler.getRxPacketError(dxl_error))
            else:
                print("Dynamixel has been successfully connected")


            # Changing operating mode
            ADDR_OPERATING_MODE= 11
            OP_MODE_POSITION= 5
            dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_OPERATING_MODE, OP_MODE_POSITION)

            # set the current limit
            ADDR_CURRENT_LIMIT = 38
            CURRENT_LIMIT_UPBOUND = 1193
            dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_CURRENT_LIMIT, CURRENT_LIMIT_UPBOUND)

            #SET THE VELOCITU LIMIT
            ADDR_VELOCITY_LIMIT = 44
            VELOCITY_LIMIT_UPBOUND = 1023
            dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_VELOCITY_LIMIT, VELOCITY_LIMIT_UPBOUND)

            #SET THE MAX POSITION LIMIT
            ADDR_MAX_POSITION_LIMIT = 48
            MAX_POSITION_LIMIT_UPBOUND = DXL_MAXIMUM_POSITION_VALUE
            dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MAX_POSITION_LIMIT, MAX_POSITION_LIMIT_UPBOUND)

            #SET THE MIN POSITION LIMIT
            ADDR_MIN_POSITION_LIMIT = 52
            MIN_POSITION_LIMIT_UPBOUND = DXL_MINIMUM_POSITION_VALUE
            dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MIN_POSITION_LIMIT, MIN_POSITION_LIMIT_UPBOUND)



            ADDR_GOAL_CURRENT = 102
            #GOAL_CURRENT_MINPOSITION = 1

            # SET THE GOAL VELOCITY
            ADDR_GOAL_VELOCITY = 104
            GOAL_VELOCITY_MAXPOSITION = 1023
            dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_VELOCITY, GOAL_VELOCITY_MAXPOSITION)


            ADDR_ACCELERATION_PROFILE = 108
            ACCELERATION_ADDRESS_POSITION= 0
            dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_ACCELERATION_PROFILE, ACCELERATION_ADDRESS_POSITION)

            ADDR_VELOCITY_PROFILE = 112
            VELOCITY_ADDRESS_POSITION= 0
            dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_VELOCITY_PROFILE, VELOCITY_ADDRESS_POSITION)


            # Enable Dynamixel Torque
            dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE)
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % packetHandler.getRxPacketError(dxl_error))
            else:
                print("Dynamixel has been successfully connected")
