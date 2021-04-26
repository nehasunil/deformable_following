#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:13:36 2019

@author: yushe
"""


import os
import sys, tty, termios
from simple_pid import PID

import cv2
import numpy as np
import _thread
import time

from dynamixel_sdk import * 


################################################################################################################
#setup for the motor
ADDR_MX_TORQUE_ENABLE      = 64               # Control table address is different in Dynamixel model
ADDR_MX_GOAL_POSITION      = 116
ADDR_MX_PRESENT_POSITION   = 132

# Protocol version
PROTOCOL_VERSION            = 2.0               # See which protocol version is used in the Dynamixel

# Default setting

BAUDRATE                    = 57600            # Dynamixel default baudrate : 57600
# DEVICENAME                  = '/dev/tty.usbserial-FT2N061F'    # Check which port is being used on your controller
                                                # ex) Windows: "COM1"   Linux: "/dev/ttyUSB0" Mac: "/dev/tty.usbserial-*"
DEVICENAME                  = '/dev/ttyUSB0'    # Check which port is being used on your controller
                                                # ex) 



TORQUE_ENABLE               = 1                 # Value for enabling the torque
TORQUE_DISABLE              = 0                 # Value for disabling the torque
DXL_MINIMUM_POSITION_VALUE  = 938           # Dynamixel will rotate between this value
DXL_MAXIMUM_POSITION_VALUE  = 1738            # and this value (note that the Dynamixel would not move when the position value is out of movable range. Check e-manual about the range of the Dynamixel you use.)
DXL_MOVING_STATUS_THRESHOLD = 5 


DXL_ID =1


# Initialize PortHandler instance
# Set the port path
# Get methods and members of PortHandlerLinux or PortHandlerWindows
portHandler = PortHandler(DEVICENAME)

# Initialize PacketHandler instance
# Set the protocol version
# Get methods and members of Protocol1PacketHandler or Protocol2PacketHandler
packetHandler = PacketHandler(PROTOCOL_VERSION)

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
    

#dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, 34, 90)
################################################################################################################


 #main code   
################################################################################################################

CurrentPosition = DXL_MINIMUM_POSITION_VALUE
PositionThreshold_UP = DXL_MAXIMUM_POSITION_VALUE
PositionThreshold_DOWN = DXL_MINIMUM_POSITION_VALUE
PositionIncrease =100

CurrentTorque = 200
TorqueThreshold_UP = 1193
TorqueThreshold_DOWN = 0
TorqueIncrease = 100


dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_CURRENT, CurrentTorque)  
dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, DXL_ID, ADDR_MX_GOAL_POSITION, CurrentPosition)

Sign_Moving = 0
while True:
    if Sign_Moving == 0:
        if CurrentPosition >= PositionThreshold_DOWN  and   CurrentPosition <= PositionThreshold_UP:
            CurrentPosition = CurrentPosition + PositionIncrease  
            time.sleep(0.01)              
        if CurrentPosition > PositionThreshold_UP:
            CurrentPosition = PositionThreshold_UP
            Sign_Moving = 1       
        if CurrentPosition < PositionThreshold_DOWN:
            CurrentPosition = PositionThreshold_DOWN
    if Sign_Moving == 1:
        if CurrentPosition >= PositionThreshold_DOWN  and   CurrentPosition <= PositionThreshold_UP:
            CurrentPosition = CurrentPosition - PositionIncrease  
            time.sleep(0.01)              
        if CurrentPosition > PositionThreshold_UP:
            CurrentPosition = PositionThreshold_UP       
        if CurrentPosition < PositionThreshold_DOWN:
            CurrentPosition = PositionThreshold_DOWN    
            Sign_Moving = 0          
    # if CurrentTorque > TorqueThreshold_DOWN and CurrentTorque < TorqueThreshold_UP:
    #     CurrentTorque = CurrentTorque + TorqueIncrease
    # if CurrentTorque >= TorqueThreshold_UP:
    #     CurrentTorque = TorqueThreshold_UP
    # if CurrentTorque <= TorqueThreshold_DOWN:
    #     CurrentTorque = TorqueThreshold_DOWN
    print("Current Position:", CurrentPosition)
    print("Current_Torque:", CurrentTorque)
    dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_CURRENT, CurrentTorque)
    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, DXL_ID, ADDR_MX_GOAL_POSITION, CurrentPosition)