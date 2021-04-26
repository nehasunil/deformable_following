import serial
import time
import ik
from math import pi, sin, cos, fabs
import numpy as np
import random
from numpy import abs
import copy
import math

ser = serial.Serial("/dev/tty.usbmodem145301", 1000000)  # open serial port
print(ser.name)  # check which port was really used

time.sleep(1)
O = np.array([[0], [0]])
# ser.write(b'torque 1\n')


def send(values):
    goal_str = " ".join([str(_) for _ in values])
    # print(goal_str)
    ser.write(str.encode(("goal {}\n".format(goal_str))))  # write a string


def setxy(values, end_angle, x, y):
    global O
    O = ik.ik(end_angle, x, y, O)
    joint = 2048 - O / pi * 2048

    values[1] = int(joint[0, 0])
    values[2] = int(joint[1, 0])
    values[3] = int(joint[2, 0])
    # print(values[3])


def gogo(values, x, y, ang, goal_x, goal_y, goal_ang, goal_rot, timestamp=30.0):
    global O

    # timestamp = 30.
    dx = (goal_x - x) / timestamp
    dy = (goal_y - y) / timestamp
    da = (goal_ang - ang) / timestamp
    dr = (goal_rot - values[-2]) / timestamp

    for t in range(int(timestamp)):
        x += dx
        y += dy
        ang += da
        values[-2] += dr
        # print(dx, dy, da, dr)
        setxy(values, ang, x, y)
        send(values)
        time.sleep(0.01)

    x = goal_x
    y = goal_y
    ang = goal_ang
    values[-2] = goal_rot
    # print(dx, dy, da, dr)
    setxy(values, ang, x, y)
    send(values)
    # time.sleep(0.03)


def robot_grasp(pos, gripper_close):
    global O
    if pos == 1:
        grasp_pos = 305
        down_pos = -75
    else:
        grasp_pos = 300
        down_pos = -77
    # O = np.array([[0], [0]])
    # ser.write(b'torque 1\n')
    g_close = gripper_close
    g_open = 2000
    values = [2045, 2549, 1110, 1400, 2048, g_open]
    x, y = 310, 120
    end_angle = 0.0

    # gogo(values, x, y, end_angle, x, y, end_angle, g_open)
    # print(values, x, y, end_angle)
    # time.sleep(2)

    gogo(values, x, y, end_angle, 280, 50, 1.5, 0)
    x = 280
    y = 50
    end_angle = 1.5
    # print(values, x, y, end_angle)
    # time.sleep(2)

    # down_pos = down_pos + random.randint(-2, 2)
    down_pos = down_pos
    gogo(values, x, y, end_angle, grasp_pos, down_pos, 1.5, 0)
    x = grasp_pos
    print(x)
    y = down_pos
    end_angle = 1.5
    # print(values, x, y, end_angle)
    # time.sleep(2)

    values[-1] = g_close
    setxy(values, end_angle, x, y)
    send(values)
    time.sleep(1)

    gogo(values, x, y, end_angle, 280, 120, 1.5, 0)
    x = 280
    y = 120
    end_angle = 1.5
    # print(values, x, y, end_angle)
    # time.sleep(2)

    gogo(values, x, y, end_angle, 300, 180, 0, 2050)
    x = 300
    y = 180
    end_angle = 0
    # print(values, x, y, end_angle)

    # time.sleep(4)

    # ser.close()             # close port

    # over.put('stop')


def robot_prepare(gripper_close):
    g_close = gripper_close
    g_open = 2000
    values = [2045, 2549, 1110, 2136, 2048, g_close]
    x, y = 300, 180
    end_angle = 0
    gogo(values, x, y, end_angle, 300, 100, 0.2, 2050)
    x, y = 300, 100
    end_angle = 0.2
    # print(values, x, y, end_angle)


def robot_ready(final_angle=0):
    g_close = 1000
    g_open = 2000
    values = [2045, 2549, 1110, 2136, 2048, g_open]
    if final_angle < 150:
        x, y = 310, 120
        end_angle = 0.0

        setxy(values, end_angle, x, y)
        send(values)
    else:
        x, y = 330, 120
        end_angle = 0.0

        setxy(values, end_angle, x, y)
        send(values)

        time.sleep(0.8)

        x, y = 310, 120
        end_angle = 0.0

        setxy(values, end_angle, x, y)
        send(values)

    # gogo(values, x, y, end_angle, x, y, end_angle, g_open)


def robot_sysid60(gripper_close):
    g_close = gripper_close
    g_open = 2000
    values = [2045, 2549, 1110, 2136, 2048, g_close]
    x, y = 300, 180
    end_angle = 0.0

    gogo(values, x, y, end_angle, x, y, -0.4, 2050)
    end_angle = -0.4


def robot_sysid90(gripper_close):
    g_close = gripper_close
    g_open = 2000
    values = [2045, 2549, 1110, 2136, 2048, g_close]
    x, y = 300, 180
    end_angle = -0.4

    gogo(values, x, y, end_angle, x, y, -0.9, 2050)
    end_angle = -0.9


def robot_sysidback(gripper_close):
    g_close = gripper_close
    g_open = 2000
    values = [2045, 2549, 1110, 2136, 2048, g_close]
    x, y = 300, 180
    end_angle = -0.9

    gogo(values, x, y, end_angle, x, y, 0.0, 2050)
    end_angle = 0.0


def robot_shake(gripper_close, theta):
    g_close = gripper_close
    g_open = 2000
    values = [2045, 2549, 1110, 2136, 2048, g_close]
    x, y = 300, 180
    end_angle = 0.0
    setxy(values, end_angle, x, y)

    flag = 1
    # shake_range_x = 15
    # shake_range_y = 25

    # dx = flag * math.cos(theta/180.0*math.pi) * shake_range_x
    # dy = flag * math.sin(theta/180.0*math.pi) * shake_range_y
    # for k in range(6):
    #     gogo(values, x, y, end_angle, x + dx, y + dy, 0.0, 2050, 5.0)
    #     x = x + dx
    #     y = y + dy
    #     flag *= -1
    #     dx = flag * math.cos(theta/180.0*math.pi) * shake_range_x
    #     dy = flag * math.sin(theta/180.0*math.pi) * shake_range_y

    shake_range_angle = int(10 / 360 * 4096)
    mv_range = 30

    values[-1] += 150
    goal_str = " ".join([str(_) for _ in values])
    ser.write(str.encode(("goal {}\n".format(goal_str))))

    ser.write(str.encode(("goal_clear \n")))

    steps = 10
    for k in range(8):
        #         if k % 2 == 0:
        #             values[-1] += 15
        #         values[-3] += flag*shake_range_angle

        for _ in range(steps):
            x = x + (mv_range / steps) * flag
            setxy(values, end_angle, x, y)

            goal_str = " ".join([str(_) for _ in values])
            ser.write(str.encode(("goal_stack {}\n".format(goal_str))))
            time.sleep(0.01)

        flag *= -1

    ser.write(str.encode(("goal_run \n")))

    time.sleep(1.5)
    values[-1] = gripper_close - 20
    goal_str = " ".join([str(_) for _ in values])
    ser.write(str.encode(("goal {}\n".format(goal_str))))

    time.sleep(0.4)


def robot_shake_origin(gripper_close, theta):
    g_close = gripper_close
    g_open = 2000
    values = [2045, 2549, 1110, 2136, 2048, g_close]
    x, y = 300, 180
    end_angle = 0.0
    setxy(values, end_angle, x, y)

    flag = 1
    # shake_range_x = 15
    # shake_range_y = 25

    # dx = flag * math.cos(theta/180.0*math.pi) * shake_range_x
    # dy = flag * math.sin(theta/180.0*math.pi) * shake_range_y
    # for k in range(6):
    #     gogo(values, x, y, end_angle, x + dx, y + dy, 0.0, 2050, 5.0)
    #     x = x + dx
    #     y = y + dy
    #     flag *= -1
    #     dx = flag * math.cos(theta/180.0*math.pi) * shake_range_x
    #     dy = flag * math.sin(theta/180.0*math.pi) * shake_range_y

    shake_range_angle = int(10 / 360 * 4096)
    for k in range(6):
        values[-3] += flag * shake_range_angle
        goal_str = " ".join([str(_) for _ in values])
        ser.write(str.encode(("goal {}\n".format(goal_str))))
        time.sleep(0.2)
        flag *= -1


def robot_ready2():
    g_open = 2000
    values = [2045, 2549, 1110, 2136, 2048, g_open]
    x, y = 300, 180
    end_angle = -0.9
    setxy(values, end_angle, x, y)
    send(values)
    # gogo(values, x, y, end_angle, x, y, end_angle, g_open)


def robot_sysidback2(gripper_close):
    g_close = gripper_close
    g_open = 2000
    values = [2045, 2549, 1110, 2136, 2048, g_close]
    x, y = 300, 180
    end_angle = -0.9

    gogo(values, x, y, end_angle, 310, 120, 0.0, 2050)
    x, y = 310, 120
    end_angle = 0.0


def robot_go(gripper_close, gripper_open, ratio=1.0):
    global O
    # O = np.array([[0], [0]])
    # ser.write(b'torque 1\n')

    values = [2045, 2549, 1110, 2136, 2048, 2000]
    values0 = None

    scale = 2.0

    ddy = 0.8 * scale

    vel = 15 * ratio
    dy = vel
    g_close = gripper_close
    g_open = gripper_open
    distance = 220 * ratio
    end_angle = 0.2
    ang_vel = 0.08 * ratio
    da = ang_vel

    x, y = 300, 100
    init_y = 100

    # values[-1] = 2000
    # setxy(values, end_angle, x, y)
    # send(values)
    # time.sleep(2)

    # values[-1] = g_close
    # setxy(values, end_angle, x, y)
    # send(values)
    # time.sleep(2)

    # for t in range(40):
    #     end_angle += 0.01
    #     setxy(values, end_angle, x, y)
    #     send(values)
    #     time.sleep(0.008)
    # time.sleep(2)
    cnt = 0

    ser.write(str.encode(("goal_clear \n")))
    while True:
        y += dy
        end_angle -= da

        # if dy > 0:
        #     values[-1] = g_close
        # if dy < 0:
        #     values[-1] = g_open

        if dy < 0:
            values[-1] = g_open
        else:
            values[-1] = g_close

        if y > init_y + distance:
            dy = -vel
            da = -ang_vel
        elif y < init_y:
            # dy = vel
            # da = ang_vel

            dy = 0
            da = 0
            values[-1] = g_close

        # print(y, end_angle, dy)

        O = ik.ik(end_angle, x, y, O)
        joint = 2048 - O / pi * 2048

        values[1] = int(joint[0, 0])
        values[2] = int(joint[1, 0])
        values[3] = int(joint[2, 0])

        if values0 is None:
            values0 = copy.deepcopy(values)

        goal_str = " ".join([str(_) for _ in values])
        # ser.write(str.encode(('goal {}\n'.format(goal_str))))
        ser.write(str.encode(("goal_stack {}\n".format(goal_str))))

        # print(goal_str)
        if dy == 0:
            break
        # time.sleep(0.008)
        time.sleep(0.001)

        cnt += 1
    # print(cnt, "steps")

    ser.write(str.encode(("goal_run \n")))

    time.sleep(1.0)
    gogo(values, x, y, end_angle, 300, 180, 0.0, 2050)
    x, y = 300, 180
    end_angle = 0.0

    # goal_str = ' '.join([str(_) for _ in values0])
    # ser.write(str.encode(('goal {}\n'.format(goal_str))))

    # ser.close()             # close port

    # over.put('stop')


def read_addr(addr, bit):
    ser.write(str.encode("readreg {} {}\n".format(addr, bit)))
    for i in range(2):
        line = ser.readline()
        print(line)
    data = int(str(line).split("Data: ")[-1][:-4])
    return data


def set_servoid(id):
    ser.write(str.encode("setcurrentservoid {}\n".format(id)))
    for i in range(1):
        line = ser.readline()
        print(line)


def read_goal_current():
    return read_addr(102, 2)


def read_current():
    return read_addr(126, 2)


def read_position():
    return read_addr(132, 4)


def write_goal_current(current):
    ser.write(str.encode("writereg 102 2 {}\n".format(current)))
    for i in range(2):
        line = ser.readline()
        print(line)


def write_position(position):
    ser.write(str.encode("writereg 116 4 {}\n".format(position)))
    for i in range(2):
        line = ser.readline()
        print(line)


def readpos():
    ser.write(str.encode("readpos \n"))
    for i in range(1):
        line = ser.readline()
        print(line)


def torque(enable=1):
    ser.write(str.encode("torque {}\n".format(enable)))
    for i in range(6):
        line = ser.readline()
        print(line)


# robot_go()

# readpos()
torque(1)
pos = 0
gripper_close_min = 900
robot_grasp(pos, gripper_close_min)
