import serial
import cv2
import numpy as np
from math import sin, cos, pi
import time
import random
from threading import Thread

img = np.zeros([300, 300])




class RX150_IK:
    def __init__(self):
        self.l2 = 150
        self.lT = 50
        self.l3 = 150

        self.O = np.array([[0], [0]])

    def fk(self, t2, t3):
        l2, lT, l3 = self.l2, self.lT, self.l3

        x0 = (150, 0)
        x1 = (x0[0] + l2 * sin(t2), x0[1] + l2 * cos(t2))
        x2 = (x1[0] + lT * cos(t2), x1[1] - lT * sin(t2))
        x3 = (x2[0] + l3 * cos(t2 + t3), x2[1] - l3 * sin(t2 + t3))
        return x3

    def get_JaccobianTranspose(self, t2, t3):
        l2, lT, l3 = self.l2, self.lT, self.l3

        J = np.array(
            [
                [l2 * cos(t2) - lT * sin(t2) - l3 * sin(t2 + t3), -l3 * sin(t2 + t3)],
                [-l2 * sin(t2) - lT * cos(t2) - l3 * cos(t2 + t3), -l3 * cos(t2 + t3)],
            ]
        )
        return J.T

    def ik(self, end_angle, x, y):
        l2, lT, l3 = self.l2, self.lT, self.l3
        O_last = self.O.copy()

        angle = end_angle
        l_end = 150.0
        fix_end_angle = -0.3
        # fix_end_angle = 0

        x -= l_end * cos(angle)
        y += l_end * sin(angle)

        if len(O_last) == 3:
            O = O_last[:-1]
        else:
            O = O_last

        alpha = 0.00001

        i = 0

        while True:
            i += 1

            V = self.fk(O[0, 0], O[1, 0])
            JT = self.get_JaccobianTranspose(O[0, 0], O[1, 0])

            dV = np.array([[x - V[0]], [y - V[1]]])
            O = O + alpha * JT.dot(dV)

            if (dV ** 2).sum() < 1e-4:
                break

        O = np.array(
            [O[0].tolist(), O[1].tolist(), [-O[0, 0] - O[1, 0] + angle - fix_end_angle]]
        )

        self.O = O.copy()
        return O

class RX150_Driver:
    def __init__(self, port="/dev/tty.usbmodem145301", baudrate=1000000):
        self.rx150_ik = RX150_IK()

        self.ser = serial.Serial(port, baudrate)  # open serial port
        print(self.ser.name)  # check which port was really used

        # initialize initial joint angle for ik
        time.sleep(0.2)
        joints = self.readpos_float()

        t2 = (2048 - joints[1]) / 2048. * pi
        t3 = (2048 - joints[2]) / 2048. * pi

        self.rx150_ik.O = np.array([[t2], [t3]])


        # x, y, angle
        self.last_joint = None

    def update_pos(self):
        fix_end_angle = -0.3

        time.sleep(0.2)
        joints = self.readpos_float()

        t2 = (2048 - joints[1]) / 2048. * pi
        t3 = (2048 - joints[2]) / 2048. * pi
        t4 = (2048 - joints[3]) / 2048. * pi
        end_angle = t2 + t3 + t4 + fix_end_angle

        x = np.array(self.rx150_ik.fk(t2, t3))

        angle = end_angle
        l_end = 150.0

        x[0] += l_end * cos(angle)
        x[1] -= l_end * sin(angle)

        self.last_joint = (x[0], x[1], end_angle)

    def readpos(self):
        self.ser.write(str.encode("readpos \n"))
        for i in range(1):
            line = self.ser.readline()
        return line

    def readpos_float(self):
        line = self.readpos()
        elems = line.decode('utf-8').split(' ')[:-1]
        elems = [float(_) for _ in elems]
        return elems

    def torque(self, enable=1):
        self.ser.write(str.encode("torque {}\n".format(enable)))
        for i in range(6):
            line = self.ser.readline()
            print(line)


    def send(self, values):
        goal_str = " ".join([str(_) for _ in values])
        self.ser.write(str.encode(("goal {}\n".format(goal_str))))  # write a string


    def setxy(self, values, end_angle, x, y):
        O = self.rx150_ik.ik(end_angle, x, y)
        joint = 2048 - O / pi * 2048

        values[1] = int(joint[0, 0])
        values[2] = int(joint[1, 0])
        values[3] = int(joint[2, 0])


    def gogo(self, values, goal_x, goal_y, goal_ang, goal_rot, timestamp=30.0):
        if self.last_joint is None:
            self.update_pos()

        # resolve ambiguity for rotation angle
        joints = self.readpos_float()
        values[-2] = joints[4]
        if np.abs(goal_rot - joints[4] + 4096) < np.abs(goal_rot - joints[4]):
            goal_rot += 4096
        elif np.abs(goal_rot - joints[4] - 4096) < np.abs(goal_rot - joints[4]):
            goal_rot -= 4096

        x, y, ang = self.last_joint
        dx = (goal_x - x) / timestamp
        dy = (goal_y - y) / timestamp
        da = (goal_ang - ang) / timestamp
        dr = (goal_rot - values[-2]) / timestamp

        self.last_joint = (goal_x, goal_y, goal_ang)

        for t in range(int(timestamp)):
            x += dx
            y += dy
            ang += da
            values[-2] += dr
            print("Goal rot", goal_rot, "Cur rot", values[-2])
            # print(dx, dy, da, dr)
            self.setxy(values, ang, x, y)
            self.send(values)
            time.sleep(0.01)

        x = goal_x
        y = goal_y
        ang = goal_ang
        values[-2] = goal_rot
        # print(dx, dy, da, dr)
        self.setxy(values, ang, x, y)
        self.send(values)
        # time.sleep(0.03)

    def move(self, goal_x, goal_y, goal_ang, goal_rot, timestamp=30.0):
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
            self.setxy(values, ang, x, y)
            self.send(values)
            time.sleep(0.01)

        x = goal_x
        y = goal_y
        ang = goal_ang
        values[-2] = goal_rot
        # print(dx, dy, da, dr)
        self.setxy(values, ang, x, y)
        self.send(values)
        # time.sleep(0.03)


class RX150_Driver_Thread(Thread):
    def __init__(self, port="/dev/ttyACM0", baudrate=1000000):
        Thread.__init__(self)
        self.rx150 = RX150_Driver(port=port, baudrate=baudrate)
        self.command = None
        self.last_command = None
        self.running = True

    def gogo(self, values, goal_x, goal_y, goal_ang, goal_rot, timestamp=30.0):
        self.command = (values, goal_x, goal_y, goal_ang, goal_rot, timestamp)

    def follow(self):
        while self.running:
            if self.command is not None and (self.last_command is None or self.command != self.last_command):
                self.last_command = self.command
                print(self.command)
                self.rx150.gogo(*self.command)
            time.sleep(0.01)


    def run(self):
        self.follow()


def main():
    rx150 = RX150_Driver(port="/dev/ttyACM0", baudrate=1000000)
    print(rx150.readpos())

    rx150.torque(enable=1)
    # g_open = 780
    # g_open = 780
    g_open = 780
    values = [1300, 2549, 1110, 1400, 3072+4096, g_open]
    # x = 420
    # x = 380
    x = 280
    y = 120
    end_angle = 0
    # 270 - 420
    inc = 1
    rx150.gogo(values, x, y, end_angle, 3072+4096, timestamp=100)

    print(rx150.readpos_float())

def main_thread():
    rx150_thread = RX150_Driver_Thread(port="/dev/ttyACM0", baudrate=1000000)
    rx150_thread.rx150.torque(enable=1)
    rx150_thread.start()

    ################################### 90 degrees
    # g_open = 1200
    # values = [1024, 2549, 1110, 1400, 0, g_open]
    # x = 320
    # y = 30
    # end_angle = 90 / 180. * np.pi # in pi
    # rx150_thread.rx150.gogo(values, x, y, end_angle, 0, timestamp=100)

    ################################### 0 degrees
    g_open = 1200
    values = [1024, 2549, 1110, 1400, 0, g_open]
    x = 320
    y = 90
    end_angle = 0 / 180. * np.pi # in pi
    rx150_thread.rx150.gogo(values, x, y, end_angle, 0, timestamp=100)


    for i in (list(range(30)) + list(range(30, -1, -1)))*1:
        values = [1024, 2549, 1110, 1400, 100, g_open]
        rx150_thread.gogo(values, x, y+i*2, end_angle, 0, timestamp=10)
        time.sleep(0.05)


    # for theta in (list(range(10)) + list(range(10, -1, -1)))*1:
    #     theta = theta / 180. * np.pi
    #     values = [1024+theta/np.pi*2048, 2549, 1110, 1400, 0, g_open]
    #     rx150_thread.gogo(values, x / np.cos(theta), y, end_angle, theta/np.pi*2048, timestamp=5)
    #     time.sleep(0.05)

    rx150_thread.running = False
    rx150_thread.join()

if __name__ == "__main__":
    # main_thread()
    main()
