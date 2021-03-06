from threading import Thread
import numpy as np
import socket
import time
import urx
from scipy.spatial.transform import Rotation as R


class UR_Controller(Thread):
    def __init__(self, HOST="10.42.0.121", PORT=30003):
        Thread.__init__(self)

        self.rob = urx.Robot(HOST, use_rt=True)

        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((HOST, PORT))

        self.flag_terminate = False
        self.pose_following = None

        # Check whether the robot has reach the target pose
        self.last_p = []
        self.len_p = 10
        self.total_error = 0

    def getl_rt(self):
        # get current pose with urx urrtmon with 125 Hz
        return self.rob.rtmon.getTCF(True)

    def send(self, cmd):
        cmd = str.encode(cmd)
        self.s.send(cmd)

    def speedl(self, v, a=0.5, t=0.05):
        # send speedl command in socket
        cmd =  "speedl({}, a={}, t={})\n".format(str(list(v)), a, t)
        self.send(cmd)


    def movel_wait(self, pose, a=1, v=0.04):
        # linear move in tool space and wait
        s = self.s

        cmd = "movel(p{}, a={}, v={})\n".format(str(list(pose)), a, v)
        cmd = str.encode(cmd)
        s.send(cmd)

        last_p = []
        len_p = 20
        quat_goal = R.from_rotvec(pose[3:]).as_quat()
        while True:
            p = self.getl_rt()
            # quat_current = R.from_rotvec(p[3:]).as_quat()
            #
            # error_pos = np.sum((p[:3] - pose[:3])**2)
            # error_ori = np.sum((quat_goal - quat_current)**2)
            #
            # print("pos error", error_pos)
            # print("orientation error", error_ori)
            # diff = error_pos + error_ori

            last_p.append(p.copy())
            if len(last_p) > len_p:
                last_p = last_p[-len_p:]

            diff = np.sum((p - np.mean(last_p, axis=0))**2)

            if len(last_p) == len_p and diff < 1e-12:
                break
            time.sleep(0.02)

    def follow(self):
        if self.pose_following is None:
            return
        cur_pose = self.getl_rt()
        error = self.pose_following - cur_pose
        kp = 5
        ki = 0.1
        self.total_error = self.total_error * 0.9 + error
        v = error * kp + self.total_error * ki
        cmd = np.zeros(6)
        cmd[:3] = v[:3]
        print("cur", cur_pose[2], "goal", self.pose_following[2])
        self.speedl(cmd, a=1, t=0.05)

    def movel_nowait(self, pose, a=1, v=0.04):
        # linear move in tool space and wait
        s = self.s

        cmd = "movel(p{}, a={}, v={})\n".format(str(list(pose)), a, v)
        cmd = str.encode(cmd)
        s.send(cmd)

    def check_stopped(self):
        p = self.getl_rt()

        self.last_p.append(p.copy())
        if len(self.last_p) > self.len_p:
            self.last_p = self.last_p[-self.len_p:]

        diff = np.sum((p - np.mean(self.last_p, axis=0))**2)

        if len(self.last_p) == self.len_p and diff < 1e-12:
            return True

        return False

    def run(self):
        while not self.flag_terminate:
            self.follow()
            time.sleep(0.01)
        self.rob.close()


def main():
	urc = UR_Controller()
	urc.start()

	pose0 = np.array([-0.431, 0.05, 0.21, -2.23, -2.194, -0.019])
	urc.movel_wait(pose0)
	time.sleep(2)
	pose = pose0.copy()
	for i in range(120):
		# print(urc.getl_rt())
		pose[2] = pose0[2] + np.sin(i / 40 * np.pi) * 0.03
		urc.pose_following = pose
		time.sleep(0.05)
	urc.flag_terminate = True
	urc.join()

def test_thread():
	urc = UR_Controller()
	urc.start()

	pose0 = np.array([-0.431, 0.05, 0.21, -2.23, -2.194, -0.019])
	urc.movel_wait(pose0)
	time.sleep(2)
	pose = pose0.copy()
	for i in range(120):
		# print(urc.getl_rt())
		pose[2] = pose0[2] + np.sin(i / 40 * np.pi) * 0.03
		urc.pose_following = pose
		time.sleep(0.05)
	urc.flag_terminate = True
	urc.join()

if __name__ == "__main__":
    test_thread()
