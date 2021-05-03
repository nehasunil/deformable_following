from threading import Thread
import numpy as np
import socket
import time
import urx

class UR_Controller(Thread):
    def __init__(self, HOST="10.42.0.121", PORT=30003):
        Thread.__init__(self)

        self.rob = urx.Robot(HOST, use_rt=True)

        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((HOST, PORT))

        self.flag_terminate = False

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

        while True:
            p = self.getl_rt()
            diff = np.sum((p - pose)**2)
            if diff < 1e-5:
                break
            time.sleep(0.02)

    def run(self):
        while not self.flag_terminate:
            time.sleep(1)

if __name__ == "__main__":
	urc = UR_Controller()
	urc.start()
	for i in range(10):
		print(urc.getl_rt())
		time.sleep(0.01)

