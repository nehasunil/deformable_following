import time
from datetime import datetime
import numpy as np
import pickle
import os
import cv2
import matplotlib.pyplot as plt

def get_timestamp():
    return datetime.now().strftime("%Y%m%d%H%M%S%f")

def read_logs(filename):
    # data = np.load(filename, allow_pickle=True)
    logs = pickle.load(open(filename, "rb"))
    # gelsight_url = logs[0]['gelsight_url']
    return logs

class Logger():
    def __init__(self):
        # raw data
        self.gelsight = None
        # self.cable_pose = None
        self.ur_velocity = None
        self.ur_pose = None

        # states
        self.x = None
        self.y = None
        self.theta = None
        self.dt = None

        # action
        self.phi = None

        # last_log
        self.last_log = None

        self.data_dir = './data/'
        self.id = '20210611'
        self.img_dir = os.path.join(self.data_dir, 'imgs', self.id)
        self.log_dir = os.path.join(self.data_dir, 'logs', self.id)
        self.logs = []

        self.prefix = get_timestamp()

    def save_img(self):
        dirname = self.img_dir
        timestamp = get_timestamp()

        filename = os.path.join(dirname, "{}_{}.jpg".format(self.prefix, timestamp))
        # cv2.imwrite(filename, self.gelsight)
        return filename

    def update_timestamp(self):
        self.prefix = get_timestamp()

    def save_logs(self):
        if len(self.logs) < 10:
            print("Logs < 10, not saving")
            return

        print("Log length: ", len(self.logs))

        filename = os.path.join(self.log_dir, self.prefix) + '.p'
        # np.savez(filename, logs=self.logs)

        self.update_timestamp()
        pickle.dump(self.logs, open(filename, "wb"))
        self.logs = []

    def add(self):
        self.gelsight_url = self.save_img()

        log = {
            # 'gelsight_url'  : self.gelsight_url,
            # 'cable_pose'    : self.cable_pose,
            'ur_velocity'   : self.ur_velocity,
            'ur_pose'       : self.ur_pose,
            'x'             : self.x,
            'y'             : self.y,
            'theta'         : self.theta,
            # 'phi'           : self.phi,
            'dt'            : self.dt,
        }

        self.logs.append(log)


def update_log(logger):
    logger.gelsight = np.random.random([200,300,3]) * 255
    logger.pose = np.array([1., 1., 1., 2., 2., 2.])

def draw(logs):
    x = []
    y = []
    thetas = []
    last_xy = [0, 0]
    for i in range(0, len(logs)):
        log = logs[i]
        ur_pose = log['ur_pose']
        theta = log['theta']
        cable_center = log['x']


        # dist = np.sum((ur_pose[:2] - last_xy)**2)**0.5
        # print(dist)
        # if (i > 1 and (dist > 1e-3 or dist == 0.)):
        #     last_xy = [ur_pose[0], ur_pose[1]]
        #     continue
        #     print("XXXXX")

        last_xy = [ur_pose[0], ur_pose[1]]
        thetas.append(theta)
        x.append(ur_pose[0])
        y.append(ur_pose[1])

        # x.append(cable_center[0])
        # y.append(cable_center[1])
    # x = x[120:-150]
    # y = y[120:-150]
    # plt.plot(x, y, 'x')
    # plt.plot(thetas)
    plt.figure()
    plt.plot(x, y, 'x')
    plt.figure()
    plt.plot(thetas)
    plt.show()

if __name__ == "__main__":
    # logger = Logger()
    # update_log(logger)
    # logger.add()
    # logger.save_logs()

    # logs = read_logs('../data/logs/1908290930/20190829144117332498.p')
    # logs = read_logs('~/Code/Fabric/src/data/logs/1908290930/20210427.p')

    draw(logs)
    # print(get_timestamp())
