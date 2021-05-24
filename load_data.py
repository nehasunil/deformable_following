import numpy as np
import pickle
import glob
import matplotlib.pyplot as plt
from  scipy.ndimage.filters import gaussian_filter1d

def read_logs(filename):
    logs = pickle.load(open(filename, "rb"))
    # gelsight_url = logs[0]['gelsight_url']
    return logs

def draw(logs):
    ur_xy = []
    cable_xy = []
    thetas = []
    ur_v = []
    x = []
    y = []
    dt = []

    for i in range(30, len(logs)):
        log = logs[i]
        ur_pose = log['ur_pose']
        theta = log['theta']
        cable_center = log['x']
        ur_velocity = log['ur_velocity']
        t_diff = log['dt']

        thetas.append(theta)
        ur_xy.append([ur_pose[0], ur_pose[1]])
        cable_xy.append(cable_center)
        ur_v.append([ur_velocity[0], ur_velocity[1]])
        dt.append(t_diff)

        x.append(ur_pose[0])
        y.append(ur_pose[1])

    return ur_xy, cable_xy, thetas, ur_v, dt

def remove_end(x):
    return x[1:-1]

def parse(logs):
    pixel_size = 0.2e-3
    ur_xy, cable_xy, thetas, ur_v, dt = draw(logs)

    pose0 = np.array([-0.539, -0.226, 0.092])
    fixpoint_x = pose0[0] + 0.006 # - 15e-3
    fixpoint_y = pose0[1] - 0.039 # 0.2 measure distance between 2 grippers at pose0
    cable_xy = np.array(cable_xy)
    # cable_xy = np.zeros((len(cable_y), 2))
    # for y in range(len(cable_y)):
    #     cable_xy[y,1] = cable_y[y]
    # cable_xy[:,1] *= -1
    cable_real_xy = np.array(ur_xy) + np.array([0., -0.039]) + cable_xy*pixel_size
    alpha = np.arctan((cable_real_xy[:,0] - fixpoint_x)/(cable_real_xy[:,1] - fixpoint_y))

    ur_v = np.asarray(ur_v)
    beta = np.arcsin((ur_v[:,0])/((ur_v[:,0]**2+ur_v[:,1]**2)**0.5)) # use test case to check signs

    # print("UR_V", ur_v[:,:2], "BETA", beta)

    dt.pop(0)
    dt.append(np.mean(dt))
    dt = np.array(dt)

    # print(cable_xy.shape)
    # x = gaussian_filter1d(cable_real_xy[:,1], 2)
    x = gaussian_filter1d(cable_xy[:,0]*pixel_size, 30)

    # phi = gaussian_filter1d(alpha - np.array(thetas),2)
    # phi = gaussian_filter1d(beta - alpha,2)
    phi = gaussian_filter1d(beta - np.array(thetas), 2)

    # print("UR_xy", cable_real_xy, "UR_V", ur_v, "X", x)
    # print("BETA", beta, "ALPHA", alpha, "PHI", phi)

    theta = gaussian_filter1d(np.array(thetas),2)
    theta_ = gaussian_filter1d(np.array(thetas),30)
    x_dot = gaussian_filter1d(np.gradient(x)/np.mean(dt), 30)#dt
    theta_dot = gaussian_filter1d(np.gradient(theta_)/np.mean(dt), 30)#/dt
    alpha_dot = gaussian_filter1d(np.gradient(alpha)/np.mean(dt), 30)#/dt

    x = remove_end(x)
    theta = remove_end(theta)
    phi = remove_end(phi)
    x_dot = remove_end(x_dot)
    theta_dot = remove_end(theta_dot)
    alpha = remove_end(alpha)
    alpha_dot = remove_end(alpha_dot)


    # plt.figure()
    # plt.plot(alpha)

    # plt.figure()
    # plt.plot(beta)


    # plt.figure()
    # ur_xy = np.array(ur_xy)
    # plt.plot(ur_xy[:, 0], ur_xy[:, 1], 'x')

    # plt.show()

    # print(alpha.shape, x.shape)
    X = np.hstack([np.array([x]).T, np.array([theta]).T, np.array([alpha]).T, np.array([phi]).T])
    Y = np.hstack([np.array([x_dot]).T, np.array([theta_dot]).T, np.array([alpha_dot]).T])
    # print(X.shape, Y.shape)

    return X, Y

def loadall():
    X, Y = np.empty((0,4), np.float32), np.empty((0,3), np.float32)
    # for filename in glob.glob('logs/1908290930/*.p'):
    # for filename in glob.glob('data/logs/20210503/*.p'):
    #     logs = read_logs(filename)
    #     try:
    #         x, y = parse(logs)
    #         X = np.vstack([X, x])
    #         Y = np.vstack([Y, y])
    #     except:
    #         print(filename)
    for filename in glob.glob('data/logs/20210520/*.p'):
        logs = read_logs(filename)
        try:
            x, y = parse(logs)
            X = np.vstack([X, x])
            Y = np.vstack([Y, y])
        except:
            print(filename)
    print(X.shape, Y.shape)
    return X, Y

if __name__ == "__main__":
    loadall()
