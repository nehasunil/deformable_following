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
    # cable_xy = []
    fabric_x0 = []
    fabric_y0 = []
    thetas = []
    ur_v = []
    x = []
    y = []
    dt = []

    for i in range(20, len(logs)):
        log = logs[i]
        ur_pose = log['ur_pose']
        theta = log['theta']
        fabric_x0 = log['x']
        fabric_y0 = log['y']
        ur_velocity = log['ur_velocity']
        t_diff = log['dt']

        thetas.append(theta)
        ur_xy.append([ur_pose[0], ur_pose[1]]) 
        # cable_xy.append(cable_center)
        ur_v.append([ur_velocity[0], ur_velocity[1]])
        dt.append(t_diff)

        x.append(ur_pose[0])
        y.append(ur_pose[1])

    # plt.plot(x, y, 'x')
    # plt.plot(thetas)
    # plt.figure()
    # plt.plot(x, y, 'x')
    # plt.figure()
    # plt.plot(thetas)
    # plt.show()
    return ur_xy, fabric_x0, fabric_y0, thetas, ur_v, dt

def remove_end(x):
    return x[1:-1]

def parse(logs):
    # pixel_size = 0.2e-3 
    # pixel_size = 0.1e-3 # caliper + gelsight examples
    pixel_size = 0.2e-3
    ur_xy, fabric_x0, fabric_y0, thetas, ur_v, dt = draw(logs)

    # pose0 = np.array([-0.556, -0.227, 0.092])
    # pose0 = np.array([-0.539, -0.226, 0.092])
    pose0 = np.array([-0.505, -0.219, 0.235])
    fixpoint_x = pose0[0] #+ 0.006 #0.0178 # - 15e-3
    fixpoint_y = pose0[1] - 0.12 #0.039 #0.0382 # 0.2 measure distance between 2 grippers at pose0
    fabric_x = 0.2 - fabric_x0 + 0.5*(1-2*fabric_y0)*np.tan(thetas)
    alpha = np.arctan((np.array(ur_xy)[:, 0] - fixpoint_x) / (np.array(ur_xy)[:, 1] - fixpoint_y)) * np.cos(np.pi * 30 / 180)

    ur_v = np.asarray(ur_v)
    # beta = np.arcsin((ur_v[:,0])/((ur_v[:,0]**2+ur_v[:,1]**2)**0.5)) # use test case to check signs
    beta = np.arcsin((ur_v[:, 0]) * np.cos(np.pi*30/180) / (((ur_v[:, 0]*np.cos(np.pi*30/180)) ** 2 + ur_v[:, 1] ** 2) ** 0.5))

    # print("UR_V", ur_v[:,:2], "BETA", beta)

    # for i in range(5):
    dt.pop(0)
    dt.append(np.mean(dt))
    dt = np.array(dt)

    # print(cable_xy.shape)
    # x = gaussian_filter1d(cable_real_xy[:,1], 2)
    x = gaussian_filter1d(fabric_x*pixel_size, 30)

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

    for i in range(5):
        x = remove_end(x)
        theta = remove_end(theta)
        phi = remove_end(phi)
        x_dot = remove_end(x_dot)
        theta_dot = remove_end(theta_dot)
        alpha = remove_end(alpha)
        alpha_dot = remove_end(alpha_dot)
    for i in range(4):
        dt = remove_end(dt)


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
    # X = np.hstack([np.array([x]).T, np.array([phi]).T])
    # print(X.shape, Y.shape)

    return X, Y

def loadall():
    X, Y = np.empty((0,4), np.float32), np.empty((0,3), np.float32)
    # for filename in glob.glob('logs/1908290930/*.p'):
    # for filename in glob.glob('data/logs/20210503/*.p'):
    for filename in glob.glob('data/20210611_30/*.p'):
        # print(filename)
        logs = read_logs(filename)
        try:
            x, y = parse(logs)
            X = np.vstack([X, x])
            Y = np.vstack([Y, y])
        except:
            print(filename)
        # break
    # for filename in glob.glob('data/logs/initialtesting/linearGPLQR/*.p'):
    #     logs = read_logs(filename)
    #     x, y = parse(logs)
    #     X = np.vstack([X, x])
    #     Y = np.vstack([Y, y])
    # for filename in glob.glob('data/logs/initialtesting/linearLQR/*.p'):
    #     logs = read_logs(filename)
    #     x, y = parse(logs)
    #     X = np.vstack([X, x])
    #     Y = np.vstack([Y, y])
    # for filename in glob.glob('data/logs/initialtesting/tvLQR/*.p'):
    #     logs = read_logs(filename)
    #     x, y = parse(logs)
    #     X = np.vstack([X, x])
    #     Y = np.vstack([Y, y])

    print(X.shape, Y.shape)
    return X, Y

if __name__ == "__main__":
    loadall()

