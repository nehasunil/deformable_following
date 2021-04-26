import numpy as np
from math import cos, sin, pi

from .fast_poisson import poisson_reconstruct

import pickle

import cv2


def img2grad(frame0, frame):

    diff = frame * 1.0 - frame0

    # dx = (diff[:,:,2] * cos(pi/6) - diff[:,:,0] * cos(pi/6)) / 255.
    # dy = (diff[:,:,0] * sin(pi/6) + diff[:,:,2] * sin(pi/6) - diff[:,:,1]) / 255.

    dx = (diff[:, :, 1] - (diff[:, :, 0] + diff[:, :, 2]) * 0) / 255.0
    dy = (diff[:, :, 2] - diff[:, :, 0] * 2) / 255.0
    # dy = -(diff[:,:,0] - (diff[:,:,1]  + diff[:,:,2]) * 1) / 255.

    dx = dx / (1 - dx ** 2) ** 0.5 / 32 * 2
    dy = dy / (1 - dy ** 2) ** 0.5 / 32

    # cv2.imshow('dx',dx*32+0.5)
    # cv2.imshow('dy',dy*32+0.5)

    return dx, dy


def img2depth(frame0, frame):
    dx, dy = img2grad(frame0, frame)

    zeros = np.zeros_like(dx)
    depth = poisson_reconstruct(dy, dx * 0, zeros)

    # dx_poisson, dy_poisson = np.gradient(depth)
    # dy[dy_poisson>1] = dy_poisson[dy_poisson>1]

    # depth = poisson_reconstruct(dy, dx, zeros)

    return depth

    # return dx, dy


# bias = model.predict([[np.dstack([np.zeros([48, 48], dtype=np.float32), np.zeros([48, 48], dtype=np.float32)])]])


def img2depth_nn_dy(frame0, frame):

    frame_small = frame
    frame0_small = frame0
    dx, dy = img2grad(frame0_small, frame_small)

    dx = dx / 4 * 0
    dy = dy / 4

    pred = model.predict([[np.dstack([dy, dx])]]) - bias
    zeros = np.zeros_like(dx)
    depth = poisson_reconstruct(dy * 2, pred[0, :, :, 1] * 2, zeros)

    # dx = dx / 4
    # dy = dy / 4 * 0
    # pred = model.predict([[np.dstack([dx, dy])]]) - bias

    # zeros = np.zeros_like(dx)
    # depth = poisson_reconstruct(pred[0,:,:,1] * 2, dx * 2, zeros)

    # depth = poisson_reconstruct(pred[0,:,:,0] * 4, pred[0,:,:,1] * 4, zeros)
    # depth = poisson_reconstruct(dy, dx, zeros)

    # cv2.imshow('dy', cv2.resize(dy*40+0.5, (300, 300)))
    # cv2.imshow('dx_predict', cv2.resize(pred[0,:,:,1]*40+0.5, (300, 300)))

    return depth


def img2depth_nn(frame0, frame):
    LUT = pickle.load(open("LUT.pkl", "rb"))

    # frame_small = cv2.resize(frame, (48, 48))
    # frame0_small = cv2.resize(frame0, (48, 48))
    frame_small = frame
    frame0_small = frame0
    # dx, dy = img2grad(frame0_small, frame_small)

    # dx = dx / 32 * 0
    # dy = dy / 128.

    img = frame * 1.0 - frame0 + 127

    W, H = img.shape[0], img.shape[1]

    X = np.reshape(img, [W * H, 3])
    Y = LUT.predict(X)

    dy = np.reshape(Y[:, 0], [W, H])
    dx = np.reshape(Y[:, 1], [W, H])

    print(dx.max(), dx.min())

    dx = dx / 256.0 / 32 * 0
    dy = dy / 256.0 / 32

    pred = model.predict([[np.dstack([dy, dx])]])
    z_reshape = np.reshape(pred[0], [frame_small.shape[0], frame_small.shape[1]])

    print("MAX", z_reshape.max())

    return z_reshape * 4.0



def warp_perspective(img, corners, output_sz=(210, 270)):
    TOPLEFT, TOPRIGHT, BOTTOMLEFT, BOTTOMRIGHT = corners

    WARP_W = output_sz[0]
    WARP_H = output_sz[1]

    points1=np.float32([TOPLEFT,TOPRIGHT,BOTTOMLEFT,BOTTOMRIGHT])
    points2=np.float32([[0,0],[WARP_W,0],[0,WARP_H],[WARP_W,WARP_H]])

    matrix=cv2.getPerspectiveTransform(points1,points2)

    result = cv2.warpPerspective(img, matrix, (WARP_W, WARP_H))

    return result

def ini_frame(frame):
    frame_rect = warp_perspective(raw_img, (252, 137), (429, 135), (197, 374), (500, 380))
    return frame
