import numpy as np
from math import cos, sin, pi
from .fast_poisson import poisson_reconstruct

import cv2



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
