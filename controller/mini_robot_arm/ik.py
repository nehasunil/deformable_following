import cv2
import numpy as np
from math import sin, cos, pi
import time
import random

img = np.zeros([300, 300])

l2 = 150
lT = 50
l3 = 150


def fk(t2, t3):
    x0 = (150, 0)
    x1 = (x0[0] + l2 * sin(t2), x0[1] + l2 * cos(t2))
    x2 = (x1[0] + lT * cos(t2), x1[1] - lT * sin(t2))
    x3 = (x2[0] + l3 * cos(t2 + t3), x2[1] - l3 * sin(t2 + t3))
    return x3


def draw_fk(t2, t3):
    img = np.zeros([300, 500, 3])
    x0 = (150, 0)
    x1 = (x0[0] + l2 * sin(t2), x0[1] + l2 * cos(t2))
    x2 = (x1[0] + lT * cos(t2), x1[1] - lT * sin(t2))
    x3 = (x2[0] + l3 * cos(t2 + t3), x2[1] - l3 * sin(t2 + t3))
    cv2.line(img, x0, (int(x1[0]), int(x1[1])), (255, 0, 0), 3)
    cv2.line(img, (int(x1[0]), int(x1[1])), (int(x2[0]), int(x2[1])), (255, 0, 0), 3)
    cv2.line(img, (int(x2[0]), int(x2[1])), (int(x3[0]), int(x3[1])), (255, 0, 0), 3)
    return img


def get_JaccobianTranspose(t2, t3):
    J = np.array(
        [
            [l2 * cos(t2) - lT * sin(t2) - l3 * sin(t2 + t3), -l3 * sin(t2 + t3)],
            [-l2 * sin(t2) - lT * cos(t2) - l3 * cos(t2 + t3), -l3 * cos(t2 + t3)],
        ]
    )
    return J.T


def ik(end_angle, x, y, O_):
    angle = end_angle
    l_end = 150.0
    fix_end_angle = -0.3

    x -= l_end * cos(angle)
    y += l_end * sin(angle)

    if len(O_) == 3:
        O = O_[:-1]
    else:
        O = O_
    alpha = 0.00001

    i = 0

    while True:
        i += 1

        V = fk(O[0, 0], O[1, 0])
        JT = get_JaccobianTranspose(O[0, 0], O[1, 0])

        dV = np.array([[x - V[0]], [y - V[1]]])
        O = O + alpha * JT.dot(dV)

        if (dV ** 2).sum() < 1e-4:
            break

    return np.array(
        [O[0].tolist(), O[1].tolist(), [-O[0, 0] - O[1, 0] + angle - fix_end_angle]]
    )


tm = time.time()
N = 360
end_angle = 0.2

O = np.array([[0], [0]])
O0 = np.array([[0], [0]])

for t in range(360):
    x, y = 300 + sin(t / 180 * pi * 10) * 50, 100 + cos(t / 180 * pi * 10) * 30
    O = ik(end_angle, x, y, O)
    # img = draw_fk(O[0, 0], O[1, 0])
    # cv2.circle(img, (int(x), int(y)), 20, (0, 0, 255), 2)
    # cv2.imshow("img", img[::-1])
    # cv2.waitKey(1)

# for i in range(N):
#     t2, t3 = random.random() * pi, random.random()*pi
#     # x, y = fk(-50/180*pi, -50/180*pi)
#     x, y = fk(t2, t3)
#     O = ik(x, y)
#     # print(t2, t3, O)
#     V = fk(O[0,0], O[1,0])
#     print((t2-O[0,0])**2+(t3-O[1,0])**2,
#         (V[0]-x)**2+(V[1]-y)**2)
# print((time.time()-tm)/N)

# img = draw_fk(-20/180*pi, -20/180*pi)
# cv2.imshow(
#     "img",
#     img[
#         ::-1,
#     ],
# )
# cv2.waitKey(0)
