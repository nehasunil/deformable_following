import cv2
import numpy as np


def segmentation(img):
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40, 50, 40])
    upper_green = np.array([80, 255, 255])
    lower_red_1 = np.array([0, 100, 80])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([170, 100, 80])
    upper_red_2 = np.array([180, 255, 255])
    lower_yellow = np.array([20, 150, 80])
    upper_yellow = np.array([40, 255, 255])

    mask_green = cv2.inRange(img_hsv, lower_green, upper_green)
    mask_red = cv2.inRange(img_hsv, lower_red_1, upper_red_1) | cv2.inRange(
        img_hsv, lower_red_2, upper_red_2
    )
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

    mask = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.float32)
    mask[mask_yellow == 255] = [0, 1, 1]
    mask[mask_green == 255] = [0, 1, 0]
    mask[mask_red == 255] = [0, 0, 1]

    return mask
