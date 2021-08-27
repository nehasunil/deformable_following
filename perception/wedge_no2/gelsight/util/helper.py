import cv2
import numpy as np


def draw_circle(img, cx, cy, r):
    cimg = img.copy()
    # draw circle boundary
    cv2.circle(cimg, (cy, cx), r, (255, 255, 255), -1)
    cimg = img * 1.0 - cimg / 6
    cimg = np.clip(cimg, 0, 255).astype(np.uint8)
    # draw the center of the circle
    cv2.circle(cimg, (cy, cx), 2, (0, 0, 255), 3)
    return cimg


def label_circle(img, cx=200, cy=150, r=10):
    # label the circle info: r (radius), cx (center x), cy (center y)
    dx, dy, dr = 1, 1, 1
    while True:
        cimg = draw_circle(img, cx, cy, r)
        cv2.imshow("label_circle", cimg)

        c = cv2.waitKey(1)
        if c == ord("q") or c == 27:
            # save
            return cx, cy, r
        elif c == ord("w"):
            # Up
            cx -= dx
        elif c == ord("s"):
            # Down
            cx += dx
        elif c == ord("a"):
            # Left
            cy -= dy
        elif c == ord("d"):
            # Right
            cy += dy
        elif c == ord("="):
            # Increase radius
            r += dr
        elif c == ord("-"):
            # Decrese radius
            r -= dr


def find_marker(frame, threshold_list=(40, 40, 40)):
    RESCALE = 400.0 / frame.shape[0]
    frame_small = frame
    
    kern = round(63 / RESCALE)
    if kern%2 == 0:
        kern += 1
        
    # Blur image to remove noise
    blur = cv2.GaussianBlur(frame_small, (kern, kern), 0)
    
    # Subtract the surrounding pixels to magnify difference between markers and background
    diff = blur - frame_small.astype(np.float32)
    
    diff *= 16.0
    diff[diff < 0.0] = 0.0
    diff[diff > 255.0] = 255.0
    
    kern = round(31 / RESCALE)
    if kern%2 == 0:
        kern += 1
    
    diff = cv2.GaussianBlur(diff, (kern, kern), 0)
    # cv2.imshow("diff_marker", diff / 255.0)

    mask = (
        (diff[:, :, 0] > threshold_list[0])
        & (diff[:, :, 2] > threshold_list[1])
        & (diff[:, :, 1] > threshold_list[2])
    )

    # mask = (diff[:, :, 0] > 0) & (diff[:, :, 2] > 0) & (diff[:, :, 1] > 0)

    # def sigmoid(x, rng=1.0):
    #     return 1.0 / (1 + np.exp(-x * rng))

    # soft_mask = np.mean(sigmoid(diff - 30, rng=0.1), axis=-1)
    # cv2.imshow("soft_mask", soft_mask)

    mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]))
    # mask = cv2.resize((mask * 255).astype(np.uint8), (frame.shape[1], frame.shape[0]))

    # mask = erode(mask, ksize=5)
    # mask = dilate(mask, ksize=5)
    return mask
