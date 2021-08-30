import os
import pickle
import random

import cv2
import numpy as np
from numpy import cos, pi, sin
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from util.fast_poisson import poisson_reconstruct

from scipy.interpolate import griddata
from util import helper


def dilate(img, ksize=5):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)


def erode(img, ksize=5):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.erode(img, kernel, iterations=1)


def interpolate_grad(img, mask):
    # mask = (soft_mask > 0.5).astype(np.uint8) * 255
    # cv2.imshow("mask_hard", mask)
    # pixel around markers
    mask_around = (dilate(mask, ksize=3) > 0) & (mask != 1)
    # mask_around = mask == 0
    mask_around = mask_around.astype(np.uint8)
#     cv2.imshow("mask_around", mask_around * 255)

    x, y = np.arange(img.shape[0]), np.arange(img.shape[1])
    yy, xx = np.meshgrid(y, x)

    # mask_zero = mask == 0
    mask_zero = mask_around == 1
    mask_x = xx[mask_zero]
    mask_y = yy[mask_zero]
    points = np.vstack([mask_x, mask_y]).T
    values = img[mask_x, mask_y]
    markers_points = np.vstack([xx[mask != 0], yy[mask != 0]]).T
    method = "nearest"
    # method = "linear"
    # method = "cubic"
    x_interp = griddata(points, values, markers_points, method=method)
    x_interp[x_interp != x_interp] = 0.0
    ret = img.copy()
    ret[mask != 0] = x_interp
    return ret


def demark(diff, gx, gy):
#     cv2.imshow("diff_before", diff / 255.0)

    mask = helper.find_marker(diff, threshold_list=(40, 40, 40))
#     cv2.imshow("mask", mask * 255.0)

    # gx[mask == 1] = 0
    # gy[mask == 1] = 0
    gx = interpolate_grad(gx, mask)
    gy = interpolate_grad(gy, mask)

    return gx, gy


class Class_3D:
    def __init__(self, model_fn="LUT.pkl", features_type="RGBXY"):
        self.features_type = features_type
        self.output_dir = "output/"
        self.load_lookup_table(model_fn)

    def load_lookup_table(self, model_fn):
        self.model = pickle.load(open(model_fn, "rb"))

    def infer(
        self,
        img,
        resolution=(0, 0),
        save_id=None,
        refine_gxgy=None,
        demark=None,
        display=True,
        reverse_RB=False,
    ):
        if reverse_RB:
            img = img[:, :, ::-1]
        if resolution != (0, 0):
            img = cv2.resize(img, resolution)

        if "G" not in self.features_type:
            img[:, :, 1] = 127.5
        if "R" not in self.features_type:
            img[:, :, 2] = 127.5
        if "B" not in self.features_type:
            img[:, :, 0] = 127.5

        W, H = img.shape[0], img.shape[1]

        X = np.reshape(img, [W * H, 3]) / 255 - 0.5

        x = np.arange(W)
        y = np.arange(H)
        yy, xx = np.meshgrid(y, x)
        xx, yy = np.reshape(xx, [W * H, 1]), np.reshape(yy, [W * H, 1])
        xx, yy = xx / W - 0.5, yy / H - 0.5

        X_features = [X, xx, yy]
        X = np.hstack(X_features)

        Y = self.model.predict(X)

        gx = np.reshape(Y[:, 0], [W, H])
        gy = np.reshape(Y[:, 1], [W, H])

        if demark is not None:
            gx, gy = demark(img, gx, gy)

        gx_raw, gy_raw = gx.copy(), gy.copy()

        if refine_gxgy is not None:
            print("before: gx.max() gy.max()", gx.max(), gy.max())
            gy_raw *= 0.0
            gx, gy = refine_gxgy(gx, gy)
            print("gx.max() gy.max()", gx.max(), gy.max())

        gx_img = (np.clip(gx / 2 + 0.5, 0, 1) * 255).astype(np.uint8)
        gy_img = (np.clip(gy / 2 + 0.5, 0, 1) * 255).astype(np.uint8)

        zeros = np.zeros_like(gx)
        depth = poisson_reconstruct(gx, gy, zeros)
        depth = cv2.resize(depth, (H, W))
        depth_single = poisson_reconstruct(gx_raw, gy_raw, zeros)
        depth_single = cv2.resize(depth_single, (H, W))
#         print(depth.max())
        # depth_img = (np.clip(depth*255/15-80, 0, 255)).astype(np.uint8)
        scale = W / 200
        # depth_img = (np.clip(depth * 255 / 12 / scale - 50, 0, 255)).astype(np.uint8)
        # depth_single_img = (
        #     np.clip(depth_single * 255 / 12 / scale - 50, 0, 255)
        # ).astype(np.uint8)

        depth_img = (np.clip(depth * 255 / 24 / scale + 63, 0, 255)).astype(np.uint8)
        depth_single_img = (
            np.clip(depth_single * 255 / 24 / scale + 63, 0, 255)
        ).astype(np.uint8)

        depth_img_heat = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)

        if display:
            pass
            # cv2.imshow(
            #     "img", np.clip((img - 127.5) * 2 + 127.5, 0, 255).astype(np.uint8)
            # )

            cv2.imshow("gx", gx_img)
            cv2.imshow("gy", gy_img)

            # cv2.imshow("depth_calibrated_heat", depth_img_heat)
            # cv2.imshow("depth_single_img_gray", depth_single_img)
            # cv2.imshow("depth_calibrated_gray", depth_img)

        # depth = img2depth(img)
        # depth_img = (np.clip(depth*255*2.5, 0, 255)).astype(np.uint8)
        # # depth_img = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)

        # img[:, :, 1] = img[:, :, 0]
        # img[:, :, 2] = img[:, :, 0]

        # cv2.imshow('depth_est', depth_img)
        if save_id is not None:
            self.save_img(
                raw=img,
                gx_img=gx_img,
                gy_img=gy_img,
                z_gray=depth_img,
                z_heat=depth_img_heat,
                save_id=save_id,
            )
        return depth, gx, gy

    def save_img(
        self,
        raw=None,
        gx_img=None,
        gy_img=None,
        z_gray=None,
        z_heat=None,
        combine=True,
        save_id=None,
    ):
        if raw is not None:
            output_fn = os.path.join(self.output_dir, save_id + "_depth_calibrated.jpg")
            cv2.imwrite(output_fn, raw)

        if gx_img is not None:
            output_fn = os.path.join(self.output_dir, save_id + "_gx.jpg")
            cv2.imwrite(output_fn, gx_img)

        if gy_img is not None:
            output_fn = os.path.join(self.output_dir, save_id + "_gy.jpg")
            cv2.imwrite(output_fn, gy_img)

        if z_gray is not None:
            output_fn = os.path.join(self.output_dir, save_id + "_depth_gray.jpg")
            cv2.imwrite(output_fn, z_gray)

        if z_heat is not None:
            output_fn = os.path.join(self.output_dir, save_id + "_depth_heat.jpg")
            cv2.imwrite(output_fn, z_heat)

        if combine is True:
            # image sizes
            W, H = raw.shape[0], raw.shape[1]
            interval = 50

            # create canvas
            combine_img = (
                np.ones([W * 2 + interval * 2, H * 3 + interval * 2, 3], dtype=np.uint8)
                * 255
            )

            # image placement
            xy_list = [
                (0, 0),
                (W + interval, H + interval),
                (0, H + interval),
                (0, (H + interval) * 2),
                (W + interval, (H + interval) * 2),
            ]
            image_list = [raw, gx_img, gy_img, z_gray, z_heat]

            # draw to canvas
            for i in range(len(image_list)):
                image = image_list[i]
                x, y = xy_list[i]

                # gray to RGB
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

                print("image.shape", image.shape)
                combine_img[x : x + W, y : y + H] = image

            output_fn = os.path.join(self.output_dir, save_id + "_combine.jpg")
            cv2.imwrite(output_fn, combine_img)


if __name__ == "__main__":
    c3d = Class_3D()

    img = cv2.imread("cali_data/20200930_164410.jpg")
    c3d.infer(img)
    cv2.waitKey(0)
