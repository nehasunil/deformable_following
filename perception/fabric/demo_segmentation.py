import cv2
import pyk4a
from helpers import colorize
from pyk4a import Config, PyK4A
import deepdish as dd
import numpy as np
from util.segmentation import segmentation
from run import Run
from PIL import Image
import skimage.measure
from copy import deepcopy
import torchvision.transforms as T

model_id = 29
epoch = 200
pretrained_model = "/home/gelsight/Code/Fabric/models/%d/chkpnts/%d_epoch%d" % (
    model_id,
    model_id,
    epoch,
)


def seg_output(depth, model=None):
    max_d = np.nanmax(depth)
    depth[np.isnan(depth)] = max_d
    # depth_min, depth_max = 400.0, 1100.0
    # depth = (depth - depth_min) / (depth_max - depth_min)
    # depth = depth.clip(0.0, 1.0)

    img_depth = Image.fromarray(depth)
    transform = T.Compose([T.ToTensor()])
    img_depth = transform(img_depth)
    img_depth = np.array(img_depth[0])

    out = model.evaluate(img_depth).squeeze()
    seg_pred = out[:, :, :3]

    #         prob_pred *= mask
    # seg_pred_th = deepcopy(seg_pred)
    # seg_pred_th[seg_pred_th < 0.8] = 0.0

    return seg_pred


t1 = Run(model_path=pretrained_model, n_features=3)


def main():
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_1080P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
        )
    )
    k4a.start()

    exp_dict = {-11: 500, -10: 1250, -9: 2500, -8: 8330, -7: 16670, -6: 33330}
    exp_val = -6  # to be changed when running
    k4a.exposure = exp_dict[exp_val]

    data = {"color": [], "depth": []}
    count = 0

    sz = 650
    topleft = [300, 255]

    # crop_x = [421, 960]
    # crop_y = [505, 897]

    bottomright = [topleft[0] + sz, topleft[1] + sz]

    while True:
        capture = k4a.get_capture()

        if capture.color is not None:
            color = capture.color
            color_small = cv2.resize(color, (0, 0), fx=1, fy=1)
            color_small = color_small[
                topleft[1] : bottomright[1], topleft[0] : bottomright[0]
            ]
            cv2.imshow("Color", color_small)

        if capture.transformed_depth is not None:
            depth_transformed = capture.transformed_depth
            depth_transformed_small = cv2.resize(depth_transformed, (0, 0), fx=1, fy=1)
            depth_transformed_small = depth_transformed_small[
                topleft[1] : bottomright[1], topleft[0] : bottomright[0]
            ]
            # cv2.imshow(
            #     "Transformed Depth", colorize(depth_transformed_small, (600, 1100))
            # )

        mask = segmentation(color_small)
        cv2.imshow("mask", mask)

        depth_min = 400.0
        depth_max = 1100.0
        depth_transformed_small_img = (depth_transformed_small - depth_min) / (
            depth_max - depth_min
        )
        depth_transformed_small_img = depth_transformed_small_img.clip(0, 1)

        cv2.imshow("depth", depth_transformed_small_img)

        depth_transformed_100x100 = skimage.measure.block_reduce(
            depth_transformed_small_img, (4, 4), np.mean
        )
        seg_pred = seg_output(depth_transformed_100x100, model=t1)
        # seg_pred_th = deepcopy(seg_pred)
        # seg_pred_th[seg_pred_th < 0.8] = 0.0

        mask = seg_pred.copy()
        # mask = np.zeros(
        #     (seg_pred_th.shape[0], seg_pred_th.shape[1], 3), dtype=np.float32
        # )
        # mask[seg_pred_th[:, :, 1] > 0.8] = [0, 1, 1]
        # mask[seg_pred_th[:, :, 2] > 0.8] = [0, 1, 0]
        # mask[seg_pred_th[:, :, 0] > 0.8] = [0, 0, 1]

        # mask[seg_pred_th[:, :, 1] > 0.8] = [0, 1, 1]
        # mask[seg_pred_th[:, :, 2] > 0.8] = [0, 1, 0]
        # mask[seg_pred_th[:, :, 0] > 0.8] = [0, 0, 1]
        H, W = mask.shape[0], mask.shape[1]
        mask = (
            mask.reshape((H * W, 3))
            @ np.array([[0, 0, 1.0], [0, 1.0, 1.0], [0, 1.0, 0]])
        ).reshape((H, W, 3))
        # mask = 1.0 / (1 + np.exp(-5 * (mask - 0.8)))

        mask = cv2.resize(mask, (sz, sz))

        cv2.imshow("model", mask)

        key = cv2.waitKey(1)
        if key != -1:
            cv2.destroyAllWindows()
            break

    k4a.stop()
    # dd.io.save("data/videos/data_prelim.h5", data)


if __name__ == "__main__":
    main()
