import cv2
import pyk4a
from helpers import colorize
from pyk4a import Config, PyK4A
import deepdish as dd
import numpy as np
from util.segmentation import segmentation
import os


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

    dir_data = "data/imgs/0625_10k"
    os.makedirs(dir_data, exist_ok=True)

    sz = 600
    topleft = [350, 305]
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
            cv2.imshow(
                "Transformed Depth", colorize(depth_transformed_small, (600, 900))
            )
        data["color"].append(color_small)
        data["depth"].append(depth_transformed_small)

        if count > 1000:
            break
        if count % 100 == 0:
            print(count)
        count += 1

        key = cv2.waitKey(5)
        if key != -1:
            cv2.destroyAllWindows()
            break

        mask = segmentation(color_small)
        cv2.imshow("mask", mask)

        cv2.imwrite(f"{dir_data}/color_{count}.jpg", color_small)
        depth_min = 600.0
        depth_max = 900.0
        depth_transformed_small_img = (
            (depth_transformed_small - depth_min) / (depth_max - depth_min) * 255
        )
        depth_transformed_small_img = depth_transformed_small_img.clip(0, 255)
        cv2.imwrite(
            f"{dir_data}/depth_{count}.jpg",
            depth_transformed_small_img.astype(np.uint8),
        )
        np.save(f"{dir_data}/depth_{count}.npy", depth_transformed_small)

    k4a.stop()
    # dd.io.save("data/videos/data_prelim.h5", data)


if __name__ == "__main__":
    main()
