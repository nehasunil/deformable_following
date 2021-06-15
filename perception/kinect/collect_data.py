import cv2
import pyk4a
from helpers import colorize
from pyk4a import Config, PyK4A
import numpy as np


def main():
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
        )
    )
    k4a.start()

    exp_dict = {-11: 500, -10: 1250, -9: 2500, -8: 8330, -7: 16670, -6: 33330}
    exp_val = -7  # to be changed when running
    k4a.exposure = exp_dict[exp_val]

    id = 0

    while True:
        capture = k4a.get_capture()

        if capture.color is not None:
            color = capture.color
            cv2.imshow("Color", color)

        if capture.transformed_depth is not None:
            depth_transformed = capture.transformed_depth
            cv2.imshow(
                "Transformed Depth", colorize(depth_transformed, (600, 1100))
            )

        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite(f"data/intrinsic_test/color_{id}.png", color)
            id += 1
        elif key == ord('q'):
            cv2.destroyAllWindows()
            break

    k4a.stop()


if __name__ == "__main__":
    main()
