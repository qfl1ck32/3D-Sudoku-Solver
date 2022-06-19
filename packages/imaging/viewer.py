import cv2 as cv

import numpy as np


def show_image(image: np.ndarray, window_name="-", window_size=(480, 480)):
    image = cv.resize(image, window_size)
    cv.imshow(window_name, image)

    cv.waitKey(0)

    cv.destroyWindow(window_name)

    return image
