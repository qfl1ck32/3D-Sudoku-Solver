import cv2 as cv

import numpy.typing as npt

from packages.imaging.reader import read_image
from packages.imaging.viewer import show_image

from solution.constants import pattern_matching_digits_path

digit_to_image_map = {}


def generate_digit_to_image_map():
    classes = [i for i in range(1, 10, 1)]

    mp = {k: [] for k in [i for i in range(1, 10, 1)]}

    for cls in classes:
        image = read_image(f"{pattern_matching_digits_path}/{cls}.png")

        mp[cls] = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    return mp


def classify_digit(image: npt.NDArray):
    global digit_to_image_map

    if not digit_to_image_map:
        digit_to_image_map = generate_digit_to_image_map()

    digit_to_score_map = {k: 0 for k in [i for i in range(1, 10, 1)]}

    for i in digit_to_image_map.keys():
        max_probability = 0

        current_template = digit_to_image_map[i]

        probability = cv.matchTemplate(
            image=image,
            templ=current_template,
            method=cv.TM_CCOEFF_NORMED
        )[0][0]

        if probability > max_probability:
            max_probability = probability

        digit_to_score_map[i] = max_probability

    # for key in digit_to_score_map.keys():
    #     print((key, digit_to_score_map.get(key)))

    # show_image(image)
    # cv.imwrite("./overleaf/classify_digit-image.jpg", image)
    # input("a:")

    best_digit = max(digit_to_score_map, key=digit_to_score_map.get)

    # TODO: constant
    if digit_to_score_map[best_digit] < 0.5:
        return 0

    return best_digit
