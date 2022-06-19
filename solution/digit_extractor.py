import cv2
import numpy.typing as npt
import numpy as np
from packages.imaging.viewer import show_image

from solution.classify_digits import classify_digit
from solution.extract_face_digits import extract_face_digits

from solution.constants import cube_size
from solution.constants import *


def extract_numbers_from_face(face_image: npt.NDArray, face: int):
    digit_images = extract_face_digits(face_image, face)

    answer = [[0 for _ in range(cube_size)] for _ in range(cube_size)]

    # show_image(face_image, "face")

    # for i, img in enumerate(digit_images):
    #     if img is None:
    #         continue
    #     print(i)
    #     show_image(img, "img")

    #     cv2.imwrite(f"data/pattern_matching/digits_2/{g}.png", img)

    #     g += 1

    # x = input()

    for row in range(cube_size):
        for column in range(cube_size):
            digit_image = digit_images[3 * row + column]

            if digit_image is not None:
                answer[row][column] = classify_digit(digit_image)
                # answer[row][column] = 0 if digit_image is None else digit_image

    return np.array(answer)
