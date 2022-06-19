from typing import List
import numpy as np
import cv2 as cv
from packages.imaging.viewer import show_image

from solution.types import SudokuPath


def add_circles_for_path(path: SudokuPath, image: np.ndarray, pad_size=0):
    if pad_size != 0:
        img = np.pad(image, ((pad_size, pad_size), (pad_size,
                                                    pad_size), (0, 0)), constant_values=255)
    else:
        img = image.copy()

    for face in path.faces:
        for point in face.coordinates:
            cv.circle(img, (point[0] + pad_size,
                      point[1] + pad_size), 5, (0, 0, 255), 10)

    return img


def add_circles_for_paths(paths: List[SudokuPath], image: np.ndarray, pad_size=0):
    if pad_size != 0:
        img = np.pad(image, ((pad_size, pad_size), (pad_size,
                                                    pad_size), (0, 0)), constant_values=255)
    else:
        img = image.copy()

    faces_same_coordinates = []

    for path in paths:
        for face in path.faces:
            faces_same_coordinates.append(face)

    xx = []

    for path in paths:
        for face in path.faces:
            if faces_same_coordinates.count(face) > 1 and xx.count(face) == 0:
                xx.append(face)

    for index, path in enumerate(paths):
        for face in path.faces:
            for point in face.coordinates:
                X = point[0] + pad_size
                Y = point[1] + pad_size

                cv.circle(img, (X, Y), 5, (0, 255 if index else 0, 255), 25)

    for face in xx:
        for point in face.coordinates:
            X = point[0] + pad_size
            Y = point[1] + pad_size

            cv.circle(img, (X, Y), 5, (0, 127, 255), 25)

    return img
