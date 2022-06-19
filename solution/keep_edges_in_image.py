from re import S
from typing import List
import numpy as np
import cv2
from packages.imaging.utils import rotate_bound

from packages.imaging.viewer import show_image
from solution.constants import KEEP_EDGES_IN_IMAGE_THRESHOLD_KEY, thresholds_and_stuff


def keep_edges_in_image(image: np.ndarray, structure: np.ndarray, angles: List[int]):
    best_answer = None
    best_angle = None
    best_struct = None

    best_percentage = 0

    # cv2.imwrite("./overleaf/keep_edges_in_image-original_image.jpg", image)

    for angle in angles:
        current_structure = rotate_bound(structure, angle)
        current_structure[current_structure != 255] = 0

        # show_image(current_structure, "Structura")

        # cv2.imwrite("./overleaf/keep_edges_in_image-structure.jpg",
        #             cv2.resize(current_structure, (640, 640)))

        closed = cv2.morphologyEx(
            image, cv2.MORPH_CLOSE, current_structure, iterations=1)

        # closed = cv2.dilate(image, current_structure)

        # show_image(closed, "after close")

        # cv2.imwrite("./overleaf/keep_edges_in_image-dilated.jpg",
        #             cv2.resize(closed, (640, 640)))

        # closed = cv2.erode(closed, current_structure)

        # show_image(closed, "after close 2")

        # cv2.imwrite(
        #     "./overleaf/keep_edges_in_image-dilated-then-eroded.jpg", cv2.resize(closed, (640, 640)))

        # grayscaled = cv2.cvtColor(closed, cv2.COLOR_BGR2GRAY)

        # show_image(grayscaled, "Gray")

        thresholded = cv2.threshold(
            closed, thresholds_and_stuff[KEEP_EDGES_IN_IMAGE_THRESHOLD_KEY], 255, cv2.THRESH_BINARY)[1]

        # show_image(thresholded, "Thresh")

        answer = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)

        content_percentage: float = np.sum(
            answer == 0) / float(answer.shape[0] * answer.shape[1])

        # cv2.imwrite(
        #     f"./overleaf/keep_edges_in_image-exemplu_slash-{angle}-{non_zeros_percentage}.jpg", answer)

        if content_percentage > best_percentage:
            best_percentage = content_percentage
            best_angle = angle
            best_answer = answer
            best_struct = current_structure

    return best_answer, best_angle, best_struct
