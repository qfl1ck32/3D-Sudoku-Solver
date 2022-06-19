import math
import numpy as np
import cv2 as cv
from packages.imaging.viewer import show_image

from solution.enums import EdgeType


def extract_points_for_edges(image: np.ndarray, struct: np.ndarray, type: EdgeType, struct_original_sizes: np.ndarray, vertical_struct_sizes: tuple = None):
    first_path_points = []
    second_path_points = []

    gray_bitwise_not_image = cv.bitwise_not(
        cv.cvtColor(image, cv.COLOR_RGB2GRAY))

    contours, _ = cv.findContours(
        gray_bitwise_not_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    debug = False

    if debug:
        c = image.copy()

        cv.drawContours(c, contours, -1, (0, 0, 255), 2)

        # cv.imwrite(f"./overleaf/extract_points_for_edges-{type}-contur.jpg", c)

        show_image(c, f"Contour? - {len(contours)}")

    gray_bitwise_not_image = cv.cvtColor(
        gray_bitwise_not_image, cv.COLOR_GRAY2BGR)

    struct_distance = np.sqrt(
        struct_original_sizes.shape[0] ** 2 + struct_original_sizes.shape[1] ** 2)

    # TODO: the other way around?
    height, width = struct_original_sizes.shape[:2]

    if type == EdgeType.VERTICAL:
        for contour in contours:
            top = np.array(contour[contour[:, :, 1].argmin()][0])
            bottom = np.array(contour[contour[:, :, 1].argmax()][0])

            distance = np.linalg.norm(top - bottom)

            number_of_lines = round(distance / struct_distance)
            # print((top, bottom, struct_distance))
            # print(number_of_lines)
            # show_image(image)

            starting_point = top

            # TODO: should be [0, i * height], right?
            for i in range(number_of_lines):
                dist_to_add = np.array(
                    [width, i * height])

                first_path_points.append(
                    starting_point + dist_to_add)

                second_path_points.append(
                    starting_point + dist_to_add - (width, 0))

                for point in [first_path_points[-1], second_path_points[-1]]:
                    cv.circle(i, point, 3, (0, 0, 255), 3)

    else:
        for contour in contours:
            top = np.array(contour[contour[:, :, 1].argmin()][0])
            bottom = np.array(contour[contour[:, :, 1].argmax()][0])

            distance = np.linalg.norm(top - bottom)

            number_of_lines = round(distance / struct_distance)

            prev_point = bottom if type == EdgeType.SLASH else top

            sign = 1 if type == EdgeType.BACKSLASH else -1

            m = np.tan(np.deg2rad(sign * 30))

            c = np.array([prev_point[1] - m * prev_point[0]])

            def y(x):
                return m * x + c

            for i in range(number_of_lines):
                first_path_points.append(prev_point)

                next_orientation_point = prev_point - \
                    [0, -sign * vertical_struct_sizes[1]]

                second_path_points.append(next_orientation_point)

                x = int(prev_point[0] + 0.96 * width)

                prev_point = np.array([x, int(y(x))])

    return [tuple(item) for item in first_path_points], \
        [tuple(item) for item in second_path_points]
