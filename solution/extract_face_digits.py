import cv2 as cv
import numpy as np
import numpy.typing as npt
from packages.imaging.viewer import show_image
from packages.logging.Logger import logger

from solution.constants import digit_size, struct_width_constant, cube_face_height, cube_face_width
from solution.enums import Axis

from packages.imaging.utils import sliding_window

from scipy import ndimage
from skimage.segmentation import clear_border

import pytesseract

from solution.classify_digits import classify_digit

dilate_kernel = cv.getStructuringElement(cv.MORPH_CROSS, (4, 4))

# [13]
# [33, 38, 49, 51, 62, 66, 101, 102, 117, 125, 129, 149, 150, 156, 158, 159, 161] <-- clearBorder(cell, 2)


# [13]
# [66, 117, 125, 129, 156, 158, 159, 161]  <-- clearBorder(cell, 20)

i = 0


def extract_face_digits(image: npt.NDArray, axis: int):
    global i

    i += 1

    # debug = i >= 11
    # debug = True
    # debug = i >= 10
    debug = False

    method_name = "[extract_face_digits]"

    result = []

    if debug:
        show_image(image, f"{method_name} Image before processing")
        cv.imwrite("./overleaf/extract-face-digits-1.jpg", image)

    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    image = cv.threshold(image, 127, 255, cv.THRESH_BINARY_INV)[1]

    if debug:
        show_image(image, f"{method_name} Image after gray & threshold")
        cv.imwrite("./overleaf/extract-face-digits-2.jpg", image)

    element_horizontal = cv.getStructuringElement(
        cv.MORPH_RECT, (image.shape[1] // 3, 1))
    element_vertical = cv.getStructuringElement(
        cv.MORPH_RECT, (1, image.shape[0] // 3))

    edges_vertical = cv.morphologyEx(image, cv.MORPH_OPEN, element_vertical)
    edges_horizontal = cv.morphologyEx(
        image, cv.MORPH_OPEN, element_horizontal)

    edges = edges_vertical | edges_horizontal

    if debug:
        show_image(edges, f"{method_name} Edges")
        cv.imwrite("./overleaf/extract-face-digits-3.jpg", edges)

    image = image & ~edges

    element_noise = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    if debug:
        show_image(image, f"{method_name} Image after removing lines")
        cv.imwrite("./overleaf/extract-face-digits-4.jpg", image)

    image = cv.morphologyEx(image, cv.MORPH_OPEN, element_noise, iterations=3)

    if debug:
        show_image(image, f"{method_name} Image after removing noise")
        cv.imwrite("./overleaf/extract-face-digits-5.jpg", image)

    # show_image(image, "Init")
    # cv.imwrite("./overleaf/extract_face_digits-imagine_initiala.jpg", image)

    # hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # black_low = np.array([0, 0, 0])
    # black_high = np.array([360, 255, 100])

    # image = cv.inRange(hsv_image, black_low, black_high)

    # if debug:
    #     show_image(image, "hahaha")

    # image = cv.resize(
    #     image, (cube_face_height, cube_face_width), interpolation=cv.INTER_AREA)

    image = cv.resize(image, (image.shape[1], image.shape[1]))

    face_shape = image.shape

    cube_length = int(face_shape[0] / 3)

    if debug:
        show_image(image, f"{method_name} Image after processing")
        # cv.imwrite("./overleaf/extract_face_digits-touching_edges.jpg", image)

    for x, y, cell in sliding_window(image, cube_length, (cube_length, cube_length)):
        if cell.shape[0] != cell.shape[1] or cell.shape[0] != cube_length:
            continue

        # XXX: We really, really do not need this anymore.

        # center = int((2 * x + cube_length) / 2), int((2 * y + cube_length) / 2)

        # top_left_corner = center[0] - eps, center[1] - eps

        # bottom_right_corner = center[0] + eps, center[1] + eps

        if debug:
            image_copy = cv.cvtColor(image, cv.COLOR_GRAY2BGR).copy()

            pad_size = 20

            image_copy = np.pad(image_copy, ((
                pad_size, pad_size), (pad_size, pad_size), (0, 0)), constant_values=0)

            for point in [(x, y), (x + cube_length, y + cube_length)]:
                cv.circle(
                    image_copy, (point[0] + pad_size, point[1] + pad_size), 3, (0, 0, 255), 25)

            # cv.circle(image_copy, center, 5, (255, 255, 255), 8)

            # for point in [top_left_corner, bottom_right_corner]:
            #     cv.circle(image_copy, point, 5, (0, 255, 255), 10)

            show_image(image_copy, f"{method_name} Extracting at these points")
            cv.imwrite("./overleaf/extract-face-digits-6.jpg", image_copy)

        white_pixels_percentage = cv.countNonZero(
            cell) / float(cell.shape[0] * cell.shape[1])

        if white_pixels_percentage < 0.05:
            if debug:
                logger.info(
                    f"{method_name} Adding `None`, {white_pixels_percentage}.")

                show_image(cell, f"{method_name} < 5% white pixels")
                cv.imwrite("./overleaf/extract-face-digits-7.jpg", cell)

            result.append(None)

            continue

        contours, _ = cv.findContours(
            cell, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            if debug:
                logger.info(f"{method_name} Adding `None`.")

            result.append(None)

            continue

        contour = max(contours, key=cv.contourArea)

        if debug:
            img = cv.cvtColor(cell, cv.COLOR_GRAY2BGR)

            cv.drawContours(img, [contour], -1, (0, 0, 255), 5)

            show_image(img, f"{method_name} ROI contour")

            cv.imwrite("./overleaf/extract-face-digits-7.jpg", img)

            cv.imwrite("./overleaf/extract-face-digits-cell.jpg", cell)

        mask = np.zeros_like(cell, dtype="uint8")

        cv.drawContours(
            image=mask,
            contours=[contour],
            contourIdx=-1,
            color=255,
            thickness=-1
        )

        digit = cv.bitwise_or(
            src1=cell,
            src2=cell,
            mask=mask
        )

        x, y, w, h = cv.boundingRect(contour)

        if debug:
            img = cv.cvtColor(cell, cv.COLOR_GRAY2BGR)

            cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 5)

            show_image(img, f"{method_name} Bounding rect. over ROI")

            cv.imwrite("./overleaf/extract-face-digits-8.jpg", img)

        ROI = digit[y: y + h, x: x + w]

        if debug:
            show_image(ROI, f"{method_name} ROI")
            cv.imwrite("./overleaf/extract-face-digits-9.jpg", ROI)

        if axis == Axis.Z:
            ROI = ndimage.rotate(ROI, 45)

            if debug:
                show_image(ROI, f"{method_name} ROI after rotate")
                cv.imwrite("./overleaf/extract-face-digits-10.jpg", ROI)

            contours, _ = cv.findContours(
                image=ROI,
                mode=cv.RETR_EXTERNAL,
                method=cv.CHAIN_APPROX_SIMPLE
            )

            contour = max(contours, key=cv.contourArea)

            x, y, w, h = cv.boundingRect(contour)

            ROI = ROI[y: y + h, x: x + w]

            if debug:
                show_image(ROI, f"{method_name} ROI 2")
                cv.imwrite("./overleaf/extract-face-digits-11.jpg", ROI)

        if debug:
            show_image(ROI, f"{method_name} Region of Interest")

        result.append(cv.resize(ROI, digit_size))

    return result
