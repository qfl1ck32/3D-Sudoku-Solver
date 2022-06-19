import cv2 as cv
import numpy as np
from packages.imaging.utils import order_points

from packages.imaging.viewer import show_image
from packages.logging.Logger import logger

from solution.constants import face_image_extractor_cube_area_threshold

good_face_idx = 1
bad_face_idx = 1
i = 0


def extract_face_if_exists(face_image: np.array):
    global i
    global good_face_idx
    global bad_face_idx

    i += 1

    # debug = i >= 42
    # debug = i >= 11
    debug = False

    method_name = "[extract_face_if_exist]"

    # if debug:
    #     show_image(face_image, f"{method_name} Initial face")
    # cv.imwrite("./overleaf/does_face_exist-initial_face.jpg", face_image)

    original_image = face_image.copy()

    hsv_image = cv.cvtColor(original_image, cv.COLOR_BGR2HSV)

    black_low = np.array([0, 0, 0])
    black_high = np.array([360, 255, 100])

    range = cv.inRange(hsv_image, black_low, black_high)

    img = ~range

    if debug:
        show_image(range, f"{method_name} Pixelii din acel range")
        cv.imwrite(
            "./overleaf/extract-face-if-exists-hsv-inrange-result.jpg", hsv_image)

        show_image(hsv_image, f"{method_name} Imaginea HSV")
        cv.imwrite("./overleaf/extract-face-if-exists-hsv.jpg", hsv_image)

    contours = cv.findContours(
        img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]

    contour = max(contours, key=cv.contourArea)

    if debug:
        # show_image(original_image, f"{method_name} Original image")

        img2 = original_image.copy()

        cv.drawContours(img2, [contour], -1, (0, 0, 255), 15)

        show_image(img, f"{method_name} Gray & thresholded image")
        cv.imwrite("./overleaf/extract-face-if-exists-graynthresh.jpg", img)

        show_image(img2, f"{method_name} Contour of face")
        cv.imwrite("./overleaf/extract-face-if-exists-contour.jpg", img2)

    x, y, w, h = cv.boundingRect(contour)

    if debug:
        img = original_image.copy()

        cv.drawContours(img, [contour], -1, (0, 255, 0), 15)

        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 15)

        show_image(img, f"{method_name} Bounding box")

        cv.imwrite("./overleaf/extract-face-if-exists-bounding-box.jpg", img)

    face_image = original_image[y: y + h, x: x + w]

    area = cv.contourArea(contour) / \
        (face_image.shape[0] * face_image.shape[1])

    if area < face_image_extractor_cube_area_threshold:
        if debug:
            logger.info(f"{method_name} {i} No face detected")
        return None, None

    corners = order_points(list(map(lambda c: c[0], contour)), np.float32)

    if debug:
        img = original_image.copy()

        for point in corners:
            cv.circle(img, (int(point[0]), int(
                point[1])), 3, (0, 0, 255), 20)

            cv.imwrite(
                "./overleaf/extract-face-if-exists-corner-points.jpg", img)

        show_image(img, f"{method_name} The identified points")

    width, height = original_image.shape[:2]

    transform_end = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    transformation_matrix = cv.getPerspectiveTransform(corners, transform_end)

    face_image = cv.warpPerspective(
        original_image, transformation_matrix, (width, height),
        borderValue=(255, 255, 255),
        borderMode=cv.BORDER_CONSTANT)

    if debug:
        show_image(face_image, f"{method_name} Face")
        cv.imwrite(
            "./overleaf/extract-face-if-exists-after-warp-perspective.jpg", face_image)

    return cv.resize(face_image, (height, width)), corners
