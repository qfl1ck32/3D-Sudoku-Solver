import math
import cv2
import numpy as np

from packages.imaging.viewer import show_image


def rotate_bound(image, angle, borderValue=None):
    (h, w) = image.shape[:2]
    (cx, cy) = (w/2, h/2)

    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h*sin)+(w*cos))
    nH = int((h*cos)+(w*sin))

    M[0, 2] += (nW/2) - cx
    M[1, 2] += (nH/2) - cy

    return cv2.warpAffine(image, M, (nW, nH), borderValue=borderValue)


def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2]  # image shape has 3 dimensions
    # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    image_center = (width/2, height/2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(
        mat, rotation_mat, (bound_w, bound_h), borderValue=(255, 255, 255))
    return rotated_mat


def sliding_window(image: np.ndarray, step: int, window_size: tuple[int, int], eps=0):
    window_x, window_y = window_size[0], window_size[1]
    image_height, image_width = image.shape[0], image.shape[1]

    for y in range(0, image_height, step):
        for x in range(0, image_width, step):
            yield x, y, image[y: y + window_y + eps, x: x + window_x + eps]


def shift_image(image: np.array, shift_x: int, shift_y: int, borderValue=(255, 255, 255)):
    width, height = image.shape[:2]

    M = np.float32([
        [1, 0, shift_x],
        [0, 1, shift_y]
    ])

    return cv2.warpAffine(image, M, (height, width), borderMode=cv2.BORDER_CONSTANT, borderValue=borderValue)


def rotate_coordinates(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    return [int(qx), int(qy)]


def add_text_at_point(img: np.ndarray, txt: str, point: tuple, circle_size=48, text_scale=1.5, text_thickness=2, text_face=cv2.FONT_HERSHEY_DUPLEX, color=(0, 0, 255)):
    cv2.circle(img, point, circle_size, color, -1)

    text_size, _ = cv2.getTextSize(
        txt, text_face, text_scale, text_thickness)

    text_origin = (int(point[0] - text_size[0] / 2),
                   int(point[1] + text_size[1] / 2))

    cv2.putText(img, txt, text_origin, text_face,
                text_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)


def get_new_coordinates_after_image_resize(original_size: np.ndarray, new_size: np.ndarray, original_coordinate):
    original_size = np.array(original_size[:2])

    new_size = np.array(new_size[:2])

    original_coordinate = np.array(original_coordinate)

    new_coordinates = original_coordinate / (original_size / new_size)

    return (int(new_coordinates[0]), int(new_coordinates[1]))


def unpad_coordinates(coordinates, pad_size):
    return coordinates[0] - pad_size, coordinates[1] - pad_size


def order_points(pts, dtype=np.int32):
    pts = np.array(pts)
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype=dtype)
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    rect = np.array(pts, dtype=np.float32)

    # obtain a consistent order of the points and unpack them
    # individually
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight),
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    # return the warped image
    return warped
