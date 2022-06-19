from collections import defaultdict
import pickle
import threading
import imutils
import cv2 as cv

from matplotlib.pyplot import contour

import pytesseract

import numpy as np

from packages.debugging.add_circles_for_path import add_circles_for_path
from packages.imaging.reader import read_image

from packages.imaging.utils import add_text_at_point, four_point_transform, get_new_coordinates_after_image_resize, order_points, rotate_bound, rotate_coordinates, rotate_image, shift_image, unpad_coordinates
from packages.imaging.viewer import show_image

from solution.calculate_shifts import calculate_shifts
from solution.create_path_map import calculate_iou
from solution.exceptions import NumberOfIdentifiedFacesIsWrong, check_thick_lines_points_have_been_identified_correctly

from solution.extract_face_if_exists import extract_face_if_exists
from solution.extract_points_for_edges import extract_points_for_edges
from solution.add_padding_to_coordinates import get_face_coordinates, add_padding_to_coordinates
from solution.keep_edges_in_image import keep_edges_in_image

from solution.outer_cube_extractor import extract_outer_cube

from solution.types import Point, SquareCoordinates, SudokuFace, SudokuPath
from solution.constants import STRUCT_SIZES_ERR_KEY, cube_size, iou_threshold, struct_width_constant, thresholds_and_stuff, digits_files_path
from solution.enums import EdgeType, Axis, PathDirection

from packages.logging.Logger import logger


from solution.constants import paths_file_path, pattern_matching_digits_path, cubed_cube_size, cube_face_height, cube_face_width

from typing import Callable, Dict, List

pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/Cellar/tesseract/5.1.0/bin/tesseract'


def _create_struct(size: tuple[int]):
    struct = cv.getStructuringElement(cv.MORPH_RECT, size)
    struct[struct == 1] = 255

    return struct

def _unpad_image(image, pad_size, rgb=True):
    return image[pad_size: -pad_size, pad_size: -pad_size, :] if rgb else image[pad_size: -pad_size, pad_size: -pad_size]


class SudokuCV:
    original_image: np.ndarray
    contour_image: np.ndarray

    A: Point
    B: Point
    C: Point
    D: Point
    E: Point
    F: Point
    G: Point

    struct: np.ndarray
    vertical_struct: np.ndarray
    slash_struct: np.ndarray
    backslash_struct: np.ndarray

    transform_end: np.ndarray

    vertical_edges_left_points: List[SquareCoordinates]
    vertical_edges_right_points: List[SquareCoordinates]

    backslash_edges_right_points: List[SquareCoordinates]
    backslash_edges_down_points: List[SquareCoordinates]

    slash_edges_up_points: List[SquareCoordinates]
    slash_edges_down_points: List[SquareCoordinates]

    shift_horizontal_x: Point
    shift_vertical_x: Point

    shift_horizontal_y: Point
    shift_vertical_y: Point

    shift_horizontal_z: Point
    shift_vertical_z: Point

    paths: List[SudokuPath]

    current_path: SudokuPath

    coordinates_to_faces_map: Dict[tuple, SudokuFace]

    faces_coordinates: List[tuple]

    all_edges_image: np.ndarray

    _extract_calls: int

    steps: int

    current_step: int

    _current_cube_index: int

    verbose: bool

    def __init__(self, image: np.ndarray, current_cube_index=1, verbose=False):
        self._current_cube_index = current_cube_index

        self.steps = 9
        self.current_step = 1

        # self.contour_image = extract_outer_cube(image)
        self.contour_image = imutils.resize(image, width=640)
        self.original_image = self.contour_image

        self.contour_image = cv.cvtColor(self.contour_image, cv.COLOR_BGR2GRAY)

        self.current_path = SudokuPath()

        self.paths = []

        self.coordinates_to_faces_map = dict()
        self.faces_coordinates = []

        self._extract_calls = 0

        self.verbose = verbose

    def log(self, message: str):
        if self.verbose:
            logger.info(message)

    def setup(self):
        height, width = self.contour_image.shape[:2]

        self.A = 0, int(height / 4)
        self.B = int(width / 2), 0
        self.C = width, int(height / 4)
        self.D = width, int(3 * height / 4)  # sad D
        self.E = int(width / 2), height
        self.F = 0, int(3 * height / 4)
        self.G = int(width / 2), int(height / 2)

        self.shift_horizontal_x, self.shift_vertical_x = calculate_shifts(
            [self.A, self.G, self.F])
        self.shift_horizontal_y, self.shift_vertical_y = calculate_shifts(
            [self.G, self.C, self.E])
        self.shift_horizontal_z, self.shift_vertical_z = calculate_shifts(
            [self.A, self.B, self.G])

        # self.A, self.B, self.C, self.D, self.E, self.F, self.G = res

        # print(self.shift_horizontal_x, self.shift_vertical_x)
        # print(self.shift_horizontal_y, self.shift_vertical_y)
        # print(self.shift_horizontal_z, self.shift_vertical_z)

        # print(self.A, self.B, self.C, self.D, self.E, self.F, self.G)

        # self.shift_horizontal_x, self.shift_vertical_x = calculate_shifts(
        #     [self.A, self.G, self.F])
        # self.shift_horizontal_y, self.shift_vertical_y = calculate_shifts(
        #     [self.G, self.C, self.E])
        # self.shift_horizontal_z, self.shift_vertical_z = calculate_shifts(
        #     [self.A, self.B, self.G])

        # print(self.shift_horizontal_x, self.shift_vertical_x)
        # print(self.shift_horizontal_y, self.shift_vertical_y)
        # print(self.shift_horizontal_z, self.shift_vertical_z)

        # i = self.contour_image.copy()

        # for point in [self.A, self.B, self.C, self.D, self.E, self.F, self.G]:
        #     cv.circle(i, point, 3, (0, 255, 0), 3)

        # show_image(i, "Puncte")

        self.transform_end = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")

    def log(self, message: str):
        if self.verbose:
            logger.info(message)

    def extract_thick_edges(self):
        self.log(f"{self.get_step_message()} Extracting thick edges")

        height, width = self.contour_image.shape[:2]

        pad_size = 50

        # # vertical edges right points
        # v = [(621, 892), (621, 1116), (621, 443)]

        # a = (621, 892)

        # b = a[0] - self.shift_horizontal_y[0], a[1] - \
        #     self.shift_horizontal_y[1]
        # c = a[0] - self.shift_vertical_y[0], a[1] - self.shift_vertical_y[1]
        # d = a[0] - self.shift_horizontal_y[0] - self.shift_vertical_y[0], a[1] - \
        #     self.shift_horizontal_y[1] - self.shift_vertical_y[1]

        # i = self.contour_image.copy()

        # for z in [b, c, d]:
        #     cv.circle(i, z, 5, (0, 0, 255), 14)

        # cv.circle(i, a, 5, (0, 0, 255), 35)

        # show_image(i)

        # # cv.imwrite("./overleaf/calculate_shifts-exemplu_start.jpg", i)
        # cv.imwrite("./overleaf/calculate_shifts-exemplu_plasare_puncte.jpg", i)

        # self.contour_image = np.pad(self.contour_image, ((
        #     pad_size, pad_size), (pad_size, pad_size), (0, 0)), constant_values=255)

        # for index, point in enumerate([self.A, self.B, self.C, self.D, self.E, self.F, self.G]):
        #     point = point[0] + pad_size, point[1] + pad_size
        #     cv.circle(self.contour_image, point, 25, (0, 0, 255), -1)

        #     TEXT_FACE = cv.FONT_HERSHEY_DUPLEX
        #     TEXT_SCALE = 1
        #     TEXT_THICKNESS = 1
        #     TEXT = chr(ord("A") + index)

        #     text_size, _ = cv.getTextSize(
        #         TEXT, TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
        #     text_origin = (int(point[0] - text_size[0] / 2),
        #                    int(point[1] + text_size[1] / 2))

        #     cv.putText(self.contour_image, TEXT, text_origin, TEXT_FACE,
        #                TEXT_SCALE, (255, 255, 255), TEXT_THICKNESS, cv.LINE_AA)

        # for text, point in [["0, 0", (pad_size, pad_size)],
        #                     ["0, h", (pad_size, pad_size + width)],
        #                     ["h, w",
        #                      (height - pad_size, width + pad_size)],
        #                     ["0, w", (height - pad_size, pad_size)]]:
        #     add_text_at_point(self.contour_image, text,
        #                       point, 38, 1, 1, color=(0, 0, 0))

        # cv.imwrite("./overleaf/sudoku_cv-setup_varfuri.jpg",
        #            self.contour_image)

        # input("stop: ")

        padded_contour_image = np.pad(self.contour_image, (
            (pad_size, pad_size),
            (pad_size, pad_size)),
            'constant',
            constant_values=255)

        height_of_structs = int(height / 6)

        width_of_structs = int(struct_width_constant * height_of_structs)

        # original_contour_image_size = padded_contour_image.shape

        # padded_contour_image = imutils.resize(padded_contour_image, width=600)

        self.struct = _create_struct(
            (int(width_of_structs * thresholds_and_stuff[STRUCT_SIZES_ERR_KEY]), int(height_of_structs * thresholds_and_stuff[STRUCT_SIZES_ERR_KEY]) - 20))

        # self.struct = _create_struct((10, 40))
        struct_original_sizes = _create_struct(
            (width_of_structs, height_of_structs))

        # Slash
        slash_edges_image, angle, self.slash_struct = keep_edges_in_image(
            padded_contour_image, self.struct, [63, 62, 61, 60, 59, 58, 57])

        slash_struct_original_sizes = rotate_bound(
            struct_original_sizes, angle)
        slash_struct_original_sizes[slash_struct_original_sizes != 255] = 0

        slash_edges_image = _unpad_image(slash_edges_image, pad_size)

        # cv.imwrite("./overleaf/sudoku_cv-setup-slash_struct.jpg",
        #            cv.bitwise_not(self.slash_struct))

        # show_image(cv.addWeighted(self.contour_image, 0.8,
        #                           slash_edges_image, 0.8, 1), "Slash")

        self.slash_edges_down_points, self.slash_edges_up_points = extract_points_for_edges(
            slash_edges_image, self.slash_struct, EdgeType.SLASH, slash_struct_original_sizes, struct_original_sizes.shape)

        # i = cv.addWeighted(self.contour_image.copy(), 0.2, slash_edges_image, 0.8, 1)

        # for point in self.slash_edges_down_points + self.slash_edges_up_points:
        #     cv.circle(i, point, 2, (0, 0, 255), 5)

        # show_image(i, "Abc")

        # cv.imwrite(
        #     "./overleaf/extract_points_for_edges-puncte_imagine_slash.jpg", i)

        # Vertical
        vertical_edges_image, angle, self.vertical_struct = keep_edges_in_image(
            padded_contour_image, self.struct, [0, -1, 1, -2, 2, -3, 3])

        vertical_edges_image = _unpad_image(vertical_edges_image, pad_size)

        vertical_struct_original_sizes = rotate_bound(
            struct_original_sizes, angle)
        vertical_struct_original_sizes[vertical_struct_original_sizes != 255] = 0

        self.vertical_edges_right_points, self.vertical_edges_left_points = extract_points_for_edges(
            vertical_edges_image, self.vertical_struct, EdgeType.VERTICAL, vertical_struct_original_sizes)

        # i = cv.addWeighted(self.contour_image.copy(), 0.2, vertical_edges_image, 0.8, 1)

        # for point in self.vertical_edges_left_points + self.vertical_edges_right_points:
        #     cv.circle(i, point, 2, (0, 0, 255), 5)

        # show_image(i, "Abc")

        # cv.imwrite(
        #     "./overleaf/extract_points_for_edges-puncte_imagine_verticala.jpg", i)

        # cv.imwrite("./overleaf/sudoku_cv-setup-vertical_struct.jpg",
        #            cv.bitwise_not(self.vertical_struct))

        # show_image(cv.addWeighted(self.contour_image, 0.8,
        #                           vertical_edges_image, 0.8, 1), "Vertical")

        backslash_edges_image, angle, self.backslash_struct = keep_edges_in_image(
            padded_contour_image, self.struct, [-57, -58, -59, -60, -61, -62, -63])

        backslash_struct_original_sizes = rotate_bound(
            struct_original_sizes, angle)
        backslash_struct_original_sizes[backslash_struct_original_sizes != 255] = 0

        backslash_edges_image = _unpad_image(
            backslash_edges_image, pad_size)

        self.backslash_edges_right_points, self.backslash_edges_down_points = extract_points_for_edges(
            backslash_edges_image, self.backslash_struct, EdgeType.BACKSLASH, backslash_struct_original_sizes, struct_original_sizes.shape)

        # i = cv.addWeighted(self.contour_image.copy(), 0.2, backslash_edges_image, 0.8, 1)

        # for point in self.backslash_edges_down_points + self.backslash_edges_right_points:
        #     cv.circle(i, point, 2, (0, 0, 255), 5)

        # show_image(i, "Abc")

        # cv.imwrite(
        #     "./overleaf/extract_points_for_edges-puncte_imagine_backslash.jpg", i)

        # cv.imwrite("./overleaf/sudoku_cv-setup-backslash_struct.jpg",
        #            cv.bitwise_not(self.backslash_struct))

        # show_image(cv.addWeighted(self.contour_image, 0.8,
        #                           backslash_edges_image, 0.8, 1), "Backslash")

        debug = False

        check_thick_lines_points_have_been_identified_correctly(
            self.vertical_edges_left_points,
            self.vertical_edges_right_points,
            self.backslash_edges_right_points,
            self.backslash_edges_down_points,
            self.slash_edges_up_points,
            self.slash_edges_down_points
        )

        # self.all_edges_image = backslash_edges_image & slash_edges_image & vertical_edges_image

        # all = cv.addWeighted(self.contour_image,
        #                      0.2, self.all_edges_image, 0.8, 1)

        # points = []

        # points.extend(self.vertical_edges_left_points)
        # points.extend(self.vertical_edges_right_points)
        # points.extend(self.backslash_edges_down_points)
        # points.extend(self.backslash_edges_right_points)
        # points.extend(self.slash_edges_down_points)
        # points.extend(self.slash_edges_up_points)

        # for a in points:
        #     cv.circle(all, a, 2, (0, 255, 0), 6)

        # show_image(all, f"All - {self._current_cube_index}")

        # cv.imwrite("./overleaf/all-edges-in-image-original.jpg",
        #            self.contour_image)
        # cv.imwrite("./overleaf/all-edges-in-image.jpg", all)

        # cv.imwrite(
        #     f"./data/solution/complete/{self._current_cube_index}-points-and-edges.jpg", all)

    def get_step_message(self):
        text = f"[{self.current_step} / {self.steps}]"

        self.current_step += 1

        return text

    def extract_paths(self):
        # self.log(
        #     f"{self.get_step_message()} Identifying and extracting the faces")
        # self.generate_coordinates_to_faces_map()

        self.log(
            f"{self.get_step_message()} Extracting vertical right paths")

        # Vertical
        for a in self.vertical_edges_right_points:
            self.extract_path(self.extract_vertical_right_path_get_first_coordinates, Axis.Y,
                              self.extract_vertical_right_path_get_second_coordinates, Axis.X, self.extract_vertical_right_path_get_next_points, a)

            self.store_current_path(
                PathDirection.HORIZONTAL)

        self.log(
            f"{self.get_step_message()} Extracting vertical left paths")

        for b in self.vertical_edges_left_points:
            self.extract_path(self.extract_vertical_left_path_get_first_coordinates, Axis.Y,
                              self.extract_vertical_left_path_get_second_coordinates, Axis.X, self.extract_vertical_left_path_get_next_point, b)

            self.store_current_path(
                PathDirection.HORIZONTAL)

        # Slash
        self.log(f"{self.get_step_message()} Extracting slash up paths")
        for a in self.slash_edges_up_points:
            self.extract_path(self.extract_slash_up_path_get_first_coordinates, Axis.Y,
                              self.extract_slash_up_path_get_second_coordinates, Axis.Z, self.extract_slash_up_path_get_next_point, a)

            self.store_current_path(PathDirection.VERTICAL)

        self.log(f"{self.get_step_message()} Extracting slash down paths")

        for a in self.slash_edges_down_points:
            self.extract_path(self.extract_slash_down_path_get_first_coordinates, Axis.Y,
                              self.extract_slash_down_path_get_second_coordinates, Axis.Z, self.extract_slash_down_path_get_next_point, a)

            self.store_current_path(PathDirection.VERTICAL)

        self.log(
            f"{self.get_step_message()} Extracting backslash right paths")

        # Backslash
        for a in self.backslash_edges_right_points:
            self.extract_path(self.extract_backslash_right_path_get_first_coordinates, Axis.Z,
                              self.extract_backslash_right_path_get_second_coordinates, Axis.X, self.extract_backslash_right_path_get_next_point, a)

            is_only_x = all(
                face.axis == Axis.X for face in self.current_path.faces)

            direction = PathDirection.HORIZONTAL

            if is_only_x:
                direction = PathDirection.VERTICAL

            self.store_current_path(direction)

        self.log(
            f"{self.get_step_message()} Extracting backslash down paths")

        for b in self.backslash_edges_down_points:
            self.extract_path(self.extract_backslash_down_path_get_first_coordinates, Axis.Z,
                              self.extract_backslash_down_path_get_second_coordinates, Axis.X, self.extract_backslash_down_path_get_next_point, b)

            direction = PathDirection.VERTICAL

            is_only_z = all(
                face.axis == Axis.Z for face in self.current_path.faces)

            if is_only_z:
                direction = PathDirection.HORIZONTAL

            self.store_current_path(direction)

    def extract_digits_for_faces(self):
        self.log(f"{self.get_step_message()} Extracting digits")

        i = 0
        for path in self.paths:
            for face in path.faces:
                digits = face.extract_digits()

                if digits is not None:
                    print(digits)
                    i += 1

                    if i >= 0:
                        show_image(face.face_image, f"Face {i} [{len(digits.nonzero()[0])}]")

    i = 0

    def extract_path(self, get_coordinates_1: Callable, axis_1: Axis, get_coordinates_2: Callable, axis_2: Axis, get_next_point: Callable, current_point: Point):
        self.i += 1

        if self.should_stop_path_extraction_recursion():
            return

        # self.i == 45

        coordinates = get_coordinates_1(current_point)

        axis = axis_1

        face_image, true_coordinates = self.extract_face_given_coordinates(coordinates, axis)

        # pad_size = 250

        # img = np.pad(self.contour_image.copy(), ((pad_size, pad_size),
        #              (pad_size, pad_size), (0, 0)), constant_values=255)

        # for point in coordinates:
        #     cv.circle(img, (point[0] + pad_size,
        #               point[1] + pad_size), 4, (0, 0, 255), 20)

        # cv.circle(img, (current_point[0] + pad_size,
        #           current_point[1] + pad_size), 4, (0, 0, 255), 40)

        # show_image(img, f"First coordinates - {face}")

        # cv.imwrite(f"./overleaf/extract_path-{self.i}-face-{face}.jpg", img)

        if face_image is None:
            axis = axis_2
            coordinates = get_coordinates_2(current_point)

            face_image, true_coordinates = self.extract_face_given_coordinates(coordinates, axis)

            # pad_size = 250

            # img = np.pad(self.contour_image.copy(), ((
            #     pad_size, pad_size), (pad_size, pad_size), (0, 0)), constant_values=255)

            # for point in coordinates:
            #     cv.circle(img, (point[0] + pad_size,
            #               point[1] + pad_size), 4, (0, 0, 255), 20)

            # cv.circle(
            #     img, (current_point[0] + pad_size, current_point[1] + pad_size), 4, (0, 0, 255), 40)

            # show_image(img, f"Second coordinates - {face}")

            # cv.imwrite(
            #     f"./overleaf/extract_path-{self.i}-face-{face}.jpg", img)

        self.add_face_to_current_path(
            SudokuFace(true_coordinates, axis, face_image))

        return self.extract_path(get_coordinates_1, axis_1, get_coordinates_2, axis_2, get_next_point, get_next_point(coordinates, axis))

    # Slash down
    def extract_slash_down_path_get_first_coordinates(self, a: Point):
        b = (a[0] - self.shift_horizontal_y[0],
             a[1] - self.shift_horizontal_y[1])
        c = (a[0] - self.shift_vertical_y[0] - self.shift_horizontal_y[0],
             a[1] - self.shift_vertical_y[1] - self.shift_horizontal_y[1])
        d = (a[0] - self.shift_vertical_y[0], a[1] - self.shift_vertical_y[1])

        return (a, b, c, d)

    def extract_slash_down_path_get_second_coordinates(self, a: Point):
        b = (a[0] - self.shift_horizontal_z[0],
             a[1] - self.shift_horizontal_z[1])
        c = (a[0] - self.shift_vertical_z[0] - self.shift_horizontal_z[0],
             a[1] - self.shift_vertical_z[1] - self.shift_horizontal_z[1])
        d = (a[0] - self.shift_vertical_z[0], a[1] - self.shift_vertical_z[1])

        return (a, b, c, d)

    def extract_slash_down_path_get_next_point(self, coordinates: SquareCoordinates, _):
        (_, _, _, d) = coordinates

        return d

    # Slash up
    def extract_slash_up_path_get_first_coordinates(self, d: Point):
        a = (d[0] + self.shift_vertical_y[0], d[1] + self.shift_vertical_y[1])
        b = (a[0] - self.shift_horizontal_y[0],
             a[1] - self.shift_horizontal_y[1])
        c = (a[0] - self.shift_vertical_y[0] - self.shift_horizontal_y[0],
             a[1] - self.shift_vertical_y[1] - self.shift_horizontal_y[1])

        return (a, b, c, d)

    def extract_slash_up_path_get_second_coordinates(self, d: Point):
        a = (d[0] + self.shift_vertical_z[0], d[1] + self.shift_vertical_z[1])
        b = (a[0] - self.shift_horizontal_z[0],
             a[1] - self.shift_horizontal_z[1])
        c = (a[0] - self.shift_vertical_z[0] - self.shift_horizontal_z[0],
             a[1] - self.shift_vertical_z[1] - self.shift_horizontal_z[1])

        return (a, b, c, d)

    def extract_slash_up_path_get_next_point(self, coordinates: SquareCoordinates, _):
        (a, _, _, _) = coordinates

        return a

    # Vertical right
    def extract_vertical_right_path_get_first_coordinates(self, a: Point):
        b = (a[0] - self.shift_horizontal_y[0],
             a[1] - self.shift_horizontal_y[1])
        c = (a[0] - self.shift_vertical_y[0] - self.shift_horizontal_y[0],
             a[1] - self.shift_vertical_y[1] - self.shift_horizontal_y[1])
        d = (a[0] - self.shift_vertical_y[0], a[1] - self.shift_vertical_y[1])

        return(a, b, c, d)

    def extract_vertical_right_path_get_second_coordinates(self, a: Point):
        b = (a[0] - self.shift_horizontal_x[0],
             a[1] - self.shift_horizontal_x[1])
        c = (a[0] - self.shift_horizontal_x[0] - self.shift_vertical_x[0],
             a[1] - self.shift_vertical_x[1] - self.shift_horizontal_x[1])
        d = (a[0] - self.shift_vertical_x[0], a[1] - self.shift_vertical_x[1])

        return (a, b, c, d)

    def extract_vertical_right_path_get_next_points(self, coordinates: SquareCoordinates, _):
        (_, b, _, _) = coordinates

        return b

    # Vertical left

    def extract_vertical_left_path_get_first_coordinates(self, b: Point):
        a = (b[0] + self.shift_horizontal_y[0],
             b[1] + self.shift_horizontal_y[1])
        c = (a[0] - self.shift_vertical_y[0] - self.shift_horizontal_y[0],
             a[1] - self.shift_vertical_y[1] - self.shift_horizontal_y[1])
        d = (a[0] - self.shift_vertical_y[0], a[1] - self.shift_vertical_y[1])

        return (a, b, c, d)

    def extract_vertical_left_path_get_second_coordinates(self, b: Point):
        a = (b[0] + self.shift_horizontal_x[0],
             b[1] + self.shift_horizontal_x[1])
        c = (a[0] - self.shift_horizontal_x[0] - self.shift_vertical_x[0],
             a[1] - self.shift_vertical_x[1] - self.shift_horizontal_x[1])
        d = (a[0] - self.shift_vertical_x[0], a[1] - self.shift_vertical_x[1])

        return (a, b, c, d)

    def extract_vertical_left_path_get_next_point(self, coordinates: SquareCoordinates, _):
        (a, _, _, _) = coordinates

        return a

    # Backslash right
    def extract_backslash_right_path_get_first_coordinates(self, a: Point):
        b = (a[0] - self.shift_horizontal_z[0],
             a[1] - self.shift_horizontal_z[1])
        c = (a[0] - self.shift_vertical_z[0] - self.shift_horizontal_z[0],
             a[1] - self.shift_vertical_z[1] - self.shift_horizontal_z[1])
        d = (a[0] - self.shift_vertical_z[0], a[1] - self.shift_vertical_z[1])

        return (a, b, c, d)

    def extract_backslash_right_path_get_second_coordinates(self, d: Point):
        a = (d[0] + self.shift_vertical_x[0],
             d[1] + self.shift_vertical_x[1])
        b = (a[0] - self.shift_horizontal_x[0],
             a[1] - self.shift_horizontal_x[1])
        c = (a[0] - self.shift_horizontal_x[0] - self.shift_vertical_x[0],
             a[1] - self.shift_vertical_x[1] - self.shift_horizontal_x[1])
        # d = (a[0] - self.shift_vertical_x[0], a[1] - self.shift_vertical_x[1])

        return (a, b, c, d)

    def extract_backslash_right_path_get_next_point(self, coordinates: SquareCoordinates, face: Axis):
        (a, b, _, _) = coordinates

        return b if face == Axis.Z else a

    # Backslash left
    def extract_backslash_down_path_get_first_coordinates(self, b: Point):
        a = (b[0] + self.shift_horizontal_z[0],
             b[1] + self.shift_horizontal_z[1])
        c = (a[0] - self.shift_vertical_z[0] - self.shift_horizontal_z[0],
             a[1] - self.shift_vertical_z[1] - self.shift_horizontal_z[1])
        d = (a[0] - self.shift_vertical_z[0], a[1] - self.shift_vertical_z[1])

        return (a, b, c, d)

    def extract_backslash_down_path_get_second_coordinates(self, a: Point):
        b = (a[0] - self.shift_horizontal_x[0],
             a[1] - self.shift_horizontal_x[1])
        c = (a[0] - self.shift_horizontal_x[0] - self.shift_vertical_x[0],
             a[1] - self.shift_vertical_x[1] - self.shift_horizontal_x[1])
        d = (a[0] - self.shift_vertical_x[0], a[1] - self.shift_vertical_x[1])

        return (a, b, c, d)

    def extract_backslash_down_path_get_next_point(self, coordinates: SquareCoordinates, face: Axis):
        (a, _, _, d) = coordinates

        return a if face == Axis.Z else d

    # Next stuff
    def extract_face_given_coordinates(self, coordinates: SquareCoordinates, axis: Axis):
        self._extract_calls += 1

        # debug = len(self.coordinates_to_faces_map.keys()) == 27
        # debug = self._extract_calls >= 42
        # debug = len(self.paths) >= 14
        debug = False

        # print(self._extract_calls)

        method_name = "[extract_face_given_coordinates]"

        a, b, c, d = coordinates

        a, b, c, d = add_padding_to_coordinates(a, b, c, d, axis)

        if debug:
            self.log(f"{method_name} Axis: {axis}")

            cpy = self.contour_image.copy()

            for point in [a, b, c, d]:
                cv.circle(cpy, point, 3, (0, 0, 255), 15)

            show_image(
                cpy, f"{method_name} Extracting here")
            cv.imwrite("./overleaf/extract-face-coordinates.jpg", cpy)

        for cached_coordinates in self.coordinates_to_faces_map.keys():
            if calculate_iou(cached_coordinates, coordinates) > iou_threshold:
                face = self.coordinates_to_faces_map[cached_coordinates]

                # transformation = cv.getPerspectiveTransform(
                #     np.array([a, b, c, d], dtype=np.float32), self.transform_end)

                # face_image = cv.warpPerspective(
                #     self.contour_image, transformation, (
                #         self.contour_image.shape[1], self.contour_image.shape[0]),
                #     borderValue=(255, 255, 255),
                #     borderMode=cv.BORDER_CONSTANT)

                # show_image(face_image, "Asa arata la coordonatele calculate")
                # cv.imwrite(
                #     "./overleaf/extract_face_given_coordinates-not_cached-image.jpg", face_image)

                # show_image(face.face_image, "Cached")

                # cv.imwrite(
                #     "./overleaf/extract_face_given_coordinates-cached-image.jpg", face.face_image)

                if debug:
                    self.log(
                        f"{method_name} Got cached version for {coordinates}, at {cached_coordinates}")

                    show_image(face.face_image, f"{method_name} Cached face")

                return face.face_image, face.coordinates

        # TODO: this optimisation should work. It doesn't :|
        # if len(self.coordinates_to_faces_map.keys()) == 27:
        #     if debug:
        #         logger.info(
        #             f"{method_name} All the faces have been identified, and the given coordinates do not intersect with any.")
        #     return None

        transformation_matrix = cv.getPerspectiveTransform(
            np.array([a, b, c, d], dtype=np.float32), self.transform_end)
        
        face_image = cv.warpPerspective(
            self.original_image, transformation_matrix, (
                self.contour_image.shape[1], self.contour_image.shape[0]),
            borderValue=(255, 255, 255),
            borderMode=cv.BORDER_CONSTANT)

        if debug:
            show_image(face_image, "Extracted image")
            cv.imwrite("./overleaf/extract-face-extracted.jpg", face_image)

        # face_image = four_point_transform(self.contour_image, [a, b, c, d])

        # cv.imwrite(
        #     "./overleaf/extract_face-perspective_transform_exemplu.jpg", face_image)

        new_face_image, coordinates_correction = extract_face_if_exists(face_image)

        true_coordinates = {
            "a": None
        }
        
        if new_face_image is not None:
            a, b, c, d = coordinates_correction

            new_coordinates = np.array([a, b, c, d], dtype=np.int32)

            M_inverse = np.linalg.inv(transformation_matrix)

            points = np.float32([[a], [b], [c], [d]])

            trf = cv.perspectiveTransform(points, M_inverse)

            true_coordinates["a"] = tuple(tuple([int(x[0]), int(x[1])]) for x in [trf[0][0], trf[1][0], trf[2][0], trf[3][0]])
            
            # i = self.contour_image.copy()

            # for point in new_coordinates:
            #     cv.circle(i, point, 3, (0, 0, 255), 5)

            # show_image(i, "Wow")

            # i = self.contour_image.copy()

            # for point in coordinates:
            #     cv.circle(i, point, 3, (0, 0, 255), 5)
            
            # show_image(i, "Not wow")

        if new_face_image is not None:
            if debug:
                show_image(new_face_image, "Fata noua")
                cv.imwrite("./overleaf/extract-face-new-face.jpg",
                           new_face_image)

        if debug:
            if new_face_image is not None:
                show_image(new_face_image, f"{method_name} The face exists")

            else:
                show_image(face_image,
                           f"{method_name} The face does not exist")


        return new_face_image, true_coordinates["a"]

    def add_face_to_current_path(self, face: SudokuFace):
        for coordinates in self.coordinates_to_faces_map.keys():
            if calculate_iou(face.coordinates, coordinates) > iou_threshold:
                face = self.coordinates_to_faces_map[coordinates]
                break

        self.current_path.add_face(face)

    def should_stop_path_extraction_recursion(self):
        return len(self.current_path.faces) == cube_size

    def store_current_path(self, direction: PathDirection):
        self.current_path.set_direction(direction)

        self.paths.append(self.current_path)

        for face in self.current_path.faces:
            existing_key = {"coordinates": face.coordinates}

            for coordinates in self.coordinates_to_faces_map.keys():
                if calculate_iou(face.coordinates, coordinates) > iou_threshold:
                    existing_key["coordinates"] = coordinates
                    break

            self.coordinates_to_faces_map[existing_key["coordinates"]] = face

        # i = self.contour_image.copy()

        # i = add_circles_for_path(self.current_path, i, 250)

        # show_image(i, "Current path")
        # cv.imwrite(f"./overleaf/extract_path-{self.i}-path.jpg", i)

        self.current_path = SudokuPath()

    def save_paths(self, path=paths_file_path):
        with open(path, "wb") as file:
            pickle.dump(self.paths, file)

    def load_paths(self, path=paths_file_path) -> List[SudokuPath]:
        with open(path, "rb") as file:
            self.paths = pickle.load(file)

    def add_numbers_on_face(self, face: SudokuFace):
        image = face.face_image.copy()
        
        face_image = cv.cvtColor(face.face_image, cv.COLOR_BGR2GRAY)

        face_image = cv.threshold(face_image, 250, 255, cv.THRESH_BINARY)[1]

        face_image = cv.erode(face_image, np.ones((2, 2)), iterations=2)

        contours = cv.findContours(face_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

        for contour in contours:
            f = face_image.copy()

            f = cv.cvtColor(f, cv.COLOR_GRAY2BGR)

            corners = order_points(list(map(lambda c: c[0], contour)))

            top_left_corner, bottom_right_corner = corners[0], corners[2]

            width, height = bottom_right_corner[1] - top_left_corner[1], bottom_right_corner[0] - top_left_corner[0]

            for point in [top_left_corner, (top_left_corner[0] + height, top_left_corner[1] + width)]:
                cv.circle(f, point, 3, ( 0, 0, 255), 5)

            image[top_left_corner[1]: top_left_corner[1] + width, top_left_corner[0]: top_left_corner[0] + height] = 255

        for row in range(3):
            for column in range(3):
                number = face.digits[row][column]

                if number == 0:
                    return image

                (width, height) = image.shape[:2]

                epsilon = 8

                one_cube_size = int(width / 3)

                image = cv.resize(image, (width, width))

                digit_image = read_image(f"{digits_files_path}/{number}.jpg")

                # TODO: this could be improved, store rotated images :)
                if face.axis == Axis.Z:
                    digit_image = read_image(f"{digits_files_path}/rot/{number}.jpg")

                digit_image = cv.bitwise_not(digit_image)

                digit_image = cv.resize(
                    digit_image, (one_cube_size, one_cube_size))

                digit_image = np.bitwise_not(digit_image)

                cpy = cv.cvtColor(digit_image, cv.COLOR_BGR2GRAY)
                thresh = cv.threshold(cpy, 127, 255, cv.THRESH_BINARY_INV)[1]

                contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

                contour = max(contours, key=cv.contourArea)

                x, y, w, h = cv.boundingRect(contour)

                pad_size = 5

                x -= pad_size
                y -= pad_size
                w += 2 * pad_size
                h += 2 * pad_size

                # i = digit_image.copy()

                # cv.rectangle(i, (x, y), (x + w, y + h), (0, 0, 255), 1)

                # show_image(i)

                # digit_image = digit_image[y: y + h, x: x + w]

                y, x, w, h = int(row * one_cube_size), int(column * one_cube_size), one_cube_size, one_cube_size

                center = (2 * x + w) // 2 - epsilon, (2 * y + h) // 2

                digit_size = one_cube_size // 3
                
                top_left_corner = center[0] - digit_size, center[1] - digit_size
                bottom_right_corner = center[0] + digit_size, center[1] + digit_size

                # i = image.copy()

                # for point in [(x, y), center, top_left_corner, bottom_right_corner]:
                #     cv.circle(i, point, 3, (0, 0, 255), 5)

                # show_image(i, f"a {row} {column}")

                size = bottom_right_corner[0] - top_left_corner[0]

                digit_image = cv.resize(digit_image, (size, size), interpolation=cv.INTER_AREA)

                image[top_left_corner[1]: top_left_corner[1] + size, top_left_corner[0]: top_left_corner[0] + size] = digit_image

                image = cv.resize(image, (height, width))

        return image

    def add_face_on_image(self, face: SudokuFace, contour_image: np.ndarray):
        a, b, c, d = face.coordinates

        face_image = self.add_numbers_on_face(face)

        (width, height) = contour_image.shape[:2]

        transformation = cv.getPerspectiveTransform(
            self.transform_end, np.array([a, b, c, d], dtype="float32"))

        transformed_face_image = cv.warpPerspective(
            face_image, transformation, (height, width))

        mask = np.zeros(transformed_face_image.shape[:2], dtype="uint8")

        contours, _ = cv.findContours(
            cv.cvtColor(transformed_face_image, cv.COLOR_BGR2GRAY), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        cv.drawContours(mask, contours, -1, 255, -1)

        mask = np.bitwise_not(mask)

        good = cv.bitwise_and(contour_image, contour_image, mask=mask)

        final = good | transformed_face_image

        return final

    def generate_solution_image(self, faces):
        image = self.original_image.copy()

        for face in faces:
            image = self.add_face_on_image(face, image)

        return image

    def generate_coordinates_to_faces_map(self):
        corners_array = [
            (Axis.X, [self.A, self.G, self.E, self.F], [
                self.shift_horizontal_x, self.shift_vertical_x]),

            (Axis.Y, [self.G, self.C, self.D, self.E], [
             self.shift_horizontal_y, self.shift_vertical_y]),

            (Axis.Z, [self.A, self.B, self.C, self.G], [
                self.shift_horizontal_z, self.shift_vertical_z])
        ]

        existing_coordinates = []

        pad_size = 50

        img = np.pad(self.original_image.copy(), ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), constant_values=255)

    
        for depth, row, column, axis, (_, corners, (shift_horizontal, shift_vertical)) in [(1, 0, 0, Axis.Y, corners_array[1]), (0, 0, 2, Axis.Z, corners_array[2]), (0, 2, 1, Axis.X, corners_array[0])]:
            # if len(existing_coordinates) == cubed_cube_size or detected_faces == cubed_cube_size:
            #     break
            
            top_left = corners[0]


            a, b, c, d = get_face_coordinates(
                depth, axis, top_left, shift_horizontal, shift_vertical, row, column)

            a, b, c, d = list(a), list(b), list(c), list(d)

            a[0] += pad_size
            a[1] += pad_size
            b[0] += pad_size
            b[1] += pad_size
            c[0] += pad_size
            c[1] += pad_size
            d[0] += pad_size
            d[1] += pad_size

            for point in [a, b, c, d]:
                cv.circle(img, point, 5, (0, 0, 255), 20)

            for index, point in enumerate([a, b, c, d]):
                add_text_at_point(
                    img, chr(ord("A") + index), point, 8, 1, 1)

            # cv.imwrite(
            #     "./overleaf/extract_face-coordonate_fata_exemplu.jpg", img)


            continue


        cv.imwrite(
            f"./overleaf/conventii_puncte_fete.jpg", img)

        # if len(existing_coordinates) != cubed_cube_size:
        #     raise NumberOfIdentifiedFacesIsWrong(len(existing_coordinates))