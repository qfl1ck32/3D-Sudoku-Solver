from re import S
from typing import List
import numpy as np


import cv2 as cv

from scipy.ndimage import rotate
from packages.debugging.add_circles_for_path import add_circles_for_path, add_circles_for_paths
from packages.imaging.reader import read_image
from packages.imaging.viewer import show_image
from solution.SudokuCV import SudokuCV, _unpad_image
from solution.create_path_map import create_path_map

from solution.enums import EdgeType, Axis, PathDirection
from solution.types import PathMap, SquareCoordinates, SudokuFace, SudokuPath

from solution.constants import cube_size
from solution.constants import train_data_path


class SudokuSolver:
    path_map: PathMap

    faces: List[SudokuFace]

    recursive_calls: int

    sudoku_cv: SudokuCV

    completed_image: np.ndarray

    _debug_number: int

    def __init__(self, sudoku_cv: SudokuCV):
        self.path_map = create_path_map(sudoku_cv.paths)

        self.sudoku_cv = sudoku_cv

        self.faces = []

        # self._debug_number = 211
        self._debug_number = float('+inf')
        # self._debug_number = 0

        for i, paths in enumerate(self.path_map.values()):
            for path in paths:
                # img = self.sudoku_cv.contour_image.copy()

                # add_circles_for_path(path, img)

                # show_image(img, "This path")
                for face in path.faces:
                    if face not in self.faces:
                        self.faces.append(face)
            # break

        self.recursive_calls = 0

        self.completed_image = sudoku_cv.contour_image.copy()

    def _is_debug(self):
        return self.recursive_calls > self._debug_number

    def _debug_is_move_legal(self, given_face: SudokuFace, row: int, column: int, number: int, answer: bool):
        if not self._is_debug():
            return

        cube_image = self.completed_image.copy()

        print(f"Putting {number} on {row} / {column} => {answer}")

        paths = self.path_map[given_face.coordinates]

        for point in given_face.coordinates:
            cv.circle(cube_image, point, 8, (0, 255, 0), 8)

        add_circles_for_path(paths[0], cube_image)

        show_image(cube_image, "This is the path")

    def is_state_valid(self, given_face: SudokuFace, row: int, column: int, number: int):
        # print((row, column, number))
        if given_face.digits[row][column] != 0:
            # print("Nu este 0.")
            self._debug_is_move_legal(given_face, row, column, number, False)
            return False

        if number in given_face.digits:
            # print("Este assigned deja.")
            self._debug_is_move_legal(given_face, row, column, number, False)
            return False

        paths = self.path_map[given_face.coordinates]

        # j = self.sudoku_cv.contour_image.copy()

        # for point in given_face.coordinates:
        #     cv.circle(j, point, 3, (0, 0, 255), 25)

        # show_image(j, "J")
        # cv.imwrite("./overleaf/two-paths-main-face.jpg", j)

        # i = self.sudoku_cv.contour_image.copy()

        # i = add_circles_for_path(paths[0], i, 50)

        # show_image(i)
        # cv.imwrite("./overleaf/two-paths-1.jpg", i)

        # i = self.sudoku_cv.contour_image.copy()

        # i = add_circles_for_path(paths[1], i, 50)

        # show_image(i)
        # cv.imwrite("./overleaf/two-paths-2.jpg", i)

        for path in paths:
            all_axis = list(map(lambda f: f.axis, path.faces))

            has_x_z_perspective_change = Axis.X in all_axis and Axis.Z in all_axis

            is_only_x = all(axis == Axis.X for axis in all_axis)
            is_only_z = all(axis == Axis.Z for axis in all_axis)

            direction = path.direction

            old_direction = direction

            # print(direction)

            # or is_only_x or is_only_z
            if has_x_z_perspective_change:
                if given_face.axis == Axis.X:
                    direction = PathDirection.VERTICAL
                elif given_face.axis == Axis.Z:
                    direction = PathDirection.HORIZONTAL

            # if old_direction != direction:
            #     print(old_direction, direction)

            #     i = self.sudoku_cv.contour_image.copy()

            #     i = add_circles_for_path(path, i)

            #     for point in given_face.coordinates:
            #         cv.circle(i, point, 3, (0, 0, 255), 30)

            #     show_image(i, "Path")

            # else:
            #     if is_only_x:
            #         direction = PathDirection.VERTICAL
            #     # elif is_only_z:
            #     #     direction = PathDirection.HORIZONTAL

            for face in path.faces:

                digits = face.digits

                if has_x_z_perspective_change:
                    if given_face.axis == Axis.Z and face.axis == Axis.X:
                        digits = rotate(digits, -90)

                    elif given_face.axis == Axis.X and face.axis == Axis.Z:
                        digits = rotate(digits, 90)

                slice_to_verify = digits[:,
                                         column] if direction == PathDirection.VERTICAL else digits[row, :]

                if self._is_debug():
                    print("Verific pt fatza")
                    print(digits)
                    print(
                        f"Si extrag, pt {str(direction)} (original {path.direction}) ")
                    print(slice_to_verify)

                    print("--------------------------------")

                if number in slice_to_verify:
                    # print((row, column, number))
                    self._debug_is_move_legal(
                        given_face, row, column, number, False)

                    return False

        # print((row, column, number))
        self._debug_is_move_legal(given_face, row, column, number, True)

        return True

    def find_empty_face_and_coordinates(self):
        for face in self.faces:
            rows, cols = np.where(face.digits == 0)

            # print(face.digits)

            # show_image(face.original_face_image, "Uite")

            if len(rows) == 0:
                continue

            return face, rows[0], cols[0]

        return None

    def solve(self):
        # if self.recursive_calls % 100 == 0:
        #     print(self.recursive_calls)
        #     show_image(self.completed_image, "Step")

        # if self._is_debug():
        #     show_image(self.completed_image, f"Hehey - {self.recursive_calls}")

        self.recursive_calls += 1

        find_result = self.find_empty_face_and_coordinates()

        if find_result is None:
            return True
        else:
            face, row, column = find_result

        for number in range(1, 10):
            if self.is_state_valid(face, row, column, number):
                # print("Pe fata ")
                # print(face.digits)
                # print(f"Este legal {row} {column} -> {number}")
                # input("a: ")

                face.digits[row][column] = number

                if self._is_debug():
                    self.completed_image = self.sudoku_cv.add_face_on_image(
                        face, self.completed_image)

                if self.solve():
                    return True

                face.digits[row][column] = 0

                if self._is_debug():
                    self.completed_image = self.sudoku_cv.add_face_on_image(
                        face, self.completed_image)

        return False
