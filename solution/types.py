import numpy as np
from packages.imaging.viewer import show_image

from solution.digit_extractor import extract_numbers_from_face

from solution.enums import EdgeType, Axis, PathDirection

from typing import List
from collections import defaultdict

from scipy.ndimage import rotate

from solution.exceptions import ThePathsCouldNotHaveBeenProperlyIdentified, ThickLinesCouldNotBeIdentified

Point = (int, int)

SquareCoordinates = tuple[Point]


class SudokuFace:
    coordinates: SquareCoordinates
    digits: np.ndarray

    axis: Axis
    face_image: np.ndarray

    def __init__(self, coordinates: SquareCoordinates, axis: Axis, face_image: np.ndarray):
        self.coordinates = coordinates
        self.axis = axis
        self.face_image = face_image

        if face_image is None:
            raise ThePathsCouldNotHaveBeenProperlyIdentified()

        self.digits = None

    def extract_digits(self):
        if self.digits is not None:
            return None

        self.digits = extract_numbers_from_face(
            self.face_image, self.axis)

        return self.digits


class SudokuPath:
    direction: PathDirection

    faces: List[SudokuFace]

    def __init__(self, faces: List[SudokuFace] = None):
        if faces is None:
            faces = []

        self.faces = faces

    def add_face(self, face: SudokuFace):
        self.faces.append(face)

    def set_direction(self, direction: PathDirection):
        self.direction = direction


PathMap = defaultdict[SquareCoordinates, List[SudokuPath]]
