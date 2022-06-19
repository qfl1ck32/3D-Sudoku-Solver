from collections import defaultdict
from typing import List

from solution.types import PathMap, SudokuPath

from solution.constants import iou_threshold

from shapely.geometry import Polygon


def calculate_iou(box_1, box_2):
    poly_1 = Polygon(box_1)

    poly_2 = Polygon(box_2)

    return poly_1.intersection(poly_2).area / poly_1.union(poly_2).area


def create_path_map(paths: List[SudokuPath]):
    paths_by_coordinates: PathMap = defaultdict(lambda: [])

    for path in paths:
        for face in path.faces:
            paths_by_coordinates[face.coordinates].append(path)

    return paths_by_coordinates
