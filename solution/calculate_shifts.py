from typing import List
from solution.constants import cube_size


def calculate_shifts(corners: List):
    top_left, top_right, bottom_left = corners

    shift_horizontal = int(
        (top_left[0] - top_right[0]) / cube_size), int((top_left[1] - top_right[1]) / cube_size)

    shift_vertical = int((top_left[0] - bottom_left[0]) /
                         cube_size), int((top_left[1] - bottom_left[1]) / cube_size)

    return shift_horizontal, shift_vertical
