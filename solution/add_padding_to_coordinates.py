from typing import List
import numpy as np

from solution.types import Point
from solution.enums import Axis


def get_padding_amount(axis: int):
    return 20

    if axis == Axis.Z:
        return 8
    if axis == Axis.X:
        return 8
    if axis == Axis.Y:
        return 8


def get_distance(x: tuple[int, int], y: tuple[int, int]):
    return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


def get_y_equation(p1: tuple[int, int], p2: tuple[int, int]):
    x_p1, y_p1 = p1
    x_p2, y_p2 = p2

    def eq(x: float, should_round=True):
        result = (x - x_p1) / (x_p2 - x_p1) * (y_p2 - y_p1) + y_p1

        return int(result) if should_round else result

    return eq


def get_shifts(corners: List[tuple]):
    top_left, top_right, bottom_left = corners

    shift_horizontal = int((top_left[0] - top_right[0]) /
                           3), int((top_left[1] - top_right[1]) / 3)

    shift_vertical = int((top_left[0] - bottom_left[0]) /
                         3), int((top_left[1] - bottom_left[1]) / 3)

    return shift_horizontal, shift_vertical


def get_face_coordinates(depth: int, axis: int, top_left: tuple[int, int],
                         shift_horizontal: tuple[int, int], shift_vertical: tuple[int, int], row: int, column: int):

    if axis == Axis.Z:
        a = [top_left[0] - shift_vertical[0] * row - shift_horizontal[0] * column,
             top_left[1] - shift_vertical[1] * (row + 2 * depth) - shift_horizontal[1] * column]

    elif axis == Axis.X:
        a = [top_left[0] - shift_vertical[0] * row - shift_horizontal[0] * (column + depth),
             top_left[1] - shift_vertical[1] * row - shift_horizontal[1] * (column - depth)]

    elif axis == Axis.Y:
        a = [top_left[0] - shift_vertical[0] * row - shift_horizontal[0] * (column - depth),
             top_left[1] - shift_vertical[1] * row - shift_horizontal[1] * (column + depth)]

    b = [a[0] - shift_horizontal[0], a[1] - shift_horizontal[1]]
    c = [a[0] - shift_vertical[0] - shift_horizontal[0],
         a[1] - shift_vertical[1] - shift_horizontal[1]]
    d = [a[0] - shift_vertical[0], a[1] - shift_vertical[1]]

    return tuple(a), tuple(b), tuple(c), tuple(d)


def add_padding_to_coordinates(a: Point, b: Point, c: Point, d: Point, axis: Axis):
    padding_amount = get_padding_amount(axis)

    a, b, c, d = list(a), list(b), list(c), list(d)

    if axis == Axis.Z:
        ac = get_distance(a, c)

        bd = get_distance(b, d)

        rep = ac / bd

        to_add = int(padding_amount / rep)

        a[0] -= padding_amount
        c[0] += padding_amount

        b[1] -= to_add
        d[1] += to_add

    else:
        ac_y_eq = get_y_equation(a, c)
        bd_y_eq = get_y_equation(b, d)

        a[0] -= padding_amount
        a[1] = ac_y_eq(a[0])

        c[0] += padding_amount
        c[1] = ac_y_eq(c[0])

        b[0] += padding_amount
        b[1] = bd_y_eq(b[0])

        d[0] -= padding_amount
        d[1] = bd_y_eq(d[0])

    return tuple(a), tuple(b), tuple(c), tuple(d)
