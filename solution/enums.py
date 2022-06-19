from enum import Enum


class EdgeType(Enum):
    VERTICAL = 0
    SLASH = 1
    BACKSLASH = 2


class PathDirection(Enum):
    VERTICAL = 0
    HORIZONTAL = 1


class Axis(Enum):
    X = "X"
    Y = "Y"
    Z = "Z"
