# XXX Conventions
#   When playing with coordinates, we always start at the top left corner
#       X = inside from left
#       Y = inside from right
#       Z = inside from above
#   Squares:
#       top left corner - a
#       top right corner - b
#       bottom right corner - c
#       bottom left corner - d


import numpy as np


data_path = "./data"

solution_path = f"{data_path}/solution"

train_data_path = f"{data_path}/training"
validation_data_path = f"{data_path}/validation"

pattern_matching_path = f"{data_path}/pattern_matching"

pattern_matching_cube_countour_image_path = f"{pattern_matching_path}/outer_cube_detection/contour-2.jpg"

pattern_matching_digits_path = f"{pattern_matching_path}/digits"

digits_files_path = f"{data_path}/digits/"

paths_file_path = f"{solution_path}/paths.pkl"

sudoku_validation_files_path = f"{data_path}/sudoku_validation_files"

face_image_extractor_cube_area_threshold = .75

iou_threshold = 0.5

cube_size = 3

cubed_cube_size = np.power(cube_size, 3)

digit_size = (128, 128)

struct_width_constant = 0.09

cube_face_width, cube_face_height = (128, 128)

KEEP_EDGES_IN_IMAGE_THRESHOLD_KEY = "keep_edges_in_image_threshold"
STRUCT_SIZES_ERR_KEY = "struct_sizes_err"

thresholds_and_stuff = dict({
    "iou_threshold": 0.5,
    "cube_grid_pattern_matching_threshold": 0.63,
    KEEP_EDGES_IN_IMAGE_THRESHOLD_KEY: 127,
    STRUCT_SIZES_ERR_KEY: 0.8
})
