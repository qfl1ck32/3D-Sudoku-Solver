from asyncio import ALL_COMPLETED
import os
import itertools
import pickle
import shutil
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

from solution.constants import KEEP_EDGES_IN_IMAGE_THRESHOLD_KEY, STRUCT_SIZES_ERR_KEY, \
    train_data_path, validation_data_path, solution_path, sudoku_validation_files_path, thresholds_and_stuff
from packages.imaging.reader import read_image
from packages.imaging.viewer import show_image
from solution.SudokuCV import SudokuCV
from solution.create_path_map import create_path_map
from solution.outer_cube_extractor import extract_outer_cube
from solution.types import SudokuFace
from solution.SudokuSolver import SudokuSolver


class SudokuValidation:
    def __init__(self):
        pass

    def generate_validation_set(self, size=100):
        count = 0

        for i in range(1, 326, 1):
            if count == size:
                break
            x = np.random.random(1)[0]

            if x > 0.5:
                shutil.copy(f"{train_data_path}/deskewed/{i}.jpg",
                            f"{validation_data_path}/{i}.jpg")
                count += 1
        return

    def get_validation_files_paths(self):
        return list(map(lambda s: f"{validation_data_path}/{s}", os.listdir(validation_data_path)))

    # de la 360 a inceput sa mearga bine
    def find(self):
        """
        Doar 9 și 319 au probleme
        0
        (0.7999999999999999, 127)
        """
        struct_sizes_err = np.arange(0.5, 0.9, 0.1)
        keep_edges_in_image_threshold = np.arange(127, 250, 1)

        all_combinations = list(itertools.product(
            struct_sizes_err, keep_edges_in_image_threshold))

        max_err_count = 5
        best_err_count = max_err_count
        best_values = []

        validation_files_paths = self.get_validation_files_paths()

        for values in tqdm(all_combinations):
            struct_size_err, keep_edges_in_image_threshold = values

            thresholds_and_stuff[STRUCT_SIZES_ERR_KEY] = struct_size_err
            thresholds_and_stuff[KEEP_EDGES_IN_IMAGE_THRESHOLD_KEY] = keep_edges_in_image_threshold

            err_count = 0

            for file_path in validation_files_paths:
                image = read_image(file_path)

                try:
                    SudokuCV(image, verbose=False).extract_thick_edges()
                except Exception:
                    err_count += 1

                if err_count > max_err_count:
                    break

            if err_count < best_err_count:
                best_values = values
                best_err_count = err_count

        print(best_err_count)
        print(best_values)

    def extract_faces(self):
        for file_path in tqdm(self.get_validation_files_paths()):
            image = read_image(file_path)

            sudoku_cv = SudokuCV(image)

            sudoku_cv.setup()

            sudoku_cv.extract_thick_edges()

            sudoku_cv.extract_paths()

            pickle.dump()

    def calculate_accuracies(self):
        pass
