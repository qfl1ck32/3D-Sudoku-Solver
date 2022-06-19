from collections import defaultdict
from glob import glob
import os
import pickle
from timeit import default_timer
from typing import List

import cv2 as cv
from cv2 import boundingRect
import numpy as np
from solution.ImageTransformer import ImageTransformer

from solution.constants import train_data_path, validation_data_path

from packages.debugging.add_circles_for_path import add_circles_for_path, add_circles_for_paths

from packages.imaging.reader import read_image, read_images
from packages.imaging.viewer import show_image
from solution.SudokuValidation import SudokuValidation

from solution.create_path_map import create_path_map
from solution.SudokuCV import SudokuCV

from solution.constants import paths_file_path
from solution.SudokuSolver import SudokuSolver
from solution.types import PathMap, SudokuFace
from solution.constants import cube_size, solution_path
from solution.deskew_images import deskew_images

import matplotlib.pyplot as plt


solve_errors = []
numbers_errors = []
paths_errors = []
weird_errors = []


def main_test():
    # return deskew_images()

    global paths_errors
    global numbers_errors
    global solve_errors
    global weird_errors

    # for i in [13, 316]:
    for i in range(1, 326, 1):
        # if len(glob(f"{solution_path}/images/{i}-*")):
        #     continue

        if not os.path.exists(f"{train_data_path}/deskewed/{i}.jpg"):
            continue

        print(f"Solving for Cube {i}")

        start = default_timer()

        image = read_image(f"{train_data_path}/deskewed/{i}.jpg")

        sudoku_cv = SudokuCV(image, i, True)

        try:
            sudoku_cv.setup()
        except Exception as e:
            print(e)
            weird_errors.append(i)
            continue

        sudoku_cv.generate_coordinates_to_faces_map()

        try:
            sudoku_cv.extract_thick_edges()
        except Exception as e:
            print(e)
            weird_errors.append(i)
            continue

        sudoku_cv

        path_path = f"./data/solution/paths/paths-{i}.pkl"

        try:
            if os.path.exists(path_path):
                sudoku_cv.load_paths(path_path)
            else:
                sudoku_cv.extract_paths()
                # sudoku_cv.save_paths(path_path)

        except Exception as e:
            print(i)
            print(e)

            paths_errors.append(i)
            continue

        try:
            sudoku_cv.extract_digits_for_faces()
        except Exception as e:
            numbers_errors.append(i)
            print(e)
            continue

        # print(len(faces))
        # for i, face in enumerate(faces):
        #     if i < 18:
        #         continue

        #     print(face.digits)
        #     show_image(face.face_image, "Face")

        # sudoku_cv.load_paths("./test.pkl")

        # i = 0
        # for path in sudoku_cv.paths:
        #     img = sudoku_cv.contour_image.copy()

        #     add_circles_for_path(path, img)

        #     show_image(img, "path")

        # if digits is not None:
        #     i += 1
        #     print(i)
        #     print(face.digits)
        #     show_image(face.original_face_image)

        path_map = create_path_map(sudoku_cv.paths)

        # faces = []

        # for i, path in enumerate(sudoku_cv.paths):
        #     img = sudoku_cv.contour_image.copy()

        #     img = add_circles_for_path(path, img, 50)

        #     show_image(img, f"{i}")

        #     cv.imwrite("./overleaf/perspective-x-z-direction.jpg", img)

        # XXX useful
        # faces = []

        # for paths in path_map.values():
        #     for path in paths:
        #         for face in path.faces:
        #             if face not in faces:
        #                 faces.append(face)

        # for index, face in enumerate(faces):
        #     print(index)

        #     paths = path_map[face.coordinates]

        #     i = sudoku_cv.contour_image.copy()

        #     pad_size = 50

        #     i = np.pad(i, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), constant_values=255)

        #     for point in face.coordinates:
        #         cv.circle(i, (point[0] + pad_size, point[1] + pad_size), 3, (0, 0, 255), 10)

        #     show_image(i)

        #     cv.imwrite("./overleaf/non_trivial_cases-face.jpg", i)

        #     i = sudoku_cv.contour_image.copy()

        #     i1 = add_circles_for_path(paths[0], i, 50)

        #     show_image(i1)

        #     cv.imwrite("./overleaf/non_trivial_cases-path1.jpg", i1)

        #     i = sudoku_cv.contour_image.copy()

        #     i2 = add_circles_for_path(paths[1], i, 50)

        #     for indeex, face in enumerate(paths[1].faces):
        #         cv.imwrite(f"./overleaf/non_trivial_cases-face-{indeex}.jpg", face.face_image)

        #     show_image(i2)

        #     cv.imwrite("./overleaf/non_trivial_cases-path2.jpg", i2)
        # XXX end of useful

        solver = SudokuSolver(sudoku_cv)

        c = 0

        for face in solver.faces:
            c += len(face.digits.nonzero()[0])

        print(c)

        input("a:")

        # for path in sudoku_cv.paths:
        #     img = sudoku_cv.contour_image.copy()

        #     img = add_circles_for_path(path, img)

        #     show_image(img, "Img for path")

        print("Generating the solution...")

        solved = solver.solve()

        print(f"Solved: {solved}")

        if not solved:
            solve_errors.append(i)

        end = default_timer()

        print(end - start)

        print("Done")

        print(solver.recursive_calls)

        if not solved:
            continue

        img = sudoku_cv.generate_solution_image(solver.faces)

        # show_image(img)

        cv.imwrite(
            f"./data/solution/images/{i}-{end - start}.jpg", img)

        print(solve_errors)
        print(numbers_errors)
        print(paths_errors)
        print(weird_errors)

    print(solve_errors)
    print(numbers_errors)
    print(paths_errors)
    print(weird_errors)


def main_errors():
    # for i in solve_errors:  # 323 de rezolvat
    for i in [323]:
        image = read_image(f"{train_data_path}/deskewed/{i}.jpg")

        sudoku_cv = SudokuCV(image, i, True)

        sudoku_cv.extract_thick_edges()

        path_path = f"./data/solution/paths/paths-{i}.pkl"

        if os.path.exists(path_path):
            sudoku_cv.load_paths(path_path)
        else:
            sudoku_cv.extract_paths()
            sudoku_cv.save_paths(path_path)

        sudoku_cv.extract_digits_for_faces()

        path_map = create_path_map(sudoku_cv.paths)

        solver = SudokuSolver(path_map, sudoku_cv)

        img = sudoku_cv.contour_image.copy()

        print("Generating the solution...")

        solved = solver.solve()

        print(f"Solved: {solved}")

        print("Done")

        print(solver.recursive_calls)

        if not solved:
            continue

        img = sudoku_cv.contour_image.copy()

        for face in solver.faces:
            img = sudoku_cv.add_face_on_image(face, img)

        show_image(img, "Solved")


def main_validation():
    pass


def main_final():
    if os.path.exists(f"{solution_path}/timing.pkl"):
        with open(f"{solution_path}/timing.pkl", "rb") as f:
            data = pickle.load(f)
    else:
        data = []

    for i in range(1 + len(data), 326, 1):
        print(f"Solving Cube {i}")

        image = read_image(f"{train_data_path}/deskewed/{i}.jpg")

        sudoku_cv = SudokuCV(image)

        sudoku_cv.setup()

        time_info = default_timer()

        sudoku_cv.extract_thick_edges()

        sudoku_cv.extract_paths()

        sudoku_cv.extract_digits_for_faces()

        end_time_info = default_timer()

        solver = SudokuSolver(sudoku_cv)

        time_solve = default_timer()

        solved = solver.solve()

        end_time_solve = default_timer()

        print(f"Solved! Solution found: {solved}")

        data.append([a := end_time_info - time_info, b :=
                    end_time_solve - time_solve])

    with open(f"{solution_path}/timing.pkl", "wb") as file:
        pickle.dump(data, file)


def main_times():
    with open(f"{solution_path}/timing.pkl", "rb") as file:
        data = pickle.load(file)

    info_extraction_times = np.array(list(map(lambda c: c[0], data)))
    solution_times = np.array(list(map(lambda c: c[1], data)))

    full_times = info_extraction_times + solution_times

    print(np.argmin(full_times))
    
    min_index = info_extraction_times.argmin()
    max_index = info_extraction_times.argmax()

    min_index_2 = solution_times.argmin()
    max_index_2 = solution_times.argmax()

    mean_value_info = info_extraction_times.mean()
    mean_value_solution = solution_times.mean()

    standard_items = 163

    info_standard = np.array(info_extraction_times.tolist()[:standard_items])
    info_challenging = np.array(info_extraction_times.tolist()[standard_items:])

    solution_standard = np.array(solution_times.tolist()[:standard_items])
    solution_challengine = np.array(solution_times.tolist()[standard_items:])

    all_standard  = info_standard + solution_standard
    all_challenging = info_challenging + solution_challengine

    x = 1 + np.arange(len(info_extraction_times))

    fig = plt.figure(figsize=(12, 6), dpi=150)

    plt.plot(x, info_extraction_times)
    plt.ylabel("Timpul necesar extragerii informațiilor vizuale (s)")
    plt.xlabel("Numărul imaginii")

    plt.axhline(y=mean_value_info, color='r', linestyle='dashed', label='Media pe toate imaginile')
    plt.axhline(y=info_standard.mean(), color='orange', linestyle='dashed', label='Media pe imaginile de dificultate standard')
    plt.axhline(y=info_challenging.mean(), color='purple', linestyle='dashed', label='Media pe imaginile de dificultate ridicată')

    plt.axvline(x=163.5, color='g', label='Schimbarea nivelului de dificultate')

    plt.legend(bbox_to_anchor=(0.808, 1), loc='upper center')

    fig.savefig("./overleaf/figure-times-info-extraction.jpg")

    #

    fig = plt.figure(figsize=(12, 6), dpi=150)

    # solution_times = np.delete(solution_times, solution_times.argmax())

    x = 1 + np.arange(len(data))

    plt.plot(x, solution_times)
    plt.ylabel("Timpul necesar generării soluției (s)")
    plt.xlabel("Numărul imaginii")

    plt.axhline(y=mean_value_solution, color='r', linestyle='dashed', label='Media pe toate imaginile')
    plt.axhline(y=solution_standard.mean(), color='orange', linestyle='dashed', label='Media pe imaginile de dificultate standard')
    plt.axhline(y=solution_challengine.mean(), color='purple', linestyle='dashed', label='Media pe imaginile de dificultate ridicată')

    plt.axvline(x=163.5, color='g', label='Schimbarea nivelului de dificultăte')
    plt.legend(bbox_to_anchor=(0.194, 1), loc='upper center')

    fig.savefig("./overleaf/figure-times-solution.jpg")

    print(all_standard.min(), all_standard.mean(), all_standard.max())
    print(all_challenging.min(), all_challenging.mean(), all_challenging.max())

def main_test2():
    mp = []

    for path in glob("./data/solution/images/*"):
        name = path.split("/images/")[1].split("-")[0]

        time = float(path.split("-")[1].split(".jpg")[0])
        
        mp.append((int(name), time))

    mp = sorted(mp, key=lambda x: x[1], reverse=True)

    print(mp)

if __name__ == '__main__':
    main_times()
    # main_test()

    # main_final()

    # main_test2()
    # for path in glob("./data/digits/*"):
    #     img = read_image(path)

    #     original_img = img.copy()

    #     img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #     thresh = cv.threshold(img, 5, 255, cv.THRESH_BINARY_INV)[1]

    #     contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

    #     contour = max(contours, key=cv.contourArea)

    #     x, y, w, h = boundingRect(contour)

    #     pad_size = 5

    #     x -= pad_size
    #     y -= pad_size
    #     w += 2 * pad_size
    #     h += 2 * pad_size

    #     print((x, y, w, h))

    #     img = original_img[y: y + h, x: x + w]

    #     # show_image(img, 'img')

    #     t = path.split("/digits/")[1].split(".")[0]

    #     cv.imwrite(f"./data/digits/{t}.jpg", img)
