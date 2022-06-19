import cv2
import os

from timeit import default_timer

from packages.imaging.reader import read_image
from packages.logging.Logger import logger

from solution.constants import train_data_path
from solution.outer_cube_extractor import extract_outer_cube


def deskew_images():
    for file_name in os.listdir(train_data_path):
        if not file_name == "299.jpg":
            continue

        start_time = default_timer()

        logger.info(f"Deskewing {file_name}...")

        image = read_image(f"{train_data_path}/{file_name}")

        cube = extract_outer_cube(image)

        [name, extension] = file_name.split(".")

        cv2.imwrite(f"{train_data_path}/deskewed/{name}.{extension}", cube)

        logger.info(f"Done - {default_timer() - start_time}.\n")
