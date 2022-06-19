import cv2 as cv

from glob import glob


def read_image(file_path: str, flags=None):
    return cv.imread(file_path, flags=flags)


def read_images(folder_path: str, limit=None):
    return list(map(read_image, glob(f"{folder_path}/*")[:limit]))
