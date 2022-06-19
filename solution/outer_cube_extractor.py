import cv2 as cv
import numpy as np
from packages.imaging.reader import read_image
from packages.imaging.utils import rotate_bound, rotate_image

from solution.ImageTransformer import ImageTransformer

from solution.constants import pattern_matching_cube_countour_image_path

from packages.imaging.viewer import show_image

contour_template_image = read_image(
    pattern_matching_cube_countour_image_path)

template_size = (400, 400)

contour_template_image = cv.resize(contour_template_image, template_size)
contour_template_image = cv.cvtColor(contour_template_image, cv.COLOR_BGR2GRAY)
contour_template_image[contour_template_image < 127] = 0
contour_template_image[contour_template_image >= 127] = 255

contour_template_image_number_of_white_pixels = np.sum(
    contour_template_image == 255)


def extract_outer_cube(image: np.ndarray):
    best_top = None
    best_left = None
    best_right = None
    best_bottom = None

    best_theta = 0
    best_phi = 0
    best_gamma = 0

    best_match = 0

    grayscale = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    blurred = cv.GaussianBlur(grayscale,
                              ksize=(3, 3),
                              sigmaX=3)

    thresholded = cv.adaptiveThreshold(blurred,
                                       maxValue=255,
                                       adaptiveMethod=cv.ADAPTIVE_THRESH_MEAN_C,
                                       thresholdType=cv.THRESH_BINARY,
                                       blockSize=25,
                                       C=2)

    negated = cv.bitwise_not(thresholded)

    contours, _ = cv.findContours(negated,
                                  mode=cv.RETR_EXTERNAL,
                                  method=cv.CHAIN_APPROX_SIMPLE)

    contour = max(contours, key=cv.contourArea)

    mask = np.zeros(shape=image.shape[:2], dtype=np.uint8)

    contour_image = cv.drawContours(image=mask,
                                    contours=[contour],
                                    contourIdx=-1,
                                    color=255,
                                    thickness=-1)

    # cv.imwrite(
    #     "./overleaf/outer_cube_extractor-imagine_contur_mascat.jpg", contour_image)

    # img = cv.drawContours(image, [contour], -1, (0, 0, 255), 10)

    # cv.imwrite("./overleaf/outer_cube_extractor-imagine_cu_contur.jpg", img)

    # x = input("b:")

    # TODO: should we add any kind of check, to make sure we identified the cube, and not something else?
    area = cv.countNonZero(contour_image) / \
        float(contour_image.shape[0] * contour_image.shape[1])

    # for angle in np.arange(-1.5, 1.5, 0.05):
    for theta in np.arange(-2, 2.1, 0.5):
        # for theta in [-2]:
        theta_image = ImageTransformer(
            contour_image).rotate_along_axis(theta=theta)

        for gamma in np.arange(-2, 2.1, 0.5):
            # for gamma in [-1]:
            gamma_image = ImageTransformer(
                theta_image).rotate_along_axis(gamma=gamma)
            for phi in np.arange(-2, 2.1, 0.5):
                # for phi in [0.99]:
                current_image = ImageTransformer(
                    gamma_image).rotate_along_axis(phi=phi)

                contours = cv.findContours(image=current_image,
                                           mode=cv.RETR_EXTERNAL,
                                           method=cv.CHAIN_APPROX_SIMPLE)[0]

                contour = max(contours, key=cv.contourArea)

                top = contour[contour[:, :, 1].argmin()][0]
                right = contour[contour[:, :, 0].argmax()][0]
                bottom = contour[contour[:, :, 1].argmax()][0]
                left = contour[contour[:, :, 0].argmin()][0]

                area = cv.countNonZero(
                    current_image) / float(current_image.shape[1] * current_image.shape[0])

                if area > 0.6:
                    continue

                contours, _ = cv.findContours(image=current_image,
                                              mode=cv.RETR_EXTERNAL,
                                              method=cv.CHAIN_APPROX_SIMPLE)

                contour = max(contours, key=cv.contourArea)

                top = contour[contour[:, :, 1].argmin()][0]
                right = contour[contour[:, :, 0].argmax()][0]
                bottom = contour[contour[:, :, 1].argmax()][0]
                left = contour[contour[:, :, 0].argmin()][0]

                cube = current_image[top[1]: bottom[1], left[0]: right[0]]

                cube = cv.resize(cube, template_size)

                cube[cube < 127] = 2
                cube[cube >= 127] = 255

                match = np.sum(
                    cube == contour_template_image) / contour_template_image_number_of_white_pixels

                if match > best_match:
                    best_match = match

                    best_top = top
                    best_right = right
                    best_bottom = bottom
                    best_left = left

                    best_theta = theta
                    best_gamma = gamma
                    best_phi = phi

                if match > 0.99:
                    break

    print((best_theta, best_gamma, best_phi, best_match))

    image = ImageTransformer(image).rotate_along_axis(
        best_theta, best_phi, best_gamma)

    # show_image(image, "Transformed")

    cube = image[best_top[1]: best_bottom[1], best_left[0]: best_right[0]]

    # show_image(cube, "Dupa best")

    cube_for_contour = cv.bitwise_not(cv.cvtColor(cube, cv.COLOR_BGR2GRAY))

    cube_for_contour[cube_for_contour < 196] = 0

    # print(np.unique(cube_for_contour))

    # show_image(cube_for_contour, "Cfc")

    contours, _ = cv.findContours(
        cube_for_contour, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    contour = max(contours, key=cv.contourArea)

    # cv.drawContours(cube, [contour], -1, (0, 255, 0), 2)

    # show_image(cube, "Cube")

    top = contour[contour[:, :, 1].argmin()][0]
    right = contour[contour[:, :, 0].argmax()][0]
    bottom = contour[contour[:, :, 1].argmax()][0]
    left = contour[contour[:, :, 0].argmin()][0]

    # print(top)
    # print(right)
    # print(bottom)
    # print(left)

    # for point in [top, right, bottom, left]:
    #     cv.circle(cube, point, 3, (0, 255, 0), 3)

    cube = cube[top[1]: bottom[1], left[0]: right[0]]

    # show_image(cube, "Cube final")

    return cube
