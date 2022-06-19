import matplotlib

import matplotlib.pyplot as plt
import numpy as np


def plot_image(image: np.ndarray):
    plt.figure(figsize=(8, 6))

    plt.imshow(image, "gray")

    plt.show()

    return image
