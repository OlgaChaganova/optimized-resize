import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from skimage import color, data, feature, morphology


def transform(img: np.array, radius: int) -> np.array:
    if img.ndim == 3:
        img = color.rgb2gray(img)
    img = feature.canny(img, sigma=1)
    img = morphology.binary_dilation(img, footprint=morphology.disk(radius=radius))
    return img.astype(np.uint8)


if __name__ == '__main__':
    img = data.coins()
    img = transform(img, radius=1)
    img_path = os.path.join('tests', 'images', '1.jpg')
    plt.imsave(img_path, img, cmap=cm.gray)

    img = data.astronaut()
    img = transform(img, radius=2)
    img_path = os.path.join('tests', 'images', '2.jpg')
    plt.imsave(img_path, img, cmap=cm.gray)

    img = data.horse()
    img = transform(img, radius=3)
    img_path = os.path.join('tests', 'images', '3.jpg')
    plt.imsave(img_path, img, cmap=cm.gray)
