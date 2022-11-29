import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from skimage import color, data, feature


def transform(img: np.array) -> np.array:
    if img.ndim == 3:
        img = color.rgb2gray(img)
    img = feature.canny(img, sigma=1)
    return img.astype(np.uint8)


if __name__ == '__main__':
    img = data.coins()
    img = transform(img)
    plt.imsave(f'../tests/test_images/1.jpg', img, cmap=cm.gray)

    img = data.astronaut()
    img = transform(img)
    plt.imsave(f'../tests/test_images/2.jpg', img, cmap=cm.gray)

    img = data.horse()
    img = transform(img)
    plt.imsave(f'../tests/test_images/3.jpg', img, cmap=cm.gray)