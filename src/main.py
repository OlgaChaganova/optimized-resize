import argparse
import os
import typing as tp
from collections import Counter

import cv2
import numpy as np


class ImageResizer(object):
    def __init__(self, img_filename: str, new_shape: tp.Tuple[int, int]):
        self._img_filename = img_filename
        self._new_shape = new_shape

    def resize(self):
        image = self._read_image()
        resized_image = self._resize(image)
        self._save_image(resized_image)

    # @classmethod
    # def nearest_neighbour(cls, image: np.array, new_shape: tp.Tuple[int, int]):
    #     """Only DOWNSCALE"""
    #     img_width, img_height = image.shape
    #     new_width, new_height = new_shape
    #     x_ratio = img_width / new_width
    #     y_ratio = img_height / new_height
    #     x_coords = np.ceil(np.arange(new_width) * x_ratio).astype(np.int32)
    #     y_coords = np.ceil(np.arange(new_height) * y_ratio).astype(np.int32)
    #     return image[x_coords][:, y_coords]

    @classmethod
    def nearest_neighbour(cls, image: np.array, new_shape: tp.Tuple[int, int]):
        """DOWNSCALE AND UPSCALE"""
        img_width, img_height = image.shape
        new_width, new_height = new_shape

        x_ratio = img_width / new_width
        y_ratio = img_height / new_height

        resized = np.zeros([new_width, new_height])

        for i in range(new_width):
            for j in range(new_height):
                resized[i, j] = image[int(i * x_ratio), int(j * y_ratio)]
        return resized

    def _read_image(self) -> np.array:
        """Read image from file."""
        image = cv2.imread(self._img_filename)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _save_image(self, image: np.array) -> np.array:
        """Save resized image to file with new shape in written in filename."""
        base_path, filename = os.path.split(self._img_filename)
        name, ext = filename.split('.')
        filename = name + str(self._new_shape) + '.' + ext
        new_filename = os.path.join(base_path, 'resized', filename)
        cv2.imwrite(new_filename, image)

    def _resize(self, image: np.array) -> np.array:
        """Whole pipeline of image resizing."""
        new_h, new_w = self._new_shape
        img_nearest_shape = (2 * new_h, 2 * new_w)
        image = self.nearest_neighbour(image, img_nearest_shape)
        assert image.shape == img_nearest_shape
        # assert np.allclose(image, cv2.resize(image, dsize=img_nearest_shape, interpolation=cv2.INTER_NEAREST))
        # image = cv2.resize(image, dsize=img_nearest_shape, interpolation=cv2.INTER_NEAREST)
        image = cv2.resize(image, dsize=self._new_shape, interpolation=cv2.INTER_CUBIC)
        return image


def parse():
    """Parse command line."""
    parser = argparse.ArgumentParser('Effective resize of the binary image')
    parser.add_argument('input_img', type=str, help='Input image to be resized.')
    parser.add_argument('img_w_h', type=int, nargs='+', help='Size of the image after resize')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    image_resizer = ImageResizer(
        img_filename=args.input_img,
        new_shape=args.img_w_h,
    )
    image_resizer.resize()
