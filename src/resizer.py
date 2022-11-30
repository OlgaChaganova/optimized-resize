import os
import typing as tp

import cv2
import numpy as np


class ImageResizer(object):
    def __init__(
        self,
        img_filename: str,
        new_shape: tp.Tuple[int, int],
        mode: tp.Literal['naive', 'vectorized'],
    ):
        self._img_filename = img_filename
        self._new_shape = new_shape
        self._mode = mode

    def resize(self, return_image: bool = False):
        """Main method for resizing."""
        image = self._read_image()
        resized_image = self._resize(image)
        if return_image:
            return resized_image
        self._save_image(resized_image)

    @classmethod
    def nn_inter_vectorized(cls, image: np.array, new_shape: tp.Tuple[int, int]) -> np.array:
        """Nearest neigbour interpolation (upscale and downscale). Vectorized implementation"""
        img_width, img_height = image.shape
        new_width, new_height = new_shape

        # -1 because we need to get coordinates
        x_ratio = (img_width - 1) / (new_width - 1)
        y_ratio = (img_height - 1) / (new_height - 1)
        x_coords = np.ceil(np.arange(new_width - 1) * x_ratio).astype(np.int32)
        y_coords = np.ceil(np.arange(new_height - 1) * y_ratio).astype(np.int32)
        return image[x_coords][:, y_coords]

    @classmethod
    def nn_inter_naive(cls, image: np.array, new_shape: tp.Tuple[int, int]) -> np.array:
        """Nearest neigbour interpolation (upscale and downscale). Naive implementation"""
        img_width, img_height = image.shape
        new_width, new_height = new_shape
        x_ratio = img_width / new_width
        y_ratio = img_height / new_height
        resized_image = np.zeros([new_width, new_height])
        for i in range(new_width):
            for j in range(new_height):
                resized_image[i, j] = image[int(i * x_ratio), int(j * y_ratio)]
        return resized_image

    def _read_image(self) -> np.array:
        """Read image from file."""
        image = cv2.imread(self._img_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.astype(np.uint8)

    def _save_image(self, image: np.array) -> np.array:
        """Save resized image to file with new shape in written in filename."""
        base_path, filename = os.path.split(self._img_filename)
        name, ext = filename.split('.')
        filename = name + str(self._new_shape) + '.' + ext
        new_filename = os.path.join(base_path, 'resized', filename)
        cv2.imwrite(new_filename, image)

    def _resize(self, image: np.array) -> np.array:
        """Whole pipeline of image resizing."""
        if self._mode == 'naive':
            image = self.nn_inter_naive(image, self._new_shape)
        else:
            image = self.nn_inter_vectorized(image, self._new_shape)
        return image
