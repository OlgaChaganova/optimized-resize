import os
import typing as tp

import cv2
import numpy as np


class ImageResizer(object):
    def __init__(self, img_filename: str, new_shape: tp.Tuple[int, int]):
        self._img_filename = img_filename
        self._new_shape = new_shape

    def resize(self, return_image: bool = False):
        image = self._read_image()
        resized_image = self._resize(image)
        if return_image:
            return resized_image
        self._save_image(resized_image)

    @classmethod
    def nearest_neighbour(cls, image: np.array, new_shape: tp.Tuple[int, int]) -> np.array:
        """Only DOWNSCALE"""
        img_width, img_height = image.shape
        new_width, new_height = new_shape
        x_ratio = img_width / new_width
        y_ratio = img_height / new_height
        x_coords = np.ceil(np.arange(new_width) * x_ratio).astype(np.int32)
        y_coords = np.ceil(np.arange(new_height) * y_ratio).astype(np.int32)
        return image[x_coords[:-5]][:, y_coords[:-5]]

    @classmethod
    def bicubic_interpolation(cls, image: np.array, new_shape: tp.Tuple[int, int]) -> np.array:
        ...

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
        image = cv2.resize(image, dsize=self._new_shape, interpolation=cv2.INTER_CUBIC)
        return image