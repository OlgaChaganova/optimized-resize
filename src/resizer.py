import math
import os
import typing as tp

import cv2
import numpy as np


class ImageResizer(object):
    def __init__(
        self,
        img_filename: str,
        new_shape: tp.Tuple[int, int],
        mode: tp.Literal['naive_nearest', 'vectorized_nearest', 'naive_bilinear', 'vectorized_bilinear'],
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
        img_height, img_width = image.shape
        new_height, new_width = new_shape

        # -1 because we need to get coordinates
        x_ratio = (img_width - 1) / (new_width - 1)
        y_ratio = (img_height - 1) / (new_height - 1)
        x_coords = np.ceil(np.arange(new_width - 1) * x_ratio).astype(np.int32)
        y_coords = np.ceil(np.arange(new_height - 1) * y_ratio).astype(np.int32)
        return image[y_coords][:, x_coords]

    @classmethod
    def nn_inter_naive(cls, image: np.array, new_shape: tp.Tuple[int, int]) -> np.array:
        """Nearest neigbour interpolation (upscale and downscale). Naive implementation"""
        img_height, img_width = image.shape
        new_height, new_width = new_shape
        x_ratio = img_width / new_width
        y_ratio = img_height / new_height
        resized_image = np.zeros([new_width, new_height])
        for i in range(new_width):
            for j in range(new_height):
                resized_image[i, j] = image[int(j * y_ratio), int(i * x_ratio)]
        return resized_image

    @classmethod
    def bilinear_inter_vectorized(cls, image: np.array, new_shape: tp.Tuple[int, int]) -> np.array:
        """Bilinear image interpolation. Vectorized implementation"""
        img_height, img_width = image.shape
        new_height, new_width = new_shape

        image = image.ravel()  # flatten to 1d array
        total_image_len = len(image)

        x_ratio = float(img_width - 1) / (new_width - 1)
        y_ratio = float(img_height - 1) / (new_height - 1)

        y, x = np.divmod(np.arange(new_height * new_width), new_width)
        x_l = np.floor(x_ratio * x).astype('int32')
        y_l = np.floor(y_ratio * y).astype('int32')

        x_h = np.ceil(x_ratio * x).astype('int32')
        y_h = np.ceil(y_ratio * y).astype('int32')

        x_weight = (x_ratio * x) - x_l
        y_weight = (y_ratio * y) - y_l

        a = image[np.clip(y_l * img_width + x_l, 0, total_image_len - 1)]
        b = image[np.clip(y_l * img_width + x_h, 0, total_image_len - 1)]
        c = image[np.clip(y_h * img_width + x_l, 0, total_image_len - 1)]
        d = image[np.clip(y_h * img_width + x_h, 0, total_image_len - 1)]

        resized = a * (1 - x_weight) * (1 - y_weight) + \
                  b * x_weight * (1 - y_weight) + \
                  c * y_weight * (1 - x_weight) + \
                  d * x_weight * y_weight

        return resized.reshape(new_height, new_width).astype(np.uint8)

    @classmethod
    def bilinear_inter_naive(cls, image: np.array, new_shape: tp.Tuple[int, int]) -> np.array:
        """Bilinear image interpolation. Naive implementation"""
        img_height, img_width = image.shape
        new_height, new_width = new_shape
        resized_image = np.empty([new_height, new_width])
        x_ratio = float(img_width - 1) / (new_width - 1)
        y_ratio = float(img_height - 1) / (new_height - 1)
        for i in range(new_height):
            for j in range(new_width):
                x_l, y_l = math.floor(x_ratio * j), math.floor(y_ratio * i)
                x_h, y_h = math.ceil(x_ratio * j), math.ceil(y_ratio * i)
                if x_h >= img_width:
                    x_h = img_width - 1
                if y_h >= img_height:
                    y_h = img_height - 1
                x_weight = (x_ratio * j) - x_l
                y_weight = (y_ratio * i) - y_l
                a = image[y_l, x_l]
                b = image[y_l, x_h]
                c = image[y_h, x_l]
                d = image[y_h, x_h]
                pixel = a * (1 - x_weight) * (1 - y_weight) + \
                        b * x_weight * (1 - y_weight) + \
                        c * y_weight * (1 - x_weight) + \
                        d * x_weight * y_weight
                resized_image[i][j] = pixel
        return resized_image.astype(np.uint8)

    def _read_image(self) -> np.array:
        """Read image from file and make it binary."""
        image = cv2.imread(self._img_filename, cv2.IMREAD_GRAYSCALE)
        _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        image = (image // 255).astype(np.uint8)
        assert (np.unique(image) == np.array([0, 1], dtype=np.uint8)).all()
        return image

    def _save_image(self, image: np.array) -> np.array:
        """Save resized image to file with new shape in written in filename."""
        base_path, filename = os.path.split(self._img_filename)
        name, ext = filename.split('.')
        filename = name + str(self._new_shape) + '-' + self._mode.split('_')[-1] + '.' + ext
        new_filename = os.path.join(base_path, 'resized', filename)
        cv2.imwrite(new_filename, image * 255)

    def _resize(self, image: np.array) -> np.array:
        """Whole pipeline of image resizing."""
        if self._mode == 'naive_nearest':
            image = self.nn_inter_naive(image, self._new_shape)
        elif self._mode == 'vectorized_nearest':
            image = self.nn_inter_vectorized(image, self._new_shape)
        elif self._mode == 'naive_bilinear':
            image = self.bilinear_inter_naive(image, self._new_shape)
        elif self._mode == 'vectorized_bilinear':
            image = self.bilinear_inter_vectorized(image, self._new_shape)
        assert (np.unique(image) == np.array([0, 1], dtype=np.uint8)).all()
        return image
