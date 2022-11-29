import argparse
import os

import cv2
import numpy as np


def read_image(filename: str) -> np.array:
    image = cv2.imread(filename)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def save_image(filename: str, image: np.array) -> np.array:
    base_path, filename = os.path.split(filename)
    new_filename = os.path.join(base_path, 'resized' + filename)
    cv2.imwrite(new_filename, image)


def parse():
    parser = argparse.ArgumentParser('Effective resize of the binary image')
    parser.add_argument('input_img', type=str, help='Input image to be resized.')
    parser.add_argument('img_w_h', type=int, nargs='+', help='Size of the image after resize')
    return parser.parse_args()


def resize(image: np.array, new_shape: list | tuple) -> np.array:
    image = cv2.resize(image, dsize=new_shape)
    return image


if __name__ == '__main__':
    args = parse()
    image = read_image(args.input_img)
    resized_image = resize(image, args.img_w_h)
    save_image(args.input_img, resized_image)
