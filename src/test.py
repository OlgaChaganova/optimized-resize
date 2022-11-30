import logging
import os
import typing as tp
from datetime import datetime
from time import time

import cv2
import numpy as np
from omegaconf import OmegaConf, DictConfig, ListConfig

from resizer import ImageResizer
from utils.system_info import get_system_info


def read_binary_image(filename: str) -> np.array:
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    image = (image // 255).astype(np.uint8)
    return image

def timeit(func: tp.Callable, **kwargs) -> float:
    """Measure inference time. Return time in ms."""
    start = time()
    _ = func(**kwargs)
    end = time()
    return (end - start) * 1e3


def test_latency(test_cases: DictConfig | ListConfig):
    system_info = get_system_info()
    for param, desc in system_info.items():
        logging.info(f'{param}: {desc}')
    logging.info('\n')
    for test_num, test_case in test_cases.items():
        image = read_binary_image(test_case['img'])
        logging.info(f'Test for {test_num} with original shape {image.shape}')
        for size in test_case['sizes'].values():
            latency_nn_naive = timeit(ImageResizer.nn_inter_naive, image=image, new_shape=size)
            latency_nn_vector = timeit(ImageResizer.nn_inter_vectorized, image=image, new_shape=size)
            latency_bilinear_naive = timeit(ImageResizer.bilinear_inter_naive, image=image, new_shape=size)
            latency_bilinear_vector = timeit(ImageResizer.bilinear_inter_vectorized, image=image, new_shape=size)
            logging.info(
                f'New size {size}; '
                f'NAIVE NEAREST: {latency_nn_naive:.4f} ms; '
                f'VECTORIZED NEAREST: {latency_nn_vector:.4f} ms; '
                f'NAIVE BILINEAR: {latency_bilinear_naive:.4f} ms; '
                f'VECTORIZED BILINEAR: {latency_bilinear_vector:.4f} ms'
            )
        logging.info('\n')


if __name__ == '__main__':
    log_file = 'inference-test-' + datetime.today().strftime('%Y-%m-%d--%H:%M:%S')
    logging.basicConfig(filename=os.path.join('tests', 'results', log_file), level=logging.INFO)

    test_cases = OmegaConf.load(os.path.join('tests', 'test_cases.yml'))
    test_latency(test_cases)
