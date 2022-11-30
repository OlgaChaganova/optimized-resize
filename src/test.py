import logging
import os
import typing as tp
from datetime import datetime
from time import time

import cv2
from omegaconf import OmegaConf, DictConfig, ListConfig

from resizer import ImageResizer
from utils.system_info import get_system_info


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
        image = cv2.imread(test_case['img'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logging.info(f'Test for {test_num} with original shape {image.shape}')
        for size in test_case['sizes'].values():
            latency_vector = timeit(ImageResizer.nn_inter_vectorized, image=image, new_shape=size)
            latency_naive = timeit(ImageResizer.nn_inter_naive, image=image, new_shape=size)
            logging.info(
                f'New size {size}; NAIVE latency: {latency_naive:.4f} ms; VECTORIZED latency: {latency_vector:.4f} ms',
            )
        logging.info('\n')


if __name__ == '__main__':
    log_file = 'inference-test-' + datetime.today().strftime('%Y-%m-%d--%H:%M:%S')
    logging.basicConfig(filename=os.path.join('tests', 'results', log_file), level=logging.INFO)

    test_cases = OmegaConf.load(os.path.join('tests', 'test_cases.yml'))
    test_latency(test_cases)
