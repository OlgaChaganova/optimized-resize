import logging
import os
import typing as tp
from datetime import datetime
from time import time

import cv2
from omegaconf import OmegaConf, DictConfig, ListConfig


def timeit(func: tp.Callable, **kwargs) -> float:
    start = time()
    _ = func(**kwargs)
    end = time()
    return end - start


def test_latency(test_cases: DictConfig | ListConfig):
    for test_num, test_case in test_cases.items():
        image = cv2.imread(test_case['img'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logging.info(f'Test for {test_num} with original shape {image.shape}')
        for size in test_case['sizes'].values():
            latency = timeit(cv2.resize, src=image, dsize=size)
            logging.info(f'New size {size}; latency: {latency}')


if __name__ == '__main__':
    log_file = 'inference-test-' + datetime.today().strftime('%Y-%m-%d--%H:%M:%S')
    logging.basicConfig(filename=os.path.join('results', log_file), level=logging.INFO)

    test_cases = OmegaConf.load('test_cases.yml')
    test_latency(test_cases)
