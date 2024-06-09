# Built-int
import importlib.util
import logging
import sys
import random
import shutil

from pathlib import Path
from typing import Optional, Union, NoReturn, TypeVar
from types import ModuleType

# Extern libs
import torch
import numpy as np


logging.getLogger(__name__)

PathLike = TypeVar("PathLike", str, Path)


def set_root_logger(logfile: Optional[Union[str, Path]] = None,
                    loglevel: int = logging.DEBUG) -> logging.Logger:
    # Setup formats
    logging_format = '[%(asctime)s - %(levelname)-8s - %(funcName)s:%(module)s] %(message)s'
    logging_datefrmt = '%Y-%m-%d %H:%M:%S'
    h_format = logging.Formatter(logging_format, datefmt=logging_datefrmt)

    # Setup Logger
    logger = logging.getLogger('root')
    logger.setLevel(loglevel)

    # Setup handlers
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(h_format)
    stream_handler.setLevel(loglevel)
    logger.addHandler(stream_handler)

    if logfile:
        file_handler = logging.FileHandler(logfile)
        file_handler.setFormatter(h_format)
        file_handler.setLevel(loglevel)
        logger.addHandler(file_handler)

    return logger


def make_it_reproducible(random_seed: int = 42,
                         cudnn_determinisitc: bool = True,
                         cudnn_benchmark: bool = False,
                         cudnn_enabled: bool = False) -> NoReturn:
    '''
        Params:
            random_seed: integer value of random seed
            cudnn_determinisitc: if set to True then CuDNN will pick deterministic algorithms
            cudnn_benchmark: if set to True then CuDNN library will benchmark several algorithms
                             and pick that which it found to be fastest, what may lead to non-reproducible results

            Check for more details:
                https://discuss.pytorch.org/t/what-is-the-differenc-between-cudnn-deterministic-and-cudnn-benchmark/38054/3
    '''

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    torch.cuda.manual_seed_all(random_seed)

    # torch.backends.cudnn.deterministic = cudnn_determinisitc
    # torch.backends.cudnn.benchmark = cudnn_benchmark
    print('Fixed all random things with randow seed', random_seed)


def load_cfg(cfg_path: PathLike) -> ModuleType:
    cfg_path_str = str(cfg_path)
    spec = importlib.util.spec_from_file_location("config", cfg_path_str)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


def copy_cfg(src: PathLike, dst: PathLike) -> NoReturn:
    shutil.copy(src, dst)
