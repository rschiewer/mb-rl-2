from inspect import stack
from pathlib import Path
from typing import Union
import sys

import numpy as np
import yaml


def here() -> Path:
    filename = stack()[1].filename
    parent_path = Path(filename).parent
    return parent_path


def add_to_pythonpath(relative_path: str):
    relative_path = Path(relative_path)
    caller_path = Path(stack()[1].filename).parent
    full_path = caller_path / relative_path
    sys.path.append(full_path)


def load_yaml(path: Union[str, Path]):
    path = Path(path)
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config


def np_one_hot(x: np.array, n_categories: int):
    if not np.issubdtype(x.dtype, np.integer):
        raise ValueError('Only integer arrays can be converted to one-hot encoding')

    x = np.squeeze(x, axis=-1)  # remove possible redundant 1-dim data dimension
    x_onehot = np.zeros((*x.shape, n_categories))
    x = x[..., np.newaxis]  # make sure x_onehot and x have same number of dimensions
    np.put_along_axis(x_onehot, x, 1, axis=-1)  # use x as index array for x_onehot and put 1 at respective indices

    return x_onehot