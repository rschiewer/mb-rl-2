from inspect import stack
from pathlib import Path
from typing import Union
import sys

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