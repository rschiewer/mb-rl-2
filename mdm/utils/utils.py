from inspect import stack
from pathlib import Path
import sys


def here() -> Path:
    filename = stack()[1].filename
    parent_path = Path(filename).parent
    return parent_path

def add_to_pythonpath(relative_path: str):
    relative_path = Path(relative_path)
    caller_path = Path(stack()[1].filename).parent
    full_path = caller_path / relative_path
    sys.path.append(full_path)