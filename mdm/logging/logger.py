from abc import ABC, abstractmethod
import re
from typing import Any, Dict

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class Scope:
    DEFAULT = 'default'
    TRAIN = 'train'
    TEST = 'test'
    PARAMETERS = 'parameters'
    MISC = 'misc'
    scope_check = re.compile('^[a-zA-Z0-9_/]+$')

    def __init__(self,
                 description: str):
        if not self.scope_check.match(description):
            raise ValueError(f'Provided scope description {description} contains invalid characters.')
        self._descr = description.lower()

    def __str__(self):
        return self._descr


class Logger(ABC):

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def teardown(self):
        pass

    @abstractmethod
    def log(self,
            message: Dict[str, Any],
            scope: Scope,
            time_step: int = None):
        pass

    @abstractmethod
    def log_object(self,
                   object: Any,
                   scope: Scope,
                   time_step: int = None):
        pass

    @abstractmethod
    def log_plot(self,
                 figure: Figure,
                 scope: Scope,
                 time_step: int = None):
        pass