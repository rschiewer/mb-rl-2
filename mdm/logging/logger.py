from __future__ import annotations

from abc import ABC, abstractmethod
import re
from typing import Any, Dict, Union
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class Scope:
    _DEFAULT = 'default'
    _TRAIN = 'train'
    _TEST = 'test'
    _PARAMETERS = 'parameters'
    _HYPERPARAMETERS = 'hyperparameters'
    _DATA = 'data'
    _MISC = 'misc'
    scope_check = re.compile('^[a-zA-Z0-9_/()=]+$')

    def __init__(self,
                 description: Union[Scope, str]):
        self._set_descr(description)

    def _set_descr(self, description: Union[Scope, str]):
        description = str(description)
        if not self.scope_check.match(description):
            raise ValueError(f'Scope description {description} contains invalid characters.')
        self._descr = description.lower()

    def __str__(self):
        return self._descr

    def __truediv__(self, other):
        other = str(other)
        return Scope(self._descr + '/' + other)

    @staticmethod
    def DEFAULT():
        return Scope(Scope._DEFAULT)

    @staticmethod
    def TRAIN():
        return Scope(Scope._TRAIN)

    @staticmethod
    def TEST():
        return Scope(Scope._TEST)

    @staticmethod
    def PARAMETERS():
        return Scope(Scope._PARAMETERS)

    @staticmethod
    def HYPERPARAMETERS():
        return Scope(Scope._HYPERPARAMETERS)

    @staticmethod
    def DATA():
        return Scope(Scope._DATA)

    @staticmethod
    def MISC():
        return Scope(Scope._MISC)


class Logger(ABC):

    def __init__(self):
        self._run_id = -1

    def __del__(self):
        self.stop_session()

    @abstractmethod
    def start_session(self):
        pass

    @abstractmethod
    def stop_session(self):
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
    def log_file(self,
                 path: Union[str, Path],
                 scope: Scope,
                 time_step: int = None):
        pass

    @abstractmethod
    def log_plot(self,
                 figure: Figure,
                 scope: Scope,
                 time_step: int = None):
        pass