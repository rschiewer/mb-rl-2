from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class Scope(Enum):
    DEFAULT = 0
    TRAIN = 1
    TEST = 2
    PARAMETERS = 3
    MISC = 4

    def __str__(self):
        return self.name.lower()


class Logger(ABC):

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def teardown(self):
        pass

    @abstractmethod
    def log(self, message: Dict[str, Any], scope: Scope, time_step: int = None):
        pass

    @abstractmethod
    def log_object(self, object: Any, scope: Scope, time_step: int = None):
        pass

    @abstractmethod
    def log_plot(self, figure: Figure, scope: Scope, time_step: int = None):
        pass