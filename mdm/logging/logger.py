from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict


class Scope(Enum):
    DEFAULT = 0
    TRAIN = 1
    TEST = 2
    MISC = 3


class Logger(ABC):

    @abstractmethod
    def log(self, message: Dict[str, Any], scope: Scope, time_step: int = None):
        pass