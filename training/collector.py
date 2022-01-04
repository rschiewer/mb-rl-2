from abc import ABC, abstractmethod
from typing import Any


class Collector(ABC):

    def __init__(self, num_collect: int):
        self.num_collect = num_collect

    @abstractmethod
    def collect(self) -> Any:
        """
        Collect some experience.
        :return: The experience data, depending on the collector this can be e.g. trajectories or single transitions
        """
        pass
