from abc import ABC, abstractmethod
from typing import Any


class Collector(ABC):

    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    @abstractmethod
    def get_batch(self) -> Any:
        """
        Provide a batch of experience.
        :return: The experience data, depending on the collector this can be e.g. trajectories or single transitions
        """
        pass
