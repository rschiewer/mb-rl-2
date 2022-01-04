from abc import ABC, abstractmethod
from typing import Any


class Driver(ABC):

    @abstractmethod
    def interact(self, *args, **kwargs) -> Any:
        """
        Collect some experience by interacting with the environment.
        :return: The experience data, depending on the driver this can be e.g. trajectories or single transitions
        """
        pass
