from abc import ABC, abstractmethod
from typing import Dict, List, AnyStr

import numpy as np

from memory.trajectory_memory import TrajectoryMemory


class Driver(ABC):

    @abstractmethod
    def interact(self,
                 n_episodes: int,
                 *args,
                 **kwargs) -> TrajectoryMemory:
        """
        Collect some experience by interacting with the environment.
        :return: A trajectory memory object containing the collected data
        """
        pass
