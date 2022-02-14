from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np


class Driver(ABC):

    @abstractmethod
    def interact(self,
                 *args,
                 **kwargs) -> List[Dict[str: np.ndarray, str: np.ndarray, str: np.ndarray, str: np.ndarray]]:
        """
        Collect some experience by interacting with the environment.
        :return: A list containing for every episode a dict {'s': states, 'a': actions, 'r': rewards,
            'terminal': done_flags} where every value in the dict is a numpy array
        """
        pass
