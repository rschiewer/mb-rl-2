from typing import List, Dict
from abc import ABC, abstractmethod

from mdm.utils.utils import DataType


class Driver(ABC):

    @abstractmethod
    def interact(self,
                 n_episodes: int,
                 *args,
                 **kwargs) -> List[Dict[str, DataType]]:
        """
        Collect some experience by interacting with the environment.
        :return: A list of dicts containing the collected data
        """
        pass
