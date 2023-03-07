from typing import Union
from abc import abstractmethod

from mdm.utils.gym_wrappers import *


class Policy:

    @abstractmethod
    def __call__(self,
                 env: Union[CacheLastStepEnv, CacheLastStepVecEnv]):
        pass
