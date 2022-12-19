from typing import List, Dict
from enum import Enum, auto
import random

from mdm.training.driver import Driver
from mdm.memory.trajectory_memory import TrajectoryMemory
from mdm.utils.utils import DataType


class SamplingType(Enum):
    SEQUENTIAL = auto()
    RANDOM = auto()


class OfflineRLDriver(Driver):

    def __init__(self,
                 memory: List[Dict[str, DataType]],
                 sampling_type: SamplingType = SamplingType.RANDOM,
                 shuffle: bool = False):
        self.memory = memory
        self.sampling_type = sampling_type
        self._i_curr = 0

        if shuffle:
            random.shuffle(self.memory)

    def interact(self,
                 n_episodes: int,
                 *args,
                 **kwargs) -> List[Dict[str, DataType]]:
        if self.sampling_type == SamplingType.SEQUENTIAL:
            i_start = self._i_curr
            diff = self._i_curr + n_episodes - len(self.memory)
            if diff > 0:
                i_end = len(self.memory)
            else:
                i_end = self._i_curr + n_episodes
            self._incr_ptr(n_episodes)
            sub_memory = self.memory[i_start: i_end]
        else:
            sub_memory = random.choices(self.memory, k=n_episodes)

        return sub_memory

    def _incr_ptr(self,
                  step: int):
        self._i_curr += step
        if self._i_curr > len(self.memory):
            self._i_curr = 0
            #if self.shuffle:
            #    self.memory = self.memory.shuffle()

