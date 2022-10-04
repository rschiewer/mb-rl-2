from enum import Enum, auto

from mdm.training.driver import Driver
from mdm.memory.trajectory_memory import TrajectoryMemory


class SamplingType(Enum):
    SEQUENTIAL = auto()
    RANDOM = auto()
    PRIORITIZED = auto()


class OfflineRLDriver(Driver):

    def __init__(self,
                 memory: TrajectoryMemory,
                 sampling_type: SamplingType = SamplingType.RANDOM):
        self.memory = memory
        self.sampling_type = sampling_type
        self._i_curr = 0

        #if shuffle:
        #    self.memory = self.memory.shuffle()

    def interact(self,
                 n_episodes: int,
                 *args,
                 **kwargs) -> TrajectoryMemory:
        if self.sampling_type == SamplingType.SEQUENTIAL:
            i_start = self._i_curr
            diff = self._i_curr + n_episodes - len(self.memory)
            if diff > 0:
                i_end = len(self.memory)
            else:
                i_end = self._i_curr + n_episodes
            self._incr_ptr(n_episodes)
            sub_memory = self.memory[i_start: i_end]
        elif self.sampling_type == SamplingType.RANDOM:
            sub_memory = self.memory.sample(n_episodes, prioritized=False)
        else:
            sub_memory = self.memory.sample(n_episodes, prioritized=True)

        return sub_memory

    def _incr_ptr(self,
                  step: int):
        self._i_curr += step
        if self._i_curr > len(self.memory):
            self._i_curr = 0
            #if self.shuffle:
            #    self.memory = self.memory.shuffle()

