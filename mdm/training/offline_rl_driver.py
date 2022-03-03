from torch.utils.data import Dataset, DataLoader

from mdm.training.driver import Driver
from mdm.memory.trajectory_memory import TrajectoryMemory


class OfflineRLDriver(Driver):

    def __init__(self,
                 memory: TrajectoryMemory,
                 shuffle: bool = True):
        self.memory = memory
        self._i_curr = 0

    def interact(self,
                 n_episodes: int,
                 *args,
                 **kwargs) -> TrajectoryMemory:
        i_start = self._i_curr
        diff = self._i_curr + n_episodes - len(self.memory)
        if diff > 0:
            i_end = len(self.memory)
        else:
            i_end = self._i_curr + n_episodes
        sub_memory = self.memory[i_start: i_end]

        self._incr_ptr(n_episodes)

        return sub_memory

    def _incr_ptr(self,
                  step: int):
        self._i_curr += step
        if self._i_curr > len(self.memory):
            self._i_curr = 0

