from torch.utils.data import Dataset, DataLoader

from mdm.training.driver import Driver
from mdm.memory.trajectory_memory import TrajectoryMemory


class OfflineRLDriver(Driver):

    def __init__(self,
                 dataset: Dataset,
                 batch_size: int,
                 shuffle: bool = True):
        super(OfflineRLDriver, self).__init__(batch_size)

        self.dataset = dataset
        self.batch_size = batch_size
        self.data_loader = DataLoader(dataset, batch_size, shuffle, num_workers=0)
        self._data_iter = iter(self.data_loader)

    def interact(self,
                 n_episodes: int,
                 *args,
                 **kwargs) -> TrajectoryMemory:
        if n_episodes != self.batch_size:
            raise ValueError(f'Expected n_episodes ({n_episodes}) to equal self.batch_size ({self.batch_size})')

        batch = next(self._data_iter)

        return TrajectoryMemory(batch)

