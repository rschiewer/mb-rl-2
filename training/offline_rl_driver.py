from typing import Any

from torch.utils.data import Dataset, DataLoader

from training.driver import Driver


class OfflineRLDriver(Driver):

    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True):
        super(OfflineRLDriver, self).__init__(batch_size)

        self.dataset = dataset
        self.data_loader = DataLoader(dataset, batch_size, shuffle, num_workers=0)
        self._data_iter = iter(self.data_loader)

    def interact(self) -> Any:
        batch = next(self._data_iter)
        return batch

