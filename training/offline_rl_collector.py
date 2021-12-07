from typing import Any

from torch.utils.data import Dataset, DataLoader

from training.collector import Collector


class OfflineRLCollector(Collector):

    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True):
        super(OfflineRLCollector, self).__init__(batch_size)

        self.dataset = dataset
        self.data_loader = DataLoader(dataset, batch_size, shuffle, num_workers=0)
        self._data_iter = iter(self.data_loader)

    def get_batch(self) -> Any:
        batch = next(self._data_iter)
        return batch

