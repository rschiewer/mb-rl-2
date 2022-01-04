from typing import Any

from torch.utils.data import Dataset, DataLoader

from training.collector import Collector


class OfflineRLCollector(Collector):

    def __init__(self, dataset: Dataset, num_collect: int, shuffle: bool = True):
        super(OfflineRLCollector, self).__init__(num_collect)

        self.dataset = dataset
        self.data_loader = DataLoader(dataset, num_collect, shuffle, num_workers=0)
        self._data_iter = iter(self.data_loader)

    def collect(self) -> Any:
        batch = next(self._data_iter)
        return batch

