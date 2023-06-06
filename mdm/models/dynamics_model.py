from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
from functools import reduce
from collections import namedtuple

import torch

from mdm.models.building_blocks import ManagedStatefulTrainingModule


class DynamicsModel(torch.nn.Module, ABC):

    def __init__(self):
        super(DynamicsModel, self).__init__()
        self._current_train_step = None

    def prepare_for_training(self):
        self._current_train_step = 0
        for m in self.modules():
            if isinstance(m, ManagedStatefulTrainingModule):
                m.prepare_for_training()

    def train_step(self,
                   training_data: Dict[str, torch.Tensor],
                   optimizer: torch.optim.Optimizer,
                   **kwargs) -> Any:
        results = self._train_step(training_data, optimizer, **kwargs)
        self._current_train_step += 1
        return results

    def eval_step(self,
                  training_data: Dict[str, torch.Tensor],
                  **kwargs) -> Any:
        return self._eval_step(training_data, **kwargs)

    @abstractmethod
    def _train_step(self,
                    training_data: Dict[str, torch.Tensor],
                    optimizer: torch.optim.Optimizer,
                    **kwargs) -> Any:
        pass

    @abstractmethod
    def _eval_step(self,
                   training_data: Dict[str, torch.Tensor],
                   **kwargs) -> Any:
        pass
