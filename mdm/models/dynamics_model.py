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
                   o_ground_truth: torch.Tensor,
                   a_ground_truth: torch.Tensor,
                   r_ground_truth: torch.Tensor,
                   term_ground_truth: torch.Tensor,
                   mask: torch.Tensor,
                   optimizer: torch.optim.Optimizer,
                   **kwargs) -> Dict[str, torch.Tensor]:
        results = self._train_step(o_ground_truth, a_ground_truth, r_ground_truth, term_ground_truth, mask, optimizer,
                                   **kwargs)
        self._current_train_step += 1
        return results

    def eval_step(self,
                  o_ground_truth: torch.Tensor,
                  a_ground_truth: torch.Tensor,
                  r_ground_truth: torch.Tensor,
                  term_ground_truth: torch.Tensor,
                  mask: torch.Tensor,
                  **kwargs) -> Dict[str, torch.Tensor]:
        return self._eval_step(o_ground_truth, a_ground_truth, r_ground_truth, term_ground_truth, mask, **kwargs)

    @abstractmethod
    def _train_step(self,
                    o_ground_truth: torch.Tensor,
                    a_ground_truth: torch.Tensor,
                    r_ground_truth: torch.Tensor,
                    term_ground_truth: torch.Tensor,
                    mask: torch.Tensor,
                    optimizer: torch.optim.Optimizer,
                    **kwargs) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def _eval_step(self,
                   o_ground_truth: torch.Tensor,
                   a_ground_truth: torch.Tensor,
                   r_ground_truth: torch.Tensor,
                   term_ground_truth: torch.Tensor,
                   mask: torch.Tensor,
                   **kwargs) -> Dict[str, torch.Tensor]:
        pass
