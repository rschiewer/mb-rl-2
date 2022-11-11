from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
from functools import reduce
from collections import namedtuple

import torch

from mdm.utils.torch_tools import StatefulTrainingModule


class DynamicsModel(StatefulTrainingModule, ABC):

    def __init__(self):
        super(DynamicsModel, self).__init__()

    @abstractmethod
    def train_step(self,
                   o_ground_truth: torch.Tensor,
                   a_ground_truth: torch.Tensor,
                   r_ground_truth: torch.Tensor,
                   term_ground_truth: torch.Tensor,
                   mask: torch.Tensor,
                   optimizer: torch.optim.Optimizer) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def eval_step(self,
                  o_ground_truth: torch.Tensor,
                  a_ground_truth: torch.Tensor,
                  r_ground_truth: torch.Tensor,
                  term_ground_truth: torch.Tensor,
                  mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        pass