from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
from functools import reduce
from collections import namedtuple

import torch
from torch import device, dtype


from mdm.utils.torch_tools import DeviceMixin


class DynamicsModel(torch.nn.Module, DeviceMixin, ABC):

    def __init__(self):
        super(DynamicsModel, self).__init__()

    @abstractmethod
    def train_step(self,
                   s_ground_truth: torch.Tensor,
                   a_ground_truth: torch.Tensor,
                   r_ground_truth: torch.Tensor,
                   term_ground_truth: torch.Tensor,
                   optimizer: torch.optim.Optimizer,
                   n_warmup: int = 1) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def eval_step(self,
                  s_ground_truth: torch.Tensor,
                  a_ground_truth: torch.Tensor,
                  r_ground_truth: torch.Tensor,
                  term_ground_truth: torch.Tensor,
                  n_warmup: int = 1) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def input_compatible(self,
                         s_ground_truth: torch.Tensor,
                         a_ground_truth: torch.Tensor,
                         r_ground_truth: torch.Tensor) -> Tuple[bool, str]:
        # TODO: add term_ground_truth here as well
        pass
