from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
from functools import reduce
from collections import namedtuple

import torch
from torch import device, dtype

Placeholder = namedtuple('placeholder', 'device')


class DynamicsModel(torch.nn.Module, ABC):

    def __init__(self):
        super(DynamicsModel, self).__init__()

    @property
    def device(self):
        ph = Placeholder(None)
        first_param = reduce(lambda a, b: a if a.device == b.device else ph, self.parameters())
        if type(first_param) is Placeholder:
            raise RuntimeError('Model has parameters on multiple devices')

        return first_param.device

    @abstractmethod
    def forward(self,
                start_states: torch.Tensor,
                actions: torch.Tensor,
                context: torch.Tensor = None) -> Any:
        pass

    @abstractmethod
    def train_step(self,
                   s_ground_truth: torch.Tensor,
                   a_ground_truth: torch.Tensor,
                   r_ground_truth: torch.Tensor,
                   optimizer: torch.optim.Optimizer,
                   n_warmup: int = 1) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def eval_step(self,
                  s_ground_truth: torch.Tensor,
                  a_ground_truth: torch.Tensor,
                  r_ground_truth: torch.Tensor,
                  n_warmup: int = 1) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def input_compatible(self,
                         s_ground_truth: torch.Tensor,
                         a_ground_truth: torch.Tensor,
                         r_ground_truth: torch.Tensor) -> Tuple[bool, str]:
        pass
