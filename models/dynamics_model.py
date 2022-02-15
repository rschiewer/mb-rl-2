from abc import ABC, abstractmethod
from typing import Dict, Any

import torch
import pytorch_lightning as pl


class DynamicsModel(torch.nn.Module, ABC):

    def __init__(self):
        super(DynamicsModel, self).__init__()

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
                   n_warmup: int = 1) -> Any:
        pass

