from abc import ABC, abstractmethod
from typing import Dict

import torch
import pytorch_lightning as pl


class RLModel(torch.nn.Module, ABC):

    def __init__(self, config: Dict):
        super(RLModel, self).__init__()
        self.config = config

    @abstractmethod
    def forward(self, trajectory: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """
        Given a trajectory, predict the future. Which information of the trajectory is needed depends on the specific
        subclass implementation. This means actions as well as states, rewards and done flags can potentially be
        predicted. Thus, this can be a base class for dynamics models as well as policies or value functions.
        :param trajectory: A tensor containing all information needed by the environment model to predict the future
        :param context: A tensor containing additional information for generating the future
        :return: A tensor representing the predicted future given `trajectory`
        """
        pass

