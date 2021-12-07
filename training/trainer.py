from typing import Callable

import torch
import gym


class Trainer:

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, get_batch: Callable,
                 eval_fn: Callable = None, scheduler: object = None):
        self.model = model
        self.optimizer = optimizer
        self.get_batch = get_batch
        self.eval_fn = eval_fn
        self.scheduler = scheduler