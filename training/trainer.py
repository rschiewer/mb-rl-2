from typing import Callable

import torch


from training.driver import Driver


class Trainer:

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, collector: Driver,
                 eval_fn: Callable = None, scheduler: object = None):
        self.model = model
        self.optimizer = optimizer
        self.collector = collector
        self.eval_fn = eval_fn
        self.scheduler = scheduler