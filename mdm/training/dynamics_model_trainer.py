from typing import Callable, Tuple, Dict
from abc import ABC

import torch
from tqdm import tqdm
import numpy as np

from mdm.logging.logger import Logger, Scope
from mdm.models.dynamics_model import DynamicsModel


def one_hot(x: np.array, n_categories: int):
    if not np.issubdtype(x.dtype, np.integer):
        raise ValueError('Only integer arrays can be converted to one-hot encoding')

    x = np.squeeze(x, axis=-1)  # remove possible redundant 1-dim data dimension
    x_onehot = np.zeros((*x.shape, n_categories))
    x = x[..., np.newaxis]  # make sure x_onehot and x have same number of dimensions
    np.put_along_axis(x_onehot, x, 1, axis=-1)  # use x as index array for x_onehot and put 1 at respective indices

    return x_onehot


def flatten_and_unsqueeze(s: np.array, a: np.array, r: np.array, terminal: np.array):
    # flatten data dimensions if multiple or add explicit 1-sized data dimension if there is none
    d_s = np.prod(s.shape[2:]) if len(s.shape) >= 3 else 1
    d_a = np.prod(a.shape[2:]) if len(a.shape) >= 3 else 1
    d_r = np.prod(r.shape[2:]) if len(r.shape) >= 3 else 1
    d_term = np.prod(terminal.shape[2:]) if len(terminal.shape) >= 3 else 1

    s = s.reshape((*s.shape[:2], d_s))
    a = a.reshape((*a.shape[:2], d_a))  # mind: there is one more state than actions, rewards and terminal flags
    r = r.reshape((*r.shape[:2], d_r))
    terminal = terminal.reshape((*terminal.shape[:2], d_term))

    return s, a, r, terminal


class DynamicsModelTrainer(ABC):

    def __init__(self,
                 model: DynamicsModel,
                 optimizer: torch.optim.Optimizer,
                 get_batch_train: Callable[..., Tuple[np.array, np.array, np.array, np.array]],
                 get_batch_test: Callable[..., Tuple[np.array, np.array, np.array, np.array]],
                 warmup_steps: int,
                 eval_interval: int = None,
                 scheduler: object = None,
                 logger: Logger = None):
        self.model = model
        self.optimizer = optimizer
        self.get_batch_train = get_batch_train
        self.get_batch_test = get_batch_test
        self.warmup_steps = warmup_steps
        self.eval_interval = eval_interval
        self.scheduler = scheduler
        self.logger = logger

    def train(self,
              n_train_steps: int,
              progress_bar: bool = False):

        step_iter = range(n_train_steps)
        if progress_bar:
            step_iter = tqdm(step_iter, desc='Training Progress')
            last_eval_losses = {'N/A': torch.zeros(0)}

        for i_step in step_iter:
            s, a, r, terminal = self.get_batch_train()
            s, a, r = [torch.from_numpy(x).to(device=self.model.device, dtype=torch.float32) for x in (s, a, r)]

            compatible, msg = self.model.input_compatible(s, a, r)
            if not compatible:
                raise ValueError(msg)

            train_losses = self.model.train_step(s, a, r, self.optimizer, self.warmup_steps)

            if progress_bar:
                self._update_progressbar_descr(last_eval_losses, train_losses, step_iter)

            if self.logger is not None:
                train_losses.update({'r_raw': r, 'r_sum_ep': r.sum(axis=1), 'r_sum': r.sum(), 'r_mean': r.mean()})
                self.logger.log(train_losses, Scope.TRAIN, i_step)

            if self.scheduler is not None:
                self.scheduler.step()

            if self.eval_interval is not None and i_step % self.eval_interval == 0:
                s, a, r, terminal = self.get_batch_test()
                s, a, r = [torch.from_numpy(x).to(device=self.model.device, dtype=torch.float32) for x in (s, a, r)]

                compatible, msg = self.model.input_compatible(s, a, r)
                if not compatible:
                    raise ValueError(msg)

                eval_losses = self.model.eval_step(s, a, r, self.warmup_steps)

                if self.logger is not None:
                    eval_losses.update({'r_raw': r, 'r_sum_ep': r.sum(axis=1), 'r_sum': r.sum(), 'r_mean': r.mean()})
                    self.logger.log(eval_losses, Scope.TEST, i_step)

                if progress_bar:
                    last_eval_losses = eval_losses

    @staticmethod
    def _update_progressbar_descr(last_eval_losses: Dict[str, torch.Tensor],
                                  train_losses: Dict[str, torch.Tensor],
                                  pbar: tqdm):
        train_losses_stripped = [str(loss.detach().cpu().numpy()) for loss in train_losses.values()]
        eval_losses_stripped = [str(loss.detach().cpu().numpy()) for loss in last_eval_losses.values()]
        descr = 'train_losses = ' + ', '.join(train_losses_stripped) + ' | val_losses = ' + \
                ', '.join(eval_losses_stripped)
        pbar.set_description(descr)





