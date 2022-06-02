from typing import Callable, Tuple, Dict, Union
from abc import ABC

import torch
from tqdm import tqdm
import numpy as np
from pathlib import Path

from mdm.logging.logger import Logger, Scope
from mdm.models.dynamics_model import DynamicsModel


class DynamicsModelTrainer(ABC):

    def __init__(self,
                 model: DynamicsModel,
                 optimizer: torch.optim.Optimizer,
                 get_batch_train: Callable[..., Tuple[np.array, np.array, np.array, np.array]],
                 get_batch_test: Callable[..., Tuple[np.array, np.array, np.array, np.array]],
                 n_warmup_steps: int,
                 eval_interval: int = None,
                 scheduler: object = None,
                 logger: Logger = None,
                 **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.get_batch_train = get_batch_train
        self.get_batch_test = get_batch_test
        self.n_warmup_steps = n_warmup_steps
        self.eval_interval = eval_interval
        self.scheduler = scheduler
        self.logger = logger

    def train(self,
              n_train_steps: int,
              progress_bar: bool = False,
              checkpoint_path: Union[str, Path] = None,
              **kwargs):
        device = self.model.device
        step_iter = range(n_train_steps)
        last_eval_losses = {'N/A': torch.tensor(0, device=self.model.device)}

        if progress_bar:
            step_iter = tqdm(step_iter, desc='Training Progress')

        if self.logger:
            self.logger.setup()

        for i_step in step_iter:
            s, a, r, term = self.get_batch_train()
            s, a, r, term = [torch.from_numpy(x).to(device=device, dtype=torch.float32) for x in (s, a, r, term)]

            compatible, msg = self.model.input_compatible(s, a, r)
            if not compatible:
                raise ValueError(msg)

            #with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            #    with record_function("model_training"):
            train_losses = self.model.train_step(s, a, r, term, self.optimizer, self.n_warmup_steps)
            #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

            if progress_bar:
                self._update_progressbar_descr(last_eval_losses, train_losses, step_iter)

            if self.logger:
                r_stats = {'r_raw': r, 'r_sum_ep': r.sum(axis=1), 'r_sum': r.sum(), 'r_mean': r.mean()}
                #self.logger.log(self._to_np(r_stats), Scope.TRAIN, i_step)
                self.logger.log(self._to_np(train_losses), Scope.TRAIN(), i_step)

            if self.scheduler:
                self.scheduler.step()

            if self.eval_interval is not None and i_step % self.eval_interval == 0:
                s, a, r, term = self.get_batch_test()
                s, a, r, term = [torch.from_numpy(x).to(device=device, dtype=torch.float32) for x in (s, a, r, term)]

                compatible, msg = self.model.input_compatible(s, a, r)
                if not compatible:
                    raise ValueError(msg)

                eval_losses = self.model.eval_step(s, a, r, term, self.n_warmup_steps)

                last_total_loss = last_eval_losses.get('total', np.inf)
                if checkpoint_path and eval_losses['total'] < last_total_loss:
                    torch.save(self.model, Path(checkpoint_path).parent / 'checkpoint.ptmdl')

                last_eval_losses = eval_losses

                if self.logger:
                    r_stats = {'r_raw': r, 'r_sum_ep': r.sum(axis=1), 'r_sum': r.sum(), 'r_mean': r.mean()}
                    #self.logger.log(self._to_np(r_stats), Scope.TEST, i_step)
                    self.logger.log(self._to_np(eval_losses), Scope.TEST(), i_step)

        if self.logger:
            self.logger.teardown()

    @staticmethod
    def _update_progressbar_descr(eval_losses: Dict[str, torch.Tensor],
                                  train_losses: Dict[str, torch.Tensor],
                                  pbar: tqdm):
        train_losses = {k: v for k, v in train_losses.items() if v.ndim <= 1}
        eval_losses = {k: v for k, v in eval_losses.items() if v.ndim <= 1}
        l_train = [f'{n}: {l:2.3e}' for n, l in train_losses.items()]
        l_eval = [f'{n}: {l:2.3e}' for n, l in eval_losses.items()]
        descr = 'train_losses = ' + ', '.join(l_train) + ' | val_losses = ' + ', '.join(l_eval)
        pbar.set_description(descr)

    @staticmethod
    def _to_np(data_dict: Dict[str, torch.Tensor]):
        return {k: v.detach().cpu().numpy() for k, v in data_dict.items()}




