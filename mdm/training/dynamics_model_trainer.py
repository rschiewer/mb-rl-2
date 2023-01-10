import re
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
                 get_batch_train: Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
                 get_batch_test: Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
                 eval_interval: int = None,
                 scheduler: object = None,
                 logger: Logger = None,
                 train_callback: Callable = None,
                 eval_callback: Callable = None,
                 **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.get_batch_train = get_batch_train
        self.get_batch_test = get_batch_test
        self.eval_interval = eval_interval
        self.scheduler = scheduler
        self.logger = logger
        self.train_callback = train_callback
        self.eval_callback = eval_callback

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
            self.logger.start_session()

        self.model.prepare_for_training()
        for i_step in step_iter:
            s, a, r, term, trunc, mask = self.get_batch_train(i_step)

            #with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            #    with record_function("model_training"):
            self.model.train()
            train_losses = self.model.train_step(s, a, r, term, mask, self.optimizer)
            #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

            if progress_bar:
                self._update_progressbar_descr(last_eval_losses, train_losses, step_iter)

            if self.logger:
                self.logger.log(self._to_np(train_losses), Scope.TRAIN(), i_step)

            if self.scheduler:
                self.scheduler.step()

            if self.eval_interval is not None and i_step % self.eval_interval == 0:
                s, a, r, term, trunc, mask = self.get_batch_test(i_step)
                #s, a, r, term, mask = [torch.from_numpy(x).to(device=device, dtype=torch.float32) for x in
                #                       (s, a, r, term, r.mask)]

                self.model.eval()
                eval_losses = self.model.eval_step(s, a, r, term, mask)
                self.model.train()

                last_total_loss = last_eval_losses.get('total', np.inf)
                if checkpoint_path and eval_losses['total'] < last_total_loss:
                    torch.save(self.model, Path(checkpoint_path).parent / 'checkpoint.ptmdl')

                last_eval_losses = eval_losses

                if self.logger:
                    self.logger.log(self._to_np(eval_losses), Scope.TEST(), i_step)

                    means, stds = {}, {}
                    for name, param in self.model.named_parameters():
                        if torch.numel(param) == 0:
                            continue
                        name = re.sub('[\s.]+', '_', name)
                        means[name + '_mean'] = param.detach().mean().cpu().numpy()
                        stds[name + '_std'] = param.detach().std(unbiased=False).cpu().numpy()
                    means['MEAN_TOTAL'] = np.mean([v for v in means.values()])
                    stds['STD_TOTAL'] = np.std([v for v in stds.values()])
                    means.update(stds)
                    self.logger.log(means, Scope.PARAMETERS(), i_step)

                if self.eval_callback:
                    self.eval_callback(s, a, r, term, mask, i_step)

            if self.train_callback:
                self.train_callback(s, a, r, term, mask, i_step)

        if self.logger:
            self.logger.stop_session()

    @staticmethod
    def _update_progressbar_descr(eval_losses: Dict[str, torch.Tensor],
                                  train_losses: Dict[str, torch.Tensor],
                                  pbar: tqdm):
        train_losses = {k: v for k, v in train_losses.items() if v.ndim <= 1 and k.startswith('monitoring_')}
        #eval_losses = {k: v for k, v in eval_losses.items() if v.ndim <= 1 and k.startswith('monitoring_')}
        l_train = [f'{n.replace("monitoring_", "")}: {l:2.3e}' for n, l in train_losses.items()]
        #l_eval = [f'{n.replace("monitoring_", "")}: {l:2.3e}' for n, l in eval_losses.items()]
        descr = 'train_losses = ' + ', '.join(l_train) #+ ' | val_losses = ' + ', '.join(l_eval)
        pbar.set_description(descr)

    @staticmethod
    def _to_np(data_dict: Dict[str, torch.Tensor]):
        return {k: v.detach().cpu().numpy() for k, v in data_dict.items()}

