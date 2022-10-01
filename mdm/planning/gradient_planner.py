from math import ceil

import torch
from typing import Union, Callable, Tuple, Optional, Dict
import numpy as np

from mdm.models.multiscale_model_mk2 import MultiscaleDynamicsModelMK2
from mdm.planning.cem_planner import DistributionType, process_terminal_flag_mat, compute_episode_returns
from mdm.utils.torch_tools import TensorData

class GradientPlanner:

    def __init__(self,
                 type: DistributionType,
                 device: torch.device = 'cpu'):
        self.type = type
        self.device = device

        if type is DistributionType.NORMAL:
            self._init_dist = self._init_normal
            self._build_dist = self._build_normal
        elif type is DistributionType.CATEGORICAL:
            self._init_dist = self._init_categorical
            self._build_dist = self._build_categorical

    def _init_normal(self,
                     d_batch: int,
                     n_time_steps: int,
                     d_dist: int):
        mu = 2 * torch.rand(d_batch, n_time_steps, d_dist, device=self.device) - 1
        sigma = torch.rand(d_batch, n_time_steps, d_dist, device=self.device) + 5.0
        return torch.stack([mu, sigma], dim=0)

    def _build_normal(self,
                      dist_params: torch.Tensor):
        mu, sigma = torch.unbind(dist_params, dim=0)
        sigma = torch.abs(sigma) + 0.01
        return torch.distributions.Normal(loc=mu, scale=sigma)

    def _init_categorical(self,
                          d_batch: int,
                          n_time_steps: int,
                          d_dist: int):
        params = torch.rand(d_batch, n_time_steps, d_dist, device=self.device)
        params_norm = params / params.sum(dim=-1, keepdim=True)
        return params_norm

    def _build_categorical(self,
                           dist_params: torch.Tensor):
        cond = torch.tensor(0, dtype=torch.float32, device=dist_params.device)
        dist_params_interval = torch.where(dist_params < cond, cond, dist_params)
        dist_params_norm = dist_params_interval / dist_params_interval.sum(dim=-1, keepdim=True)
        return torch.distributions.RelaxedOneHotCategorical(temperature=0.1, probs=dist_params_norm)

    def plan(self,
             rollout_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, TensorData]]],
             d_dist: int,
             n_rollouts: int,
             n_plan_steps: int,
             n_evolution_steps: int,
             winning_perc: float,
             discount: float,
             act_noise: float = 0,
             init_act_params: Union[torch.Tensor, np.ndarray] = None):
            n_winners = ceil(n_rollouts * winning_perc)
            act_dist_params = self._init_dist(n_rollouts, n_plan_steps, d_dist)
            act_dist_params.requires_grad_(True)

            actions, i_winners, R_winners, rollout_data = None, None, None, None
            optimizer = torch.optim.Adam([act_dist_params], lr=0.01)
            for i_ev in range(n_evolution_steps):
                actions = self._build_dist(act_dist_params).rsample()
                criterion, terminal_flag_mat, rollout_data = rollout_fn(actions)

                disc_mat = torch.cumprod(torch.full_like(criterion, fill_value=discount), dim=1)
                disc_mat = torch.roll(disc_mat, 1, dims=1)
                disc_mat[:, 0] = 1

                if terminal_flag_mat is not None:
                    final_disc_mat = process_terminal_flag_mat(terminal_flag_mat) * disc_mat
                else:
                    final_disc_mat = disc_mat
                disc_ret = compute_episode_returns(criterion, final_disc_mat)
                loss = -torch.mean(disc_ret)

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            # get the top performer at the very end
            disc_ret_sorted = torch.sort(disc_ret, dim=0, descending=True)
            i_winners, R_winners = disc_ret_sorted.indices[:n_winners], disc_ret_sorted.values[:n_winners]

            return actions, self._build_dist(act_dist_params), i_winners.tolist(), R_winners.tolist(), rollout_data


