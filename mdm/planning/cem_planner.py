from enum import Enum
from typing import Callable, Union, Tuple, Optional, Dict
from functools import reduce
from math import ceil

import torch
import numpy as np


def compute_episode_returns(step_rewards: torch.Tensor, disc_mat: Union[None, torch.Tensor]):
    # in case of multi-dim rewards, sum reward dimension to one scalar
    if step_rewards.ndim > 2 and step_rewards.shape[-1] > 1:
        step_rewards = step_rewards.sum(dim=(-1), keepdim=True)
    elif step_rewards.ndim == 1:
        step_rewards = step_rewards.unsqueeze(-1)
    # if disc_mat is None, no discounting is necessary, just sum rewards per run
    if disc_mat is None:
        discounted_returns = step_rewards.sum(dim=1)
    else:
        discounted_returns = torch.sum(step_rewards * disc_mat, dim=1)
    return discounted_returns


class DistributionType(Enum):
    NORMAL = 0
    CATEGORICAL = 1
    ONE_HOT_CATEGORICAL = 2


class CrossentropyPlanner:

    def __init__(self,
                 type: DistributionType = DistributionType.NORMAL,
                 device: torch.device = 'cpu'):
        self.type = type
        self.device = device
        if type is DistributionType.NORMAL:
            self._act_dist = torch.distributions.Normal
            self._init_dist = self._init_normal
            self._update_dist = self._update_normal
            self._build_dist = self._build_normal
        elif type is DistributionType.CATEGORICAL:
            self._act_dist = torch.distributions.Categorical
            self._init_dist = self._init_categorical
            self._update_dist = self._update_categorical
            self._build_dist = self._build_categorical

    def plan(self,
             rollout_fn: Callable[[torch.Tensor, torch.Tensor],
                                  Tuple[torch.Tensor, Union[torch.Tensor, None], Dict[str, torch.Tensor]]],
             start_states: torch.Tensor,
             d_dist: int,
             n_plan_steps: int,
             n_evolution_steps: int,
             winning_perc: float,
             discount: float,
             act_noise: float = 0,
             init_act_params: Union[torch.Tensor, np.ndarray] = None):
        if start_states.device != self.device:
            raise ValueError(f'Expected device for start_states is {self.device} but was {start_states.device}')

        d_batch = start_states.shape[0]  # n_batch equals number of rollouts
        n_winners = ceil(d_batch * winning_perc)
        act_dist_params = self._init_params(d_batch, n_plan_steps, d_dist, init_act_params)

        if discount != 0:
            exponents = torch.arange(n_plan_steps, device=self.device)
            disc_mat = torch.tile(torch.pow(discount, exponents), (d_batch, 1))
        else:
            disc_mat = None

        actions, i_winners, rollout_data = None, None, None
        for i_ev in range(n_evolution_steps):
            actions = self._build_dist(act_dist_params).sample()
            criterion, rollout_disc_mat, rollout_data = rollout_fn(start_states, actions)

            if rollout_disc_mat is not None:
                disc_mat = rollout_disc_mat
            disc_ret = compute_episode_returns(criterion, disc_mat)
            disc_ret_sorted = torch.sort(disc_ret, dim=0, descending=True)

            i_winners, R_winners = disc_ret_sorted.indices[:n_winners], disc_ret_sorted.values[:n_winners]

            if i_ev == n_evolution_steps - 1:  # disable action noise for the last update
                act_noise = 0

            # update distribution parameters with MLE parameters of the winner samples
            act_dist_params = self._update_dist(actions, act_dist_params, i_winners, act_noise)

        print(disc_ret_sorted.values[:n_winners])

        return actions, self._build_dist(act_dist_params), i_winners.tolist(), rollout_data

    def _init_normal(self,
                     d_batch: int,
                     n_time_steps: int,
                     d_dist: int):
        mu = 2 * torch.rand(d_batch, n_time_steps, d_dist, device=self.device) - 1
        sigma = torch.maximum(torch.rand(d_batch, n_time_steps, d_dist, device=self.device),
                              torch.tensor(0.25, device=self.device))
        return torch.stack([mu, sigma], dim=0)

    def _build_normal(self,
                      dist_params: torch.Tensor):
        mu, sigma = torch.unbind(dist_params, dim=0)
        return torch.distributions.Normal(loc=mu, scale=sigma)

    def _update_normal(self,
                       actions: torch.Tensor,
                       dist_params: torch.Tensor,
                       i_winners: torch.Tensor,
                       noise: float):
        winner_actions = actions[i_winners.tolist()]
        noise = torch.tensor(noise, device=self.device)
        n_batch = dist_params.shape[1]
        n_winners = winner_actions.shape[0]

        # compute prototype mu and sigma
        mu_ml = winner_actions.mean(dim=0)
        sigma_ml = torch.sqrt(torch.mean((winner_actions - mu_ml.unsqueeze(0)) ** 2, dim=0))

        # just copy prototype values along batch axis
        mu_ml = torch.tile(mu_ml, dims=(n_batch, 1, 1))
        sigma_ml = torch.tile(sigma_ml, dims=(n_batch, 1, 1))

        # add noise to diversify
        mu_ml_noise = mu_ml + (2 * torch.rand_like(mu_ml, device=self.device) - 1) * noise
        sigma_ml_noise = sigma_ml + (2 * torch.rand_like(sigma_ml, device=self.device) - 1) * noise
        sigma_ml_noise = torch.where(sigma_ml_noise <= 0, sigma_ml, sigma_ml_noise)  # don't accidentally make sigma < 0

        return torch.stack([mu_ml_noise, sigma_ml_noise], dim=0)

    def _init_categorical(self,
                          d_batch: int,
                          n_time_steps: int,
                          d_dist: int):
        params = torch.rand(d_batch, n_time_steps, d_dist, device=self.device)
        params /= params.sum(dim=-1, keepdim=True)
        return params

    def _build_categorical(self,
                           dist_params: torch.Tensor):
        return self._act_dist(probs=dist_params)

    def _update_categorical(self,
                            actions: torch.Tensor,
                            dist_params: torch.Tensor,
                            i_winners: torch.Tensor,
                            noise: float):
        winner_actions = actions[i_winners.tolist()]
        noise = torch.tensor(noise, device=self.device)
        n_batch = dist_params.shape[0]
        n_actions = dist_params.shape[-1]

        actions_onehot = torch.nn.functional.one_hot(winner_actions, num_classes=n_actions)
        dist_params = torch.mean(actions_onehot.float(), dim=(0))  # yields one list of distributions, one per time step
        dist_params = torch.tile(dist_params, dims=(n_batch, 1, 1))  # this copies the list to all batch indices
        #dist_params[dist_params.shape[0] // 2 :] = torch.rand_like(dist_params[dist_params.shape[0] // 2:])
        # add noise to diversify
        dist_params = dist_params + (2 * torch.rand_like(dist_params, device=self.device) - 1) * noise
        dist_params = torch.clamp(dist_params, torch.tensor(0.0), torch.tensor(1.0))
        dist_params /= dist_params.sum(dim=-1, keepdim=True)
        return dist_params

    def _init_params(self,
                     d_batch: int,
                     n_time_steps: int,
                     d_dist: int,
                     init_act_params: Union[torch.Tensor, np.ndarray]):
        if init_act_params is None:
            act_params = self._init_dist(d_batch, n_time_steps, d_dist)
        else:
            #if init_act_params.shape != (d_batch, n_time_steps, d_dist):
            #    raise ValueError(f'Initial action parameters argument shape mismatch, found: {init_act_params.shape}, '
            #                     f'expected: {(d_batch, n_time_steps, d_dist)}')
            if isinstance(init_act_params, np.ndarray):
                act_params = torch.from_numpy(init_act_params).to(self.device)
            else:
                act_params = init_act_params.to(self.device)
        return act_params
