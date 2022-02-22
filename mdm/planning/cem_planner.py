from enum import Enum
from typing import Callable, Union
from functools import reduce
from math import ceil

import torch
import numpy as np


def compute_episode_returns(step_rewards: torch.Tensor, gamma: float):
    # in case of multi-dim rewards, sum reward dimension to one scalar
    if len(step_rewards.shape) > 2 and step_rewards.shape[-1] > 1:
        step_rewards = step_rewards.sum(dim=(-1))
    # if gamma is smaller 1 there is some work to do, else use torch builtin sum()
    if gamma < 1:
        d_time = step_rewards.shape[1]
        step_rewards_bw = step_rewards.flip(dims=(1,))
        discounted_returns = [reduce(lambda disc_sum, r: disc_sum * gamma + r, batch) for batch in step_rewards_bw]
        discounted_returns = torch.stack(discounted_returns, dim=0)
    else:
        discounted_returns = step_rewards.sum(dim=1)
    return discounted_returns


class DistributionType(Enum):
    NORMAL = 0
    CATEGORICAL = 1


class CrossentropyPlanner:

    def __init__(self,
                 type: DistributionType = DistributionType.NORMAL):
        self.type = type
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
             rollout_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
             start_states: torch.Tensor,
             d_dist: int,
             n_plan_steps: int,
             n_evolution_steps: int,
             winning_perc: float,
             discount: float,
             act_noise: float = 0,
             init_act_params: Union[torch.Tensor, np.ndarray] = None):
        d_batch = start_states.shape[0]  # n_batch equals number of rollouts
        n_winners = ceil(d_batch * winning_perc)
        act_dist_params = self._init_params(d_batch, n_plan_steps, d_dist, init_act_params)

        winner_actions = None
        for i_ev in range(n_evolution_steps):
            actions = self._build_dist(act_dist_params).sample()
            step_rewards = rollout_fn(start_states, actions)

            disc_ret = compute_episode_returns(step_rewards, discount)
            disc_ret_sorted = torch.sort(disc_ret, dim=0, descending=True)
            i_winners, R_winners = disc_ret_sorted.indices[:n_winners], disc_ret_sorted.values[:n_winners]
            winner_actions = actions[i_winners.tolist()]

            # update distribution parameters with MLE parameters of the winner samples
            act_dist_params = self._update_dist(winner_actions, act_dist_params, act_noise)

        return winner_actions, self._build_dist(act_dist_params), i_winners

    def _init_normal(self,
                     d_batch: int,
                     n_time_steps: int,
                     d_dist: int):
        mu = 2 * torch.rand(d_batch, n_time_steps, d_dist) - 1
        sigma = torch.maximum(torch.rand(d_batch, n_time_steps, d_dist), torch.tensor(0.1))
        return torch.stack([mu, sigma], dim=0)

    def _build_normal(self,
                      dist_params: torch.Tensor):
        mu, sigma = torch.unbind(dist_params, dim=0)
        return torch.distributions.Normal(loc=mu, scale=sigma)

    def _update_normal(self,
                       winner_actions: torch.Tensor,
                       dist_params: torch.Tensor,
                       noise: float):
        noise = torch.tensor(noise)
        n_batch = dist_params.shape[1]
        n_winners = winner_actions.shape[0]

        # compute prototype mu and sigma
        mu_ml = winner_actions.mean(dim=0)
        sigma_ml = torch.mean((winner_actions - torch.tile(mu_ml, dims=(n_winners, 1, 1))) ** 2, dim=0)

        # just copy prototype values along batch axis
        mu_ml = torch.tile(mu_ml, dims=(n_batch, 1, 1))
        sigma_ml = torch.tile(sigma_ml, dims=(n_batch, 1, 1))

        # add noise to diversify
        mu_ml = mu_ml + (2 * torch.rand_like(mu_ml) - 1) * noise
        sigma_ml = sigma_ml + (2 * torch.rand_like(sigma_ml) - 1) * noise
        sigma_ml = torch.maximum(sigma_ml, torch.tensor(0.1))

        return torch.stack([mu_ml, sigma_ml], dim=0)

    def _init_categorical(self,
                          d_batch: int,
                          n_time_steps: int,
                          d_dist: int):
        params = torch.rand(d_batch, n_time_steps, d_dist)
        params /= params.sum(dim=-1, keepdim=True)
        return params

    def _build_categorical(self,
                           dist_params: torch.Tensor):
        return self._act_dist(probs=dist_params)

    def _update_categorical(self,
                            winner_actions: torch.Tensor,
                            dist_params: torch.Tensor,
                            noise: float):
        noise = torch.tensor(noise)
        n_batch = dist_params.shape[0]
        n_actions = dist_params.shape[-1]

        actions_onehot = torch.nn.functional.one_hot(winner_actions, num_classes=n_actions)
        dist_params = torch.mean(actions_onehot.float(), dim=(0))  # yields one list of distributions, one per time step
        dist_params = torch.tile(dist_params, dims=(n_batch, 1, 1))  # this copies the list to all batch indices
        #dist_params[dist_params.shape[0] // 2 :] = torch
        # add noise to diversify
        dist_params = dist_params + (2 * torch.rand_like(dist_params) - 1) * noise
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
                act_params = torch.from_numpy(init_act_params)
            else:
                act_params = init_act_params
        return act_params
