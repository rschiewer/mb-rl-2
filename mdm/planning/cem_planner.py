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
            self._update_dist_params = self._update_dist_params_normal
        elif type is DistributionType.CATEGORICAL:
            self._act_dist = torch.distributions.Categorical
            self._update_dist_params = self._update_dist_params_categorical

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
        act_dist_params = self._act_dist_params(d_batch, n_plan_steps, d_dist, init_act_params)

        winner_actions = None
        for i_ev in range(n_evolution_steps):
            actions = self._act_dist(probs=act_dist_params).sample()
            step_rewards = rollout_fn(start_states, actions)

            disc_ret = compute_episode_returns(step_rewards, discount)
            disc_ret_sorted = torch.sort(disc_ret, dim=0, descending=True)
            i_winners, R_winners = disc_ret_sorted.indices[:n_winners], disc_ret_sorted.values[:n_winners]
            winner_actions = actions[i_winners.tolist()]

            # update distribution parameters with MLE parameters of the winner samples
            act_dist_params = self._update_dist_params(winner_actions, act_dist_params, act_noise)

        return winner_actions[0]

    def _update_dist_params_normal(self,
                                   winner_actions: torch.Tensor,
                                   dist_params: torch.Tensor,
                                   noise: float):
        raise NotImplementedError('You have to do this')

    def _update_dist_params_categorical(self,
                                        winner_actions: torch.Tensor,
                                        dist_params: torch.Tensor,
                                        noise: float):
        n_batch = dist_params.shape[0]
        n_actions = dist_params.shape[-1]
        actions_onehot = torch.nn.functional.one_hot(winner_actions, num_classes=n_actions)
        numerator = torch.sum(actions_onehot, dim=0)
        denominator = torch.sum(actions_onehot, dim=[0, 2])[..., None]
        dist_params = numerator / denominator  # this yields one list of distributions with shape (d_time, d_action)
        dist_params = torch.tile(dist_params, dims=(n_batch, 1, 1))  # this copies the one list to all batch items
        dist_params = torch.clamp(dist_params + torch.rand_like(dist_params) * noise, 0.0, 1.0)  # this diversifies
        return dist_params

    @staticmethod
    def _act_dist_params(d_batch: int,
                         n_time_steps: int,
                         d_dist: int,
                         init_act_params: Union[torch.Tensor, np.ndarray]):
        if init_act_params is None:
            act_params = torch.rand(d_batch, n_time_steps, d_dist)
        else:
            if init_act_params.shape != (d_batch, n_time_steps, d_dist):
                raise ValueError(f'Initial action parameters argument shape mismatch, found: {init_act_params.shape}, '
                                 f'expected: {(d_batch, n_time_steps, d_dist)}')
            if isinstance(init_act_params, np.ndarray):
                act_params = torch.from_numpy(init_act_params)
            else:
                act_params = init_act_params
        return act_params
