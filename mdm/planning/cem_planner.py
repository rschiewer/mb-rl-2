import copy
from typing import Callable, Union, Tuple, Optional, Dict
from math import ceil

import torch
import numpy as np
import gym
import matplotlib.pyplot as plt


from mdm.utils.torch_tools import TensorData
from mdm.utils.utils import DistributionType


def compute_episode_returns(step_rewards: torch.Tensor,
                            disc_mat: Union[None, torch.Tensor]):
    # in case of multi-dim rewards, sum reward dimension to one scalar
    if step_rewards.ndim > 3 and step_rewards.shape[-1] > 1:
        step_rewards = step_rewards.sum(dim=(-1), keepdim=True)
    # if disc_mat is None, no discounting is necessary, just sum rewards per run
    if disc_mat is None:
        discounted_returns = step_rewards.sum(dim=2)
    else:
        discounted_returns = torch.sum(step_rewards * disc_mat, dim=2)
    return discounted_returns


def process_terminal_flag_mat(terminal_flags: torch.Tensor):
    # shift terminal flags matrix one to the right to not zero out final reward
    terminal_flags = torch.roll(terminal_flags, shifts=1, dims=2)
    terminal_flags[:, :, 0] = 0
    disc_mat = torch.cumprod(1 - terminal_flags, dim=2)
    return disc_mat


class CrossentropyPlanner:

    def __init__(self,
                 type: DistributionType,
                 d_dist: int,
                 n_evolution_steps: int = 10,
                 winning_perc: float = 0.3,
                 discount: float = 0.99,
                 act_noise: float = 0,
                 alpha: float = 1.0,
                 device: torch.device = 'cpu',
                 debug_env: gym.Env = None,
                 a_min: float = None,
                 a_max: float = None,
                 **dist_args):
        self.type = type
        self.d_dist = d_dist
        self.n_evolution_steps = n_evolution_steps
        self.winning_perc = winning_perc
        self.discount = discount
        self.act_noise = act_noise
        self.device = device
        self.dist_args = dist_args
        self.alpha = alpha
        self._debug_env = copy.deepcopy(debug_env)
        self.a_min = torch.tensor(a_min, device=device) if a_min is not None else None
        self.a_max = torch.tensor(a_max, device=device) if a_max is not None else None
        if type is DistributionType.NORMAL:
            self._init_dist = self._init_normal
            self._update_dist = self._update_normal
            self._build_dist = self._build_normal
        elif type is DistributionType.CATEGORICAL:
            self._init_dist = self._init_categorical
            self._update_dist = self._update_categorical
            self._build_dist = self._build_categorical

    def _maybe_get_real_reward(self, actions: torch.Tensor, i_winners: torch.Tensor, average: bool = True):
        if self._debug_env:
            if average:
                actions = actions[i_winners.tolist()].to(torch.float32).mean(dim=0).detach().cpu().numpy()
            else:
                actions = actions[i_winners[0]].detach().cpu().numpy()
            if self.type == DistributionType.CATEGORICAL:
                actions = actions.round().astype(int)

            rewards = [0.0]
            self._debug_env.reset()
            for a in actions:
                o, r, term, trunc, info = self._debug_env.step(a)
                rewards.append(r)
            rewards = np.array(rewards)
            disc_mat = np.cumprod(np.full_like(rewards, self.discount, dtype=float))
            disc_mat = np.roll(disc_mat, 1, axis=0)
            disc_mat[0] = 1
            R = np.sum(disc_mat * rewards)
            return R

    def plan(self,
             rollout_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, TensorData]]],
             n_rollouts: int,
             n_plan_steps: int,
             n_envs: int = 1,
             init_act_params: Union[torch.Tensor, np.ndarray] = None):
        n_winners = ceil(n_rollouts * self.winning_perc)
        act_dist_params = self._init_params(n_envs, n_rollouts, n_plan_steps, self.d_dist, init_act_params)

        #exponents = torch.arange(n_plan_steps, device=self.device)
        #if discount != 0:
        #    disc_mat = torch.tile(torch.pow(discount, exponents), (n_rollouts, 1))
        #else:
        #    disc_mat = None

        actions, i_winners, R_winners, rollout_data = None, None, None, None
        R_real_evolution, R_winners_evolution = [], []
        if self._debug_env:
            self._debug_env.reset()
        for i_ev in range(self.n_evolution_steps):
            actions = self._build_dist(act_dist_params).sample()  # shape: (n_envs, n_rollouts, n_plan_steps, d_dist)
            if self.a_min is not None:
                actions = torch.max(actions, self.a_min)
            if self.a_max is not None:
                actions = torch.min(actions, self.a_max)
            criterion, terminal_flag_mat, rollout_data = rollout_fn(actions)

            assert criterion.shape[0] == n_envs
            assert criterion.shape[1] == n_rollouts
            assert criterion.ndim == 3

            disc_mat = torch.cumprod(torch.full_like(criterion, fill_value=self.discount), dim=2)
            disc_mat = torch.roll(disc_mat, 1, dims=2)
            disc_mat[:, :, 0] = 1

            if terminal_flag_mat is not None:
                final_disc_mat = process_terminal_flag_mat(terminal_flag_mat) * disc_mat
            else:
                final_disc_mat = disc_mat
            disc_ret = compute_episode_returns(criterion, final_disc_mat)
            disc_ret_sorted = torch.sort(disc_ret, dim=1, descending=True)

            i_winners, R_winners = disc_ret_sorted.indices[:, :n_winners], disc_ret_sorted.values[:, :n_winners]

            #R_real = self._maybe_get_real_reward(actions, i_winners, average=True)
            #R_real_evolution.append(R_real)
            #R_winners_evolution.append(R_winners.mean(dim=0).detach().cpu().numpy())

            if i_ev == self.n_evolution_steps - 1:  # disable action noise for the last update
                act_noise = 0
            else:
                act_noise = self.act_noise

            # update distribution parameters with MLE parameters of the winner samples
            act_dist_params = self._update_dist(actions, act_dist_params, i_winners, act_noise)

        #print(disc_ret_sorted.values[:n_winners])
        #plt.plot(R_real_evolution, label='R real')
        #plt.plot(R_winners_evolution, label='R rollout')
        #plt.legend()
        #plt.show()

        return actions, self._build_dist(act_dist_params), i_winners, R_winners, rollout_data

    def _init_normal(self,
                     n_envs: int,
                     d_batch: int,
                     n_time_steps: int,
                     d_dist: int):
        mu_spread = self.dist_args.get('mu_init_spread', 2.0)
        sigma_min = self.dist_args.get('sigma_init_min', 1.0)
        mu = mu_spread * torch.rand(n_envs, d_batch, n_time_steps, d_dist, device=self.device) - mu_spread / 2
        sigma = torch.rand(n_envs, d_batch, n_time_steps, d_dist, device=self.device) + sigma_min
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
        # actions before:         (n_rollouts, n_time_steps, d_dist)
        # actions new:    (n_envs, n_rollouts, n_time_steps, d_dist)
        n_envs, n_batch = dist_params.shape[1:3]
        # broadcasting taken from https://discuss.pytorch.org/t/how-to-select-particular-elements-of-a-3d-tensor-based-on-indices-along-dim-1-in-pytorch/129632/9
        winner_actions = actions[torch.arange(n_envs).unsqueeze(1), i_winners]
        noise = torch.tensor(noise, device=self.device)
        lower_bound = torch.tensor(0.0001, device=actions.device)

        # compute prototype mu and sigma
        mu_ml = winner_actions.mean(dim=1)
        sigma_ml = torch.sqrt(torch.mean((winner_actions - mu_ml.unsqueeze(1)) ** 2, dim=1)) + 0.001

        # just copy prototype values along batch axis
        mu_ml = torch.tile(mu_ml.unsqueeze(1), dims=(1, n_batch, 1, 1))
        sigma_ml = torch.tile(sigma_ml.unsqueeze(1), dims=(1, n_batch, 1, 1))

        # add noise to diversify half of distributions
        mu_ml[:, n_batch//2:] += (2 * torch.rand_like(mu_ml[:, n_batch//2:], device=self.device) - 1) * noise
        sigma_ml[:, n_batch//2:] += (2 * torch.rand_like(sigma_ml[:, n_batch//2:], device=self.device) - 1) * noise
        sigma_ml = torch.where(sigma_ml <= lower_bound, lower_bound, sigma_ml)  # don't accidentally make sigma < 0

        mu_old, sigma_old = torch.unbind(dist_params, dim=0)
        mu_new = (1 - self.alpha) * mu_old + self.alpha * mu_ml
        sigma_new = (1 - self.alpha) * sigma_old + self.alpha * sigma_ml

        return torch.stack([mu_new, sigma_new], dim=0)

    def _init_categorical(self,
                          n_envs: int,
                          d_batch: int,
                          n_time_steps: int,
                          d_dist: int):
        params = torch.rand(n_envs, d_batch, n_time_steps, d_dist, device=self.device)
        params /= params.sum(dim=-1, keepdim=True)
        return params

    def _build_categorical(self,
                           dist_params: torch.Tensor):
        return torch.distributions.Categorical(probs=dist_params)

    def _update_categorical(self,
                            actions: torch.Tensor,
                            dist_params: torch.Tensor,
                            i_winners: torch.Tensor,
                            noise: float):
        n_envs, d_batch = dist_params.shape[:2]
        n_actions = dist_params.shape[-1]
        # broadcasting taken from https://discuss.pytorch.org/t/how-to-select-particular-elements-of-a-3d-tensor-based-on-indices-along-dim-1-in-pytorch/129632/9
        winner_actions = actions[torch.arange(n_envs).unsqueeze(1), i_winners]
        noise = torch.tensor(noise, device=self.device)

        actions_onehot = torch.nn.functional.one_hot(winner_actions, num_classes=n_actions)
        dist_params_new = torch.mean(actions_onehot.float(), dim=1)  # yields one list of distributions, one per time step
        dist_params_new = torch.tile(dist_params_new.unsqueeze(1), dims=(1, d_batch, 1, 1))  # this copies the list to all batch indices
        # add noise to diversify
        dist_params_new[:, d_batch//2:] += (2 * torch.rand_like(dist_params_new[:, d_batch//2:], device=self.device) - 1) * noise
        dist_params_new[:, d_batch//2:] = torch.clamp(dist_params_new[:, d_batch//2:], torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device))
        dist_params_new /= dist_params_new.sum(dim=-1, keepdim=True)
        dist_params = (1 - self.alpha) * dist_params + self.alpha * dist_params_new
        return dist_params

    def _init_params(self,
                     n_envs: int,
                     n_rollouts: int,
                     n_time_steps: int,
                     d_dist: int,
                     init_act_params: Union[torch.Tensor, np.ndarray]):
        if init_act_params is None:
            act_params = self._init_dist(n_envs, n_rollouts, n_time_steps, d_dist)
        else:
            #if init_act_params.shape != (d_batch, n_time_steps, d_dist):
            #    raise ValueError(f'Initial action parameters argument shape mismatch, found: {init_act_params.shape}, '
            #                     f'expected: {(d_batch, n_time_steps, d_dist)}')
            if isinstance(init_act_params, np.ndarray):
                act_params = torch.from_numpy(init_act_params).to(self.device)
            else:
                act_params = init_act_params.to(self.device)
        return act_params

    def get_winner_actions(self,
                           final_actions: torch.Tensor,
                           final_act_dist: torch.distributions.Distribution,
                           i_winners: torch.Tensor,
                           resample: bool = True):
        n_envs, n_rollouts = final_actions.shape[:2]
        if resample:
            actions = final_act_dist.sample()
        else:
            actions = final_actions

        winner_actions = actions[torch.arange(n_envs), i_winners[:, 0]]

        if self.a_min is not None:
            actions = torch.max(actions, self.a_min)
        if self.a_max is not None:
            actions = torch.min(actions, self.a_max)
        return winner_actions

