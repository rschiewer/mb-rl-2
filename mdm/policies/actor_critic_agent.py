import copy
import random
from itertools import chain
from typing import Tuple, Sequence, Optional, Dict, List
from collections import namedtuple, OrderedDict
from copy import deepcopy

import gymnasium as gym
import torch
import torch.distributions as torchd
from torch.distributions import Distribution
from torch.distributions import kl_divergence
import numpy as np
import matplotlib.pyplot as plt

from mdm.utils.torch_tools import layers_with_activation as lwa, SquashedNormal, RunningMeanStd
from mdm.utils.torch_tools import FuzzyDeviceMixin, compute_mask, detach_dist, compile_if_not_debug, stack_dists, \
    concat_dists
from mdm.utils.utils import fig_to_img
from mdm.logging.logger import GlobalLogger, Scope


class ActorCriticAgent(FuzzyDeviceMixin, torch.nn.Module):

    def __init__(self,
                 level: int,
                 observation_key: str,
                 d_o: int,
                 d_a: int,
                 min_a: float | Sequence[float] = None,
                 max_a: float | Sequence[float] = None,
                 actor_lws: Sequence[int] = (),
                 actor_act_fn: str = 'relu',
                 actor_layer_norm: bool = True,
                 critic_lws: Sequence[int] = (),
                 critic_act_fn: str = 'relu',
                 critic_layer_norm: bool = True,
                 tr_policy_ema_update_coeff: float = 0.99,
                 tr_policy_kl_coeff: float = 0.0,
                 eps_exploration_init: float = 0,
                 eps_exploration_coeff: float = 0,
                 act_entropy_exploration_coeff: float = 0.0,
                 learn_act_entropy_exploration_coeff: bool = False,
                 novelty_exploration_coeff: float = 0.0,
                 use_ema_world_model: bool = False,
                 goal_seeking: bool = False,
                 dynamics_loss: bool = True,
                 normalize_observations: str | bool = False,
                 normalize_rewards: str | bool = False,
                 **kwargs):
        super().__init__()

        if goal_seeking:
            d_o = 2 * d_o

        self.level = level
        self.observation_key = observation_key
        self.d_a = d_a
        self.d_o = d_o
        if min_a:
            if isinstance(min_a, float):
                min_a = [min_a for _ in range(d_a)]
            self.min_a = torch.nn.Parameter(torch.tensor(min_a, dtype=torch.float32), requires_grad=False)
        else:
            self.min_a = None

        if max_a:
            if isinstance(max_a, float):
                max_a = [max_a for _ in range(d_a)]
            self.max_a = torch.nn.Parameter(torch.tensor(max_a, dtype=torch.float32), requires_grad=False)
        else:
            self.max_a = None
        self.ema_coeff = tr_policy_ema_update_coeff
        self.beta = tr_policy_kl_coeff
        self.eps = torch.tensor(eps_exploration_init, device=self.device, dtype=torch.float32, requires_grad=False)
        self.eps_mul = eps_exploration_coeff
        self.alpha = torch.nn.Parameter(torch.tensor(act_entropy_exploration_coeff))
        if not learn_act_entropy_exploration_coeff:
            self.alpha.requires_grad = False
        self.mu = novelty_exploration_coeff
        self.use_slow_world_model = use_ema_world_model
        self.goal_seeking = goal_seeking

        self.actor_net = torch.nn.Sequential(lwa(lws=[d_o, *actor_lws, d_a * 2], activation=actor_act_fn,
                                                 layer_norm=actor_layer_norm, name='actor_net'))
        self.critic_net = torch.nn.Sequential(lwa(lws=[d_o, *critic_lws, 1], activation=critic_act_fn,
                                                  layer_norm=critic_layer_norm, name='critic_net'))

        self._ema_actor_net = copy.deepcopy(self.actor_net)
        self._ema_critic_net = copy.deepcopy(self.critic_net)
        for param in self._ema_actor_net.parameters(): param.detach_()
        for param in self._ema_critic_net.parameters(): param.detach_()
        self._current_train_step = 0
        self.dynamics_loss = dynamics_loss
        self.normalize_observations = normalize_observations
        self.normalize_rewards = normalize_rewards

        if normalize_observations == 'running_average':
            self.o_running_average = RunningMeanStd(shape=(d_o,))
        if normalize_rewards == 'running_average':
            self.r_running_average = RunningMeanStd(shape=(1,))

    # @torch.compile
    def scale_action(self, action):
        if self.min_a is not None and self.max_a is not None:
            action = action * (self.max_a - self.min_a) / 2 + (self.min_a + self.max_a) / 2
        return action

    @compile_if_not_debug
    def _act_dist(self,
                  actor_net: torch.nn.Module,
                  x: torch.Tensor):
        # TODO: test penalty for too large pre-scaled activations
        x = actor_net(x)
        mu, logvar = torch.tensor_split(x, 2, dim=-1)
        sigma = torch.nn.functional.softplus(logvar) + 0.01
        d = SquashedNormal(loc=mu, scale=sigma)
        return d

    @compile_if_not_debug
    def forward(self,
                o: torch.Tensor,
                use_ema_modules: bool = False,
                sample: bool = True,
                disable_exploration: bool = False,
                **kwargs):
        if o.shape == self.d_o:  # add batch dim if not there already
            o = o.unsqueeze(0)

        if self.normalize_observations:
            o = o - self.o_running_average.mean.to(torch.float32)[None, ...]
            o = o / torch.sqrt(self.o_running_average.var.to(torch.float32) + 1e-6)[None, ...]

        if use_ema_modules:
            state_values = self._ema_critic_net(o)
            a_dist = self._act_dist(self._ema_actor_net, o)
        else:
            state_values = self.critic_net(o)
            a_dist = self._act_dist(self.actor_net, o)

        if sample:
            a_smpl = a_dist.rsample()
        else:
            a_smpl = a_dist.mean

        # scale a_smpl to allowed action interval
        if self.min_a is not None and self.max_a is not None:
            a_smpl = a_smpl * (self.max_a - self.min_a) / 2 + (self.min_a + self.max_a) / 2

        # adding random variance directly to the SquashedNormal destabilizes training a lot (not sure why),
        # so we add noise after sampling from it and clamp the result to prevent invalid actions
        if self.eps > 0 and self.training and not disable_exploration:
            a_smpl = a_smpl + torch.normal(torch.zeros_like(a_smpl), torch.full_like(a_smpl, self.eps))
            a_smpl = torch.clamp(a_smpl, self.min_a + torch.finfo().eps, self.max_a - torch.finfo().eps)

        return a_dist, a_smpl, state_values

    def act_in_sim(self,
                   env_state: Dict[str, torch.Tensor],
                   sim_env: 'HierarchicalRSSM',
                   n_steps: int,
                   goal: torch.Tensor = None,
                   agent_memory: Dict[str, List[torch.Tensor]] | None = None,
                   env_memory: Dict[str, List[torch.Tensor]] | None = None,
                   disable_exploration: bool = False,
                   sample_actions: bool = True,
                   sample_model: bool = True,
                   reconstruct: bool = True):
        agent_memory = {} if agent_memory is None else agent_memory
        env_mem = {} if env_memory is None else env_memory
        ema_env_mem = {}
        for t in range(n_steps):
            agent_o = self.preproc_o(env_state, goal)

            a_dist, a, v = self(agent_o, sample=sample_actions, disable_exploration=disable_exploration)

            if torch.isnan(env_state['z']).any() or torch.isinf(env_state['z']).any():
                raise RuntimeError(f'Invalid env state in act_in_sim: {env_state["z"]}')
            if torch.isnan(agent_o).any() or torch.isinf(agent_o).any():
                raise RuntimeError(f'Invalid agent observation in act_in_sim: {agent_o}')
            if torch.isnan(a).any() or torch.isinf(a).any():
                raise RuntimeError(f'Invalid agent action in act_in_sim: {a}')

            ema_a_dist, _, ema_v = self(agent_o, use_ema_modules=True, sample=sample_actions,
                                        disable_exploration=disable_exploration)
            trajectory = {'o': None, 'a': a.unsqueeze(0), 'r': None, 'terminal': None}
            mem, mem_ema, next_env_state = sim_env.forward_static(trajectory=trajectory, start_state=env_state,
                                                                  level=self.level, n_steps=1, n_warmup=0,
                                                                  use_ema_modules=self.use_slow_world_model,
                                                                  memory=env_mem, memory_other=ema_env_mem,
                                                                  sample_state=sample_model, sample_output=False,
                                                                  reconstruct=reconstruct)

            # TODO: check if the correct time step's z is used for computing rewards and if the correct time step's z is stored
            # r = self.build_step_reward(mem['z_dist'][-1], mem['r'][-1], goal)
            r = self.build_step_reward(mem['z'][-1], mem['r'][-1], goal, self.goal_seeking)
            novelty = kl_divergence(mem_ema['z_dist'][-1], mem['z_dist'][-1]).sum(dim=-1)
            timestep = {'o_env': env_state['z'], 'o_env_next': mem['z'][-1], 'goal': goal, 'o': agent_o, 'a': a, 'r': r,
                        'r_raw': mem['r'][-1], 'terminal': mem['terminal'][-1], 'v': v, 'ema_v': ema_v,
                        'a_dist': a_dist, 'ema_a_dist': ema_a_dist, 'model_novelty': novelty}

            # store agent data
            for k, v in timestep.items():
                data = agent_memory.get(k, [])
                data.append(v)
                agent_memory[k] = data

            # prepare next step
            env_state = next_env_state

        return {'agent': agent_memory, 'model': env_mem, 'model_state': env_state, 'ema_model': ema_env_mem}

    # @torch.compile
    def update_exploration(self):
        self.eps *= self.eps_mul
        self.eps = torch.max(self.eps, torch.tensor(0.05, dtype=torch.float32, device=self.eps.device))

    def eval_step(self,
                  a_dist: list[torchd.Distribution],
                  ema_a_dist: list[torchd.Distribution],
                  model_novelty: list[torch.Tensor],
                  a: list[torch.Tensor],
                  r: list[torch.Tensor],
                  terminal: list[torch.Tensor],
                  v: list[torch.Tensor],
                  ema_v: list[torch.Tensor],
                  o_env: list[torch.Tensor],
                  o_env_next: list[torch.Tensor],
                  goal: list[torch.Tensor | None],
                  first_step_mask: torch.Tensor | None = None,
                  **kwargs):
        # if a terminal transition occurs, the terminal flag is close to 1 and would block out the reward in that step
        # so shift terminals list one to the right and make first element zeros
        terminal = torch.stack(terminal)
        mask = compute_mask(terminal, first_step_mask=first_step_mask, disable=False)

        if GlobalLogger.can_log('mask_agent', self._current_train_step):
            fig = plt.figure(figsize=(5, 5))
            plt.matshow(mask.detach().cpu().numpy().squeeze(), fignum=fig, aspect='auto')
            plt.colorbar()
            agent_name = 'goal_seeking' if self.goal_seeking else 'r_max'
            GlobalLogger.logger.log_plot(fig_to_img(fig),
                                         Scope.TRAIN() / f'agent/{agent_name}_agent_l{self.level}_mask')
            plt.close(fig)
            del fig

        # terminal = torch.stack(terminal)
        # terminal = torch.roll(terminal, shifts=0, dims=0)
        # terminal[0] = 0
        # del terminal[-1]
        # terminal.insert(0, torch.zeros_like(terminal[0]))

        v = torch.stack(v)
        ema_v = torch.stack(ema_v)
        r = torch.stack(r)
        a = torch.stack(a[:-1])
        a_dist = stack_dists(a_dist[:-1])
        ema_a_dist = stack_dists(ema_a_dist[:-1])

        # if self.ema_reg:
        # v = torch.stack([torch.min(v, ema_v) for v, ema_v in zip(v, ema_v)])  # use this for policy targets

        if self.normalize_rewards:
            r = r - self.r_running_average.mean.to(torch.float32)[None, ...]
            r = r / torch.sqrt(self.r_running_average.var.to(torch.float32) + 1e-6)[None, ...]

        valid = (1 - mask)
        valid_r = valid * r
        valid_v = valid * v

        # valid_bootstrap = (valid * torch.minimum(v, ema_v))[-1]
        if self.goal_seeking:
            # we only train a single chunk, no bootstrapping needed beyond that
            # CAUTION: we don't train the last step and should get one step more than chunk size
            # valid_bootstrap = torch.zeros_like((valid_r[-1]))
            gamma = 1.00
        else:
            # valid_bootstrap = ((1 - terminal) * valid_v + terminal * valid_r)[-1]
            gamma = 0.99
        valid_bootstrap = ((1 - terminal) * valid_v + terminal * valid_r)[-1]
        returns = self._calc_returns_simple(valid_r[:-1], valid_bootstrap, gamma=gamma)

        # advantage = returns - torch.minimum(v, ema_v).detach()
        advantage = returns - valid_v[:-1].detach()
        if self.dynamics_loss:
            policy_loss = valid[:-1] * (-advantage)  # validity mask multiplied directly
            # policy_loss = valid[:-1] * (-returns)  # validity mask multiplied directly
        else:
            a_log_prob = a_dist.log_prob(a.detach())
            policy_loss = valid[:-1] * -a_log_prob * advantage.detach()

        model_novelty_reward_aug = valid[:-1] * self.mu * torch.stack(model_novelty[:-1]).unsqueeze(-1)
        act_entropy_reward_aug = valid[:-1] * self.alpha * a_dist.entropy().sum(dim=-1, keepdims=True)
        policy_loss -= act_entropy_reward_aug
        value_target = returns + model_novelty_reward_aug  # validity mask is multiplied with value_loss
        value_loss = valid[:-1] * torch.nn.functional.smooth_l1_loss(v[:-1], value_target.detach(), reduction='none')
        ppo_loss = valid[:-1] * self.beta * torchd.kl_divergence(detach_dist(ema_a_dist),
                                                                 a_dist).sum(dim=-1, keepdims=True)

        policy_loss = torch.mean(policy_loss)
        value_loss = torch.mean(value_loss)
        ppo_loss = torch.mean(ppo_loss)
        model_novelty_reward_aug = torch.mean(model_novelty_reward_aug)
        entropy_reward_aug = torch.mean(act_entropy_reward_aug)
        loss = policy_loss + value_loss + ppo_loss

        if self.normalize_rewards:
            running_r = self.r_running_average.mean.mean()
        else:
            running_r = torch.tensor(0.0, device=self.device)

        if self.normalize_observations:
            running_o = self.o_running_average.mean.mean()
        else:
            running_o = torch.tensor(0.0, device=self.device)

        losses = {'total': loss, 'policy': policy_loss, 'value': value_loss, 'policy_trust_region_loss': ppo_loss,
                  'model_novelty_reward_aug': model_novelty_reward_aug, 'action_entropy_reward_aug': entropy_reward_aug,
                  'eps_exploration': self.eps, 'monitoring_a_dist_mean': a_dist.mean.mean(),
                  'monitoring_a_dist_std': a_dist.scale.mean(), 'monitoring_r_running_average': running_r,
                  'monitoring_o_running_average': running_o}

        if kwargs.get('for_train_step', False):  # ugly hack
            losses['mask'] = mask

        return losses

    def train_step(self,
                   simulation_data: Dict[str, List[torch.Tensor]],
                   actor_optimizer: torch.optim.Optimizer,
                   critic_optimizer: torch.optim.Optimizer,
                   **kwargs):
        actor_optimizer.zero_grad(set_to_none=True)
        critic_optimizer.zero_grad(set_to_none=True)

        losses = self.eval_step(**simulation_data, for_train_step=True)

        with torch.no_grad():
            mask = losses.pop('mask')
            if self.normalize_observations:
                self.o_running_average.update(torch.stack(simulation_data['o']), mask=mask)
            if self.normalize_rewards:
                self.r_running_average.update(torch.stack(simulation_data['r']), mask=mask)

        # invalid_losses = ''
        # for k, v in losses.items():
        #    if torch.isnan(v).any() or torch.isinf(v).any():
        #        invalid_losses += f'{k}: {v}, '
        # if len(invalid_losses) > 0:
        #    raise RuntimeError(f'Invalid loss in {self._agent_repr} detected: {invalid_losses}')

        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        actor_optimizer.step()
        critic_optimizer.step()

        self._update_ema_modules()

        # update exploration
        self.update_exploration()
        self._current_train_step += 1

        return losses

    @compile_if_not_debug
    def _update_ema_modules(self):
        with torch.no_grad():
            params = chain.from_iterable([m.parameters() for m in self.actor_net] +
                                         [m.parameters() for m in self.critic_net])
            ema_params = chain.from_iterable([m.parameters() for m in self._ema_actor_net] +
                                             [m.parameters() for m in self._ema_critic_net])
            for param, ema_param in zip(params, ema_params):
                ema_param[:] = self.ema_coeff * ema_param + (1 - self.ema_coeff) * param

    @staticmethod
    @compile_if_not_debug
    def _calc_returns(rewards, state_values, timestep_mask, gamma):
        returns = []
        discounts = []
        R = state_values[-1].detach()
        # for r, mask in zip(rewards[::-1], timestep_mask[::-1]):
        for r, mask in zip(torch.flip(rewards, dims=(0,)), torch.flip(timestep_mask, dims=(0,))):
            gamma_final = ((1 - mask) * gamma).detach()
            R = r + gamma_final * R
            returns.insert(0, R)
            discounts.insert(0, gamma_final)
        return torch.stack(returns), torch.stack(discounts)

    @staticmethod
    def _calc_returns_simple(rewards, state_value_bootstrap, gamma):
        R = state_value_bootstrap.detach()
        returns = []
        for r in torch.flip(rewards, dims=(0,)):
            R = r + gamma * R
            returns.insert(0, R)
        return torch.stack(returns)

    @staticmethod
    @compile_if_not_debug
    def _calc_gae(rewards, state_values, timestep_mask, gamma, lambda_):
        advantages = []
        discounts = []
        last_value = state_values[-1].detach()
        last_advantage = 0
        for r, v, mask in zip(rewards[::-1], state_values[::-1], timestep_mask[::-1]):
            gamma_final = ((1 - mask) * gamma).detach()
            delta = r + gamma_final * last_value - v.detach()
            last_advantage = delta + gamma * lambda_ * last_advantage
            last_value = v.detach()
            advantages.insert(0, last_advantage)
            discounts.insert(0, gamma_final)
        return torch.stack(advantages), torch.stack(discounts)

    @staticmethod
    def goal_similarity(o: torch.Tensor | torchd.Distribution, goal: torch.Tensor | torchd.Distribution):
        if isinstance(o, torch.Tensor) and isinstance(goal, torch.Tensor):
            return - torch.mean(torch.abs(o - goal) ** 2, dim=-1, keepdim=True)
        elif isinstance(o, torch.Tensor) and isinstance(goal, Distribution):
            return torch.mean(goal.log_prob(o), dim=-1, keepdim=True)
        elif isinstance(o, Distribution) and isinstance(goal, torch.Tensor):
            return torch.mean(o.log_prob(goal), dim=-1, keepdim=True)
        else:
            return - torch.mean(torchd.kl_divergence(goal, o) + torchd.kl_divergence(o, goal), dim=-1, keepdim=True)

    def preproc_o(self,
                  step: Dict[str, torch.Tensor | torchd.Distribution],
                  goal: torch.Tensor | torchd.Distribution | None = None):
        o = step[self.observation_key]
        if isinstance(o, torchd.Distribution):
            o = o.mean  # use most probable o if a distribution is provided

        o = o.detach()  # don't propagate through multiple time steps

        if self.goal_seeking:
            # detach goal to avoid propagating gradients to upper level model into other agents
            if isinstance(goal, torchd.Distribution):
                goal = goal.mode.detach()
            else:
                goal = goal.detach()
            return torch.concat([o, goal], dim=-1)
        else:
            return o

    def build_step_reward(self,
                          o: torch.Tensor | torchd.Distribution,
                          r: torch.Tensor | torchd.Distribution,
                          goal: torch.Tensor | torchd.Distribution | None = None,
                          use_goal_reward: bool = False):
        if use_goal_reward:
            # detach goal to avoid propagating gradients to upper level model into other agents
            if isinstance(goal, torchd.Distribution):
                goal = detach_dist(goal)
            else:
                goal = goal.detach()
            # return 0.5 * self.goal_similarity(o, goal) + 0.5 * r
            return self.goal_similarity(o, goal)
        else:
            return r

    @property
    def _agent_repr(self):
        agent_name = 'agent'
        if self.goal_seeking:
            agent_name = 'goal seeking ' + agent_name
        else:
            agent_name = 'r max ' + agent_name
        agent_name = f'L{self.level} ' + agent_name

        return agent_name
