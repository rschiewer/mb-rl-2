import copy
import math
import random
from itertools import chain
from typing import Tuple, Sequence, Optional, Dict, List, Union
from collections import namedtuple, OrderedDict
from copy import deepcopy

import gymnasium as gym
import torch
import torch.distributions as torchd
from torch.distributions import Distribution
from torch.distributions import kl_divergence
import numpy as np
import matplotlib.pyplot as plt

from mdm.models.building_blocks import rssm_add_labels, rssm_detach_state, RSSMStateType
from mdm.utils.torch_tools import layers_with_activation as lwa, SquashedNormal, RunningMeanStd
from mdm.utils.torch_tools import (FuzzyDeviceMixin, compute_mask, detach_dist, disable_torch_compile, stack_dists,
                                   concat_dists, TanhBijector, clip_but_pass_gradient, FreezeParameters,
                                   plot_grad_flow, masked_mean, masked_var)
from mdm.utils.utils import fig_to_img, append_memory
from mdm.logging.logger import GlobalLogger, Scope, Logger


class ActorCriticAgent(torch.nn.Module):

    def __init__(self,
                 level: int,
                 observation_type: str,
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
                 ema_update_coeff: float = 0.99,
                 tr_policy_kl_coeff: float = 0.0,
                 eps_exploration_init: float = 0,
                 eps_exploration_coeff: float = 0,
                 eps_exploration_min: float = 0,
                 act_entropy_exploration_coeff: float = 0.0,
                 learn_act_entropy_exploration_coeff: bool = False,
                 novelty_exploration_coeff: float = 0.0,
                 min_scale: float = 0.1,
                 use_slow_world_model: bool = False,
                 use_slow_value_target: bool = False,
                 goal_seeking: bool = False,
                 dynamics_loss: bool = True,
                 normalize_observations: str | bool = False,
                 normalize_rewards: str | bool = False,
                 init_action_variance: float = 5.0,
                 **kwargs):
        super().__init__()

        if goal_seeking:
            d_o = 2 * d_o

        self.level = level
        self.observation_type = observation_type
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
        self.ema_coeff = ema_update_coeff
        self.beta = tr_policy_kl_coeff
        self.eps = torch.nn.Parameter(torch.tensor(eps_exploration_init, dtype=torch.float32), requires_grad=False)
        self.eps_mul = torch.nn.Parameter(torch.tensor(eps_exploration_coeff, dtype=torch.float32), requires_grad=False)
        self.eps_min = torch.nn.Parameter(torch.tensor(eps_exploration_min, dtype=torch.float32, requires_grad=False))
        self.alpha = torch.nn.Parameter(torch.tensor(act_entropy_exploration_coeff, dtype=torch.float32))
        if not learn_act_entropy_exploration_coeff:
            self.alpha.requires_grad = False
        self.mu = novelty_exploration_coeff
        self.min_scale = min_scale
        self.use_slow_world_model = use_slow_world_model
        self.use_slow_value_target = use_slow_value_target
        self.goal_seeking = goal_seeking

        self.actor_net = torch.nn.Sequential(lwa(lws=[d_o, *actor_lws, d_a * 2], activation=actor_act_fn,
                                                 layer_norm=actor_layer_norm, name='actor_net'))
        self.critic_net = torch.nn.Sequential(lwa(lws=[d_o, *critic_lws, 1], activation=critic_act_fn,
                                                  layer_norm=critic_layer_norm, name='critic_net'))

        self._ema_actor_net = copy.deepcopy(self.actor_net)
        self._ema_critic_net = copy.deepcopy(self.critic_net)
        for param in self._ema_actor_net.parameters(): param.requires_grad = False
        for param in self._ema_critic_net.parameters(): param.requires_grad = False
        self._current_train_step = 0
        self.dynamics_loss = dynamics_loss
        self.normalize_observations = normalize_observations
        self.normalize_rewards = normalize_rewards
        self.init_action_variance = init_action_variance

        self.o_running_average = RunningMeanStd(shape=(d_o,))
        self.r_running_average = RunningMeanStd(shape=(1,))
        self.return_running_average = RunningMeanStd(shape=(1,))

        self.goal_reached_eps = 0.05
        self.min_float = torch.finfo().eps

    def forward(self,
                o: torch.Tensor,
                use_ema_modules: bool = False,
                sample: bool = True,
                disable_exploration: bool = True):
        o = o.detach()
        if o.shape == self.d_o:  # add batch dim if not there already
            o = o.unsqueeze(0)

        if self.normalize_observations:
            _o_mean, _o_var = self.o_running_average.mean, self.o_running_average.var
            o = self.o_running_average.normalize(o, _o_mean, _o_var)

        a_dist = self._a_dist_params(o, use_ema_modules)
        a_smpl = self._a_smpl(dist_params=a_dist, sample=sample, explore=not disable_exploration)

        return a_dist, a_smpl

    @torch.jit.ignore
    def act_in_sim(self,
                   env_start_state: RSSMStateType,
                   sim_env: 'HierarchicalRSSM',
                   n_steps: int,
                   goal: torch.Tensor = None,
                   agent_memory: Optional[Dict[str, List[torch.Tensor]]] = None,
                   env_memory: Optional[Dict[str, List[torch.Tensor]]] = None,
                   disable_exploration: bool = False,
                   sample_actions: bool = True,
                   sample_states: bool = True,
                   reconstruct: bool = True):
        if self.use_slow_world_model:
            world = sim_env._ema_rssm_modules[self.level]
            other_world = sim_env.rssm_modules[self.level]
        else:
            world = sim_env.rssm_modules[self.level]
            other_world = sim_env._ema_rssm_modules[self.level]

        agent_memory = {} if agent_memory is None else agent_memory
        env_mem = {} if env_memory is None else env_memory
        ema_env_mem = {}

        if goal is None:  # for torchscript
            goal = torch.zeros_like(env_start_state[0])
        goal = goal.detach()  # goal is absolute and should not be altered through gradients

        env_state = env_start_state  # note: first observation is not added to agent memory
        with FreezeParameters([world, other_world]):
            for t in range(n_steps):
                agent_o = self.fuse_o_with_goal(env_state, goal)
                a_dist, a = self(agent_o.detach(), sample=sample_actions, disable_exploration=disable_exploration)
                # ema_a_dist, _, ema_v = self(agent_o, use_ema_modules=True, sample=sample_actions,
                #                            disable_exploration=disable_exploration)
                ema_a_dist = torch.ones_like(a_dist)

                if torch.isnan(env_state[0]).any() or torch.isinf(env_state[0]).any():
                    raise RuntimeError(f'Invalid env state in act_in_sim: {env_state[0]}')
                if torch.isnan(agent_o).any() or torch.isinf(agent_o).any():
                    raise RuntimeError(f'Invalid agent observation in act_in_sim: {agent_o}')
                if torch.isnan(a).any() or torch.isinf(a).any():
                    raise RuntimeError(f'Invalid agent action in act_in_sim: {a}')

                # env_state = rssm_detach_state(*env_state)
                next_env_state = world(a=a, last_state=env_state, use_posterior=False, sample_state=sample_states)
                pred = world.decode(next_env_state[-1], sample=False, reconstruct_observation=reconstruct)
                # with torch.no_grad():
                #    _, next_slow_env_state = other_world(a=a, last_state=env_state, use_posterior=False,
                #                                        sample_state=sample_states)
                #    slow_z_dist = other_world.z_dist(next_slow_env_state[1])
                #    z_dist = world.z_dist(next_env_state[1])
                #    novelty = kl_divergence(z_dist, slow_z_dist).detach()
                novelty = torch.zeros_like(agent_o[:, 0])
                # TODO: check if correct time step's z is used for calc rewards and if correct time step's z is stored
                # r = self.build_step_reward(mem['z_dist'][-1], mem['r'][-1], goal)
                r = self.build_step_reward(next_env_state, pred['r'], goal, self.goal_seeking)
                terminal = self.build_step_terminal(next_env_state, goal, pred['terminal'])

                timestep = {
                    'o': agent_o, #self.fuse_o_with_goal(next_env_state, goal), #agent_o,
                    'a': a,
                    'r': r,
                    'terminal': pred['terminal'],
                    'a_dist': a_dist,
                    'ema_a_dist': ema_a_dist,
                    'model_novelty': novelty
                }

                append_memory(agent_memory, **timestep)
                append_memory(env_mem, **pred, **rssm_add_labels(next_env_state))
                env_state = next_env_state

        #append_memory(agent_memory, o=self.fuse_o_with_goal(next_env_state, goal))  # add final observation to memory

        # agent novelty reward, could be computed in eval_step to save some compute
        # a_ema = torch.stack(agent_memory['a'][-n_steps:])
        # _, ema_state_mem = other_world.scan(a_ema, start_state=env_start_state, sample_state=sample_states)
        # z_dist_params = torch.stack(env_mem['z_post'][-n_steps:])
        # slow_z_dist_params = torch.stack([ema_state[2] for ema_state in ema_state_mem])  # always try to get posterior
        # novelty = kl_divergence(world.z_dist(z_dist_params), world.z_dist(slow_z_dist_params))
        # model_novelty = agent_memory.get('model_novelty', [])
        # model_novelty += list(novelty.unbind(0))
        # agent_memory['model_novelty'] = model_novelty

        return {'agent': agent_memory, 'model': env_mem, 'model_state': env_state, 'ema_model': ema_env_mem}

    @torch.jit.export
    def update_exploration(self):
        with torch.no_grad():
            self.eps.copy_(self.eps * self.eps_mul)
            self.eps.copy_(torch.max(self.eps, self.eps_min))

    """
    def scale_to_a_interval(self, action):
        # if self.min_a is not None and self.max_a is not None:
        #    action = clip_but_pass_gradient(action, self.min_a + self.min_float, self.max_a - self.min_float)
        # action = torch.clamp(action, -1.0 + self.min_float, 1.0 - self.min_float)
        if self.min_a is not None and self.max_a is not None:
            action = action * (self.max_a - self.min_a) / 2 + (self.min_a + self.max_a) / 2
        # if self.min_a is not None and self.max_a is not None:
        #    action = torch.clamp(action, self.min_a + self.min_float, self.max_a - self.min_float)
        return action

    def scale_to_unit_interval(self, action):
        if self.min_a is not None and self.max_a is not None:
            action = 2 * (action - (self.min_a + self.max_a) / 2) / (self.max_a - self.min_a)
        # stability reasons
        # action = torch.clamp(action, -1.0 + self.min_float, 1.0 - self.min_float)
        return action
    """

    def _a_dist_params(self,
                       o: torch.Tensor,
                       use_ema_modules: bool = False):
        if use_ema_modules:
            params = self._ema_actor_net(o)
        else:
            params = self.actor_net(o)
        mu, logvar = torch.tensor_split(params, 2, dim=-1)
        mu = torch.nn.functional.tanh(mu)
        sigma = torch.nn.functional.softplus(logvar + self.init_action_variance) + self.min_scale
        # sigma = torch.ones_like(sigma)
        d = torch.stack([mu, sigma], dim=-1)
        return d

    def _a_dist(self,
                mu: torch.Tensor,
                sigma: torch.Tensor):
        # sigma = torch.full_like(sigma, 0.1)
        d = torch.distributions.Normal(loc=mu, scale=sigma)
        d = torch.distributions.TransformedDistribution(d, [TanhBijector()])
        # d = torch.distributions.Independent(d, 1)
        return d

    @torch.jit.ignore
    def _a_dist_stats(self,
                      dist_params: torch.Tensor,
                      valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, sigma = dist_params.unbind(-1)
        # from https://github.com/rlworkgroup/garage/blob/master/src/garage/torch/distributions/tanh_normal.py
        mu = TanhBijector()(mu)
        mean = torch.sum(mu * valid) / torch.sum(valid)
        scale = torch.sum(sigma * valid) / torch.sum(valid)
        return mean, scale

    @torch.jit.ignore
    def _a_smpl(self,
                dist_params: torch.Tensor,
                sample: bool,
                explore: bool = False):
        mu, sigma = dist_params.unbind(-1)
        d = self._a_dist(mu, sigma)

        if sample:
            s = d.rsample()
        else:
            s = torch.nn.functional.tanh(mu)

        if explore:
            noise = torch.distributions.Normal(torch.zeros_like(mu), torch.full_like(sigma, self.eps.data)).sample()
            s = s + noise
            #s = torch.clamp(s, -1.0, 1.0)
            s = torch.tanh(s)
            #sigma = torch.where(sigma < self.eps_min + self.min_float, self.eps_min + self.min_float, sigma)

        # s = self.scale_to_a_interval(s)

        # if self.min_a is not None and self.max_a is not None:
        #   s = torch.clamp(s, self.min_a + self.min_float, self.max_a - self.min_float)

        return s

    @torch.jit.ignore
    def _a_log_prob(self,
                    a_dist_params: torch.Tensor,
                    a: torch.Tensor):
        mu, sigma = a_dist_params.unbind(-1)
        d = self._a_dist(mu, sigma)
        # a = self.scale_to_unit_interval(a)
        a_log_prob = d.log_prob(a)
        return a_log_prob

    @torch.jit.ignore
    def _a_dist_entropy(self,
                        a_dist_params: torch.Tensor):
        mu, sigma = a_dist_params.unbind(-1)
        # from https://arxiv.org/pdf/2006.05990.pdf Appendix B.8 bulletpoint 5
        # and https://math.stackexchange.com/questions/4116762/is-there-a-closed-form-expression-for-entropy-on-tanh-transform-of-gaussian-rand
        # it becomes clear that we don't have a closed form solution for a transformed Gaussian
        # so we use https://github.com/rlworkgroup/garage/blob/master/src/garage/torch/distributions/tanh_normal.py
        d = torch.distributions.Normal(loc=mu, scale=sigma)
        entropy = d.entropy()
        return entropy

    @torch.jit.export
    def eval_step(self,
                  a_dist: list[torch.Tensor],
                  ema_a_dist: list[torch.Tensor],
                  model_novelty: list[torch.Tensor],
                  a: list[torch.Tensor],
                  r: list[torch.Tensor],
                  # r_raw: list[torch.Tensor],
                  terminal: list[torch.Tensor],
                  o: List[torch.Tensor],
                  # o_env: list[torch.Tensor],
                  # o_env_next: list[torch.Tensor],
                  # goal: Union[List[torch.Tensor], List[None]],
                  first_step_mask: Optional[torch.Tensor] = None,
                  for_train_step: bool = False):

        # s_0            s_1            s_2
        #  |  __________^ |  __________^ |
        #  v /            v /            v
        # a_0            a_1            a_2
        # r_0(a_0, s_0), r_1(a_1, s_1), r_2(a_2, s_2)
        # t_0(a_0, s_0), t_1(a_1, s_1), t_2(a_2, s_2)

        o = torch.stack(o)
        terminal = torch.stack(terminal)
        r = torch.stack(r)

        #if self.goal_seeking:
        #    with torch.no_grad():
        #        terminal = torch.zeros_like(terminal)
        #        first_step_mask = torch.zeros_like(first_step_mask)

        # use compute_mask to produce validity matrix that honors first_step_mask, which might mean to make a complete
        # trajectory invalid if the first time step is already after a terminal step.
        mask = compute_mask(terminal, first_step_mask=first_step_mask, disable=False, gamma=0.99)

        if self.normalize_observations:
            _o_mean, _o_var = self.o_running_average(o, mask)
            o = self.o_running_average.normalize(o, _o_mean, _o_var)

        if self.normalize_rewards:
            _r_mean, _r_var = self.r_running_average.update(r, mask)
            r = self.r_running_average.normalize(r, _r_mean, _r_var)

        # if GlobalLogger.can_log('mask_agent', self._current_train_step):
        #    fig = plt.figure(figsize=(5, 5))
        #    plt.matshow(mask.detach().cpu().numpy().squeeze(), fignum=fig, aspect='auto')
        #    plt.colorbar()
        #    agent_name = 'goal_seeking' if self.goal_seeking else 'r_max'
        #    GlobalLogger.logger.log_plot(fig_to_img(fig),
        #                                 Scope.TRAIN() / f'agent/{agent_name}_agent_l{self.level}_mask')
        #    plt.close(fig)
        #    del fig

        # terminal = torch.stack(terminal)
        # terminal = torch.roll(terminal, shifts=0, dims=0)
        # terminal[0] = 0
        # del terminal[-1]
        # terminal.insert(0, torch.zeros_like(terminal[0]))

        # prevent critic net parameters from being updated through policy loss but let gradients of policy loss flow
        # through value network back into simulated environment
        if self.use_slow_value_target:
            with FreezeParameters([self._ema_critic_net]):
                v_actor = self._ema_critic_net(o)
        else:
            with FreezeParameters([self.critic_net]):
                v_actor = self.critic_net(o)

        a = torch.stack(a)
        a_dist = torch.stack(a_dist)
        ema_a_dist = torch.stack(ema_a_dist)
        a_log_prob = self._a_log_prob(a_dist, a.detach())

        # valid_bootstrap = (valid * torch.minimum(v_actor, ema_v))[-1]
        if self.goal_seeking:
            # we only train a single chunk, no bootstrapping needed beyond that
            # CAUTION: we don't train the last step and should get one step more than chunk size
            #bootstrap = (1 - mask)[-1] * r[-1]  # (1 - mask[-1]) * v_actor[-1]
            gamma = 0.90
            #lambda_returns = calc_returns_simple(r[:-1], terminal[:-1], bootstrap, gamma=gamma)
            #bootstrap = torch.zeros_like(r[-1])  # (1 - mask[-1]) * v_actor[-1]
            #lambda_returns = calc_lambda_returns(r, terminal, v_actor, bootstrap, gamma, 0.95)
            # bootstrap = terminal[-1] * valid[-1] * r[-1]
        else:
            gamma = 0.99
            #bootstrap = (1 - mask)[-1] * v_actor[-1]
            #bootstrap = (1 - mask)[-1] * v_actor[-1]  # (1 - mask[-1]) * v_actor[-1]
        bootstrap = v_actor[-1] #(1 - terminal)[-1] * v_actor[-1] + terminal[-1] * r[-1]
        #bootstrap = (1 - terminal)[-1] * v_actor[-1] + terminal[-1] * r[-1]
        lambda_returns = calc_lambda_returns(r[:-1], terminal[:-1], v_actor[:-1], bootstrap, gamma, 0.99)
        #lambda_returns = calc_lambda_returns(r, terminal, v_actor[:-1], bootstrap, gamma, 0.99)
        #lambda_returns = calc_returns_simple(r[:-1], terminal[:-1], bootstrap, gamma=gamma)
        #bootstrap = (1 - mask)[-1] * v_actor[-1]  # (1 - mask[-1]) * v_actor[-1]
        #bootstrap = (1 - mask)[-1] * (((1 - terminal) * v_actor + terminal * r))[-1]
        mask = mask[:-1]
        v_actor = v_actor[:-1]
        o = o[:-1]
        a = a[:-1]
        a_dist = a_dist[:-1]
        a_log_prob = a_log_prob[:-1]
        r = r[:-1]

        # valid_bootstrap = ((1 - terminal) * valid_v + terminal * valid_r)[-1]
        # returns = self._calc_returns_simple(valid_r[:-1], valid_bootstrap, gamma=gamma)
        # bootstrap = valid[-1] * ((1 - terminal[-1]) * v_active[-1] + terminal[-1] * r[-1])
        # lambda_returns = self._calc_returns_simple((r * valid)[:-1], bootstrap, gamma=gamma)
        # lambda_returns = calc_returns_simple(r[:-1], terminal[:-1], bootstrap, gamma=gamma)

        # normalize returns and state values
        ret_mean, ret_std = self.return_running_average(lambda_returns, mask=mask)  # update and return stats
        lambda_returns_actor = self.return_running_average.normalize(lambda_returns, ret_mean, ret_std)
        v_actor = self.return_running_average.normalize(v_actor, ret_mean, ret_std)
        # ret_mean, ret_std = lambda_returns.mean(dim=-1).detach(), lambda_returns.std(dim=-1).detach()
        # lambda_returns_actor = (lambda_returns - ret_mean) / ret_std
        # v_actor = (v_actor - ret_mean) / ret_std
        advantage_actor = lambda_returns_actor - v_actor  # advantage
        #advantage_actor = lambda_returns #- v_actor

        # ACTOR
        if self.dynamics_loss:
            if self.goal_seeking:
                policy_loss = -advantage_actor  # -v_actor[:-1]  # -lambda_returns
            else:
                policy_loss = -advantage_actor  # -v_actor[:-1]  # -lambda_returns
        else:
            # advantage = (lambda_returns - (v_actor * valid)[:-1]).detach()
            policy_loss = -a_log_prob * advantage_actor.detach()

        act_entropy = self._a_dist_entropy(a_dist)
        act_entropy_reward_aug = self.alpha * torch.sum(act_entropy, dim=-1, keepdim=True)
        policy_loss = policy_loss - act_entropy_reward_aug

        # CRITIC
        with torch.no_grad():
            value_target = lambda_returns  # + act_entropy_reward_aug  # + model_novelty_reward_aug[:-1]
        #v_critic = self.critic_net(o.detach()[:-1])
        v_critic = self.critic_net(o.detach())
        value_loss = torch.nn.functional.smooth_l1_loss(v_critic, value_target.detach(), reduction='none')
        # ppo_loss = valid[:-1] * self.beta * torchd.kl_divergence(detach_dist(ema_a_dist),
        #                                                         a_dist).sum(dim=-1, keepdims=True)

        policy_loss = masked_mean(policy_loss, mask)
        value_loss = masked_mean(value_loss, mask)
        entropy_reward_aug = masked_mean(act_entropy_reward_aug, mask)
        ppo_loss = torch.zeros_like(value_loss)  # torch.mean(ppo_loss)
        loss = policy_loss + value_loss  # + ppo_loss

        if self.normalize_rewards:
            running_r = self.r_running_average.mean.mean()
        else:
            running_r = torch.tensor(0.0, device=loss.device)

        if self.normalize_observations:
            running_o = self.o_running_average.mean.mean()
        else:
            running_o = torch.tensor(0.0, device=loss.device)

        with torch.no_grad():
            a_dist_mean = masked_mean(a, mask)
            a_dist_std = masked_var(a, mask)
            a_min = a.min()
            a_max = a.max()
            a_entropy = masked_mean(self._a_dist_entropy(a_dist), mask)
            ep_r = (r * (1 - mask)).sum(dim=0)
            ep_r_mask = torch.where(mask.mean(dim=0) < 1.0, 0.0, 1.0)
            valid_r = masked_mean(ep_r, ep_r_mask)
            a_log_prob_mean = masked_mean(a_log_prob, mask)
            v_mean = masked_mean(v_actor, mask)
            lambda_returns_mean = masked_mean(lambda_returns, mask)

        losses = {'total': loss, 'policy': policy_loss, 'value': value_loss, 'policy_trust_region_loss': ppo_loss,
                  'action_entropy_reward_aug': entropy_reward_aug,
                  'eps_exploration': self.eps,
                  'monitoring_a_dist_mean': a_dist_mean,
                  'monitoring_a_dist_std': a_dist_std,
                  'monitoring_a_min': a_min,
                  'monitoring_a_max': a_max,
                  'monitoring_a_entropy': a_entropy,
                  'monitoring_r_running_average': running_r,
                  'monitoring_o_running_average': running_o,
                  'monitoring_obtained_reward': valid_r,
                  'monitoring_a_log_prob': a_log_prob_mean,
                  'monitoring_v': v_mean,
                  'monitoring_lambda_returns': lambda_returns_mean,
                  'monitoring_faction_valid': torch.sum(1 - mask) / torch.sum(torch.ones_like(mask)),
                  'monitoring_bootstrap': bootstrap.mean()}

        if for_train_step:
            losses['mask'] = mask

        return losses

    @torch.jit.ignore
    def update_step(self,
                    simulation_data: Dict[str, List[torch.Tensor]],
                    first_step_mask: Optional[torch.Tensor],
                    actor_optimizer: torch.optim.Optimizer,
                    critic_optimizer: torch.optim.Optimizer,
                    logger: Logger = None):
        losses = self.eval_step(a_dist=simulation_data['a_dist'],
                                ema_a_dist=simulation_data['ema_a_dist'],
                                model_novelty=simulation_data['model_novelty'],
                                a=simulation_data['a'],
                                r=simulation_data['r'],
                                # r_raw=simulation_data['r_raw'],
                                terminal=simulation_data['terminal'],
                                o=simulation_data['o'],
                                # o_env=simulation_data['o_env'],
                                # o_env_next=simulation_data['o_env_next'],
                                # goal=simulation_data['goal'],
                                first_step_mask=first_step_mask,
                                for_train_step=True)

        with torch.no_grad():
            mask = losses.pop('mask')
            if self.normalize_observations:
                self.o_running_average.update(torch.stack(simulation_data['o']), mask=mask)
            if self.normalize_rewards:
                self.r_running_average.update(torch.stack(simulation_data['r']), mask=mask)

        invalid_losses = ''
        for k, v in losses.items():
            if torch.isnan(v).any() or torch.isinf(v).any():
                invalid_losses += f'{k}: {v}, '
        if len(invalid_losses) > 0:
            raise RuntimeError(f'Invalid loss in {self._agent_repr} detected: {invalid_losses}')

        actor_optimizer.zero_grad(set_to_none=True)
        critic_optimizer.zero_grad(set_to_none=True)

        #val_bef = np.sum([p.detach().cpu().numpy().mean() for p in self.critic_net.parameters()])
        #pol_bef = np.sum([p.detach().cpu().numpy().mean() for p in self.actor_net.parameters()])

        losses['total'].backward()

        # actor_grads = np.concatenate([p.grad.flatten().detach().cpu().numpy() if p.grad is not None else 0
        #                             for p in self.actor_net.parameters()])
        # plt.hist(actor_grads, bins=100)
        # plt.show()
        #torch.nn.utils.clip_grad_value_(self.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.parameters(), 100.0)

        # if logger:
        #    message = {}
        #    for name, p in self.named_parameters():
        #        if p.grad is not None:
        #            message[name] = p.grad.detach().cpu().abs().numpy().mean()
        #    logger.log(message, Scope.PARAMETERS() / 'gradients' / self._agent_repr.replace(' ', '_'))

        actor_optimizer.step()
        critic_optimizer.step()

        #val_aftr = np.sum([p.detach().cpu().numpy().mean() for p in self.critic_net.parameters()])
        #pol_aftr = np.sum([p.detach().cpu().numpy().mean() for p in self.actor_net.parameters()])

        # if self.goal_seeking:
        #_val_diff = val_aftr - val_bef
        #_pol_diff = pol_aftr - pol_bef
        #print(_val_diff)
        #print(_pol_diff)

        self._update_ema_modules()

        # update exploration
        self.update_exploration()
        self._current_train_step += 1

        return losses

    def _update_ema_modules(self):
        with torch.no_grad():
            params = chain.from_iterable([m.parameters() for m in self.actor_net] +
                                         [m.parameters() for m in self.critic_net])
            ema_params = chain.from_iterable([m.parameters() for m in self._ema_actor_net] +
                                             [m.parameters() for m in self._ema_critic_net])
            for param, ema_param in zip(params, ema_params):
                ema_param[:] = self.ema_coeff * ema_param + (1 - self.ema_coeff) * param

    @staticmethod
    def goal_similarity(o: torch.Tensor,
                        goal: torch.Tensor):
        # if isinstance(o, torch.Tensor) and isinstance(goal, torch.Tensor):
        #    return - torch.mean(torch.abs(o - goal) ** 2, dim=-1, keepdim=True)
        # elif isinstance(o, torch.Tensor) and isinstance(goal, Distribution):
        #    return torch.mean(goal.log_prob(o), dim=-1, keepdim=True)
        # elif isinstance(o, Distribution) and isinstance(goal, torch.Tensor):
        #    return torch.mean(o.log_prob(goal), dim=-1, keepdim=True)
        # else:
        #    return - torch.mean(torchd.kl_divergence(goal, o) + torchd.kl_divergence(o, goal), dim=-1, keepdim=True)
        #similarity = torch.pow(0.99, torch.mean(torch.abs(o - goal) ** 2, dim=-1, keepdim=True))
        #o_norm = torch.linalg.vector_norm(o, dim=-1, keepdim=True)
        #goal_norm = torch.linalg.vector_norm(goal, dim=-1, keepdim=True)
        #norm = torch.maximum(o_norm, goal_norm)
        #similarity = torch.linalg.vecdot(goal / norm, o / norm, dim=-1).unsqueeze(-1)
        similarity = - torch.mean(torch.abs(o - goal) ** 2, dim=-1, keepdim=True)
        return similarity

    @torch.jit.export
    def o_from_state(self,
                     step: RSSMStateType):
        if self.observation_type == 'z':
            o = step[1]
        elif self.observation_type == 'h':
            o = step[0]
        elif self.observation_type == 's_embedding':
            o = step[5]
        else:
            raise ValueError(f'Unknown observation_typ: {self.observation_type}')
        return o

    @torch.jit.export
    def fuse_o_with_goal(self,
                         step: RSSMStateType,
                         goal: Optional[torch.Tensor] = None):
        o = self.o_from_state(step)
        if self.goal_seeking:
            # detach goal to avoid propagating gradients to upper level model into other agents
            return torch.concat([o, goal], dim=-1)
        else:
            return o

    @torch.jit.export
    def build_step_reward(self,
                          step: RSSMStateType,
                          r: torch.Tensor,
                          goal: Optional[torch.Tensor] = None,
                          use_goal_reward: bool = False):
        # TODO: should we detach o as well?
        if use_goal_reward:
            o = self.o_from_state(step)
            # return 0.5 * self.goal_similarity(o, goal) + 0.5 * r
            return self.goal_similarity(o, goal)
        else:
            return r

    @torch.jit.export
    def build_step_terminal(self,
                            step: RSSMStateType,
                            goal: torch.Tensor,
                            terminal: torch.Tensor):
        return terminal
        term_prob = terminal
        if self.goal_seeking:
            o = self.o_from_state(step)
            similarity = torch.exp(1000 * - torch.mean(torch.abs(o - goal) ** 2, dim=-1, keepdim=True))
            #similarity = torch.exp(1000 * self.goal_similarity(o, goal))
            term_prob = torch.maximum(similarity, terminal)
        assert torch.all(term_prob >= 0)
        assert torch.all(term_prob <= 1)
        return term_prob

    @property
    def _agent_repr(self):
        agent_name = 'agent'
        if self.goal_seeking:
            agent_name = 'goal seeking ' + agent_name
        else:
            agent_name = 'r max ' + agent_name
        agent_name = f'L{self.level} ' + agent_name

        return agent_name

def calc_lambda_returns(
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        values: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: float,
        lambda_: float
):
    """
    Compute the discounted reward for a batch of data.
    reward, value, and discount are all shape [horizon - 1, batch, 1] (last element is cut off)
    Bootstrap is [batch, 1]
    """
    disc_mat = 1.0 - torch.concat([torch.zeros_like(terminals[0:1]), terminals[:-1]])
    next_values = torch.cat([values[1:], bootstrap[None]], 0)
    target = rewards + disc_mat * discount * next_values * (1 - lambda_)
    timesteps = list(range(rewards.shape[0] - 1, -1, -1))
    outputs = []
    accumulated_reward = bootstrap
    for t in timesteps:
        inp = target[t]
        final_discount = disc_mat[t] * discount
        accumulated_reward = inp + final_discount * lambda_ * accumulated_reward
        outputs.append(accumulated_reward)
    returns = torch.flip(torch.stack(outputs), [0])
    return returns


def calc_returns_simple(rewards: torch.Tensor,
                        terminals: torch.Tensor,
                        state_value_bootstrap: torch.Tensor,
                        gamma: float):
    disc_mat = 1.0 - torch.concat([torch.zeros_like(terminals[0:1]), terminals[:-1]])
    R = state_value_bootstrap
    returns = []
    for t in reversed(range(len(rewards))):
        R = rewards[t] + gamma * disc_mat[t] * R
        returns.insert(0, R)
    return torch.stack(returns)

    # R = state_value_bootstrap
    # returns = []
    # for r in torch.flip(rewards, dims=(0,)):
    #    R = r + gamma * R
    #    returns.insert(0, R)
    # return torch.stack(returns)
