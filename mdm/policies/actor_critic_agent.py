import copy
import math
import random
from itertools import chain
from typing import Tuple, Sequence, Optional, Dict, List, Union
from collections import namedtuple, OrderedDict
from copy import deepcopy

#from torchrl.objectives.value.functional import (td_lambda_return_estimate, _fast_td_lambda_return_estimate,
#                                                 vec_td_lambda_return_estimate)

import gymnasium as gym
import torch
import torch.distributions as torchd
from torch.distributions import Distribution
from torch.distributions import kl_divergence
import numpy as np
import matplotlib.pyplot as plt
from torchviz import make_dot
from colorama import Fore, Back, Style

from mdm.models.rssm_cell import RSSMStateType, rssm_detach_state, rssm_add_labels
from mdm.utils.torch_tools import layers_with_activation as lwa, SquashedNormal, RunningMeanStd
from mdm.utils.torch_tools import (FuzzyDeviceMixin, compute_mask, detach_dist, stack_dists,
                                   concat_dists, TanhBijector, clip_but_pass_gradient, FreezeParameters,
                                   plot_grad_flow, masked_mean, masked_var, check_tensor)
from mdm.utils.utils import fig_to_img, append_memory, numpyfy, extend_memory, list_of_tuples_to_tuple_of_lists
from mdm.logging.logger import GlobalLogger, Scope, Logger


class ActorCriticAgent(torch.nn.Module):

    def __init__(self,
                 level: int,
                 observation_type: str,
                 d_o: int,
                 d_a: int,
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
                 target_act_entropy: str | float = 'auto',
                 novelty_exploration_coeff: float = 0.0,
                 min_scale: float = 0.1,
                 use_slow_world_model: bool = False,
                 use_slow_value_target: bool = False,
                 goal_seeking: bool = False,
                 dynamics_loss: bool = True,
                 init_action_variance: float = 2.0,
                 **kwargs):
        super().__init__()

        if goal_seeking:
            d_o = 2 * d_o

        self.level = level
        self.observation_type = observation_type
        self.d_a = d_a
        self.d_o = d_o
        self.ema_coeff = ema_update_coeff
        self.beta = tr_policy_kl_coeff
        self.eps = torch.nn.Parameter(torch.tensor(eps_exploration_init, dtype=torch.float32), requires_grad=False)
        self.eps_mul = torch.nn.Parameter(torch.tensor(eps_exploration_coeff, dtype=torch.float32), requires_grad=False)
        self.eps_min = torch.nn.Parameter(torch.tensor(eps_exploration_min, dtype=torch.float32, requires_grad=False))
        self.alpha = torch.nn.Parameter(torch.tensor(act_entropy_exploration_coeff, dtype=torch.float32))
        self.learn_aplha = learn_act_entropy_exploration_coeff
        # see https://github.com/ray-project/ray/blob/5edabc7b2f92712982b0e590f739eb0110d80176/rllib/algorithms/sac/sac.py#L165
        if target_act_entropy == 'auto':
            target_act_entropy = float(-d_a)
        self.target_act_entropy = target_act_entropy
        # if not learn_act_entropy_exploration_coeff:
        #    self.alpha.requires_grad = False
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
        self.ema_critic_net = copy.deepcopy(self.critic_net)
        for param in self._ema_actor_net.parameters(): param.requires_grad = False
        for param in self.ema_critic_net.parameters(): param.requires_grad = False
        self._current_train_step = 0
        self.dynamics_loss = dynamics_loss
        self.init_action_variance = init_action_variance

        self.return_running_average = RunningMeanStd(shape=(1,))

        self.goal_reached_eps = 0.05
        self.min_float = torch.finfo().eps

    def forward(self,
                o: torch.Tensor,
                sample: bool,
                expl_noise: float = 0.0,
                use_ema_modules: bool = False):
        if o.shape == self.d_o:  # add batch dim if not there already
            o = o.unsqueeze(0)

        a_dist_params = self._a_dist_params(o)
        mu, sigma = a_dist_params.unbind(-1)
        a_dist = self._a_dist(mu, sigma)

        if sample:
            a_smpl = a_dist.rsample()
        else:
            #a_smpl = torch.nn.functional.tanh(mu)
            a_smpl = a_dist.mean

        if expl_noise > 0.0:
            noise = torch.distributions.Normal(loc=torch.zeros_like(mu), scale=torch.full_like(mu, expl_noise)).sample()
            #a_smpl = a_smpl + noise
            a_smpl = torch.clamp(a_smpl + noise, -1.0 + 1e-5, 1.0 - 1e-5)

        #a_smpl = torch.tanh(a_smpl)

        if torch.any(a_smpl > 1.0) or torch.any(a_smpl < -1.0):
            raise RuntimeError(f'Invalid action: {a_smpl}')

        # clipped = torch.clamp(a_smpl, -1.0 + 1e-6, 1.0 - 1e-6)
        # a_smpl = clipped.detach() + a_smpl - a_smpl.detach()
        # a_smpl = clipped
        # a_smpl = torch.tanh(a_smpl)

        return a_dist_params, a_smpl

    @torch.jit.ignore
    def act_in_sim(self,
                   env_start_state: RSSMStateType,
                   sim_env: 'HierarchicalRSSM',
                   n_steps: int,
                   goal: torch.Tensor = None,
                   agent_memory: Optional[Dict[str, List[torch.Tensor]]] = None,
                   env_memory: Optional[Dict[str, List[torch.Tensor]]] = None,
                   sample_actions: bool = True,
                   sample_states: bool = True,
                   reconstruct: bool = True,
                   expl_noise: float = 0.0):
        if self.use_slow_world_model:
            world = sim_env._ema_rssm_modules[self.level]
            other_world = sim_env.rssm_modules[self.level]
        else:
            world = sim_env.rssm_modules[self.level]
            other_world = sim_env._ema_rssm_modules[self.level]

        agent_memory = {} if agent_memory is None else agent_memory
        env_memory = {} if env_memory is None else env_memory
        ema_env_mem = {}

        env_states = [env_start_state]
        agent_obs = [self.fuse_o_with_goal(env_start_state, goal)]
        agent_acts = [torch.zeros(env_start_state[-1].shape[0], self.d_a, device=env_start_state[-1].device)]
        agent_act_dists = [torch.ones(env_start_state[-1].shape[0], self.d_a, 2, device=env_start_state[-1].device)]
        obs_autoenc = sim_env.goal_autoencoder
        with FreezeParameters([world, obs_autoenc]):
            for t in range(n_steps):
                a_dist, a = self(agent_obs[-1].detach(), sample=sample_actions, expl_noise=expl_noise)
                # ema_a_dist, _, ema_v = self(agent_o, use_ema_modules=True, sample=sample_actions,
                #                            disable_exploration=disable_exploration)
                current_env_state = world(a=a, last_state=env_states[-1], use_posterior=False,
                                          sample_state=sample_states)

                env_states.append(current_env_state)
                agent_obs.append(self.fuse_o_with_goal(current_env_state, goal))
                agent_acts.append(a)
                agent_act_dists.append(a_dist)

            # transform list of RSSM state tuples to a tuple of lists, each element of the tuple being a list that
            # containing all time steps of the previous tuple's elements at that position
            # [(a_0, b_0), (a_1, b_1), ...] -> ([a_0, a_1, ...], [b_0, b_1, ...])
            env_states_tpl = list_of_tuples_to_tuple_of_lists(env_states)
            s_embed_stacked = torch.stack(env_states_tpl[-1])  # use last RSSM state element for decoding
            pred = world.decode(s_embed_stacked, sample=False, reconstruct_observation=True)
            pred = {k: list(v.unbind(0)) for k, v in pred.items()}  # make first dimension to list again
            pred.update(rssm_add_labels(env_states_tpl))  # add rssm states to the memory
            extend_memory(env_memory, pred)  # write contents of this act_in_sim() call to env memory

            # repeat the goal over the time dimension to match the format in env_states
            if self.goal_seeking:
                #goal = goal.detach()  # goals come from upper level management and should not be changed by workers
                # encode goal and observations for easier comparison
                #_, goal_enc = obs_autoenc.encode(goal, sample=False)
                #_, obs_enc = obs_autoenc.encode(s_embed_stacked, sample=False)
                # insert observation and goal encodings into their respective tensors
                #goal = torch.zeros_like(goal)
                #goal[:, :goal_enc.shape[-1]] = goal_enc
                #s_embed_stacked = torch.zeros_like(s_embed_stacked)
                #s_embed_stacked[:, :, :obs_enc.shape[-1]] = obs_enc

                #goal = goal.detach()  # goals come from upper level management and should not be changed by workers
                # repeat the same goal for each time step
                #goal_tiled = goal.unsqueeze(0).expand(n_steps + 1, -1, -1)
                #agent_r = self.goal_similarity(s_embed_stacked, goal_tiled)
                #agent_r = list(agent_r.unbind(0))
                #agent_term = pred['terminal']
                goal = world.decode(goal.detach(), sample=False, reconstruct_observation=True)['o']
                goal_tiled = goal.unsqueeze(0).expand(n_steps + 1, -1, -1).clone()  # clone to be on the safe side for now
                state_obs = torch.stack(env_memory['o'])
                #goal[:, 2:] = 0
                #state_obs[:, :, 2:] = 0
                agent_r = self.goal_similarity(state_obs, goal_tiled)
                agent_term = self.goal_terminal(agent_r)

                agent_r = list(agent_r.unbind(0))
                agent_term = list(agent_term.unbind(0))
                #agent_term = [torch.zeros_like(t) for t in pred['terminal']]  # completely disable actual env term flag
                pred['r'] = None
            else:
                agent_r = pred['r']
                agent_term = pred['terminal']

            #if self.level > 0:
            #    _, _, _, s_embed_rec = obs_autoenc(s_embed_stacked, sample=False)
            #    novelty_loss = torch.mean((s_embed_stacked - s_embed_rec) ** 2, dim=-1, keepdim=True)
            #    agent_r = [r_t + reconstr_loss_t for r_t, reconstr_loss_t in zip(agent_r, novelty_loss)]

            ema_a_dist_mock = [torch.zeros_like(x) for x in agent_act_dists]
            extend_memory(agent_memory, {'o': agent_obs, 'a': agent_acts, 'r': agent_r, 'terminal': agent_term,
                                         'a_dist': agent_act_dists, 'ema_a_dist': ema_a_dist_mock, 'model_novelty': []})

        return {'agent': agent_memory, 'model': env_memory, 'model_state': env_states[-1], 'ema_model': ema_env_mem}

    @staticmethod
    def _check(a, agent_o, last_env_state, pred):
        if torch.isnan(last_env_state[0]).any() or torch.isinf(last_env_state[0]).any():
            raise RuntimeError(f'Invalid env state in act_in_sim: {last_env_state[0]}')
        if torch.isnan(agent_o).any() or torch.isinf(agent_o).any():
            raise RuntimeError(f'Invalid agent observation in act_in_sim: {agent_o}')
        if torch.isnan(a).any() or torch.isinf(a).any():
            raise RuntimeError(f'Invalid agent action in act_in_sim: {a}')
        for k, v in pred.items():
            if torch.isnan(v).any():
                print(f'found nan value in {k}')

    @torch.jit.export
    def update_exploration(self):
        with torch.no_grad():
            self.eps.copy_(self.eps * self.eps_mul)
            self.eps.copy_(torch.max(self.eps, self.eps_min))

    def _a_dist_params(self,
                       o: torch.Tensor):
        params = self.actor_net(o)
        mu, logvar = torch.tensor_split(params, 2, -1)
        mu = torch.tanh(mu)  # limit total range of mu but make it easy for the actor net to saturate it

        #logvar = logvar + 3.0  # make initial variance high
        #sigma = torch.nn.functional.softplus(logvar) + self.min_scale
        #sigma = torch.nn.functional.sigmoid(logvar) + self.min_scale  # limit total range of sigma
        logvar = torch.clamp(logvar, -3, 2)  # from rllib
        sigma = torch.exp(logvar) + self.min_scale
        d = torch.stack([mu, sigma], dim=-1)
        return d

    def _a_dist(self,
                mu: torch.Tensor,
                sigma: torch.Tensor):
        # sigma = torch.full_like(sigma, 0.1)
        #d = torch.distributions.Normal(loc=mu, scale=sigma)
        #d = torch.distributions.TransformedDistribution(d, [TanhBijector()])
        d = SquashedNormal(loc=mu, scale=sigma)
        # d = torch.distributions.Independent(d, 1)
        return d

    # see https://github.com/ray-project/ray/blob/d9722d2bafe8d7fd4ffb95f4ac64701720526c56/rllib/models/torch/torch_action_dist.py#L337
    @torch.jit.ignore
    def _a_log_prob(self,
                    a_dist_params: torch.Tensor,
                    a: torch.Tensor):
        mu, sigma = a_dist_params.unbind(-1)
        d = self._a_dist(mu, sigma)
        #a = torch.clamp(a, -1.0 + 1e-5, 1.0 - 1e-5)  # clamp actions to avoid numerical instability
        #a = torch.atanh(a)
        a_log_prob = d.log_prob(a)#.unsqueeze(-1)
        #a_log_prob = a_log_prob.sum(dim=-1, keepdim=True)  # sum over action dimension

        return a_log_prob

        a_unsquashed = torch.atanh(torch.clamp(a, -1 + 0.00001, 1 - 0.00001))
        a_log_prob_unsquashed = torch.distributions.Normal(loc=mu, scale=sigma).log_prob(a_unsquashed)
        a_log_prob_unsquashed = torch.clamp(a_log_prob_unsquashed, -100, 100)
        a_log_prob_unsquashed = a_log_prob_unsquashed.sum(dim=-1)
        a_unsquashed_tanhd = torch.tanh(a_unsquashed)
        a_log_prob_2 = a_log_prob_unsquashed - torch.sum(torch.log(1 - a_unsquashed_tanhd ** 2 + 0.00001), dim=-1)
        a_log_prob_2 = a_log_prob_2.unsqueeze(-1)

        if torch.isnan(a_log_prob_2).any() or torch.isinf(a_log_prob_2).any():
            raise RuntimeError('NaN or inf in action log prob found')
        return a_log_prob_2

    @torch.jit.ignore
    def _a_dist_entropy(self,
                        a_dist_params: torch.Tensor):
        raise NotImplementedError('Doesn\'t work for SquashedGaussian')
        mu, sigma = a_dist_params.unbind(-1)
        # from https://arxiv.org/pdf/2006.05990.pdf Appendix B.8 bulletpoint 5
        # and https://math.stackexchange.com/questions/4116762/is-there-a-closed-form-expression-for-entropy-on-tanh-transform-of-gaussian-rand
        # it becomes clear that we don't have a closed form solution for a transformed Gaussian
        # so we use https://github.com/rlworkgroup/garage/blob/master/src/garage/torch/distributions/tanh_normal.py
        d = torch.distributions.Normal(loc=mu, scale=sigma)
        entropy = d.entropy()
        return entropy

    @torch.jit.ignore
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
        # In general: First time step comes from replay memory, so the first action is zero padding and not optimized by
        # the actor. The last action doesn't yield any outcome and is thus not optimized either. Only the critic can
        # optimize the first time step. It doesn't optimize the last time step howeve, as it serves as the bootstrap
        # value estimate for all previous time steps.
        o = torch.stack(o)
        r = torch.stack(r)
        terminal = torch.stack(terminal)
        a = torch.stack(a)
        a_dist = torch.stack(a_dist)
        ema_a_dist = torch.stack(ema_a_dist)
        a_log_prob = self._a_log_prob(a_dist, a.detach())

        if self.goal_seeking:
            gamma = 0.95
            lambda_ = 0.95
        else:
            gamma = 0.99
            lambda_ = 0.95

        with torch.no_grad():
            if first_step_mask is None:
                first_step_mask = torch.zeros_like(terminal[0:1])
            if first_step_mask.ndim == 2:
                first_step_mask = first_step_mask.unsqueeze(0)
            # shift terminal one to the right and apply first_step_mask if available
            terminal_shifted = torch.concat([first_step_mask, terminal[:-1].detach()], dim=0)
            # see https://github.com/ray-project/ray/blob/5ae0ddaf4360eeef0525ca33d9670251337ca930/rllib/algorithms/dreamerv3/tf/models/dreamer_model.py#L397
            # compute validity of time steps by applying cumulativie product for each time step
            valid = torch.cumprod((1 - terminal_shifted) * gamma, dim=0) / gamma
            # we need the opposite of valid array, which is a mask array
            mask = (1 - valid)
            mask_t1_to_H = mask[:-1]
            mask_t1_to_Hm1 = mask[:-2]


        # prevent critic net parameters from being updated through policy loss but let gradients of policy loss flow
        # through value network back into simulated environment
        # with FreezeParameters([self.critic_net]):
        #    v_actor = self.critic_net(o)
        # if self.use_slow_value_target:
        #    with FreezeParameters([self.ema_critic_net]):
        #        v_actor_slow = self.ema_critic_net(o)
        #    v_actor = torch.minimum(v_actor, v_actor_slow)
        with FreezeParameters([self.critic_net]):
            v_actor_online = self.critic_net(o)
        with FreezeParameters([self.ema_critic_net]):
            v_actor_ema = self.ema_critic_net(o)

        if self.use_slow_value_target:
            v_actor = torch.minimum(v_actor_online, v_actor_ema)
        else:
            v_actor = v_actor_online

        #r_mean, r_std = self.return_running_average(r, mask)
        #r = self.return_running_average.normalize(r, r_mean, r_std)

        lambda_returns = self.compute_targets_dreamer_v2(r, terminal, v_actor, gamma, lambda_)

        #lambda_returns = td_lambda_return_estimate(gamma, lambda_, v_actor[1:], r[:-1],
        #                                           terminal[:-1].to(torch.bool), time_dim=0)

        #lambda_returns = _fast_td_lambda_return_estimate(gamma, lambda_, v_actor[1:].transpose(0, 1),
        #                                                 r[:-1].transpose(0, 1),
        #                                                 terminal[:-1].to(torch.bool).transpose(0, 1),
        #                                                 terminal[:-1].to(torch.bool).transpose(0, 1))
        #lambda_returns = lambda_returns.transpose(0, 1)

        # lambda_returns = calc_lambda_returns(r[1:], terminal[1:], v_actor[:-1], bootstrap, gamma, lambda_)

        # lambda_returns = calc_returns_simple(r[:-1], terminal[:-1], bootstrap, 0.99)

        #if self.goal_seeking:
        #    lambda_returns_actor = lambda_returns
        #else:
        ret_mean, ret_std = self.return_running_average(lambda_returns, mask_t1_to_H)  # update and return stats
        lambda_returns_actor = self.return_running_average.normalize(lambda_returns, ret_mean, ret_std)
        v_actor = self.return_running_average.normalize(v_actor, ret_mean, ret_std)
        #lambda_returns_actor = lambda_returns

        #params = {**dict(self.named_parameters())}
        #make_dot(lambda_returns_actor.sum(), params).view()
        #quit()

        # ACTOR
        if self.dynamics_loss:
            # we can't optimize return for first state, as it comes from replay buffer
            # policy_loss = -advantage_actor * valid
            # policy_loss = -a_log_prob * advantage_actor.detach() * valid
            # policy_loss = -advantage_actor
            policy_loss = - masked_mean(lambda_returns_actor[1:], mask_t1_to_Hm1)
        else:
            # policy_loss = -a_log_prob[:-1] * advantage_actor[1:].detach() * valid[:-1]
            # advantage = (lambda_returns - (v_actor * valid)[:-1]).detach()
            # policy_loss = -a_log_prob * advantage_actor.detach() * valid
            # policy_loss = -a_log_prob[1:] * advantage_actor[:-1].detach() * valid[:-1]
            # according to dreamer v2 reference implementation, this should be v_actor[:-2] but that doesn't seem right
            baseline = v_actor[:-2]  # this was v_actor[:-2] before, but didn't work at all
            advantage_actor = lambda_returns_actor[1:] - baseline  # .detach()
            policy_loss = - masked_mean(a_log_prob[1:-1] * advantage_actor.detach(), mask_t1_to_Hm1)
        # policy_loss = torch.sum(policy_loss)

        # ACTION ENTROPY LOSS
        #act_entropy = self._a_dist_entropy(a_dist[1:-1])
        act_entropy = torch.sum(a_log_prob[1:-1], dim=-1, keepdim=True)  # sum over action dim
        # act_entropy_loss = torch.sum(act_entropy_loss * valid)  # sum over T and B
        act_entropy_loss = - self.alpha.detach() * masked_mean(act_entropy, mask_t1_to_Hm1)
        # act_entropy_loss = - torch.maximum(self.alpha.detach(), torch.zeros_like(self.alpha)) * act_entropy_loss

        # ALPHA LOSS
        # see https://github.com/ray-project/ray/blob/cb5bb4e7763994db4f4e765ce9dc74376d7eedca/rllib/algorithms/sac/sac_tf_policy.py#L415
        if self.learn_aplha:
            alpha_loss = self.alpha * (a_log_prob[1:-1].detach() + self.target_act_entropy)
            # alpha_loss = - torch.sum(alpha_loss * valid)
            alpha_loss = - masked_mean(alpha_loss, mask_t1_to_Hm1)
        else:
            alpha_loss = torch.zeros_like(policy_loss)

        # CRITIC
        with torch.no_grad():
            value_target = lambda_returns
            v_ema_critic = self.ema_critic_net(o.detach()[:-1])
            ema_value_loss = torch.nn.functional.smooth_l1_loss(v_ema_critic, value_target.detach(), reduction='none')
            # ema_value_loss = torch.sum(ema_value_loss * valid)
            ema_value_loss = masked_mean(ema_value_loss, mask_t1_to_H)

        v_critic = self.critic_net(o.detach()[:-1])
        value_loss = torch.nn.functional.smooth_l1_loss(v_critic, value_target.detach(), reduction='none')
        # value_loss = (v_critic - value_target.detach()) ** 2
        # value_loss = torch.sum(value_loss * valid)
        value_loss = masked_mean(value_loss, mask_t1_to_H)

        # TODO: currently last action is not trained, we can change that and record last state in act_in_sim as well

        ppo_loss = torch.zeros_like(value_loss)  # torch.mean(ppo_loss)
        loss = policy_loss + value_loss + act_entropy_loss + alpha_loss  # + ppo_loss

        # log statistics
        with torch.no_grad():
            # reward and state values are only counted from second time step on as the first one is based on a model
            # state that came from experience memory and is ground truth
            per_time_step_mask = torch.where(mask_t1_to_H.mean(dim=0) < 1.0, 0.0, 1.0)
            a_dist_mean = masked_mean(a[1:-1], mask_t1_to_Hm1)
            a_dist_std = masked_var(a[1:-1], mask_t1_to_Hm1)
            a_min = a.min()
            a_max = a.max()
            ep_r = (r[1:] * (1 - mask_t1_to_H)).sum(dim=0)
            valid_r = masked_mean(ep_r, per_time_step_mask)
            a_log_prob_mean = masked_mean(a_log_prob[1:-1], mask_t1_to_Hm1)
            a_entropy_mean = masked_mean(act_entropy, mask_t1_to_Hm1)
            v_mean = masked_mean(v_critic, mask_t1_to_H)
            ema_v_mean = masked_mean(v_ema_critic, mask_t1_to_H)
            v_result_mean = masked_mean(v_actor[1:], mask_t1_to_H)
            lambda_returns_mean = masked_mean(lambda_returns, mask_t1_to_H)
            mu, sigma = a_dist.unbind(-1)
            mu_mean = masked_mean(mu[1:-1], mask_t1_to_Hm1)
            sigma_mean = masked_mean(sigma[1:-1], mask_t1_to_Hm1)

        """
        if self.goal_seeking:
            opts = np.get_printoptions()
            np.set_printoptions(precision=3, linewidth=120, floatmode='fixed', legacy='1.13')
            step_r = masked_mean(r[1:], mask[1:], dim=1).detach().cpu().numpy().squeeze()
            step_a_magnitude = a.abs().mean(dim=[0, 1]).detach().cpu().numpy()
            # print(step_r / (step_r[0] - 1e-5))
            print(step_r, end=' ')
            print(Style.DIM, step_a_magnitude, Style.RESET_ALL)
            #np.set_printoptions(precision=1, linewidth=120, floatmode='fixed', legacy=None)
            #print(Back.BLUE, Fore.WHITE, mask.detach().cpu().numpy().squeeze(), Style.RESET_ALL)
            #print(Back.YELLOW, Fore.BLACK, terminal.detach().cpu().numpy().squeeze(), Style.RESET_ALL)
            #print(Back.RED, Fore.WHITE, r.detach().cpu().numpy().squeeze(), Style.RESET_ALL)
            np.set_printoptions(**opts)
        """

        losses = {'total': loss,
                  'policy': policy_loss,
                  'value': value_loss,
                  'ema_value': ema_value_loss,
                  'alpha': self.alpha,
                  'policy_trust_region_loss': ppo_loss,
                  'action_entropy_reward_aug': act_entropy_loss,
                  'eps_exploration': self.eps,
                  'alpha_loss': alpha_loss,
                  'monitoring_a_dist_mean': a_dist_mean,
                  'monitoring_a_dist_std': a_dist_std,
                  'monitoring_a_min': a_min,
                  'monitoring_a_max': a_max,
                  'monitoring_a_entropy': a_entropy_mean,
                  'monitoring_obtained_reward': valid_r,
                  'monitoring_a_log_prob': a_log_prob_mean,
                  'monitoring_v': v_mean,
                  'monitoring_ema_v': ema_v_mean,
                  'monitoring_v_result': v_result_mean,
                  'monitoring_a_mu': mu_mean,
                  'monitoring_a_sigma': sigma_mean,
                  'monitoring_lambda_returns': lambda_returns_mean,
                  'monitoring_faction_valid': torch.sum(1 - mask_t1_to_H) / torch.sum(torch.ones_like(mask_t1_to_H)),
                  'monitoring_bootstrap': v_actor[-1].mean()}

        # if for_train_step:
        #    losses['mask_t1_to_H'] = mask_t1_to_H

        invalid_losses = ''
        for k, v in losses.items():
            if torch.isnan(v).any() or torch.isinf(v).any():
                invalid_losses += f'{k}: {v}, '
        if len(invalid_losses) > 0:
            raise RuntimeError(f'Invalid loss in {self._agent_repr} detected: {invalid_losses}')

        return losses

    @staticmethod
    def compute_value_targets(rewards, terminals, values, gamma, lambda_):
        # TODO: this is wrong I think, look at original dreamer implementation and check against rllib
        #  (this is currently following rllib)
        r_t1_to_Hm1 = rewards[1:]
        disc_t1_to_H = (1 - terminals[1:]) * gamma
        returns = [values[-1]]
        intermediates = r_t1_to_Hm1 + disc_t1_to_H * (1 - lambda_) * values[1:]

        for t in reversed(range(disc_t1_to_H.shape[0])):
            returns.append(intermediates[t] + disc_t1_to_H[t] * lambda_ * returns[-1])

        returns = torch.stack(list(reversed(returns))[:-1], dim=0)
        return returns

    # see https://github.com/danijar/dreamerv2/blob/07d906e9c4322c6fc2cd6ed23e247ccd6b7c8c41/dreamerv2/agent.py#L307
    @staticmethod
    def compute_targets_dreamer_v2(rewards, terminals, values, gamma, lambda_):
        discount = (1 - terminals[:-1]) * gamma
        next_values = values[1:]
        inputs = rewards[:-1] + discount * next_values * (1 - lambda_)

        returns = []
        last = values[-1]
        for t in reversed(range(len(discount))):
            returns.append(inputs[t] + discount[t] * lambda_ * last)
            last = returns[-1]

        returns = torch.stack(list(reversed(returns)), dim=0)
        return returns

    @torch.jit.ignore
    def update_step(self,
                    simulation_data: Dict[str, List[torch.Tensor]],
                    first_step_mask: Optional[torch.Tensor],
                    actor_optimizer: torch.optim.Optimizer,
                    critic_optimizer: torch.optim.Optimizer,
                    other_optimizer: torch.optim.Optimizer,
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

        actor_optimizer.zero_grad(set_to_none=True)
        critic_optimizer.zero_grad(set_to_none=True)
        other_optimizer.zero_grad(set_to_none=True)

        # val_bef = np.sum([p.detach().cpu().numpy().mean() for p in self.critic_net.parameters()])
        # pol_bef = np.sum([p.detach().cpu().numpy().mean() for p in self.actor_net.parameters()])

        losses['total'].backward()

        # actor_grads = np.concatenate([p.grad.flatten().detach().cpu().numpy() if p.grad is not None else 0
        #                             for p in self.actor_net.parameters()])
        # plt.hist(actor_grads, bins=100)
        # plt.show()
        #torch.nn.utils.clip_grad_value_(self.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        # if logger:
        #    message = {}
        #    for name, p in self.named_parameters():
        #        if p.grad is not None:
        #            message[name] = p.grad.detach().cpu().abs().numpy().mean()
        #    logger.log(message, Scope.PARAMETERS() / 'gradients' / self._agent_repr.replace(' ', '_'))

        actor_optimizer.step()
        critic_optimizer.step()
        other_optimizer.step()

        # val_aftr = np.sum([p.detach().cpu().numpy().mean() for p in self.critic_net.parameters()])
        # pol_aftr = np.sum([p.detach().cpu().numpy().mean() for p in self.actor_net.parameters()])

        # val_diff = val_aftr - val_bef
        # pol_diff = pol_aftr - pol_bef
        # print(f'val: {val_diff} | pol: {pol_diff}')

        self._update_ema_modules()

        # val_ema = np.sum([p.abs().detach().cpu().numpy().sum() for p in self._ema_critic_net.parameters()])
        # print(val_ema)

        # update exploration
        self.update_exploration()
        self._current_train_step += 1

        return losses

    def _update_ema_modules(self):
        with torch.no_grad():
            params = chain.from_iterable([m.parameters() for m in self.actor_net] +
                                         [m.parameters() for m in self.critic_net])
            ema_params = chain.from_iterable([m.parameters() for m in self._ema_actor_net] +
                                             [m.parameters() for m in self.ema_critic_net])
            for param, ema_param in zip(params, ema_params):
                ema_param[:] = self.ema_coeff * ema_param + (1 - self.ema_coeff) * param

    @staticmethod
    def goal_similarity(o: torch.Tensor,
                        goal: torch.Tensor) -> torch.Tensor:
        # if isinstance(o, torch.Tensor) and isinstance(goal, torch.Tensor):
        #    return - torch.mean(torch.abs(o - goal) ** 2, dim=-1, keepdim=True)
        # elif isinstance(o, torch.Tensor) and isinstance(goal, Distribution):
        #    return torch.mean(goal.log_prob(o), dim=-1, keepdim=True)
        # elif isinstance(o, Distribution) and isinstance(goal, torch.Tensor):
        #    return torch.mean(o.log_prob(goal), dim=-1, keepdim=True)
        # else:
        #    return - torch.mean(torchd.kl_divergence(goal, o) + torchd.kl_divergence(o, goal), dim=-1, keepdim=True)
        # similarity = torch.pow(0.99, torch.mean(torch.abs(o - goal) ** 2, dim=-1, keepdim=True))

        #o_norm = torch.linalg.vector_norm(o, dim=-1, keepdim=True)
        #goal_norm = torch.linalg.vector_norm(goal, dim=-1, keepdim=True)
        #norm = torch.maximum(o_norm, goal_norm).detach()
        #similarity = torch.linalg.vecdot(goal / norm, o / norm, dim=-1).unsqueeze(-1)

        #similarity = torch.where(similarity < 0.999, 0.0, 1.0)

        similarity = - torch.nn.functional.mse_loss(o, goal, reduction='none').mean(dim=-1, keepdim=True)
        #similarity = - torch.mean(torch.sqrt((o - goal) ** 2), dim=-1, keepdim=True)
        #similarity = torch.where(similarity >= -0.0001, 1.0, 0.0)
        check_tensor(similarity)
        return similarity

    @staticmethod
    def goal_terminal(agent_r: torch.Tensor):
        # nav2d with varying reward positions
        #term_zone_core_radius = 0.0005
        #term_zone_perimeter_radius = 0.001

        # PointMaze_UMaze with varying reward positions
        #term_zone_core_radius = 0.001
        #term_zone_perimeter_radius = 0.003

        term_zone_core_radius = 0.001
        term_zone_perimeter_radius = 0.1

        # sigmoid is close to 1.0 at x=3.0 and close to 0.0 at x=-3.0
        sig_min = -5.0
        sig_max = 5.0
        # transform reward so that
        # * terminal ≈ 0.0 if distance >= term_zone_perimeter_radius
        # * terminal > 0.0 if term_zone_core_radius <= distance <= term_zone_perimeter_radius
        # * terminal ≈ 1.0 if distance <= term_zone_core_radius

        offset = (term_zone_perimeter_radius + term_zone_core_radius) / 2
        magnification = (sig_max - sig_min) / (term_zone_perimeter_radius - term_zone_core_radius)
        agent_term = (agent_r + offset) * magnification
        #agent_term = torch.clamp(agent_term, -3.0, 3.0)  # stabilize
        agent_term = torch.sigmoid(agent_term)

        check_tensor(agent_term, neg_bound=0.0, pos_bound=1.0)

        return agent_term

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
            similarity = self.goal_similarity(o, goal)
            return similarity
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
