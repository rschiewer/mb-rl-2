import copy
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

from mdm.utils.torch_tools import layers_with_activation as lwa, SquashedNormal
from mdm.utils.torch_tools import FuzzyDeviceMixin, compute_mask, detach_dist, compile_if_not_debug, stack_dists, \
    concat_dists
from mdm.utils.utils import fig_to_img
from mdm.logging.logger import GlobalLogger, Scope


class ActorCriticAgent(FuzzyDeviceMixin, torch.nn.Module):

    def __init__(self, level: int,
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
            self.min_a = torch.nn.Parameter(torch.tensor(min_a, dtype=torch.float32))
            self.min_a.requires_grad = False
        else:
            self.min_a = None

        if max_a:
            if isinstance(max_a, float):
                max_a = [max_a for _ in range(d_a)]
            self.max_a = torch.nn.Parameter(torch.tensor(max_a, dtype=torch.float32))
            self.max_a.requires_grad = False
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

    # @torch.compile
    def scale_action(self, action):
        #if self.min_a is not None and self.max_a is not None:
        #    a_scaled = torch.tanh(action) * (self.max_a - self.min_a) / 2 + (self.min_a + self.max_a) / 2
        #else:
        #    a_scaled = action
        if self.min_a is not None and self.max_a is not None:
            action = action * (self.max_a - self.min_a) / 2 + (self.min_a + self.max_a) / 2
        action = torch.clamp(action, self.min_a + torch.finfo().eps, self.max_a - torch.finfo().eps)
        return action

    @compile_if_not_debug
    def _act_dist(self,
                  actor_net: torch.nn.Module,
                  x: torch.Tensor):
        # TODO: test penalty for too large pre-scaled activations
        x = actor_net(x)
        mu, logvar = torch.tensor_split(x, 2, dim=-1)
        #mu = torch.tanh(mu) * (self.max_a - self.min_a) / 2 + (self.min_a + self.max_a) / 2
        sigma = torch.nn.functional.softplus(logvar) + 0.01
        #d = torch.distributions.Normal(loc=mu, scale=sigma)
        #transforms = [torch.distributions.TanhTransform()]
        #d = torch.distributions.TransformedDistribution(d, transforms)
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

        if use_ema_modules:
            state_values = self._ema_critic_net(o)
            a_dist = self._act_dist(self._ema_actor_net, o)
        else:
            state_values = self.critic_net(o)
            a_dist = self._act_dist(self.actor_net, o)

        if sample:
            a_smpl = a_dist.rsample()
        else:
            a_smpl = a_dist.loc

        if self.eps > 0 and self.training and not disable_exploration:
            noise = torchd.Normal(loc=torch.zeros_like(a_smpl), scale=torch.full_like(a_smpl, self.eps)).sample()
            a_smpl = a_smpl + noise

        a_smpl = self.scale_action(a_smpl)

        return a_dist, a_smpl, state_values

    # @torch.compile
    def update_exploration(self):
        self.eps *= self.eps_mul
        self.eps = torch.max(self.eps, torch.tensor(0.1, dtype=torch.float32, device=self.eps.device))

    def eval_step(self,
                  a_dist: list[torch.Tensor],
                  ema_a_dist: list[torch.Tensor],
                  model_novelty: list[torch.Tensor],
                  a: list[torch.Tensor],
                  r: list[torch.Tensor],
                  terminal: list[torch.Tensor],
                  v: list[torch.Tensor],
                  ema_v: list[torch.Tensor],
                  first_step_mask: torch.Tensor | None = None,
                  **kwargs):
        # if a terminal transition occurs, the terminal flag is close to 1 and would block out the reward in that step
        # so shift terminals list one to the right and make first element zeros
        mask = compute_mask(torch.stack(terminal), first_step_mask=first_step_mask)

        if GlobalLogger.bound and self._current_train_step % 20 == 0:
            fig = plt.figure(figsize=(8, 8))
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

        # if self.ema_reg:
        # v = torch.stack([torch.min(v, ema_v) for v, ema_v in zip(v, ema_v)])  # use this for policy targets

        # TODO: bootstrap ist value vom letzten state, sollte ich die Returns nur bis zum vorletzten step ausrechnen
        #       und mache es gerade falsch weil der value des letzten states doppeld gezählt wird?

        # value function prediction for last time step is bootstrap for return and doesn't have a target itself
        valid = (1 - mask)  #
        valid_r = (valid * r)[:-1]
        #valid_bootstrap = (valid * torch.minimum(v, ema_v))[-1]
        valid_bootstrap = (valid * v)[-1]
        v = v[:-1]
        ema_v = ema_v[:-1]
        a_dist = a_dist[:-1]
        ema_a_dist = ema_a_dist[:-1]
        model_novelty = model_novelty[:-1]
        valid = valid[:-1]
        returns = self._calc_returns_simple(valid_r, valid_bootstrap, gamma=0.99)

        act_entropy_reward_aug = self.alpha * stack_dists(a_dist).entropy().sum(dim=-1, keepdims=True)
        #advantage = returns - torch.minimum(v, ema_v).detach()
        #advantage = returns - v.detach()
        #policy_loss = valid * (-(advantage + act_entropy_reward_aug))  # validity mask multiplied directly
        policy_loss = valid * (-(returns + act_entropy_reward_aug))  # validity mask multiplied directly
        model_novelty_reward_aug = valid * self.mu * torch.stack(model_novelty).unsqueeze(-1).detach()
        value_target = returns.detach() + model_novelty_reward_aug  # validity mask is multiplied with value_loss
        value_loss = valid * torch.nn.functional.smooth_l1_loss(v, value_target, reduction='none')
        ppo_loss = valid * self.beta * torchd.kl_divergence(detach_dist(stack_dists(ema_a_dist)),
                                                            stack_dists(a_dist)).sum(dim=-1, keepdims=True)

        policy_loss = torch.mean(policy_loss)
        value_loss = torch.mean(value_loss)
        ppo_loss = torch.mean(ppo_loss)
        model_novelty_reward_aug = torch.mean(model_novelty_reward_aug)
        entropy_reward_aug = torch.mean(act_entropy_reward_aug)
        loss = policy_loss + value_loss + ppo_loss

        """
        calculate losses
        from https://github.com/pytorch/examples/blob/main/reinforcement_learning/actor_critic.py
        returns, discount = self._calc_returns(r, v, terminal, gamma=0.99)
        gae_advantages, _ = self._calc_gae(r, v, terminal, gamma=0.99, lambda_=0.95)
        # returns = torch.stack(returns)
        # returns = (returns - returns.mean()) / (returns.std() + 0.0001)

        policy_losses = []
        value_losses = []
        ppo_losses = []
        act_entropy_reward_augs = []
        model_novelty_reward_augs = []
        for a_dist_, ema_a_dist_, v_, R, gae_advantage_, model_uncertainty_, discount_ in zip(a_dist,
                                                                                              ema_a_dist,
                                                                                              v,
                                                                                              returns,
                                                                                              gae_advantages,
                                                                                              model_novelty,
                                                                                              discount):
            # ACTOR
            act_entropy_reward_aug = self.alpha * discount_ * a_dist_.entropy().sum(dim=-1, keepdims=True)
            advantage = R - v_.detach()
            policy_losses.append(-(discount_ * (advantage + act_entropy_reward_aug)))
            # policy_losses.append(-gae_advantage_)
            # policy_losses.append(-R)
            # ppo_r = a_dist.log_prob(a.detach()) / detach_dist(ema_a_dist).log_prob(a.detach())
            # ppo_actor_loss = -((R.detach() - v.detach()) * torch.clip(ppo_r, torch.tensor(0.8, device=sim_env.device),
            #                                                          torch.tensor(1.2, device=sim_env.device)))
            # policy_losses.append(ppo_actor_loss)
            # regular_actor_loss = -((R.detach() - v.detach()) * a_dist.log_prob(a.detach()))
            # policy_losses.append(-((R.detach() - v.detach()) * a_dist.log_prob(a.detach())))
            # CRITIC
            model_novelty_reward_aug = self.mu * discount_ * model_uncertainty_.unsqueeze(-1)
            value_target = R.detach() + model_novelty_reward_aug.detach()
            value_losses.append(discount_ * torch.nn.functional.smooth_l1_loss(v_, value_target, reduction='none'))

            # TRUST REGION POLICY UPDATE REGULARIZATION
            ppo_losses.append(discount_ * torchd.kl_divergence(detach_dist(ema_a_dist_), a_dist_).sum(-1, keepdim=True))

            # BOOKKEEPING
            act_entropy_reward_augs.append(act_entropy_reward_aug)
            model_novelty_reward_augs.append(model_novelty_reward_aug)

        policy_loss = torch.mean(torch.stack(policy_losses))
        value_loss = torch.mean(torch.stack(value_losses))
        ppo_loss = self.beta * torch.mean(torch.stack(ppo_losses))
        entropy_reward_aug = torch.mean(torch.stack(act_entropy_reward_augs))
        model_novelty_reward_aug = torch.mean(torch.stack(model_novelty_reward_augs))
        loss = policy_loss + value_loss + ppo_loss
        """

        return {'total': loss, 'policy': policy_loss, 'value': value_loss, 'policy_trust_region_loss': ppo_loss,
                'model_novelty_reward_aug': model_novelty_reward_aug, 'action_entropy_reward_aug': entropy_reward_aug,
                'eps_exploration': self.eps, 'monitoring_a_dist_mean': torch.stack([d.loc for d in a_dist]).mean(),
                'monitoring_a_dist_std': torch.stack([d.scale for d in a_dist]).mean()}

    def train_step(self,
                   simulation_data: Dict[str, torch.Tensor],
                   actor_optimizer: torch.optim.Optimizer,
                   critic_optimizer: torch.optim.Optimizer,
                   **kwargs):
        actor_optimizer.zero_grad(set_to_none=True)
        critic_optimizer.zero_grad(set_to_none=True)

        losses = self.eval_step(**simulation_data)

        # invalid_losses = ''
        # for k, v in losses.items():
        #    if torch.isnan(v).any() or torch.isinf(v).any():
        #        invalid_losses += f'{k}: {v}, '
        # if len(invalid_losses) > 0:
        #    raise RuntimeError(f'Invalid loss in {self._agent_repr} detected: {invalid_losses}')

        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 100.0)
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

    """
    def sim_train_step(self,
                       a_dist: list[torch.Tensor],
                       ema_a_dist: list[torch.Tensor],
                       model_novelty: list[torch.Tensor],
                       a: list[torch.Tensor],
                       r: list[torch.Tensor],
                       terminal: list[torch.Tensor],
                       v: list[torch.Tensor],
                       actor_optimizer: torch.optim.Optimizer,
                       critic_optimizer: torch.optim.Optimizer,
                       **kwargs):
        # if a terminal transition occurs, the terminal flag is close to 1 and would block out the reward in that step
        # so shift terminals list one to the right and make first element zeros
        del terminal[-1]
        terminal.insert(0, torch.zeros_like(terminal[0]))

        # if self.ema_reg:
        #    vs = [torch.min(v, ema_v) for v, ema_v in zip(vs, ema_vs)]

        # calculate losses
        # from https://github.com/pytorch/examples/blob/main/reinforcement_learning/actor_critic.py
        returns, discount = self._calc_returns(r, v, terminal, gamma=0.99)
        gae_advantages, _ = self._calc_gae(r, v, terminal, gamma=0.99, lambda_=0.99)
        # returns = torch.stack(returns)
        # returns = (returns - returns.mean()) / (returns.std() + 0.0001)

        policy_losses = []
        value_losses = []
        ppo_losses = []
        act_entropy_reward_augs = []
        model_novelty_reward_augs = []
        for a_dist_, ema_a_dist_, v_, R, gae_advantage_, model_uncertainty_, discount_ in zip(a_dist, ema_a_dist, v,
                                                                                              returns, gae_advantages,
                                                                                              model_novelty, discount):
            # ACTOR
            # advantage = R - v_.detach()
            # policy_losses.append(-advantage)
            policy_losses.append(-gae_advantage_)
            # policy_losses.append(-R)
            # ppo_r = a_dist.log_prob(a.detach()) / detach_dist(ema_a_dist).log_prob(a.detach())
            # ppo_actor_loss = -((R.detach() - v.detach()) * torch.clip(ppo_r, torch.tensor(0.8, device=sim_env.device),
            #                                                          torch.tensor(1.2, device=sim_env.device)))
            # policy_losses.append(ppo_actor_loss)
            # regular_actor_loss = -((R.detach() - v.detach()) * a_dist.log_prob(a.detach()))
            # policy_losses.append(-((R.detach() - v.detach()) * a_dist.log_prob(a.detach())))
            # CRITIC
            act_entropy_reward_aug = self.alpha * discount_ * a_dist_.entropy().sum(dim=-1, keepdims=True)
            model_novelty_reward_aug = self.mu * discount_ * model_uncertainty_.unsqueeze(-1)
            value_target = R.detach() + act_entropy_reward_aug + model_novelty_reward_aug
            value_losses.append(torch.nn.functional.smooth_l1_loss(v_, value_target, reduction='none'))

            # TRUST REGION POLICY UPDATE REGULARIZATION
            ppo_losses.append(torchd.kl_divergence(detach_dist(ema_a_dist_), a_dist_).sum(-1, keepdim=True))

            # BOOKKEEPING
            act_entropy_reward_augs.append(act_entropy_reward_aug)
            model_novelty_reward_augs.append(model_novelty_reward_aug)

        policy_loss = torch.mean(torch.stack(policy_losses))
        value_loss = torch.mean(torch.stack(value_losses))
        ppo_loss = self.beta * torch.mean(torch.stack(ppo_losses))
        entropy_reward_aug = torch.mean(torch.stack(act_entropy_reward_augs))
        model_novelty_reward_aug = torch.mean(torch.stack(model_novelty))

        # before = [p.detach().cpu().numpy() for p in self._ema_actor_net.parameters()]
        # update model
        actor_optimizer.zero_grad(set_to_none=True)
        critic_optimizer.zero_grad(set_to_none=True)
        loss = policy_loss + value_loss + ppo_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        actor_optimizer.step()
        critic_optimizer.step()
        # after = [p.detach().cpu().numpy() for p in self._ema_actor_net.parameters()]

        # diff = np.sum([np.abs(x - y).sum() for x, y in zip(before, after)])
        # print(diff)

        agent_params = [OrderedDict(m.named_parameters()) for m in (self.actor_net, self.critic_net)]
        ema_params = [OrderedDict(m.named_parameters()) for m in (self._ema_actor_net, self._ema_critic_net)]
        # ema_before = [p.detach().cpu().numpy() for p in self._ema_net_body.parameters()]
        update_ema_modules(agent_params, ema_params, self.ema_coeff)
        # ema_after = [p.detach().cpu().numpy() for p in self._ema_net_body.parameters()]
        # ema_diff = np.sum([np.abs(x - y).sum() for x, y in zip(ema_before, ema_after)])

        return {'total': loss, 'policy': policy_loss, 'value': value_loss, 'policy_trust_region_loss': ppo_loss,
                'model_novelty_reward_aug': model_novelty_reward_aug, 'action_entropy_reward_aug': entropy_reward_aug}

    """

    def act_in_sim(self,
                   env_state: Dict[str, torch.Tensor],
                   sim_env: 'HierarchicalRSSM',
                   n_steps: int,
                   goal: torch.Tensor = None,
                   agent_memory: Dict[str, List[torch.Tensor]] | None = None,
                   disable_exploration: bool = False,
                   sample_actions: bool = True,
                   sample_model: bool = True,
                   reconstruct: bool = True):
        """
        This function can
        * train agent
        * train model
        * let agent follow a goal
        * let agent maximize rewards
        """
        agent_memory = {} if agent_memory is None else agent_memory
        env_mem, ema_env_mem = {}, {}
        for t in range(n_steps):
            agent_o = self.preproc_o(env_state, goal)

            a_dist, a, v = self(agent_o, sample=sample_actions, disable_exploration=disable_exploration)

            if torch.isnan(env_state['z']).any() or torch.isinf(env_state['z']).any():
                raise RuntimeError(f'Invalid env state in act_in_sim: {env_state["z"]}')
            if torch.isnan(agent_o).any() or torch.isinf(agent_o).any():
                raise RuntimeError(f'Invalid agent observation in act_in_sim: {agent_o}')
            if torch.isnan(a).any() or torch.isinf(a).any():
                raise RuntimeError(f'Invalid agent action in act_in_sim: {a}')

            ema_a_dist, _, ema_v = self(agent_o, use_ema_modules=True)
            trajectory = {'o': None, 'a': a.unsqueeze(0), 'r': None, 'terminal': None}
            mem, mem_ema, next_env_state = sim_env.forward_static(trajectory=trajectory, start_state=env_state,
                                                                  level=self.level, n_steps=1, n_warmup=0,
                                                                  use_ema_modules=self.use_slow_world_model,
                                                                  memory=env_mem, memory_other=ema_env_mem,
                                                                  sample_state=sample_model, sample_output=sample_model,
                                                                  reconstruct=reconstruct)
            r = self.build_step_reward(mem[self.observation_key][-1], mem['r'][-1], goal)
            novelty = kl_divergence(mem_ema['z_dist'][-1], mem['z_dist'][-1]).sum(dim=-1)
            timestep = {'o': agent_o, 'a': a, 'r': r, 'terminal': mem['terminal'][-1], 'v': v, 'ema_v': ema_v,
                        'a_dist': a_dist, 'ema_a_dist': ema_a_dist, 'model_novelty': novelty}

            # store agent data
            for k, v in timestep.items():
                data = agent_memory.get(k, [])
                data.append(v)
                agent_memory[k] = data

            # prepare next step
            env_state = next_env_state

        return {'agent': agent_memory, 'model': env_mem, 'model_state': env_state, 'ema_model': ema_env_mem}

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
        returns = []
        R = state_value_bootstrap.detach()
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
            return - torch.mean((o - goal) ** 2, dim=-1, keepdim=True)
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

        o = o.detach()  # don't propagate trhough multiple time steps

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
                          goal: torch.Tensor | torchd.Distribution | None = None):
        if self.goal_seeking:
            # detach goal to avoid propagating gradients to upper level model into other agents
            if isinstance(goal, torchd.Distribution):
                goal = detach_dist(goal)
            else:
                goal = goal.detach()
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
