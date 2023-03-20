import copy
import random
from typing import Tuple, Sequence, Optional, Dict
from collections import namedtuple, OrderedDict
from copy import deepcopy

import gymnasium as gym
import torch
import numpy as np

from mdm.utils.torch_tools import layers_with_activation as lwa
from mdm.models.building_blocks import GaussianDecoder
from mdm.models.hierarchical_rssm import HierarchicalRSSM
from mdm.utils.torch_tools import FuzzyDeviceMixin, update_ema_modules, detach_dist


class ActorCriticAgent(FuzzyDeviceMixin, torch.nn.Module):

    def __init__(self,
                 level: int,
                 link: str,
                 s_a: Tuple[int],
                 s_o: Tuple[int],
                 min_a: Sequence[float] = None,
                 max_a: Sequence[float] = None,
                 eps: float = 0,
                 eps_mul: float = 0,
                 ema_reg: bool = False,
                 entropy_exploration: bool = False,
                 model_uncertainty_exploration: bool = False):
        super().__init__()
        self.level = level
        self.link = link
        self.s_a = s_a
        self.s_o = s_o
        self.actor_net = torch.nn.Sequential(lwa(lws=[np.prod(s_o).item(), 64, 64, np.prod(s_a).item() * 2],
                                                 activation='relu', layer_norm=True, name='actor_net'))
        self.critic_net = torch.nn.Sequential(lwa(lws=[np.prod(s_o).item(), 64, 64, 1],
                                                  activation='relu', layer_norm=True, name='actor_net'))

        self._ema_actor_net = copy.deepcopy(self.actor_net)
        self._ema_critic_net = copy.deepcopy(self.critic_net)
        for param in self._ema_actor_net.parameters(): param.detach_()
        for param in self._ema_critic_net.parameters(): param.detach_()

        self.eps = eps
        self.eps_mul = eps_mul
        self.ema_reg = ema_reg
        self.entropy_exploration = entropy_exploration
        self.model_uncertainty_exploration = model_uncertainty_exploration

        if min_a:
            self.min_a = torch.nn.Parameter(torch.tensor(min_a))
            self.min_a.requires_grad = False
        else:
            self.min_a = None

        if max_a:
            self.max_a = torch.nn.Parameter(torch.tensor(max_a))
            self.max_a.requires_grad = False
        else:
            self.max_a = None

    #@torch.compile
    def scale_action(self, action):
        if self.min_a is not None and self.max_a is not None:
            a_scaled = torch.tanh(action) * (self.max_a - self.min_a) / 2 + (self.min_a + self.max_a) / 2
        else:
            a_scaled = action
        return a_scaled

    #@torch.compile
    def _act_dist(self,
                  actor_head: torch.nn.Module,
                  x: torch.Tensor):
        x = actor_head(x)
        mu, logvar = torch.tensor_split(x, 2, dim=-1)
        mu = torch.tanh(mu) * (self.max_a - self.min_a) / 2 + (self.min_a + self.max_a) / 2
        sigma = torch.log(1 + torch.exp(logvar)) + 0.001
        d = torch.distributions.Normal(loc=mu, scale=sigma)
        return d

    #@torch.compile
    def forward(self,
                o: torch.Tensor,
                use_ema_modules: bool = False,
                **kwargs):
        if o.shape == self.s_o:  # add batch dim if not there already
            o = o.unsqueeze(0)

        if use_ema_modules:
            state_values = self._ema_critic_net(o)
            a_dist = self._act_dist(self._ema_actor_net, o)
            # a_dist, a_smpl = self._ema_actor_head(x)
        else:
            state_values = self.critic_net(o)
            a_dist = self._act_dist(self.actor_net, o)
            # a_dist, a_smpl = self._actor_head(x)
        a_smpl = a_dist.rsample()

        # if self.eps > random.random():
        #    a_smpl = a_smpl + torch.distributions.Normal(loc=torch.zeros_like(a_smpl), scale=torch.full_like(a_smpl, 0.1)).sample()
        #    a_dist = torch.distributions.Normal(loc=a_smpl, scale=a_dist.scale)
        if self.eps > 0 and self.training:
            a_smpl = a_smpl + torch.distributions.Normal(loc=torch.zeros_like(a_smpl),
                                                         scale=torch.full_like(a_smpl, self.eps)).sample()

        a_smpl = self.scale_action(a_smpl)
        # a_dist = torch.distributions.Normal(loc=self.scale_action(a_dist.loc), scale=a_dist.scale)

        # if self.min_a is not None:
        #   # loc = torch.maximum(self.min_a, a_dist.loc)
        #   # a_dist = torch.distributions.Normal(loc=loc, scale=a_dist.scale)
        #   a_smpl = torch.maximum(self.min_a, a_smpl)
        # if self.max_a is not None:
        #   # loc = torch.minimum(self.max_a, a_dist.loc)
        #   # a_dist = torch.distributions.Normal(loc=loc, scale=a_dist.scale)
        #   a_smpl = torch.minimum(self.max_a, a_smpl)

        return a_dist, a_smpl, state_values

    #@torch.compile
    def update_exploration(self):
        self.eps *= self.eps_mul

    def sim_train_step(self,
                       sim_env: HierarchicalRSSM,
                       init_data: Dict[str, torch.Tensor],
                       n_steps: int,
                       actor_optimizer: torch.optim.Optimizer,
                       critic_optimizer: torch.optim.Optimizer):
        sim_env.train()
        rewards = []
        terminals = []
        vs = []
        ema_vs = []
        a_dists = []
        ema_a_dists = []
        performed_actions = []
        mocel_uncertainties = []
        mem, env_state = sim_env(o=init_data['o'], a=init_data['a'], r=init_data['r'], terminal=init_data['terminal'],
                                 level=self.level, use_ema_modules=True)
        for t in range(n_steps):
            a_dist, a, v = self(mem[self.link][-1])
            ema_a_dist, _, ema_v = self(mem[self.link][-1], use_ema_modules=True)
            empty_tensor = torch.zeros_like(a)
            mem, next_env_state = sim_env(o=empty_tensor, a=a.unsqueeze(0), r=empty_tensor, terminal=empty_tensor,
                                     n_warmup=0, start_state=env_state, level=self.level, use_ema_modules=False)
            mem_ema, _ = sim_env(o=empty_tensor, a=a.unsqueeze(0), r=empty_tensor, terminal=empty_tensor,
                                   n_warmup=0, start_state=env_state, level=self.level, use_ema_modules=True)
            env_state = next_env_state
            rewards.append(mem['r'][-1])
            terminals.append(mem['terminal'][-1])
            vs.append(v)
            performed_actions.append(a)
            a_dists.append(a_dist)
            ema_a_dists.append(ema_a_dist)
            ema_vs.append(ema_v)
            disagreement = torch.distributions.kl_divergence(detach_dist(mem_ema['z_prior'][-1]),
                                                             mem['z_prior'][-1]).mean(dim=-1)
            # model_disagreement.append(mem['o_dist'][-1].entropy().mean(dim=-1, keepdim=True))
            mocel_uncertainties.append(disagreement)

        # if a terminal transition occurs, the terminal flag is close to 1 and would block out the reward in that step
        # so shift terminals list one to the right and make first element zeros
        del terminals[-1]
        terminals.insert(0, torch.zeros_like(terminals[0]))

        #if self.ema_reg:
        #    vs = [torch.min(v, ema_v) for v, ema_v in zip(vs, ema_vs)]

        # calculate losses
        # from https://github.com/pytorch/examples/blob/main/reinforcement_learning/actor_critic.py
        returns, discounts = self._calc_returns(rewards, vs, terminals, gamma=0.99)
        gae_advantages, _ = self._calc_gae(rewards, vs, terminals, gamma=0.99, lambda_=0.99)
        # returns = torch.stack(returns)
        # returns = (returns - returns.mean()) / (returns.std() + 0.0001)

        policy_losses = []
        value_losses = []
        model_uncertainty_losses = []
        ema_losses = []
        action_dist_entropies = []
        for a_dist, ema_a_dist, a, v, R, gae_advantage, model_uncertainty, discount in zip(a_dists,
                                                                                           ema_a_dists,
                                                                                           performed_actions,
                                                                                           vs,
                                                                                           returns,
                                                                                           gae_advantages,
                                                                                           mocel_uncertainties,
                                                                                           discounts):
            action_dist_entropies.append(a_dist.entropy())
            advantage = R - v.detach()
            policy_losses.append(-advantage)
            #policy_losses.append(-gae_advantage)
            # policy_losses.append(-R)
            # ppo_r = a_dist.log_prob(a.detach()) / detach_dist(ema_a_dist).log_prob(a.detach())
            # ppo_actor_loss = -((R.detach() - v.detach()) * torch.clip(ppo_r, torch.tensor(0.8, device=sim_env.device),
            #                                                          torch.tensor(1.2, device=sim_env.device)))
            # policy_losses.append(ppo_actor_loss)
            # regular_actor_loss = -((R.detach() - v.detach()) * a_dist.log_prob(a.detach()))
            # policy_losses.append(-((R.detach() - v.detach()) * a_dist.log_prob(a.detach())))
            value_target = R.detach()
            value_losses.append(torch.nn.functional.smooth_l1_loss(v, value_target, reduction='none'))
            model_uncertainty_losses.append(model_uncertainty)
            ema_losses.append(torch.distributions.kl_divergence(detach_dist(ema_a_dist), a_dist).sum(-1, keepdim=True))
            # ema_losses.append(torch.nn.functional.mse_loss(v, ema_v.detach(), reduction='none'))
            # exploration_losses.append(- 0.001 * a_dist.entropy().mean(dim=-1, keepdim=True) * discount)
        policy_loss = torch.mean(torch.stack(policy_losses))
        value_loss = torch.mean(torch.stack(value_losses))
        if self.ema_reg:
            ema_loss = torch.mean(torch.stack(ema_losses))
        else:
            ema_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        if self.entropy_exploration:
            entropy_loss = - torch.mean(torch.stack(action_dist_entropies))
        else:
            entropy_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        if self.model_uncertainty_exploration:
            model_uncertainty_loss = - torch.mean(torch.stack(model_uncertainty_losses))
        else:
            model_uncertainty_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        # before = [p.detach().cpu().numpy() for p in self._net_body.parameters()]
        # ema_before = [p.detach().cpu().numpy() for p in self._ema_net_body.parameters()]
        # update model
        actor_optimizer.zero_grad(set_to_none=True)
        critic_optimizer.zero_grad(set_to_none=True)
        loss = policy_loss + value_loss + ema_loss + entropy_loss + model_uncertainty_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        actor_optimizer.step()
        critic_optimizer.step()
        # after = [p.detach().cpu().numpy() for p in self._net_body.parameters()]
        # ema_after = [p.detach().cpu().numpy() for p in self._ema_net_body.parameters()]

        # diff = np.sum([np.abs(x - y).sum() for x, y in zip(before, after)])
        # ema_diff = np.sum([np.abs(x - y).sum() for x, y in zip(ema_before, ema_after)])

        agent_params = [OrderedDict(m.named_parameters()) for m in (self.actor_net, self.critic_net)]
        ema_params = [OrderedDict(m.named_parameters()) for m in (self._ema_actor_net, self._ema_critic_net)]
        # ema_before = [p.detach().cpu().numpy() for p in self._ema_net_body.parameters()]
        update_ema_modules(agent_params, ema_params, 0.99)
        # ema_after = [p.detach().cpu().numpy() for p in self._ema_net_body.parameters()]
        # ema_diff = np.sum([np.abs(x - y).sum() for x, y in zip(ema_before, ema_after)])

        return {'total': loss, 'policy': policy_loss, 'value': value_loss, 'exploration': model_uncertainty_loss,
                'ema': ema_loss, 'action_entropy': entropy_loss}

    #@staticmethod
    @torch.compile
    def _calc_returns(rewards, state_values, timestep_mask, gamma):
        returns = []
        discounts = []
        R = state_values[-1].detach()
        for r, mask in zip(rewards[::-1], timestep_mask[::-1]):
            gamma_final = ((1 - mask) * gamma).detach()
            R = r + gamma_final * R
            returns.insert(0, R)
            discounts.insert(0, gamma_final)
        return returns, discounts

    #@staticmethod
    @torch.compile
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
        return advantages, discounts
