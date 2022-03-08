from typing import Tuple, Dict
from collections import deque
from math import ceil

import torch
import torch.nn.functional as F

from mdm.utils.torch_tools import *
from mdm.models.dynamics_model import DynamicsModel


class old_DeterministicRecurrentModel(torch.nn.Module):

    def __init__(self,
                 d_state: int,
                 d_action: int,
                 d_hidden: int,
                 n_layers: int = 1):
        super(old_DeterministicRecurrentModel, self).__init__()

        self.d_state = d_state
        self.d_actions = d_action
        self.d_hidden = d_hidden
        self.layers = torch.nn.LSTM(d_state + d_action, d_hidden, num_layers=n_layers, batch_first=True)

    def forward(self,
                s: torch.Tensor,
                a: torch.Tensor,
                h: Tuple[torch.Tensor, torch.Tensor] = None):
        assert len(s.shape) == len(a.shape) == 2, ('Expected input with shape (batch_size, dim), ',
                                                               f'found {s.shape}, {a.shape}')
        n_batch = s.shape[0]
        if h is None:
            h = (torch.zeros(n_batch, self.d_hidden), torch.zeros(n_batch, self.d_hidden))

        x = torch.concat([s, a], dim=-1)
        x, h = self.layers(x, h)

        return x, h


class MacroActionModel(torch.nn.Module, DeviceMixin):

    def __init__(self,
                 d_action: int,
                 n_abstract_steps: int,
                 d_macro_action: int):
        super(MacroActionModel, self).__init__()

        self.flatten_layer = torch.nn.Flatten(start_dim=1)
        self.det_mdl = FeedforwardBlock(d_action * n_abstract_steps, lws=(64, d_macro_action))

    def forward(self, actions: torch.Tensor):
        # actions.shape = (d_batch, n_abstract_steps, d_action)
        macro_action = self.flatten_layer(actions)
        macro_action = self.det_mdl(macro_action)
        macro_action = F.gumbel_softmax(macro_action, hard=True)
        return macro_action


class MacroRewardModel(torch.nn.Module, DeviceMixin):

    def __init__(self,
                 d_reward: int,
                 n_abstract_steps: int,
                 d_macro_reward: int):
        super(MacroRewardModel, self).__init__()

        if d_reward != d_macro_reward:
            raise NotImplementedError('Reward and macro reward must not differ in dimension')

    def forward(self, rewards: torch.Tensor):
        macro_reward = torch.mean(rewards, dim=1)  # mean over time dimension, keep batch and data dimensions
        return macro_reward


class SingleStepModel(torch.nn.Module, DeviceMixin):

    def __init__(self,
                 det_mdl: RecurrentBlock,
                 sampling_mdl_s: GaussianBlock,
                 sampling_mdl_r: GaussianBlock):
        super(SingleStepModel, self).__init__()

        self.det_mdl = det_mdl
        self.sampling_mdl_s = sampling_mdl_s
        self.sampling_mdl_r = sampling_mdl_r

    def forward(self,
                s: torch.Tensor,
                a: torch.Tensor,
                macro_s: torch.Tensor,
                macro_a: torch.Tensor,
                macro_r: torch.Tensor,
                macro_s_next: torch.Tensor,
                h: torch.Tensor):
        s, a, macro_s, macro_a, macro_r, macro_s_next = add_time_dim(s, a, macro_s, macro_a, macro_r, macro_s_next,
                                                                     batch_first=self.det_mdl.batch_first)

        sr_next_det, h = self.det_mdl(s, a, macro_s, macro_a, macro_r, macro_s_next, h=h)
        sr_next_det = remove_time_dim(sr_next_det, batch_first=self.det_mdl.batch_first)  
        s_next_dist = self.sampling_mdl_s(sr_next_det)
        r_next_dist = self.sampling_mdl_r(sr_next_det)

        return sr_next_det, s_next_dist, r_next_dist, h


class AbstractModel(torch.nn.Module, DeviceMixin):

    def __init__(self,
                 det_mdl: RecurrentBlock,
                 sampling_mdl_s_prior: GaussianBlock,
                 sampling_mdl_s_posterior: GaussianBlock,
                 sampling_mdl_r_prior: GaussianBlock,
                 sampling_mdl_r_posterior: GaussianBlock):
        super(AbstractModel, self).__init__()

        self.det_mdl = det_mdl  # same det_mdl is used for prior and posterior
        self.sampling_mdl_s_prior = sampling_mdl_s_prior
        self.sampling_mdl_s_posterior = sampling_mdl_s_posterior
        self.sampling_mdl_r_prior = sampling_mdl_r_prior
        self.sampling_mdl_r_posterior = sampling_mdl_r_posterior

    def forward(self,
                hl_s: torch.Tensor,
                hl_a: torch.Tensor,
                ll_h: torch.Tensor,
                hl_h: torch.Tensor):
        #if h_ss is None:
        #    d_batch = macro_s.shape[0]
        #    d_memory = self.det_mdl.d_inputs[-1]
        #    h_ss = torch.zeros(d_batch, d_memory, device=self.device)

        hl_s, hl_a = add_time_dim(hl_s, hl_a, batch_first=self.det_mdl.batch_first)
        hl_sr_next_det, hl_h = self.det_mdl(hl_s, hl_a, h=hl_h)
        hl_sr_next_det = remove_time_dim(hl_sr_next_det, batch_first=self.det_mdl.batch_first)
        hl_s_next_prior = self.sampling_mdl_s_prior(hl_sr_next_det)
        hl_r_next_prior = self.sampling_mdl_r_prior(hl_sr_next_det)
        hl_s_next_posterior = self.sampling_mdl_s_posterior(hl_sr_next_det, ll_h)
        hl_r_next_posterior = self.sampling_mdl_r_posterior(hl_sr_next_det, ll_h)

        return hl_s_next_prior, hl_s_next_posterior, hl_r_next_prior, hl_r_next_posterior, hl_h


class MultiscaleDynamicsModel(DynamicsModel):

    def __init__(self,
                 single_step_model: SingleStepModel,
                 abstract_model: AbstractModel,
                 macro_action_model: torch.nn.Module,
                 abstract_step_size: int,
                 d_state: int,
                 d_action: int,
                 d_reward: int,
                 d_macro_state: int,
                 d_macro_action: int,
                 d_macro_reward: int):
        super(MultiscaleDynamicsModel, self).__init__()

        if not single_step_model.det_mdl.batch_first and macro_action_model.batch_first:
            raise ValueError('This model requires all its components to obey the batch_first convention')

        self.single_step_model = single_step_model
        self.abstract_model = abstract_model
        self.macro_action_model = macro_action_model
        self.abstract_step_size = abstract_step_size
        self.d_state = d_state
        self.d_action = d_action
        self.d_reward = d_reward
        self.d_hl_state = d_macro_state
        self.d_hl_action = d_macro_action
        self.d_hl_reward = d_macro_reward

    @staticmethod
    def reconstruction_loss(x_pred, x_true):
        l = torch.mean((x_pred - x_true) ** 2)
        return l

    @staticmethod
    def kl_loss(s_priors, s_posteriors, r_priors, r_posteriors):
        l = 0
        for s_prior, s_posterior, r_prior, r_posterior in zip(s_priors, s_posteriors, r_priors, r_posteriors):
            fixed_s_post = torch.distributions.Normal(loc=s_posterior.loc.detach(), scale=s_posterior.scale.detach())
            fixed_r_post = torch.distributions.Normal(loc=r_posterior.loc.detach(), scale=r_posterior.scale.detach())
            l += torch.distributions.kl.kl_divergence(fixed_s_post, s_prior)
            l += torch.distributions.kl.kl_divergence(fixed_r_post, r_prior)
        l /= len(s_priors)  # normalize the loss w.r.t. the number of time steps explicitly
        return torch.mean(l)

    @staticmethod
    def kl_regularizer(s_priors, r_priors):
        uniform_gauss = torch.distributions.Normal(loc=torch.zeros_like(s_priors[0].loc, requires_grad=False),
                                                   scale=torch.ones_like(s_priors[0].scale, requires_grad=False))
        l = 0
        for s_prior, r_prior in zip(s_priors, r_priors):
            l += torch.distributions.kl.kl_divergence(s_prior, uniform_gauss)
            l += torch.distributions.kl.kl_divergence(r_prior, uniform_gauss)
        l /= len(s_priors)  # normalize the loss w.r.t. the number of time steps explicitly
        return torch.mean(l)

    @staticmethod
    def macro_reward_loss(macro_r_pred, macro_r_true):
        l = torch.mean((macro_r_pred - macro_r_true) ** 2)
        return l

    @staticmethod
    def average_kstep_reward(rewards: torch.Tensor, k: int, device: torch.device):
        d_batch, d_time, d_data = rewards.shape
        n_macro_steps = ceil(d_time / k)
        d_padding = n_macro_steps * k - d_time

        padding = torch.zeros(d_batch, d_padding, d_data, device=device)
        rewards = torch.concat([rewards, padding], dim=1)

        avg = []
        for i in range(n_macro_steps):
            i_start = i * k
            i_stop = (i + 1) * k
            avg.append(torch.mean(rewards[:, i_start:i_stop], dim=1, keepdim=True))

        return torch.concat(avg, dim=1)

    def forward(self,
                start_states: torch.Tensor,
                actions: torch.Tensor,
                context: torch.Tensor = None):
        d_batch, n_steps = actions.shape[:2]
        n_start_states = start_states.shape[1]
        device = self.device

        s = torch.zeros(d_batch, self.d_state, device=device)
        hl_h = self.abstract_model.det_mdl.gen_h_placeholder(d_batch)
        ll_h = self.single_step_model.det_mdl.gen_h_placeholder(d_batch)
        hl_s = torch.zeros(d_batch, self.d_hl_state, device=device)
        #hl_a = self.macro_action_model(self._next_single_step_actions(actions, 0))
        hl_a = torch.zeros(d_batch, self.d_hl_action, device=device)
        hl_r = torch.zeros(d_batch, self.d_hl_reward, device=device)
        hl_s_next = torch.zeros_like(hl_s, device=device)

        s_mem, s_dist_mem = [], []
        r_mem, r_dist_mem = [], []
        hl_r_mem = []
        hl_s_prior_mem, hl_s_posterior_mem = [], []
        hl_r_prior_mem, hl_r_posterior_mem = [], []

        for t in range(n_steps):
            if t < n_start_states:  # if still in warmup period, use teacher forcing for states
                s = start_states[:, t]
            a = actions[:, t]

            if t % self.abstract_step_size == 0 and t > 0:  # invoke abstract model every k time steps
                # TODO: think about this model more closely
                hl_a = self.macro_action_model(self._next_single_step_actions(actions, t))
                ll_h_flat = self._flatten_h(ll_h)
                hl_s = hl_s_next  # update current macro state to previously predicted one

                # invoke abstract model
                out_abstr = self.abstract_model(hl_s, hl_a, ll_h_flat, hl_h)
                hl_s_next_prior, hl_s_next_posterior, hl_r_next_prior, hl_r_next_posterior, hl_h = out_abstr

                # sample from more informed posterior distributions to get inputs for single step model
                hl_s_next = hl_s_next_posterior.rsample()
                hl_r = hl_r_next_posterior.rsample()

                # store for loss calculation
                hl_r_mem.append(hl_r)
                hl_s_prior_mem.append(hl_s_next_prior)
                hl_r_prior_mem.append(hl_r_next_prior)
                hl_s_posterior_mem.append(hl_s_next_posterior)
                hl_r_posterior_mem.append(hl_r_next_posterior)

                # prevent memory leakage beyond macro steps
                ll_h = (torch.zeros_like(ll_h[0], device=device), torch.zeros_like(ll_h[1], device=device))
                s = torch.zeros_like(s, device=device)

            #ll_h = (torch.zeros_like(ll_h[0], device=device), torch.zeros_like(ll_h[1], device=device))
            _, s_next_dist, r_next_dist, ll_h = self.single_step_model(s, a, hl_s, hl_a, hl_r, hl_s_next,
                                                                       ll_h)
            s_next = s_next_dist.rsample()
            r_next = r_next_dist.rsample()

            # store single step states and rewards for loss calculation
            s_mem.append(s_next)
            r_mem.append(r_next)
            s_dist_mem.append(s_next_dist)
            r_dist_mem.append(r_next_dist)

            # set next state to upcoming time step's current state
            s = s_next

        # TODO: make this a dict
        return (s_mem,
                s_dist_mem,
                r_mem,
                r_dist_mem,
                hl_s_prior_mem,
                hl_s_posterior_mem,
                hl_r_mem,
                hl_r_prior_mem,
                hl_r_posterior_mem)

    def _next_single_step_actions(self, actions: torch.Tensor, t: int):
        n_steps = actions.shape[1]
        if t + self.abstract_step_size > n_steps:
            diff = t + self.abstract_step_size - n_steps
            next_actions = actions[:, t:]
            next_actions = torch.concat([next_actions, torch.zeros(actions.shape[0], diff, actions.shape[2],
                                                                   device=self.device)], dim=1)
        else:
            next_actions = actions[:, t: t + self.abstract_step_size]
        return next_actions

    def train_step(self,
                   s_ground_truth: torch.Tensor,
                   a_ground_truth: torch.Tensor,
                   r_ground_truth: torch.Tensor,
                   optimizer: torch.optim.Optimizer,
                   n_warmup: int = 1):
        optimizer.zero_grad(set_to_none=True)

        losses = self.eval_step(s_ground_truth, a_ground_truth, r_ground_truth, n_warmup)
        losses['total'].backward()
        optimizer.step()

        return losses

    def eval_step(self,
                  s_ground_truth: torch.Tensor,
                  a_ground_truth: torch.Tensor,
                  r_ground_truth: torch.Tensor,
                  n_warmup: int = 1) -> Dict:
        rec_loss = MultiscaleDynamicsModel.reconstruction_loss
        kl_loss = MultiscaleDynamicsModel.kl_loss
        kl_reg = MultiscaleDynamicsModel.kl_regularizer
        macro_r_loss = MultiscaleDynamicsModel.macro_reward_loss

        # target macro reward can be pre-computed from the single step rewards
        macro_r_target = MultiscaleDynamicsModel.average_kstep_reward(r_ground_truth, self.abstract_step_size,
                                                                      self.device)
        macro_r_target = macro_r_target[:, 1:]  # exclude first chunk since macro model is inactive there

        warmup_states = s_ground_truth[:, :n_warmup, :]
        predictions = self(warmup_states, a_ground_truth)

        s_mem, s_dist_mem, r_mem, r_dist_mem = predictions[:4]
        macro_s_prior_mem, macro_s_posterior_mem = predictions[4:6]
        macro_r_mem = predictions[6]
        macro_r_prior_mem, macro_r_posterior_mem = predictions[7:]

        rec_s = rec_loss(torch.stack(s_mem, dim=1), s_ground_truth)
        rec_r = rec_loss(torch.stack(r_mem, dim=1), r_ground_truth)
        kl = kl_loss(macro_s_prior_mem, macro_s_posterior_mem, macro_r_prior_mem, macro_r_posterior_mem)
        reg = 0.0001 * kl_reg(macro_s_prior_mem, macro_r_prior_mem)
        mr = macro_r_loss(torch.stack(macro_r_mem, dim=1), macro_r_target)
        loss = rec_s + rec_r + kl + reg + mr

        return {'total': loss, 'rec_s': rec_s, 'rec_r': rec_r, 'kl': kl, 'kl_reg': reg, 'macro_r': mr}

    def input_compatible(self,
                         s_ground_truth: torch.Tensor,
                         a_ground_truth: torch.Tensor,
                         r_ground_truth: torch.Tensor) -> Tuple[bool, str]:
        compatible = True
        tens_name = None
        shape = None

        # test lengths of shapes first
        if len(s_ground_truth.shape) != 3:
            compatible = False
            tens_name = 'state'
            shape = s_ground_truth.shape
        elif len(a_ground_truth.shape) != 3:
            compatible = False
            tens_name = 'action'
            shape = a_ground_truth.shape
        elif len(r_ground_truth.shape) != 3:
            compatible = False
            tens_name = 'reward'
            shape = r_ground_truth.shape

        if not compatible:
            msg = f'{tens_name} tensor should have 3 dimensions (d_batch, d_time, d_data) even if d_data is 1 but has '
            msg += f'shape {tuple(shape)}'
            return compatible, msg

        # test data shapes
        if s_ground_truth.shape[2] != self.d_state:
            compatible = False
            tens_name = 'state'
            shape = (s_ground_truth.shape[2], self.d_state)
        elif a_ground_truth.shape[2] != self.d_action:
            compatible = False
            tens_name = 'action'
            shape = (a_ground_truth.shape[2], self.d_action)
        elif r_ground_truth.shape[2] != self.d_reward:
            compatible = False
            tens_name = 'reward'
            shape = (r_ground_truth.shape[2], self.d_reward)

        if not compatible:
            msg = f'{tens_name} tensor\'s data dimension does not match the expected size (found {shape[0]}, '
            msg += f'expected {shape[1]})'
            return compatible, msg

        return compatible, tens_name

    def rollout_low_level(self,
                          start_states: torch.Tensor,
                          actions: torch.Tensor,
                          hl_s: torch.Tensor = None,
                          hl_a: torch.Tensor = None,
                          hl_r: torch.Tensor = None,
                          hl_s_next: torch.Tensor = None,
                          mem_ll: Tuple[torch.Tensor, torch.Tensor] = None):
        d_batch, n_steps = actions.shape[:2]
        n_start_states = start_states.shape[1]
        device = self.device

        # prepare necessary placeholders
        s = torch.zeros(d_batch, self.d_state)
        mem_ll = self.single_step_model.det_mdl.gen_h_placeholder(d_batch) if mem_ll is None else mem_ll
        hl_s = torch.zeros(d_batch, self.d_hl_state, device=device) if hl_s is None else hl_s
        hl_a = torch.zeros(d_batch, self.d_hl_action, device=device) if hl_a is None else hl_a
        hl_r = torch.zeros(d_batch, self.d_hl_reward, device=device) if hl_r is None else hl_r
        hl_s_next = torch.zeros_like(hl_s, device=device) if hl_s_next is None else hl_s_next

        s_mem, s_dist_mem = [], []
        r_mem, r_dist_mem = [], []
        for t in range(n_steps):
            if t < n_start_states:  # if still in warmup period, use teacher forcing for states
                s = start_states[:, t]
            a = actions[:, t]

            _, s_next_dist, r_next_dist, mem_ll = self.single_step_model(s, a, hl_s, hl_a, hl_r, hl_s_next, mem_ll)
            s_next = s_next_dist.rsample()
            r_next = r_next_dist.rsample()

            s_mem.append(s_next)
            r_mem.append(r_next)
            s_dist_mem.append(s_next_dist)
            r_dist_mem.append(r_next_dist)

            s = s_next

        r_mem = torch.stack(r_mem, dim=1)
        s_mem = torch.stack(s_mem, dim=1)

        return s_mem, s_dist_mem, r_mem, r_dist_mem, mem_ll

    def rollout_high_level(self,
                           hl_s_start: torch.Tensor,
                           hl_as: torch.Tensor,
                           hl_h: Tuple[torch.Tensor, torch.Tensor] = None,
                           return_samples: bool = False):
        d_batch, n_steps = hl_as.shape[:2]
        n_start_states = hl_s_start.shape[1]

        hl_s = None
        if hl_h is None:
            hl_h = self.abstract_model.det_mdl.gen_h_placeholder(d_batch)
        ll_h_flat = self._flatten_h(self.single_step_model.det_mdl.gen_h_placeholder(d_batch))

        hl_s_prior_mem, hl_r_prior_mem, hl_h_mem = [], [], []
        for t in range(n_steps):
            if t < n_start_states:
                hl_s = hl_s_start[:, t]
            hl_a = hl_as[:, t]

            out_abstr = self.abstract_model(hl_s, hl_a, ll_h_flat, hl_h)
            hl_s_next_prior, hl_s_next_posterior, hl_r_next_prior, hl_r_next_posterior, hl_h = out_abstr

            hl_s_prior_mem.append(hl_s_next_prior)
            hl_r_prior_mem.append(hl_r_next_prior)
            hl_h_mem.append(hl_h)

            hl_s = hl_s_next_prior.rsample()

        if return_samples:
            hl_s_prior_mem = [entry.rsample() for entry in hl_s_prior_mem]
            hl_r_prior_mem = [entry.rsample() for entry in hl_r_prior_mem]

        # hl_h_mem is a list of tuples, one tuple per time step. hl_h_mem shape is currently
        # (time, 2, n_rec_layers, batch, d_hidden) but we want batch dimension to be in front
        hl_h_mem = torch.stack([torch.stack(elem) for elem in hl_h_mem]).permute(3, 0, 1, 2, 4)

        return hl_s_prior_mem, hl_r_prior_mem, hl_h_mem

    def hl_next_posterior(self,
                          hl_s: torch.Tensor,
                          hl_a: torch.Tensor,
                          hl_h: Tuple[torch.Tensor, torch.Tensor],
                          ll_h: Tuple[torch.Tensor, torch.Tensor],
                          return_samples: bool = False):
        d_batch = hl_s.shape[0]
        if hl_h is None:
            hl_h = self.abstract_model.det_mdl.gen_h_placeholder(d_batch)
        ll_h_flat = self._flatten_h(ll_h)

        out_abstr = self.abstract_model(hl_s, hl_a, ll_h_flat, hl_h)
        hl_s_next_prior, hl_s_next_posterior, hl_r_next_prior, hl_r_next_posterior, hl_h = out_abstr

        if return_samples:
            hl_s_next_posterior = hl_s_next_posterior.sample()
            hl_r_next_posterior = hl_r_next_posterior.sample()

        return hl_s_next_posterior, hl_r_next_posterior, hl_h

    def _flatten_h(self, h: Tuple[torch.Tensor, torch.Tensor]):
        h = torch.concat(h, dim=0)  # concat h and c tensors of LSTM along the layer dimension, this is arbitrary
        h = torch.transpose(h, 0, 1)  # bring batch dimension to front
        h = torch.flatten(h, start_dim=1)  # fold h/c/layer dimension into d_hidden
        return h


def build_single_step_model(d_macro_state: int,
                            d_macro_action: int,
                            d_macro_reward: int,
                            d_state: int,
                            d_action: int,
                            d_reward: int,
                            d_hidden: int,
                            n_rec_layers: int,
                            batch_first: bool = True) -> SingleStepModel:
    # the deterministic model receives s, a, marco_s, macro_a, macro_s_next
    det_mdl = RecurrentBlock(d_state, d_action, d_macro_state, d_macro_action, d_macro_reward, d_macro_state,
                             d_hidden=d_hidden, n_layers=n_rec_layers, batch_first=batch_first)
    # the sampling model receives the output of the deterministic model and no additional input
    sampling_mdl_s = GaussianBlock(d_hidden, lws=(64, d_state))
    sampling_mdl_r = GaussianBlock(d_hidden, lws=(64, d_reward))
    single_step_mdl = SingleStepModel(det_mdl, sampling_mdl_s, sampling_mdl_r)

    return single_step_mdl


def build_abstract_model(d_macro_state: int,
                         d_macro_action: int,
                         d_macro_reward: int,
                         d_macro_hidden: int,
                         n_macro_rec_layers: int,
                         d_single_step_mem: int,
                         n_signle_step_mem_layers: int,
                         batch_first: bool = True) -> AbstractModel:
    # final hidden state info from single step model contains h and c of all LSTM layers
    d_memory_final = d_single_step_mem * 2 * n_signle_step_mem_layers
    det_mdl = RecurrentBlock(d_macro_state, d_macro_action, d_hidden=d_macro_hidden,
                             n_layers=n_macro_rec_layers, batch_first=batch_first)
    # the sampling model receives the output of the deterministic model and no additional input
    s_prior = GaussianBlock(d_macro_hidden, lws=(64, d_macro_state))
    s_posterior = GaussianBlock(d_macro_hidden, d_memory_final, lws=(64, d_macro_state))
    r_prior = GaussianBlock(d_macro_hidden, lws=(64, d_macro_reward))
    r_posterior = GaussianBlock(d_macro_hidden, d_memory_final, lws=(64, d_macro_reward))
    abstract_model = AbstractModel(det_mdl=det_mdl, sampling_mdl_s_prior=s_prior, sampling_mdl_s_posterior=s_posterior,
                                   sampling_mdl_r_prior=r_prior, sampling_mdl_r_posterior=r_posterior)

    return abstract_model
