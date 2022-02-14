from typing import Tuple, List
from collections import deque

import torch
import torch.nn.functional as F

from models.torch_tools import RecurrentBlock, GaussianBlock, FeedforwardBlock, add_time_dim, remove_time_dim


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


def build_single_step_model(d_macro_state: int,
                            d_macro_action: int,
                            d_macro_reward: int,
                            d_state: int,
                            d_action: int,
                            d_reward: int,
                            d_hidden: int,
                            n_rec_layers: int,
                            batch_first: bool = True):
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
                         d_memory: int,
                         n_memory_layers: int):
    # final hidden state info from single step model contains h and c of all LSTM layers
    d_hidden_final = d_memory * 2 * n_memory_layers
    # assume markovian dynamics at this level of abstraction, so no recurrency
    det_mdl = FeedforwardBlock(d_macro_state, d_macro_action, d_hidden_final, lws=(64, 64))
    # the sampling model receives the output of the deterministic model and no additional input
    s_prior = GaussianBlock(64, lws=(64, d_macro_state))
    s_posterior = GaussianBlock(64, lws=(64, d_macro_state))
    r_prior = GaussianBlock(64, lws=(64, d_macro_reward))
    r_posterior = GaussianBlock(64, lws=(64, d_macro_reward))
    abstract_model = AbstractModel(det_mdl=det_mdl, sampling_mdl_s_prior=s_prior, sampling_mdl_s_posterior=s_posterior,
                                   sampling_mdl_r_prior=r_prior, sampling_mdl_r_posterior=r_posterior)

    return abstract_model


class MacroActionModel(torch.nn.Module):

    def __init__(self,
                 d_action: int,
                 n_abstract_steps: int,
                 d_macro_action: int):
        super(MacroActionModel, self).__init__()

        self.flatten_layer = torch.nn.Flatten(start_dim=1)
        self.det_mdl = FeedforwardBlock(d_action * n_abstract_steps, lws=(64, d_macro_action))

    def forward(self, actions: torch.Tensor):
        # actions.shape = (d_batch, n_abstract_steps, d_action)
        actions = self.flatten_layer(actions)
        actions = self.det_mdl(actions)
        actions = F.gumbel_softmax(actions, hard=True)
        return actions


class SingleStepModel(torch.nn.Module):

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


class AbstractModel(torch.nn.Module):

    def __init__(self,
                 det_mdl: FeedforwardBlock,
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
                macro_s: torch.Tensor,
                macro_a: torch.Tensor,
                h: torch.Tensor = None):
        if h is None:
            d_batch = macro_s.shape[0]
            d_memory = self.det_mdl.d_inputs[-1]
            h = torch.zeros(d_batch, d_memory)

            macro_sr_next_det = self.det_mdl(macro_s, macro_a, h)
            macro_s_next_dist = self.sampling_mdl_s_prior(macro_sr_next_det)
            macro_r_next_dist = self.sampling_mdl_r_prior(macro_sr_next_det)
        else:
            macro_sr_next_det = self.det_mdl(macro_s, macro_a, h)
            macro_s_next_dist = self.sampling_mdl_s_posterior(macro_sr_next_det)
            macro_r_next_dist = self.sampling_mdl_r_posterior(macro_sr_next_det)

        return macro_sr_next_det, macro_s_next_dist, macro_r_next_dist


class MultiscaleDynamicsModel(torch.nn.Module):

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
        self.d_macro_state = d_macro_state
        self.d_macro_action = d_macro_action
        self.d_macro_reward = d_macro_reward

        def reconstruction_loss(s_pred, r_pred, s_true, r_true):
            l = torch.mean((s_pred - s_true) ** 2) + torch.mean((r_pred - r_true) ** 2)
            return l

        def kl_loss(s_priors, s_posteriors, r_priors, r_posteriors):
            l = torch.tensor(0, dtype=torch.float32)
            for s_prior, s_posterior, r_prior, r_posterior in zip(s_priors, s_posteriors, r_priors, r_posteriors):
                l += torch.distributions.kl.kl_divergence(s_prior, s_posterior)
                l += torch.distributions.kl.kl_divergence(r_prior, r_posterior)
            return torch.mean(l)

        self.rec_loss = reconstruction_loss
        self.kl_loss = kl_loss

    def forward(self,
                start_states: torch.Tensor,
                actions: torch.Tensor):
        d_batch, n_steps = actions.shape[:2]
        n_start_states = start_states.shape[1]

        s_mem, s_dist_mem = [], []
        r_mem, r_dist_mem = [], []
        macro_s_prior_mem, macro_s_posterior_mem = [], []
        macro_r_prior_mem, macro_r_posterior_mem = [], []

        s, h, macro_s, macro_a, macro_r, macro_s_next, last_actions = self._gen_placeholders(d_batch)

        for t in range(n_steps):
            if t % self.abstract_step_size == 0:  # invoke abstract model every k time steps
                # preparations
                last_actions_tens = torch.stack(list(last_actions), dim=1)
                macro_a = self.macro_action_model(last_actions_tens)  # TODO: think about this model more closely
                h_flat = self._flatten_h(h)
                macro_s = macro_s_next  # update current macro state to previously predicted one

                # invoke models
                _, macro_s_next_prior, macro_r_next_prior = self.abstract_model(macro_s, macro_a)
                _, macro_s_next_posterior, macro_r_next_posterior = self.abstract_model(macro_s, macro_a, h_flat)

                # sample from more informed posterior distributions to get inputs for single step model
                macro_s_next = macro_s_next_posterior.sample()
                macro_r = macro_r_next_posterior.sample()

                # store priors and posteriors for loss calculation
                macro_s_prior_mem.append(macro_s_next_prior)
                macro_r_prior_mem.append(macro_r_next_prior)
                macro_s_posterior_mem.append(macro_s_next_posterior)
                macro_r_posterior_mem.append(macro_r_next_posterior)

                h = (torch.zeros_like(h[0]), torch.zeros_like(h[1]))  # prevent memory leakage beyond macro steps

            if t < n_start_states:  # if still in warmup period, use teacher forcing for states
                s = start_states[:, t]
            a = actions[:, t]

            _, s_next_dist, r_next_dist, h = self.single_step_model(s, a, macro_s, macro_a, macro_r, macro_s_next, h)
            s_next = s_next_dist.sample()
            r_next = r_next_dist.sample()

            # store single step states and rewards for loss calculation
            s_mem.append(s_next)
            r_mem.append(r_next)
            s_dist_mem.append(s_next_dist)
            r_dist_mem.append(r_next_dist)
            # update action history
            last_actions.append(a)
            last_actions.popleft()

            # set next state to upcoming time step's current state
            s = s_next

        return (s_mem,
                s_dist_mem,
                r_mem,
                r_dist_mem,
                macro_s_prior_mem,
                macro_s_posterior_mem,
                macro_r_prior_mem,
                macro_r_posterior_mem)

    def rollout_abstract(self,
                         macro_start_state: torch.Tensor,
                         macro_actions: torch.Tensor,
                         return_samples: bool = True):
        d_batch, n_steps = macro_actions.shape[:2]
        macro_s = macro_start_state

        macro_s_prior_mem, macro_r_prior_mem = [], []

        for t in range(n_steps):
            _, macro_s_next_prior, macro_r_next_prior = self.abstract_model(macro_s, macro_actions[:, t])
            macro_s_prior_mem.append(macro_s_next_prior)
            macro_r_prior_mem.append(macro_r_next_prior)
            macro_s = macro_s_next_prior.sample()

        if return_samples:
            macro_s_prior_mem = [entry.sample() for entry in macro_s_prior_mem]
            macro_r_prior_mem = [entry.sample() for entry in macro_r_prior_mem]

        return macro_s_prior_mem, macro_r_prior_mem

    def _gen_placeholders(self, d_batch: int):
        n_rec_layers_1sm = self.single_step_model.det_mdl.n_rec_layers
        d_hidden_1sm = self.single_step_model.det_mdl.d_hidden

        s = torch.zeros(d_batch, self.d_state)
        h = (torch.zeros(n_rec_layers_1sm, d_batch, d_hidden_1sm), torch.zeros(n_rec_layers_1sm, d_batch, d_hidden_1sm))
        macro_s = torch.zeros(d_batch, self.d_macro_state)
        macro_a = torch.zeros(d_batch, self.d_macro_action)
        macro_r = torch.zeros(d_batch, self.d_macro_reward)
        macro_s_next = torch.zeros_like(macro_s)
        last_actions = deque([torch.zeros(d_batch, self.d_action) for _ in range(self.abstract_step_size)])

        return s, h, macro_s, macro_a, macro_r, macro_s_next, last_actions

    def _flatten_h(self, h: Tuple[torch.Tensor, torch.Tensor]):
        h = torch.concat(h, dim=0)  # concat h and c tensors of LSTM along the layer dimension, this is arbitrary
        h = torch.transpose(h, 0, 1)  # bring batch dimension to front
        h = torch.flatten(h, start_dim=1)  # fold h/c/layer dimension into d_hidden
        return h

