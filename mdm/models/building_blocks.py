from typing import Tuple, Optional, Sequence, Union, TypeVar
from enum import Enum

import torch
import haste_pytorch as haste

from mdm.utils.torch_tools import (layers_with_activation, add_time_dim, remove_time_dim, sample_from_gaussian,
                                   make_gaussian_params, get_mu, get_sigma, RnnStateType)


class AbstractActionModel(torch.nn.Module):

    def __init__(self,
                 d_action: int,
                 abstract_step_size: int,
                 d_abstract_action: int,
                 lws: tuple = (64, 64),
                 layer_norm: bool = False,
                 activation: str = 'relu'):
        super(AbstractActionModel, self).__init__()

        lws = (d_action * abstract_step_size, *lws, d_abstract_action)
        self.flatten_layer = torch.nn.Flatten(start_dim=1)
        self.det_mdl = torch.nn.Sequential(*layers_with_activation(lws, activation, layer_norm=layer_norm))

    def forward(self,
                actions: torch.Tensor) -> torch.Tensor:
        # actions.shape = (d_batch, n_abstract_steps, d_action)
        macro_action = self.flatten_layer(actions)
        macro_action = self.det_mdl(macro_action)
        #macro_action = torch.tanh(macro_action)
        # macro_action = torch.softmax(macro_action, dim=-1)
        macro_action = torch.nn.functional.gumbel_softmax(macro_action, hard=True)
        return macro_action


class RSSM(torch.nn.Module):

    def __init__(self,
                 d_state: int,
                 d_action: int,
                 d_observation: int,
                 d_reward: int,
                 d_ctx_high_level: int,
                 d_x_posterior: int,
                 d_hidden: int,
                 n_hidden_layers: int = 1,
                 hidden_dropout: float = 0.1,
                 epsilon: float = 0.01,
                 s_prior_lws: Sequence[int] = (32, 32),
                 s_post_lws: Sequence[int] = (32, 32),
                 o_lws: Sequence[int] = (32, 32),
                 r_lws: Sequence[int] = (32, 32),
                 term_lws: Sequence[int] = (32, 32),
                 layer_norm: bool = False,
                 activation: str = 'relu',
                 rnn_type: str = 'lstm'):
        super().__init__()

        self.d_state = d_state
        self.d_action = d_action
        self.d_observation = d_observation
        self.d_reward = d_reward
        self.d_high_level_ctx = d_ctx_high_level
        self.d_low_level_ctx = d_x_posterior
        self.d_hidden = d_hidden
        self.n_hidden_layers = n_hidden_layers
        self.epsilon = epsilon
        self.layer_norm = layer_norm
        self.activation = activation
        self.rnn_type = rnn_type

        s_prior_lws = (d_hidden, *s_prior_lws, d_state * 2)
        s_post_lws = (d_hidden + d_x_posterior + d_state * 2, *s_post_lws, d_state * 2)
        o_lws = (d_hidden + d_state, *o_lws, d_observation * 2)
        r_lws = (d_hidden + d_state, *r_lws, d_reward * 2)
        term_lws = (d_hidden + d_state, *term_lws, 1)

        if rnn_type == 'lstm':
            rnn_constr = torch.nn.LSTM
        elif rnn_type == 'gru':
            rnn_constr = torch.nn.GRU
        else:
            raise ValueError(f'Unsupported rnn type: {rnn_type}')

        d_det_core = d_state + d_action + d_ctx_high_level
        self._rnn = rnn_constr(d_det_core, hidden_size=d_hidden, num_layers=n_hidden_layers, batch_first=True,
                               dropout=hidden_dropout)
        #self._det_core = haste.LayerNormLSTM(d_state + d_action + d_high_level_ctx, hidden_size=d_hidden,
        #                                    zoneout=0.05, dropout=hidden_dropout, batch_first=True)

        self._s_prior = torch.nn.Sequential(*layers_with_activation(s_prior_lws, activation, layer_norm=layer_norm))
        self._s_post = torch.nn.Sequential(*layers_with_activation(s_post_lws, activation, layer_norm=layer_norm))
        self._o_dist = torch.nn.Sequential(*layers_with_activation(o_lws, activation, layer_norm=layer_norm))
        self._r_dist = torch.nn.Sequential(*layers_with_activation(r_lws, activation, layer_norm=layer_norm))
        self._term_dist = torch.nn.Sequential(*layers_with_activation(term_lws, activation, layer_norm=layer_norm))

    @property
    def top_node(self):
        return self.d_high_level_ctx == 0

    @property
    def inner_node(self):
        return self.d_high_level_ctx > 0 and self.d_observation == 0

    @property
    def bottom_node(self):
        return self. self.d_observation > 0

    def gen_init_values(self,
                        d_batch: int,
                        device: torch.device):
        s = self.zero_s(d_batch, device)
        o = self.zero_o(d_batch, device)
        a = self.zero_a(d_batch, device)
        r = self.zero_r(d_batch, device)
        term = self.zero_term(d_batch, device)
        rnn_state = self.zero_rnn_state(d_batch, device)
        return {'s': s, 'o': o, 'a': a, 'r': r, 'term': term, 'rnn_state': rnn_state}

    def zero_s(self,
               d_batch: int,
               device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, self.d_state, device=device)

    def zero_o(self,
               d_batch: int,
               device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, self.d_observation, device=device)

    def zero_a(self,
               d_batch: int,
               device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, self.d_action, device=device)

    def zero_r(self,
               d_batch: int,
               device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, self.d_reward, device=device)

    def zero_term(self,
                  d_batch: int,
                  device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, 1, device=device)

    def zero_rnn_state(self,
                       d_batch: int,
                       device: torch.device) -> RnnStateType:
        if self.rnn_type == 'lstm':
            return (torch.zeros(self.n_hidden_layers, d_batch, self.d_hidden, device=device),
                    torch.zeros(self.n_hidden_layers, d_batch, self.d_hidden, device=device))
        else:
            return torch.zeros(d_batch, self.n_hidden_layers, self.d_hidden, device=device)

    def zero_x_post(self,
                    d_batch: int,
                    device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, self.d_low_level_ctx, device=device)

    def zero_ctx_high_level(self,
                            d_batch: int,
                            device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, self.d_high_level_ctx, device=device)

    def _det_core(self,
                  s: torch.Tensor,
                  a: torch.Tensor,
                  ctx_high_level: torch.Tensor,
                  rnn_state: torch.Tensor):
        #if self.feed_back_last_prediction:
        #    inp = torch.concat([s, a, x_hat, ctx_high_level], dim=-1)
        #else:
        #    inp = torch.concat([s, a, ctx_high_level], dim=-1)
        inp = torch.concat([s, a, ctx_high_level], dim=-1)
        inp = add_time_dim(inp)
        x_det, next_rnn_state = self._rnn(inp, rnn_state)
        x_det = remove_time_dim(x_det)
        return x_det, next_rnn_state

    def forward(self,
                s: torch.Tensor,
                a: torch.Tensor,
                x_current_groundtruth: torch.Tensor,  # ground truth o, r, term for this prediction step for the posterior
                ctx_high_level: torch.Tensor,  # abstr_s, abstr_h in primitive model
                rnn_state: RnnStateType,
                use_posterior: bool = True,
                sample: bool = True):
        h, next_rnn_state = self._det_core(s, a, ctx_high_level, rnn_state)
        s_prior = self.build_s_prior(h)
        s_post = self.build_s_post(h, x_current_groundtruth, s_prior)

        if use_posterior:
            s_dist = s_post
        else:
            s_dist = s_prior
        if sample:
            s_smpl = sample_from_gaussian(s_dist)
        else:
            s_smpl = get_mu(s_dist)

        o_dist = self._build_o_dist(h, s_smpl)
        r_dist = self._build_r_dist(h, s_smpl)
        term_dist = self._build_terminal_dist(h, s_smpl)

        if sample:
            o_smpl = sample_from_gaussian(o_dist)
            r_smpl = sample_from_gaussian(r_dist)
        else:
            o_smpl = get_mu(o_dist)
            r_smpl = get_mu(r_dist)

        term_smpl = term_dist
        return {'s': s_smpl, 's_prior': s_prior, 's_post': s_post, 'o_dist': o_dist, 'o': o_smpl, 'r_dist': r_dist,
                'r': r_smpl, 'term': term_smpl, 'rnn_state': next_rnn_state}

    def build_s_prior(self,
                      h: torch.Tensor) -> torch.Tensor:
        s_prior_params = self._s_prior(h)
        s_prior_params = make_gaussian_params(s_prior_params, self.epsilon)
        return s_prior_params

    def build_s_post(self,
                     h: torch.Tensor,
                     x_posterior: torch.Tensor,
                     s_prior: torch.Tensor) -> torch.Tensor:
        inp = torch.concat([h, x_posterior, get_mu(s_prior), get_sigma(s_prior)], dim=-1)
        s_post_params = self._s_post(inp)
        s_post_params = make_gaussian_params(s_post_params, self.epsilon)
        return s_post_params

    def _build_o_dist(self,
                      h: torch.Tensor,
                      s_smpl: torch.Tensor) -> torch.Tensor:
        inp = torch.concat([h, s_smpl], dim=-1)
        o_params = self._o_dist(inp)
        o_params = make_gaussian_params(o_params, self.epsilon)
        return o_params

    def _build_r_dist(self,
                      h: torch.Tensor,
                      s_smpl: torch.Tensor,
                      sample: bool = True) -> torch.Tensor:
        x_in = torch.concat([h, s_smpl], dim=-1)
        r_params = self._r_dist(x_in)
        r_params = make_gaussian_params(r_params, self.epsilon)
        return r_params

    def _build_terminal_dist(self,
                             h: torch.Tensor,
                             s_smpl: torch.Tensor) -> torch.Tensor:
        x_in = torch.concat([h, s_smpl], dim=-1)
        term_params = self._term_dist(x_in)
        #term_params = torch.sigmoid(term_params)
        #term_params = torch.clamp(term_params, 0.01, 0.99)
        #term_dist = torch.distributions.ContinuousBernoulli(logits=term_params)
        #term_smpl = term_dist.rsample() if sample else term_params
        #return term_dist, term_smpl
        return torch.sigmoid(term_params)
