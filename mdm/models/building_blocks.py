from typing import Tuple, Optional, Sequence, Union, TypeVar
from enum import Enum

import torch
import haste_pytorch as haste
from RIM import RIM

from mdm.utils.torch_tools import (layers_with_activation, add_time_dim, remove_time_dim, sample_from_gaussian,
                                   make_gaussian_params, get_mu, get_sigma, RnnStateType, sample_from_categorical)
from mdm.utils.utils import DistributionType


class AbstractActionModel(torch.nn.Module):

    def __init__(self,
                 d_a: int,
                 abstract_step_size: int,
                 d_a_abstract: int,
                 lws: tuple = (64, 64),
                 layer_norm: bool = False,
                 activation: str = 'relu',
                 distribution_type: DistributionType = DistributionType.NONE):
        super(AbstractActionModel, self).__init__()

        self.d_abstract_action = d_a_abstract
        self.distribution_type = distribution_type

        self.flatten_layer = torch.nn.Flatten(start_dim=1)
        if distribution_type is DistributionType.NONE:
            lws = (d_a * abstract_step_size, *lws, d_a_abstract)

            def prob_mdl(x: torch.Tensor):
                return torch.tanh(x)
        elif distribution_type is DistributionType.NORMAL:
            lws = (d_a * abstract_step_size, *lws, d_a_abstract * 2)

            def prob_mdl(x: torch.Tensor):
                x = make_gaussian_params(x, 1e-3)
                return sample_from_gaussian(x)
        elif distribution_type is DistributionType.CATEGORICAL:
            lws = (d_a * abstract_step_size, *lws, d_a_abstract)

            def prob_mdl(x: torch.Tensor):
                return sample_from_categorical(x)
        else:
            raise ValueError(f'Unsupported distribution type: {distribution_type}')

        self.det_mdl = torch.nn.Sequential(*layers_with_activation(lws, activation, layer_norm=layer_norm))
        self.prob_mdl = prob_mdl

    def forward(self,
                actions: torch.Tensor,
                sample: bool = True) -> torch.Tensor:
        # actions.shape = (d_batch, n_abstract_steps, d_action)
        x = self.flatten_layer(actions)
        x = self.det_mdl(x)
        x = self.prob_mdl(x)
        # x = torch.softmax(x, dim=-1)
        #x = torch.nn.functional.gumbel_softmax(x, hard=True, tau=0.1)
        #x = torch.softmax(x, dim=-1)
        #x = torch.nn.functional.one_hot(x.argmax(-1), x.shape[-1]) - x.detach() + x
        #x = torch.distributions.RelaxedOneHotCategorical(logits=x, temperature=0.1)
        #if sample:
        #    x = x.rsample()
        #else:
        #    x = torch.nn.functional.one_hot(x.probs.argmax(-1), self.d_abstract_action)
        #    x = x.to(torch.float32)
        return x


class RSSM(torch.nn.Module):

    def __init__(self,
                 d_z: int,
                 d_h: int,
                 d_a: int,
                 d_o: int,
                 d_r: int,
                 d_ctx_high_level: int,
                 d_x_posterior: int,
                 n_hidden_layers: int = 1,
                 hidden_dropout: float = 0.1,
                 epsilon: float = 0.01,
                 s_lws: Sequence[int] = (),
                 z_prior_lws: Sequence[int] = (32, 32),
                 z_post_lws: Sequence[int] = (32, 32),
                 o_lws: Sequence[int] = (32, 32),
                 r_lws: Sequence[int] = (32, 32),
                 term_lws: Sequence[int] = (32, 32),
                 layer_norm: bool = False,
                 activation: str = 'relu',
                 rnn_type: str = 'lstm',
                 stochastic_outputs: bool = True):
        super().__init__()

        self.d_z = d_z
        self.d_h = d_h
        self.d_action = d_a
        self.d_observation = d_o
        self.d_reward = d_r
        self.d_high_level_ctx = d_ctx_high_level
        self.d_low_level_ctx = d_x_posterior
        self.n_hidden_layers = n_hidden_layers
        self.epsilon = epsilon
        self.layer_norm = layer_norm
        self.activation = activation
        self.rnn_type = rnn_type
        self.stochastic_outputs = stochastic_outputs

        z_prior_lws = (d_h, *z_prior_lws, d_z * 2)
        z_post_lws = (d_h + d_z * 2 + d_x_posterior, *z_post_lws, d_z * 2)
        o_lws = (d_h + d_z, *o_lws, d_o * 2)
        r_lws = (d_h + d_z, *r_lws, d_r * 2)
        term_lws = (d_h + d_z, *term_lws, 1)
        z_preproc_lws = (d_z, d_z)

        if rnn_type == 'lstm':
            rnn_constr = torch.nn.LSTM
        elif rnn_type == 'gru':
            rnn_constr = torch.nn.GRU
        else:
            raise ValueError(f'Unsupported rnn type: {rnn_type}')

        d_det_core = d_z + d_a + d_ctx_high_level
        self._rnn = rnn_constr(d_det_core, hidden_size=d_h, num_layers=n_hidden_layers, batch_first=True,
                               dropout=hidden_dropout)
        #self._rnn = haste.IndRNN(d_det_core, hidden_size=d_hidden, batch_first=True)
        #self._det_core = haste.LayerNormLSTM(d_state + d_action + d_high_level_ctx, hidden_size=d_hidden,
        #                                    zoneout=0.05, dropout=hidden_dropout, batch_first=True)
        #self._rnn = RIM(device='cuda', input_size=d_det_core, hidden_size=d_hidden, num_units=n_units, k=3,
        #                n_layers=n_hidden_layers, rnn_cell='LSTM', bidirectional=False)

        self._z_preproc = torch.nn.Sequential(*layers_with_activation(z_preproc_lws, activation, layer_norm=layer_norm))
        self._s_prior = torch.nn.Sequential(*layers_with_activation(z_prior_lws, activation, layer_norm=layer_norm))
        self._s_post = torch.nn.Sequential(*layers_with_activation(z_post_lws, activation, layer_norm=layer_norm))
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
        z = self.zero_z(d_batch, device)
        #o = self.zero_o(d_batch, device)
        #a = self.zero_a(d_batch, device)
        #r = self.zero_r(d_batch, device)
        #term = self.zero_term(d_batch, device)
        rnn_state = self.zero_rnn_state(d_batch, device)
        #return {'z': z, 'o': o, 'a': a, 'r': r, 'term': term, 'rnn_state': rnn_state}
        return {'z': z, 'rnn_state': rnn_state}

    def zero_s(self,
               d_batch: int,
               device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, self.d_state, device=device)

    def zero_z(self,
               d_batch: int,
               device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, self.d_z, device=device)

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
            return (torch.zeros(self.n_hidden_layers, d_batch, self.d_h, device=device),
                    torch.zeros(self.n_hidden_layers, d_batch, self.d_h, device=device))
        else:
            return torch.zeros(self.n_hidden_layers, d_batch, self.d_h, device=device)

    def zero_x_post(self,
                    d_batch: int,
                    device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, self.d_low_level_ctx, device=device)

    def zero_ctx_high_level(self,
                            d_batch: int,
                            device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, self.d_high_level_ctx, device=device)

    def _det_core(self,
                  z: torch.Tensor,
                  rnn_state: torch.Tensor,
                  a: torch.Tensor,
                  ctx_high_level: torch.Tensor):
        z_processed = self._z_preproc(z)
        inp = torch.concat([z_processed, a, ctx_high_level], dim=-1)
        inp = add_time_dim(inp)
        x_det, next_rnn_state = self._rnn(inp, rnn_state)
        x_det = remove_time_dim(x_det)
        return x_det, next_rnn_state

    def forward(self,
                z: torch.Tensor,
                rnn_state: RnnStateType,
                a: torch.Tensor,
                x_current_groundtruth: torch.Tensor,
                ctx_high_level: torch.Tensor,
                use_posterior: bool = True,
                sample: bool = True):
        h, next_rnn_state = self._det_core(z, rnn_state, a, ctx_high_level)
        z_prior = self.build_z_prior(h)
        z_post = self.build_z_post(h, z_prior, x_current_groundtruth)

        if use_posterior:
            z_dist = z_post
        else:
            z_dist = z_prior
        if sample:
            z_smpl = sample_from_gaussian(z_dist)
        else:
            z_smpl = get_mu(z_dist)

        #h = torch.zeros_like(h)
        s = torch.concat([h, z_smpl], dim=-1)
        o_dist = self._build_o_dist(s)
        r_dist = self._build_r_dist(s)
        term_dist = self._build_terminal_dist(s)

        if sample and self.stochastic_outputs:
            o_smpl = sample_from_gaussian(o_dist)
            r_smpl = sample_from_gaussian(r_dist)
        else:
            o_smpl = get_mu(o_dist)
            r_smpl = get_mu(r_dist)
        term_smpl = term_dist

        return {'z': z_smpl, 'z_prior': z_prior, 'z_post': z_post, 'h': h, 's': s, 'o_dist': o_dist, 'o': o_smpl,
                'r_dist': r_dist, 'r': r_smpl, 'term_dist': None, 'term': term_smpl, 'rnn_state': next_rnn_state}

    def build_z_prior(self,
                      h: torch.Tensor) -> torch.Tensor:
        s_prior_params = self._s_prior(h)
        s_prior_params = make_gaussian_params(s_prior_params, self.epsilon)
        return s_prior_params

    def build_z_post(self,
                     h: torch.Tensor,
                     z_prior: torch.Tensor,
                     x_posterior: torch.Tensor) -> torch.Tensor:
        inp = torch.concat([h, z_prior, x_posterior], dim=-1)
        s_post_params = self._s_post(inp)
        s_post_params = make_gaussian_params(s_post_params, self.epsilon)
        return s_post_params

    def _build_o_dist(self,
                      s: torch.Tensor) -> torch.Tensor:
        o_params = self._o_dist(s)
        o_params = make_gaussian_params(o_params, 1e-3)
        return o_params

    def _build_r_dist(self,
                      s: torch.Tensor) -> torch.Tensor:
        r_params = self._r_dist(s)
        r_params = make_gaussian_params(r_params, 1e-3)
        return r_params

    def _build_terminal_dist(self,
                             s: torch.Tensor) -> torch.Tensor:
        term_params = self._term_dist(s)
        #term_params = torch.sigmoid(term_params)
        #term_params = torch.clamp(term_params, 0.01, 0.99)
        #term_dist = torch.distributions.ContinuousBernoulli(logits=term_params)
        #term_smpl = term_dist.rsample() if sample else term_params
        #return term_dist, term_smpl
        return torch.sigmoid(term_params)
