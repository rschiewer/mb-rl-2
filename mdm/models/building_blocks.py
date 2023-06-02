from __future__ import annotations

from typing import Tuple, Optional, Sequence, Union, TypeVar, Dict, List
from abc import ABC, abstractmethod
from itertools import product
from enum import Enum

import torch
import numpy as np
from mdm.utils.torch_tools import (layers_with_activation as lwa, get_dist_params, RnnStateType,
                                   sample_from_categorical, ManagedStatefulTrainingModule)


class DeprecatedRSSM(torch.nn.Module):

    def __init__(self,
                 d_z: int,
                 d_h: int,
                 d_a: int,
                 obs_encoder: 'InputEncoder',
                 obs_decoder: 'OutputDecoder',
                 r_decoder: 'GaussianDecoder',
                 term_decoder: 'BinomialDecoder',
                 n_hidden_layers: int = 1,
                 hidden_dropout: float = 0.1,
                 epsilon: float = 0.01,
                 z_prior_lws: Sequence[int] = (32, 32),
                 z_post_lws: Sequence[int] = (32, 32),
                 layer_norm: bool = False,
                 activation: str = 'relu',
                 rnn_type: str = 'lstm',
                 latent_dist: str = 'normal',
                 stochastic_outputs: bool = True):
        super().__init__()

        self.d_z = d_z
        self.d_h = d_h
        self.d_action = d_a
        self.obs_encoder = obs_encoder
        self.obs_decoder = obs_decoder
        self.r_decoder = r_decoder
        self.term_decoder = term_decoder
        self.n_hidden_layers = n_hidden_layers
        self.epsilon = epsilon
        self.layer_norm = layer_norm
        self.activation = activation
        self.rnn_type = rnn_type
        self.latent_dist = latent_dist
        self.stochastic_outputs = stochastic_outputs

        self.n_latent_categories = 8
        if latent_dist == 'normal':
            d_z_post_in = d_h + d_z * 2
            d_z_final = d_z * 2
            d_z_smpl = d_z
        elif latent_dist == 'bernoulli':
            d_z_post_in = d_h + d_z
            d_z_final = d_z
            d_z_smpl = d_z
        elif latent_dist == 'categorical':
            d_z_post_in = d_h + d_z * self.n_latent_categories
            d_z_final = d_z * self.n_latent_categories
            d_z_smpl = d_z * self.n_latent_categories
        else:
            raise ValueError(f'Unknown latent distribution type: {latent_dist}')
        self.d_z_smpl = d_z_smpl
        self.d_x_posterior = self.d_o_encoded + 2  # observation + reward + terminal

        z_prior_lws = (d_h, *z_prior_lws, d_z_final)
        z_post_lws = (d_z_post_in + self.d_x_posterior, *z_post_lws, d_z_final)

        if rnn_type == 'lstm':
            rnn_constr = torch.nn.LSTM
        elif rnn_type == 'gru':
            rnn_constr = torch.nn.GRU
        else:
            raise ValueError(f'Unsupported rnn type: {rnn_type}')

        d_det_core = d_z_smpl + d_a
        self._rnn = rnn_constr(d_det_core, hidden_size=d_h, num_layers=n_hidden_layers, batch_first=False,
                               dropout=hidden_dropout)
        self._z_prior = torch.nn.Sequential(lwa(z_prior_lws, activation, layer_norm=layer_norm, name='z_prior'))
        self._z_post = torch.nn.Sequential(lwa(z_post_lws, activation, layer_norm=layer_norm, name='z_post'))

    @property
    def o_shape(self):
        return self.obs_encoder.s_x_orig

    @property
    def d_o_encoded(self):
        return self.obs_encoder.d_x_encoded

    def gen_init_values(self,
                        d_batch: int,
                        device: torch.device):
        z = self.zero_z(d_batch, device)
        rnn_state = self.zero_rnn_state(d_batch, device)
        return {'z': z, 'rnn_state': rnn_state}

    def zero_s(self,
               d_batch: int,
               device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, self.d_state, device=device)

    def zero_z(self,
               d_batch: int,
               device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, self.d_z_smpl, device=device)

    def zero_o(self,
               d_batch: int,
               device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, *self.o_shape, device=device)

    def zero_a(self,
               d_batch: int,
               device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, self.d_action, device=device)

    def zero_r(self,
               d_batch: int,
               device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, 1, device=device)

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

    def zero_ctx_high_level(self,
                            d_batch: int,
                            device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, 0, device=device)

    def _det_core(self,
                  z: torch.Tensor,
                  rnn_state: torch.Tensor,
                  a: torch.Tensor,
                  ctx_high_level: torch.Tensor):
        inp = torch.concat([z, a, ctx_high_level], dim=-1)
        inp = inp.unsqueeze(0)  # add time dim
        # rnn_state = self.zero_rnn_state(inp.shape[1], inp.device)
        x_det, next_rnn_state = self._rnn(inp, rnn_state)
        x_det = x_det.squeeze(0)  # remove time dim
        return x_det, next_rnn_state

    def imagine(self,
                z: torch.Tensor,
                rnn_state: RnnStateType,
                ctx_high_level: torch.Tensor,
                a: torch.Tensor,
                sample: bool = True):
        h, next_rnn_state = self._det_core(z, rnn_state, a, ctx_high_level)
        z_prior, z_smpl = self.build_z_prior(h, sample)

        return {'z': z_smpl, 'z_prior': z_prior, 'h': h, 'rnn_state': next_rnn_state}

    def observe(self,
                z: torch.Tensor,
                rnn_state: RnnStateType,
                a: torch.Tensor,
                o_current: torch.Tensor,
                r_current: torch.Tensor,
                term_current: torch.Tensor,
                ctx_high_level: torch.Tensor,
                sample: bool = True):
        imagination = self.imagine(z, rnn_state, ctx_high_level, a, sample)
        x_current_groundtruth = torch.concat([self.obs_encoder(o_current), r_current, term_current], dim=-1)
        z_post, z_smpl = self.build_z_post(imagination['h'], imagination['z_prior'], x_current_groundtruth, sample)

        imagination['z'] = z_smpl  # overwrite with posterior sample
        imagination['z_post'] = z_post
        return imagination

    def forward(self,
                z: torch.Tensor,
                rnn_state: RnnStateType,
                a: torch.Tensor,
                o_current: torch.Tensor,
                r_current: torch.Tensor,
                term_current: torch.Tensor,
                ctx_high_level: torch.Tensor,
                use_posterior: bool = True,
                reconstruct: bool = True,
                sample: bool = True):
        if use_posterior:
            world_state = self.observe(z, rnn_state, a, o_current, r_current, term_current, ctx_high_level, sample)
        else:
            world_state = self.imagine(z, rnn_state, ctx_high_level, a, sample)
            world_state['z_post'] = None

        s = torch.concat([world_state['h'], world_state['z']], dim=-1)
        r_dist, r_smpl = self.r_decoder(s, sample)
        term_dist, term_smpl = self.term_decoder(s, sample)

        if reconstruct:
            o_dist, o_smpl = self.obs_decoder(s, sample)
        else:
            o_dist, o_smpl = None, None

        reconstruction = {'s': s, 'o': o_smpl, 'o_dist': o_dist, 'r_dist': r_dist, 'r': r_smpl, 'term_dist': term_dist,
                          'term': term_smpl}

        return {**world_state, **reconstruction}

    def build_z_prior(self,
                      h: torch.Tensor,
                      sample: bool = True) -> [torch.distributions.Distribution, torch.Tensor]:
        z_prior_params = self._z_prior(h)
        if self.latent_dist == 'normal':
            mu, logvar = torch.tensor_split(z_prior_params, 2, dim=-1)
            sigma = torch.exp(0.5 * logvar) + self.epsilon
            z_prior = torch.distributions.Normal(loc=mu, scale=sigma)
            z_smpl = z_prior.rsample() if sample else mu
        elif self.latent_dist == 'bernoulli':
            z_prior = torch.distributions.ContinuousBernoulli(logits=z_prior_params)
            z_smpl = z_prior.rsample() if sample else z_prior.probs
        #elif self.latent_dist == 'categorical':
        else:  # categorical
            z_prior_params = z_prior_params.reshape((z_prior_params.shape[0], self.d_z, self.n_latent_categories))
            z_prior = torch.distributions.OneHotCategorical(logits=z_prior_params)
            probs = torch.nn.functional.softmax(z_prior.probs, dim=-1)
            if sample:
                z_smpl = z_prior.sample() + probs - probs.detach()
            else:
                z_smpl = probs
        return z_prior, z_smpl

    def build_z_post(self,
                     h: torch.Tensor,
                     z_prior: torch.distributions.Distribution,
                     x_posterior: torch.Tensor,
                     sample: bool = True) -> [torch.distributions.Distribution, torch.Tensor]:
        z_prior_params = torch.concat(get_dist_params(z_prior), dim=-1)
        z_post_inp = torch.concat([h, z_prior_params, x_posterior], dim=-1)
        z_post_params = self._z_post(z_post_inp)
        if self.latent_dist == 'normal':
            mu, logvar = torch.tensor_split(z_post_params, 2, dim=-1)
            sigma = torch.exp(0.5 * logvar) + self.epsilon
            z_post = torch.distributions.Normal(loc=mu, scale=sigma)
            z_smpl = z_post.rsample() if sample else mu
        elif self.latent_dist == 'bernoulli':
            z_post = torch.distributions.ContinuousBernoulli(logits=z_post_params)
            z_smpl = z_post.rsample() if sample else z_prior.probs
        # elif self.latent_dist == 'categorical':
        else:  # categorical
            z_post_params = z_post_params.reshape((z_post_params.shape[0], self.d_z, self.n_latent_categories))
            z_post = torch.distributions.OneHotCategorical(logits=z_post_params)
            probs = torch.nn.functional.softmax(z_post.probs, dim=-1)
            if sample:
                z_smpl = z_post.sample() + probs - probs.detach()
            else:
                z_smpl = probs
        return z_post, z_smpl


class InputEncoder(torch.nn.Module):

    def __init__(self,
                 s_x_orig: Union[int, Sequence[int]],
                 d_x_encoded: int):
        super(InputEncoder, self).__init__()
        if isinstance(s_x_orig, int):
            s_x_orig = (s_x_orig,)
        self.s_x_orig = tuple(s_x_orig)
        self.d_x_encoded = d_x_encoded

    def forward(self,
                o: torch.Tensor):
        o = torch.flatten(o, start_dim=-len(self.s_x_orig))
        return o


class OneHotEncoder(InputEncoder):

    def __init__(self,
                 s_x_orig: Union[int, Sequence[int]],
                 d_x_encoded: int,
                 lws: Sequence[int],
                 activation: str,
                 layer_norm: bool,
                 **kwargs):
        super(OneHotEncoder, self).__init__(s_x_orig, d_x_encoded)
        lws = (np.prod(s_x_orig), *lws, d_x_encoded)
        self._mdl = torch.nn.Sequential(*lwa(lws, activation, layer_norm=layer_norm))

    def forward(self,
                o: torch.Tensor):
        o = torch.flatten(o, start_dim=-len(self.s_x_orig))
        return self._mdl(o)


class MLPEncoder(InputEncoder):

    def __init__(self,
                 s_x_orig: Union[int, Sequence[int]],
                 d_x_encoded: int,
                 lws: Sequence[int],
                 activation: str,
                 layer_norm: bool,
                 **kwargs):
        super(MLPEncoder, self).__init__(s_x_orig, d_x_encoded)
        lws = (np.prod(s_x_orig), *lws, d_x_encoded)
        self._mdl = torch.nn.Sequential(*lwa(lws, activation, layer_norm=layer_norm))

    def forward(self,
                o: torch.Tensor):
        return self._mdl(o)


class OutputDecoder(torch.nn.Module, ABC):

    def __init__(self, s_x_orig: Union[int, Sequence[int]], d_x_encoded: int):
        super(OutputDecoder, self).__init__()

        if isinstance(s_x_orig, int):
            s_x_orig = (s_x_orig,)
        self.s_x_orig = tuple(s_x_orig)
        self.d_x_encoded = d_x_encoded

    @abstractmethod
    def forward(self,
                x_enc: torch.Tensor,
                sample: bool = True):
        pass


class GaussianDecoder(OutputDecoder):

    def __init__(self,
                 s_x_orig: Union[int, Sequence[int]],
                 d_x_encoded: int,
                 lws: Sequence[int],
                 activation: str,
                 layer_norm: bool,
                 epsilon: float):
        super().__init__(s_x_orig, d_x_encoded)

        lws = (d_x_encoded, *lws, np.prod(s_x_orig) * 2)
        self._mdl = torch.nn.Sequential(*lwa(lws, activation, layer_norm=layer_norm))
        self.epsilon = epsilon

    def forward(self, x_enc: torch.Tensor, sample: bool = True):
        params = self._mdl(x_enc)
        # if self.s_x_orig != (1,):
        #    params = params.reshape(*params.shape[:-1], *self.s_x_orig, 2)
        mu, logvar = torch.tensor_split(params, 2, dim=-1)
        sigma = torch.log(1 + torch.exp(logvar)) + self.epsilon
        d = torch.distributions.Normal(loc=mu, scale=sigma)
        #d = torch.distributions.Independent(d, 1)
        if sample:
            s = d.rsample()
        else:
            s = d.mean
        return d, s


class OneHotDecoder(OutputDecoder, ManagedStatefulTrainingModule):

    def __init__(self,
                 s_x_orig: Union[int, Sequence[int]],
                 d_x_encoded: int,
                 lws: Sequence[int],
                 activation: str,
                 layer_norm: bool,
                 temperature: float,
                 temperature_min: Optional[float] = None,
                 temperature_decrease_steps: Optional[int] = 0):
        super(OneHotDecoder, self).__init__(s_x_orig, d_x_encoded)

        n_categories = s_x_orig[-1]
        lws = (d_x_encoded, *lws, np.prod(s_x_orig))
        self._mdl = torch.nn.Sequential(*lwa(lws, activation, layer_norm=layer_norm))
        self.n_categories = n_categories
        self._temp = temperature
        if temperature_min is None:
            self._temp_min = self._temp
        else:
            self._temp_min = temperature_min
        self._temp_decr_steps = temperature_decrease_steps
        self._temp_decr_per_step = (self._temp - self._temp_min) / max(self._temp_decr_steps, 1)

    @property
    def temperature(self):
        if self.training:
            return max(self._temp - self._current_train_step * self._temp_decr_per_step, self._temp_min)
        else:
            return self._temp_min

    def forward(self,
                x_enc: torch.Tensor,
                sample: bool = True):
        params = self._mdl(x_enc)
        params = params.reshape(*params.shape[:-1], *self.s_x_orig)
        d = torch.distributions.RelaxedOneHotCategorical(torch.tensor(self.temperature), logits=params)
        if sample:
            s = d.rsample()
        else:
            s = (d.probs == d.probs.max(dim=-1, keepdim=True).values).to(torch.float32) + d.probs - d.probs.detach()

        return d, s

        # d = torch.distributions.OneHotCategorical(logits=params)
        # probs = torch.softmax(params, dim=-1)
        # if sample:
        #    s = d.sample() + probs - probs.detach()
        # else:
        #    s = torch.argmax(probs) + probs - probs.detach()
        # return d, s


class BinomialDecoder(OutputDecoder):

    def __init__(self,
                 s_x_orig: Union[int, Sequence[int]],
                 d_x_encoded: int,
                 lws: Sequence[int],
                 activation: str,
                 layer_norm: bool):
        super(BinomialDecoder, self).__init__(s_x_orig, d_x_encoded)

        lws = (d_x_encoded, *lws, np.prod(s_x_orig))
        self._mdl = torch.nn.Sequential(*lwa(lws, activation, layer_norm=layer_norm))

    def forward(self,
                x_enc: torch.Tensor,
                sample: bool = True):
        params = self._mdl(x_enc)
        params = params.reshape(*params.shape[:-1], *self.s_x_orig)
        d = torch.distributions.ContinuousBernoulli(logits=params, lims=(0.49999, 0.50001))
        #d = torch.distributions.Independent(d, 1)
        #d = torch.distributions.RelaxedBernoulli(temperature=self._temperature, logits=params)
        if sample:
            s = d.rsample()
        else:
            s = d.mean
            #s = d.probs.round().to(torch.float32) + d.probs - d.probs.detach()
        return d, s
        #d = torch.distributions.Bernoulli(logits=params)
        #if sample:
        #    s = d.sample() + d.probs - d.probs.detach()
        #else:
        #    s = torch.argmax(d.probs).round().to(torch.float32) + d.probs - d.probs.detach()
        #return d, s


class MLPDecoder(OutputDecoder):

    def __init__(self,
                 s_x_orig: Union[int, Sequence[int]],
                 d_x_encoded: int,
                 lws: Sequence[int],
                 activation: str,
                 layer_norm: bool,
                 final_activation: str = None,
                 **kwargs):
        super(MLPDecoder, self).__init__(s_x_orig, d_x_encoded)

        lws = (d_x_encoded, *lws, np.prod(s_x_orig))
        self._mdl = torch.nn.Sequential(*lwa(lws, activation, layer_norm=layer_norm,
                                             final_activation_function=final_activation))

    def forward(self,
                x_enc: torch.Tensor,
                sample: bool = True):
        x = self._mdl(x_enc)
        x = x.reshape(*x.shape[:-1], *self.s_x_orig)
        return None, x


class UpwardsFilter(torch.nn.Module):

    def __init__(self,
                 window_size: int):
        super(UpwardsFilter, self).__init__()
        self.window_size = window_size

    def _preproc(self,
                 x: torch.Tensor,
                 pad_value: float = 0):
        n_timesteps = x.shape[0]
        n_pad = n_timesteps % self.window_size
        if n_pad > 0:
            x_pad = torch.full((n_pad, *x.shape[1:]), pad_value, device=x.device)
            x = torch.concat([x, x_pad], dim=0)
        x = x.reshape(x.shape[0] // self.window_size, self.window_size, *x.shape[1:])
        return x, n_pad

    def forward(self,
                x: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass


class SumUpwardsFilter(UpwardsFilter):

    def forward(self,
                x: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        x, _ = self._preproc(x, 0.0)
        x = torch.sum(x, dim=1)
        return x


class AvgUpwardsFilter(UpwardsFilter):

    def forward(self,
                x: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        x, n_pad = self._preproc(x, 0.0)
        x_filtered = torch.mean(x, dim=1)
        if n_pad:
            n_valid = self.window_size - n_pad
            x_filtered[-1] = torch.mean(x[-1, :n_valid], dim=0)
        return x_filtered


class MaxUpwardsFilter(UpwardsFilter):

    def forward(self,
                x: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        x, n_pad = self._preproc(x, 0.0)
        x_filtered = torch.max(x, dim=1).values
        if n_pad:
            n_valid = self.window_size - n_pad
            x_filtered[-1] = torch.max(x[-1, :n_valid], dim=0).values
        return x_filtered


class MinUpwardsFilter(UpwardsFilter):

    def forward(self,
                x: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        x, n_pad = self._preproc(x, 0.0)
        x_filtered = torch.min(x, dim=1).values
        if n_pad:
            n_valid = self.window_size - n_pad
            x_filtered[-1] = torch.min(x[-1, :n_valid], dim=0).values
        return x_filtered


class PickOneUpwardsFilter(UpwardsFilter):

    def __init__(self,
                 window_size: int,
                 offset: int):
        super(PickOneUpwardsFilter, self).__init__(window_size)
        self.offset = offset

    def forward(self,
                x: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        x, n_pad = self._preproc(x, 0.0)
        x_filtered = x[:, self.offset]
        if n_pad:
            n_valid = self.window_size - n_pad
            x_filtered[-1] = x[-1, n_valid - 1]
        return x_filtered


class LearnableUpwardsFilter(UpwardsFilter):

    def __init__(self,
                 window_size: int,
                 d_x_orig: int,
                 d_x_filtered: int,
                 lws: tuple = (64, 64),
                 layer_norm: bool = False,
                 activation: str = 'relu',
                 model_type: str = None):
        super(LearnableUpwardsFilter, self).__init__(window_size)

        self.d_x_orig = d_x_orig
        self.d_x_filtered = d_x_filtered
        self.model_type = model_type

        if model_type == 'det_mapping':
            lws = (1, 1)
            self._pipeline = self._det_mapping
        elif model_type == 'det_tanh':
            lws = (d_x_orig * window_size, *lws, d_x_filtered)
            self._pipeline = self._point_estimate
        elif model_type == 'prob_normal':
            lws = (d_x_orig * window_size, *lws, d_x_filtered * 2)
            self._pipeline = self._prob_mdl_normal
        elif model_type == 'prob_categorical':
            lws = (d_x_orig * window_size, *lws, d_x_filtered)
            self._pipeline = self._prob_mdl_categorical
        else:
            raise ValueError(f'Unsupported model type: {model_type}')
        self._mdl = torch.nn.Sequential(*lwa(lws, activation, layer_norm=layer_norm))

    def _point_estimate(self,
                        x: torch.Tensor,
                        sample: bool) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = torch.flatten(x, start_dim=2)
        x = self._mdl(x)

        return torch.tanh(x)

    def _prob_mdl_normal(self,
                         x: torch.Tensor,
                         sample: bool) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = torch.flatten(x, start_dim=2)
        x = self._mdl(x)

        mu, logvar = torch.tensor_split(x, 2, dim=-1)
        mu = torch.tanh(mu)  # restrict mean of Gaussians to (-1, 1)
        sigma = torch.exp(0.5 * logvar) + 0.001
        d = torch.distributions.Normal(loc=mu, scale=sigma)

        if sample:
            s = d.rsample()
        else:
            s = d.loc
        return s

    def _prob_mdl_categorical(self,
                              x: torch.Tensor,
                              sample: bool) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = torch.flatten(x, start_dim=2)
        x = self._mdl(x)

        if sample:
            x = sample_from_categorical(x)
        else:
            x = torch.nn.functional.one_hot(torch.argmax(x, dim=-1), num_classes=x.shape[-1]).to(torch.float32)
        return x

    def _det_mapping(self,
                     x: torch.Tensor,
                     sample: bool = None) -> torch.Tensor:
        x2 = x.argmax(dim=-1)
        exp_mat = torch.full_like(x2, x.shape[-1])
        exp_mat = torch.cumprod(exp_mat, dim=1)
        exp_mat = torch.flip(exp_mat, dims=(1,))
        exp_mat = torch.roll(exp_mat, -1, dims=1)
        exp_mat[:, -1] = 1
        x2 = torch.sum(x2 * exp_mat, dim=1)
        x2 = torch.nn.functional.one_hot(x2, num_classes=self.d_x_filtered).to(torch.float32)
        return x2

    def forward(self,
                x: torch.Tensor,
                sample: bool = True) -> torch.Tensor:
        x, n_pad = self._preproc(x, 0.0)
        return self._pipeline(x, sample)


class IdentityUpwardsFilter(UpwardsFilter):

    def __init__(self):
        super(IdentityUpwardsFilter, self).__init__(1)

    def forward(self,
                x: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        return x


class ConstUpwardsFilter(UpwardsFilter):

    def __init__(self, window_size: int, constant: float):
        super().__init__(window_size)
        self.constant = constant

    def forward(self,
                x: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        x, n_pad = self._preproc(x, 0.0)
        x = x[:, 0]
        return torch.zeros_like(x)


class RSSMCell(torch.nn.Module):

    def __init__(self,
                 d_z: int,
                 d_h: int,
                 d_a: int,
                 o_encoder: 'InputEncoder',
                 o_decoder: 'OutputDecoder',
                 r_decoder: 'OutputDecoder',
                 term_decoder: 'OutputDecoder',
                 d_context: int = 0,
                 n_hidden_layers: int = 1,
                 hidden_dropout: float = 0.1,
                 epsilon: float = 0.01,
                 z_prior_lws: Sequence[int] = (32, 32),
                 z_post_lws: Sequence[int] = (32, 32),
                 layer_norm: bool = False,
                 activation: str = 'relu',
                 rnn_type: str = 'lstm',
                 latent_dist: str = 'normal'):
        super().__init__()

        assert o_decoder.d_x_encoded == d_z + d_h
        assert r_decoder.d_x_encoded == d_z + d_h
        assert term_decoder.d_x_encoded == d_z + d_h

        self.d_z = d_z
        self.d_h = d_h
        self.d_a = d_a
        self.d_context = d_context
        self.o_encoder = o_encoder
        self.o_decoder = o_decoder
        self.r_decoder = r_decoder
        self.term_decoder = term_decoder
        self.n_hidden_layers = n_hidden_layers
        self.epsilon = epsilon
        self.layer_norm = layer_norm
        self.activation = activation
        self.rnn_type = rnn_type
        self.latent_dist = latent_dist

        self.n_latent_categories = 16
        if latent_dist == 'normal':
            d_z_final = d_z * 2
            d_z_smpl = d_z
        elif latent_dist == 'bernoulli':
            d_z_final = d_z
            d_z_smpl = d_z
        elif latent_dist == 'categorical':
            d_z_final = d_z * self.n_latent_categories
            d_z_smpl = d_z * self.n_latent_categories
        else:
            raise ValueError(f'Unknown latent distribution type: {latent_dist}')
        self.d_z_smpl = d_z_smpl

        z_prior_lws = (d_h, *z_prior_lws, d_z_final)
        z_post_lws = (d_h + self.d_o_encoded, *z_post_lws, d_z_final)

        if rnn_type == 'lstm':
            rnn_constr = torch.nn.LSTM
        elif rnn_type == 'gru':
            rnn_constr = torch.nn.GRU
        else:
            raise ValueError(f'Unsupported rnn type: {rnn_type}')

        d_det_core = d_z_smpl + d_a
        self._rnn = rnn_constr(d_det_core, hidden_size=d_h, num_layers=n_hidden_layers, batch_first=False,
                               dropout=hidden_dropout)
        self._z_prior = torch.nn.Sequential(lwa(z_prior_lws, activation, layer_norm=layer_norm, name='z_prior'))
        self._z_post = torch.nn.Sequential(lwa(z_post_lws, activation, layer_norm=layer_norm, name='z_post'))

    @property
    def o_shape(self):
        return self.o_encoder.s_x_orig

    @property
    def d_o_encoded(self):
        return self.o_encoder.d_x_encoded

    def init_state(self,
                   d_batch: int,
                   device: torch.device):
        z = self.zero_z(d_batch, device)
        rnn_state = self.zero_rnn_state(d_batch, device)
        return {'z': z, 'z_prior': None, 'z_post': None, 'rnn_state': rnn_state}

    def zero_s(self,
               d_batch: int,
               device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, self.d_state, device=device)

    def zero_z(self,
               d_batch: int,
               device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, self.d_z_smpl, device=device)

    def zero_o(self,
               d_batch: int,
               device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, *self.o_shape, device=device)

    def zero_a(self,
               d_batch: int,
               device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, self.d_a, device=device)

    def zero_r(self,
               d_batch: int,
               device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, 1, device=device)

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

    def zero_context(self,
                     d_batch: int,
                     device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, self.d_context, device=device)

    def imagine(self,
                a: torch.Tensor,
                last_state: Dict[str, torch.Tensor],
                context: torch.Tensor | None = None,
                sample: bool = True):
        if context is None:
            context = self.zero_context(a.shape[0], a.device)
        if last_state is None:
            last_state = {'z': self.zero_z(a.shape[0], a.device),
                          'rnn_state': self.zero_rnn_state(a.shape[0], a.device)}

        inp = torch.concat([last_state['z'], a, context], dim=-1)
        inp = inp.unsqueeze(0)  # add time dim
        h, next_rnn_state = self._rnn(inp, last_state['rnn_state'])
        h = h.squeeze(0)  # remove time dim
        z_prior, z_smpl = self.build_z_prior(h, sample)

        return h, {'z': z_smpl, 'z_dist' : z_prior, 'z_prior': z_prior, 'z_post': None, 'rnn_state': next_rnn_state}

    def observe(self,
                a: torch.Tensor,
                o: torch.Tensor,
                last_state: Dict[str, torch.Tensor],
                context: torch.Tensor | None = None,
                sample: bool = True):
        h, next_state = self.imagine(a, last_state, context, sample)
        z_post, z_smpl = self.build_z_post(h, self.o_encoder(o), sample)

        # overwrite chosen z sample and distribution with posterior
        next_state['z'] = z_smpl
        next_state['z_dist'] = z_post
        next_state['z_post'] = z_post
        return h, next_state

    def forward(self,
                a: torch.Tensor,
                o: torch.Tensor | None = None,
                last_state: Dict[str, torch.Tensor] | None = None,
                context: torch.Tensor | None = None,
                use_posterior: bool = True,
                reconstruct: bool = True,
                sample_state: bool = True,
                sample_output: bool = True):
        if o is None and last_state is None:
            raise ValueError('Need at least (o_current, r_current, term_current) or last_state')
        if o is None and use_posterior:
            raise ValueError('Can\'t use posterior if no ground truth data is provided')

        # compute next world state
        if use_posterior:
            h, next_state = self.observe(a, o, last_state, context, sample_state)
        else:
            h, next_state = self.imagine(a, last_state, context, sample_state)

        # predict outputs
        s = torch.concat([h, next_state['z']], dim=-1)
        r_dist, r_smpl = self.r_decoder(s, sample_output)
        term_dist, term_smpl = self.term_decoder(s, sample_output)
        if reconstruct:
            o_dist, o_smpl = self.o_decoder(s, sample_output)
        else:
            o_dist, o_smpl = None, None

        reconstruction = {'o': o_smpl, 'o_dist': o_dist, 'a': a, 'r_dist': r_dist, 'r': r_smpl,
                          'terminal_dist': term_dist, 'terminal': term_smpl, 's': s, 'h': h}

        return reconstruction, next_state

    def build_z_prior(self,
                      h: torch.Tensor,
                      sample: bool = True) -> [torch.distributions.Distribution, torch.Tensor]:
        z_prior_params = self._z_prior(h)
        if self.latent_dist == 'normal':
            mu, logvar = torch.tensor_split(z_prior_params, 2, dim=-1)
            sigma = torch.log(1 + torch.exp(logvar)) + self.epsilon
            z_prior = torch.distributions.Normal(loc=mu, scale=sigma)
            #z_prior = torch.distributions.Independent(z_prior, 1)
            z_smpl = z_prior.rsample() if sample else mu
        elif self.latent_dist == 'bernoulli':
            z_prior = torch.distributions.ContinuousBernoulli(logits=z_prior_params)
            z_smpl = z_prior.rsample() if sample else z_prior.probs
        else:  # categorical
            z_prior_params = z_prior_params.reshape((z_prior_params.shape[0], self.d_z, self.n_latent_categories))
            z_prior = torch.distributions.OneHotCategorical(logits=z_prior_params)
            probs = torch.nn.functional.softmax(z_prior.probs, dim=-1)
            if sample:
                z_smpl = z_prior.sample() + probs - probs.detach()
            else:
                z_smpl = probs
        return z_prior, z_smpl

    def build_z_post(self,
                     h: torch.Tensor,
                     o_enc: torch.Tensor,
                     sample: bool = True) -> [torch.distributions.Distribution, torch.Tensor]:
        z_post_inp = torch.concat([h, o_enc], dim=-1)
        z_post_params = self._z_post(z_post_inp)
        if self.latent_dist == 'normal':
            mu, logvar = torch.tensor_split(z_post_params, 2, dim=-1)
            sigma = torch.log(1 + torch.exp(logvar)) + self.epsilon
            z_post = torch.distributions.Normal(loc=mu, scale=sigma)
            #z_post = torch.distributions.Independent(z_post, 1)
            z_smpl = z_post.rsample() if sample else mu
        elif self.latent_dist == 'bernoulli':
            z_post = torch.distributions.ContinuousBernoulli(logits=z_post_params)
            z_smpl = z_post.rsample() if sample else z_post.mean
        else:  # categorical
            z_post_params = z_post_params.reshape((z_post_params.shape[0], self.d_z, self.n_latent_categories))
            z_post = torch.distributions.OneHotCategorical(logits=z_post_params)
            probs = torch.nn.functional.softmax(z_post.probs, dim=-1)
            if sample:
                z_smpl = z_post.sample() + probs - probs.detach()
            else:
                z_smpl = probs
        return z_post, z_smpl
