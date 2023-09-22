from __future__ import annotations

from typing import Tuple, Optional, Sequence, Union, TypeVar, Dict, List, Any
from abc import ABC, abstractmethod
from collections import namedtuple
from itertools import product
from enum import Enum

import torch
import torch.masked as torchm
from torch.nn import ModuleList
import numpy as np
from mdm.utils.torch_tools import (layers_with_activation as lwa, get_dist_params,
                                   sample_from_categorical, ManagedStatefulTrainingModule, detach_dist, concat_dists,
                                   disable_torch_compile, stack_tensor_dicts, concat_tensor_dicts,
                                   CustomGRUCell, unsqueeze_right)
from torch.profiler import record_function
from mdm.models.fastrnns import LayerNormLSTMCell

RnnStateType = TypeVar('RnnStateType', torch.Tensor, Tuple[torch.Tensor, torch.Tensor])


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
        # elif self.latent_dist == 'categorical':
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
        lws = (np.prod(s_x_orig).item(), *lws, d_x_encoded)
        self._mdl = torch.nn.Sequential(lwa(lws, activation, layer_norm=layer_norm, name='one_hot_encoder'))

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
        lws = (np.prod(s_x_orig).item(), *lws, d_x_encoded)
        self._mdl = torch.nn.Sequential(lwa(lws, activation, layer_norm=layer_norm, name='mlp_encoder'))

    def forward(self,
                o: torch.Tensor):
        o = torch.flatten(o, start_dim=-len(self.s_x_orig))
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

        lws = (d_x_encoded, *lws, np.prod(s_x_orig).item() * 2)
        self._mdl = torch.nn.Sequential(lwa(lws, activation, layer_norm=layer_norm, name='gaussian_decoder'))
        self.epsilon = epsilon

    def forward(self, x_enc: torch.Tensor, sample: bool = True):
        params = self._mdl(x_enc)
        mu, logvar = torch.tensor_split(params, 2, dim=-1)
        sigma = torch.nn.functional.softplus(logvar) + self.epsilon
        d = torch.stack([mu, sigma], dim=-1)
        if sample:
            s = self.sample(d)
        else:
            s = mu
        return d, s

    @torch.jit.ignore
    def dist(self,
             parameters: torch.Tensor) -> torch.distributions.Distribution:
        mu, sigma = parameters.unbind(-1)
        d = torch.distributions.Normal(loc=mu, scale=sigma)
        d = torch.distributions.Independent(d, 1)
        return d

    @torch.jit.ignore
    def sample(self,
               parameters):
        mu, sigma = parameters.unbind(-1)
        d = torch.distributions.Normal(loc=mu, scale=sigma)
        d = torch.distributions.Independent(d, 1)
        s = d.rsample()
        return s


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
        lws = (d_x_encoded, *lws, np.prod(s_x_orig).item())
        self._mdl = torch.nn.Sequential(lwa(lws, activation, layer_norm=layer_norm, name='one_hot_decoder'))
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

        lws = (d_x_encoded, *lws, np.prod(s_x_orig).item())
        self._mdl = torch.nn.Sequential(lwa(lws, activation, layer_norm=layer_norm, name='binomial_decoder'))

    def forward(self,
                x_enc: torch.Tensor,
                sample: bool = True):
        params = self._mdl(x_enc)
        s_new = params.shape[:-1] + self.s_x_orig
        params = params.reshape(s_new)
        params = torch.nn.functional.sigmoid(params)
        d = params
        # d = torch.distributions.ContinuousBernoulli(params=params)
        # d = torch.distributions.Bernoulli(params=params)
        # d = torch.distributions.Independent(d, 1)
        # temp = torch.tensor(0.1, dtype=x_enc.dtype, device=x_enc.device)
        # d = torch.distributions.RelaxedBernoulli(temperature=temp, params=params)
        if sample:
            s = self.sample(params)
            # s = d.sample()
            # s = d.sample() + params - params.detach()
        else:
            # s = d.mean
            s = self.mean(params)
            # s = d.params.round().to(torch.float32) + d.params - d.params.detach()
        return d, s
        # d = torch.distributions.Bernoulli(logits=params)
        # if sample:
        #    s = d.sample() + d.params - d.params.detach()
        # else:
        #    s = torch.argmax(d.params).round().to(torch.float32) + d.params - d.params.detach()
        # return d, s

    @torch.jit.ignore
    def dist(self,
             parameters: torch.Tensor) -> torch.distributions.Distribution:
        d = torch.distributions.ContinuousBernoulli(probs=parameters)
        # d = torch.distributions.Bernoulli(logits=parameters)
        d = torch.distributions.Independent(d, 1)
        return d

    @torch.jit.ignore
    def sample(self,
               parameters):
        d = torch.distributions.ContinuousBernoulli(probs=parameters)
        d = torch.distributions.Independent(d, 1)
        s = d.rsample()
        # s = d.sample() + parameters - parameters.detach()
        return s

    @torch.jit.ignore
    def mean(self,
             parameters):
        d = torch.distributions.ContinuousBernoulli(probs=parameters)
        d = torch.distributions.Independent(d, 1)
        s = d.mean
        # s = d.sample() + parameters - parameters.detach()
        return s


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

        lws = (d_x_encoded, *lws, np.prod(s_x_orig).item())
        self._mdl = torch.nn.Sequential(lwa(lws, activation, layer_norm=layer_norm,
                                            final_activation_function=final_activation, name='mlp_decoder'))

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

    @torch.jit.export
    def _preproc(self,
                 x: torch.Tensor,
                 mask: Optional[torch.Tensor] = None,
                 pad_value: float = 0,
                 window_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, int]:
        d_time, d_batch = x.shape[:2]
        if window_size is None:
            window_size = self.window_size
        overhang = d_time % window_size
        if overhang == 0:
            n_pad = 0
        else:
            n_pad = window_size - overhang

        if mask is None:
            mask = torch.zeros(d_time, d_batch, 1, dtype=torch.bool, device=x.device)
        else:
            assert mask.dtype == torch.bool, 'Need binary mask with bool dtype'

        if n_pad > 0:
            pad_shp = list(x.shape)
            pad_shp[0] = n_pad
            x_pad = torch.full(pad_shp, pad_value, device=x.device)
            mask_pad = torch.ones(n_pad, x.shape[1], 1, dtype=torch.bool, device=x.device)
            x = torch.concat([x, x_pad], dim=0)
            mask = torch.concat([mask, mask_pad], dim=0)

        x_shp_new = (x.shape[0] // window_size, window_size) + x.shape[1:]
        mask_shp_new = (mask.shape[0] // window_size, window_size) + mask.shape[1:]
        x = x.reshape(x_shp_new)
        mask = mask.reshape(mask_shp_new)
        return x, mask, n_pad

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor | None = None,
                context: torch.Tensor | None = None,
                window_size: int | None = None) -> torch.Tensor:
        x, mask, n_pad = self._preproc(x=x, mask=mask, window_size=window_size)
        return x


class SumUpwardsFilter(UpwardsFilter):

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor | None = None,
                context: torch.Tensor | None = None,
                window_size: int | None = None) -> torch.Tensor:
        x, mask, _ = self._preproc(x, mask, 0.0, window_size)
        x = torch.sum(x * (1 - mask), dim=1)
        return x


class AvgUpwardsFilter(UpwardsFilter):

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                context: Optional[torch.Tensor] = None,
                window_size: Optional[int] = None) -> torch.Tensor:
        x, mask, n_pad = self._preproc(x, mask, 0.0, window_size)
        nom = torch.sum(x * ~mask, dim=1)
        denom = torch.sum(~mask, dim=1).to(torch.float32)
        denom = torch.where(denom == 0, 1.0, denom)
        x = nom / denom
        # if n_pad:
        #    n_valid = self.window_size - n_pad
        #    x[-1] = torch.mean(x[-1, :n_valid], dim=0)
        return x


class MaxUpwardsFilter(UpwardsFilter):

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                context: Optional[torch.Tensor] = None,
                window_size: Optional[int] = None) -> torch.Tensor:
        x_rs, mask_rs, n_pad = self._preproc(x, mask, 0.0, window_size)
        x_rs = torch.where(mask_rs, -torch.inf, x_rs)
        x_rs = torch.max(x_rs, dim=1).values
        x_rs = torch.where(x_rs == -torch.inf, 0.0, x_rs)
        # if n_pad:
        #    n_valid = self.window_size - n_pad
        #    x[-1] = torch.max(x[-1, :n_valid], dim=0).values
        return x_rs


class MinUpwardsFilter(UpwardsFilter):

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                context: Optional[torch.Tensor] = None,
                window_size: Optional[int] = None) -> torch.Tensor:
        x, mask, n_pad = self._preproc(x, mask, 0.0, window_size)
        x = torch.where(mask, torch.inf, x)
        x = torch.min(x, dim=1).values
        x = torch.where(x == torch.inf, 0.0, x)
        # if n_pad:
        #    n_valid = self.window_size - n_pad
        #    x[-1] = torch.min(x[-1, :n_valid], dim=0).values
        return x


class PickOneUpwardsFilter(UpwardsFilter):

    def __init__(self,
                 window_size: int,
                 offset: int):
        super(PickOneUpwardsFilter, self).__init__(window_size)
        self.offset = offset

    @staticmethod
    def _check_slow(x: torch.Tensor, mask: torch.Tensor):
        """
        Only for debugging purposes and to make sure more complex implementations behave as intended
        :param x: tensor to be filtered up
        :param mask: mask that is used to find the last valid time step inside a chunk
        :return: the filtered x with valid entries for every chunk wherever possible or the first invalid one otherwise
        """
        n_chunks, d_chunk, d_batch = x.shape[:3]

        x_filtered = torch.full([n_chunks, d_batch, *x.shape[3:]], fill_value=0, device=x.device, dtype=x.dtype)
        for i_batch in range(d_batch):
            for i_chunk in range(n_chunks):
                for i_timestep in reversed(range(d_chunk)):
                    if mask[i_chunk, i_timestep, i_batch] == 0:
                        x_filtered[i_chunk, i_batch] = x[i_chunk, i_timestep, i_batch]
                        break
                    # if we went through the whole chunk and all time steps were invalid, the chunk is invalid and
                    # masked out later anyway

        return x_filtered

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                context: Optional[torch.Tensor] = None,
                window_size: Optional[int] = None) -> torch.Tensor:
        x, mask, n_pad = self._preproc(x, mask, 0.0, window_size)
        n_chunks, d_chunk, d_batch = x.shape[:3]
        # x_filtered_2 = self._check_slow(x, mask)

        if self.offset < 0:
            tmp_offset = d_chunk + self.offset
        else:
            tmp_offset = self.offset

        x_filtered = x[:, tmp_offset]  # first just filter and later care about validity
        mask = unsqueeze_right(mask, x)
        invalid = mask[:, tmp_offset]

        while invalid.any():  # worst case runtime O(d_chunk)
            tmp_offset -= 1
            if tmp_offset == -1:
                break
            x_filtered = torch.where(invalid, x[:, tmp_offset], x_filtered)
            invalid = torch.where(invalid, mask[:, tmp_offset], invalid)
        x_filtered = torch.where(invalid, torch.zeros_like(x_filtered), x_filtered)  # zero out remaining invalid chunks

        # diff = torch.abs(x_filtered - x_filtered_2).sum()
        # if not torch.isclose(diff, torch.tensor(0.0, device=x.device, dtype=x.dtype)):
        #    raise RuntimeError('unexpected difference between two methods for picking last elem from subtraj')

        # torch._assert(torch.isclose(diff, torch.tensor(0, device=x.device, dtype=x.dtype)), 'deviation from expected result')

        # i_whole = torch.nonzero(mask.sum(dim=1, keepdim=True) == 0)
        # i_partial = torch.nonzero(mask.sum(dim=1, keepdim=True) > 0)
        # for i_chunk in range(n_chunks):
        #    valid_steps = tuple(~mask[i_chunk].sum(dim=0).detach().cpu().numpy())
        #    dummy_i_batch = tuple(range(d_batch))
        #    #x_filtered[i_chunk][]

        # chunk_valid = mask.sum(dim=1)
        # if chunk_valid.sum() > 0:
        #    n_chunks, d_chunk = x.shape[0:2]
        #    for i_chunk in range(n_chunks):
        #        if not chunk_valid[i_chunk].sum() == 0:
        #            for t in reversed(range(d_chunk)):
        #                # if all time steps in a chunk are masked (i.e. invalid), chunk is invalid and we use
        #                # self.offset above, altough at this point it's irrelevant what we choose
        #                if mask[i_chunk][t].sum() == 0:
        #                    x_filtered[i_chunk] = x[i_chunk][t]

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
        self._mdl = torch.nn.Sequential(lwa(lws, activation, layer_norm=layer_norm, name='learnable_upwards_filter'))

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

    @torch.jit.ignore
    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor | None = None,
                context: torch.Tensor | None = None,
                window_size: int | None = None) -> torch.Tensor:
        return x


class ConstUpwardsFilter(UpwardsFilter):

    def __init__(self, window_size: int, constant: float):
        super().__init__(window_size)
        self.constant = constant

    @torch.jit.ignore
    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor | None = None,
                context: torch.Tensor | None = None,
                window_size: int | None = None) -> torch.Tensor:
        x, n_pad = self._preproc(x, mask, 0.0, window_size)
        x = x[:, 0]
        return torch.zeros_like(x)


RSSMStateType = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class RSSMCell(torch.nn.Module):

    def __init__(self,
                 d_z: int,
                 d_h: int,
                 d_a: int,
                 o_encoder: 'InputEncoder',
                 o_decoder: 'OutputDecoder',
                 r_decoder: 'OutputDecoder',
                 term_decoder: 'OutputDecoder',
                 d_s_embedding: int = None,
                 n_hidden_layers: int = 1,
                 hidden_dropout: float = 0.1,
                 epsilon: float = 0.01,
                 z_prior_lws: Sequence[int] = (32, 32),
                 z_post_lws: Sequence[int] = (32, 32),
                 s_embedding_lws: Sequence[int] = (32, 32),
                 layer_norm: bool = False,
                 activation: str = 'relu',
                 rnn_type: str = 'lstm',
                 latent_dist: str = 'normal',
                 n_latent_categories: int = 32,
                 name: str = 'rssm_cell'):
        super().__init__()

        # if latent_dist != 'normal':
        #    raise NotImplementedError('Check z_dist, z_dist_params, z_sample and z_mode methods first!')

        #if latent_dist == 'normal':
        #    assert o_decoder.d_x_encoded == d_z + d_h
        #    assert r_decoder.d_x_encoded == d_z + d_h
        #    assert term_decoder.d_x_encoded == d_z + d_h
        #elif latent_dist == 'categorical':
        #    assert o_decoder.d_x_encoded == d_z * n_latent_categories + d_h
        #    assert r_decoder.d_x_encoded == d_z * n_latent_categories + d_h
        #    assert term_decoder.d_x_encoded == d_z * n_latent_categories + d_h

        self.d_z = d_z
        self.d_h = d_h
        self.d_a = d_a
        self.d_s_embedding = d_s_embedding if d_s_embedding is not None else d_h + d_z
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

        self.n_latent_categories = n_latent_categories
        if latent_dist == 'normal':
            d_z_out = d_z * 2
            d_z_smpl = d_z
        elif latent_dist == 'bernoulli':
            d_z_out = d_z
            d_z_smpl = d_z
        elif latent_dist == 'categorical':
            d_z_out = d_z * self.n_latent_categories
            d_z_smpl = d_z * self.n_latent_categories
        else:
            raise ValueError(f'Unknown latent distribution type: {latent_dist}')
        self.d_z_smpl = d_z_smpl
        self.d_z_out = d_z_out

        z_prior_lws = (d_h, *z_prior_lws, d_z_out)
        z_post_lws = (d_h + self.d_o_encoded, *z_post_lws, d_z_out)

        d_det_core = d_z_smpl + d_a
        # need both to satisfy torch script
        self._lstm = ModuleList([torch.nn.LSTMCell(d_det_core, hidden_size=d_h) for _ in range(n_hidden_layers)])
        # self._lstm = LayerNormLSTMCell(d_det_core, hidden_size=d_h)
        self._gru = ModuleList([torch.nn.GRUCell(d_det_core, hidden_size=d_h) for _ in range(n_hidden_layers)])

        if self.rnn_type == 'lstm':
            for p in self._gru.parameters():
                p.requires_grad = False
        else:
            for p in self._lstm.parameters():
                p.requires_grad = False

        self._z_prior = torch.nn.Sequential(lwa(z_prior_lws, activation, layer_norm=layer_norm, name=f'{name}_z_prior'))
        self._z_post = torch.nn.Sequential(lwa(z_post_lws, activation, layer_norm=layer_norm, name=f'{name}_z_post'))

        if d_s_embedding is None:
            self.s_embedding = lambda x: x
        else:
            s_embedding_lws = (d_h + d_z, *s_embedding_lws, d_s_embedding)
            self.s_embedding = torch.nn.Sequential(lwa(s_embedding_lws, activation, layer_norm=layer_norm,
                                                       name=f'{name}_s_embedding'))

    @property
    def o_shape(self):
        return self.o_encoder.s_x_orig

    @property
    def d_o_encoded(self):
        return self.o_encoder.d_x_encoded

    @torch.jit.export
    @torch.no_grad()
    def init_state(self,
                   d_batch: int,
                   device: torch.device) -> RSSMStateType:
        z = self.zero_z(d_batch, device)
        h = self.zero_h(d_batch, device)
        rnn_state = self.zero_rnn_state(d_batch, device)
        mock = torch.zeros(d_batch, self.d_z_out, device=device)
        z_prior_params = self.z_dist_params(mock)
        z_post_params = self.z_dist_params(mock)
        s_embedding = self.zero_s_embedding(d_batch, device)
        # return z, z_prior_params, z_post_params, rnn_state[:, 0, :]  # for gru
        return h, z, z_prior_params, z_post_params, rnn_state, s_embedding

    @torch.jit.export
    @torch.no_grad()
    def zero_s_embedding(self,
                         d_batch: int,
                         device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, self.d_s_embedding, device=device)

    @torch.jit.export
    @torch.no_grad()
    def zero_z(self,
               d_batch: int,
               device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, self.d_z_smpl, device=device)

    @torch.jit.export
    @torch.no_grad()
    def zero_h(self,
               d_batch: int,
               device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, self.d_h, device=device)

    @torch.jit.export
    @torch.no_grad()
    def zero_o(self,
               d_batch: int,
               device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, self.d_o_encoded, device=device)

    @torch.jit.export
    @torch.no_grad()
    def zero_o_dist(self,
                    d_batch: int,
                    device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, self.d_z_out, device=device)

    @torch.jit.export
    @torch.no_grad()
    def zero_a(self,
               d_batch: int,
               device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, self.d_a, device=device)

    @torch.jit.export
    @torch.no_grad()
    def zero_r(self,
               d_batch: int,
               device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, 1, device=device)

    @torch.jit.export
    @torch.no_grad()
    def zero_term(self,
                  d_batch: int,
                  device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, 1, device=device)

    @torch.jit.export
    @torch.no_grad()
    def zero_rnn_state(self,
                       d_batch: int,
                       device: torch.device) -> torch.Tensor:
        if self.rnn_type == 'lstm':
            return torch.zeros(d_batch, self.n_hidden_layers, self.d_h, 2, device=device)
        else:
            return torch.zeros(d_batch, self.n_hidden_layers, self.d_h, device=device)

    def _lstm_forward(self,
                      inp: torch.Tensor,
                      last_rnn_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h, c = last_rnn_state.unbind(-1)
        h_next, c_next = [], []
        for i_l, layer in enumerate(self._lstm):
            h_layer, c_layer = h[:, i_l], c[:, i_l]
            h_layer_next, c_layer_next = layer(inp, (h_layer, c_layer))
            h_next.append(h_layer_next)
            c_next.append(c_layer_next)
        h_next = torch.stack(h_next, 1)
        c_next = torch.stack(c_next, 1)
        next_rnn_state = torch.stack([h_next, c_next], dim=-1).contiguous()
        h_out = h_next[:, -1]
        return h_out, next_rnn_state

    def _gru_forward(self,
                     inp: torch.Tensor,
                     last_rnn_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_next = self._gru[0](inp, last_rnn_state[:, 0])
        return h_next, h_next.unsqueeze(1)

    def imagine(self,
                a: torch.Tensor,
                last_z: torch.Tensor,
                last_rnn_state: torch.Tensor,
                sample: bool = True):
        # z = torch.flatten(last_state['z'], start_dim=1)  # in case of categorical latents this is necessary
        z = torch.flatten(last_z, start_dim=1)  # in case of categorical latents this is necessary
        # z_embed = self._z_embed_net(last_state['z'])

        # if torch.isnan(z_embed).any():
        #    raise RuntimeError(f'Invalid NAN embedded z in imagine: {z_embed}')

        inp = torch.concat([z, a], dim=-1)
        if self.rnn_type == 'lstm':
            h, next_rnn_state = self._lstm_forward(inp, last_rnn_state)
        else:
            h, next_rnn_state = self._gru_forward(inp, last_rnn_state)

        # if torch.isnan(h).any() or torch.isinf(h).any():
        #    raise RuntimeError(f'Invalid h in imagine: {h}')

        z_prior_params = self._z_prior(h)
        z_prior_params = self.z_dist_params(z_prior_params)
        if sample:
            z_smpl = self.z_sample(z_prior_params)
        else:
            z_smpl = self.z_mode(z_prior_params)

            # if torch.isnan(z_smpl).any() or torch.isinf(z_smpl).any():
        #    raise RuntimeError(f'Invalid z in imagine: {z_smpl}')

        return h, z_smpl, z_prior_params, z_prior_params, next_rnn_state

    def observe(self,
                a: torch.Tensor,
                o_enc: torch.Tensor,
                last_z: torch.Tensor,
                last_rnn_state: torch.Tensor,
                sample: bool = True):
        h, z_smpl, z_prior_params, _, next_rnn_state = self.imagine(a, last_z, last_rnn_state, sample)

        # if torch.isnan(o_enc).any():
        #    raise RuntimeError(f'Invalid NAN encoded observation in observe: {o_enc}')

        z_post_params = self._z_post(torch.concat([h, o_enc], dim=-1))
        z_post_params = self.z_dist_params(z_post_params)

        # overwrite sample from prior with sample from posterior
        if sample:
            z_smpl = self.z_sample(z_post_params)
        else:
            z_smpl = self.z_mode(z_post_params)

        return h, z_smpl, z_prior_params, z_post_params, next_rnn_state

    def forward(self,
                a: torch.Tensor,
                o_enc: Optional[torch.Tensor] = None,
                last_state: Optional[RSSMStateType] = None,
                use_posterior: bool = True,
                sample_state: bool = True) -> RSSMStateType:
        # if o is None and last_state is None:
        #    raise ValueError('Need at least "o" or "last_state"')
        # if o is None and use_posterior:
        #    raise ValueError('Can\'t use posterior if no ground truth data is provided')

        d_batch = a.shape[0]
        if last_state is None:
            last_state = self.init_state(d_batch, a.device)
        if o_enc is None:
            o_enc = torch.zeros(d_batch, self.o_encoder.d_x_encoded, device=a.device)

        if torch.isnan(a).any() or torch.isinf(a).any():
            raise RuntimeError(f'Invalid action in imagine: {a}')
        if torch.isnan(last_state[1]).any() or torch.isinf(last_state[1]).any():
            raise RuntimeError(f'Invalid last state in imagine: {last_state[1]}')

        if use_posterior:
            ret = self.observe(a, o_enc, last_state[1], last_state[4], sample_state)
        else:
            ret = self.imagine(a, last_state[1], last_state[4], sample_state)
        h, z_smpl, z_prior_params, z_post_params, next_rnn_state = ret

        s = torch.concat([h, z_smpl], dim=-1)
        s = self.s_embedding(s)
        return h, z_smpl, z_prior_params, z_post_params, next_rnn_state, s

    @torch.jit.export
    def scan(self,
             a: torch.Tensor,
             o_enc: Optional[torch.Tensor] = None,
             start_state: Optional[RSSMStateType] = None,
             posterior_steps: int = 0,
             sample_state: bool = True) -> List[RSSMStateType]:
        d_time, d_batch = a.shape[:2]

        if o_enc is None:
            o_enc = torch.zeros(d_time, d_batch, self.o_encoder.d_x_encoded, device=a.device)
        elif posterior_steps < d_time:  # necessary to avoid index error in the loop below if o_enc is too short
            pad = torch.zeros(d_time - posterior_steps, d_batch, self.o_encoder.d_x_encoded, device=a.device)
            o_enc = torch.concat([o_enc, pad], dim=0)

        state_mem: List[RSSMStateType] = []
        state = start_state
        for t, a_t in enumerate(a):
            use_posterior = t <= posterior_steps
            state = self(a[t], o_enc[t], state, use_posterior, sample_state)
            state_mem.append(state)

        return state_mem

    def z_dist_params(self,
                      net_output: torch.Tensor):
        if self.latent_dist == 'normal':
            mu, logvar = torch.tensor_split(net_output, 2, -1)
            sigma = torch.nn.functional.softplus(logvar) + self.epsilon
            z_dist = torch.stack([mu, sigma], dim=-1)
        elif self.latent_dist == 'bernoulli':
            probs = torch.nn.functional.sigmoid(net_output)
            z_dist = probs
        else:  # categorical
            net_output = net_output.reshape((net_output.shape[0], self.d_z, self.n_latent_categories))
            probs = torch.nn.functional.softmax(net_output, dim=-1)
            z_dist = probs.reshape(probs.shape[0], self.d_z * self.n_latent_categories)
        return z_dist

    def z_sample(self,
                 dist_params: torch.Tensor):
        if self.latent_dist == 'normal':
            mu, sigma = dist_params.unbind(-1)
            z_smpl = mu + torch.rand_like(sigma) * sigma
        elif self.latent_dist == 'bernoulli':
            # TODO: continuous bernoulli?
            z_smpl = torch.bernoulli(dist_params) + dist_params - dist_params.detach()
        else:  # categorical
            probs_reshaped = dist_params.reshape(dist_params.shape[0] * self.d_z, self.n_latent_categories)
            indices = torch.multinomial(probs_reshaped, 1, True)
            indices = indices.squeeze(-1)  # remove redundant extra dim coming from generating only one sample
            z_smpl = torch.nn.functional.one_hot(indices, self.n_latent_categories).to(dist_params)
            z_smpl = z_smpl + probs_reshaped - probs_reshaped.detach()  # straight-through gradient
            z_smpl = z_smpl.reshape(dist_params.shape[0], self.d_z * self.n_latent_categories)
            # z_smpl = torch.distributions.OneHotCategorical(probs=dist_params['probs']).sample()
            # z_smpl = z_smpl + dist_params['probs'] - dist_params['probs'].detach()
            # z_smpl = torch.flatten(z_smpl, start_dim=-2, end_dim=-1)
        return z_smpl

    def z_mode(self,
               dist_params: torch.Tensor):
        if self.latent_dist == 'normal':
            mu, sigma = dist_params.unbind(-1)
            z_smpl = mu
        elif self.latent_dist == 'bernoulli':
            z_smpl = torch.round(dist_params) + dist_params - dist_params.detach()
        else:  # categorical
            probs_reshaped = dist_params.reshape(dist_params.shape[0] * self.d_z, self.n_latent_categories)
            z_smpl = torch.argmax(probs_reshaped, dim=-1)
            z_smpl = torch.nn.functional.one_hot(z_smpl, num_classes=self.n_latent_categories)
            z_smpl = z_smpl + probs_reshaped - probs_reshaped.detach()  # straight-through gradient
            z_smpl = z_smpl.reshape(dist_params.shape[0], self.d_z * self.n_latent_categories)
        return z_smpl

    @torch.jit.ignore
    def z_dist(self,
               dist_params: torch.Tensor):
        if self.latent_dist == 'normal':
            mu, sigma = dist_params.unbind(-1)
            z_dist = torch.distributions.Normal(loc=mu, scale=sigma)
            z_dist = torch.distributions.Independent(z_dist, 1)
        elif self.latent_dist == 'bernoulli':
            z_dist = torch.distributions.Bernoulli(probs=dist_params)
            z_dist = torch.distributions.Independent(z_dist, 1)
        else:  # categorical
            z_dist = torch.distributions.OneHotCategorical(probs=dist_params)
        return z_dist

    @torch.jit.export
    def decode(self,
               s: torch.Tensor,
               sample: bool,
               reconstruct_observation: bool):
        if s.ndim == 2:
            d_batch = 0
        else:
            d_batch = 1
        if reconstruct_observation:
            o_dist, o_smpl = self.o_decoder(s, sample)
        else:
            o_dist, o_smpl = self.zero_o_dist(d_batch, s.device), self.zero_o(d_batch, s.device)
        r_dist, r_smpl = self.r_decoder(s, sample)
        # r_smpl = torch.nn.functional.tanh(r_smpl)
        term_dist, term_smpl = self.term_decoder(s, sample)

        return {'o': o_smpl, 'o_dist': o_dist, 'r': r_smpl, 'r_dist': r_dist, 'terminal': term_smpl,
                'terminal_dist': term_dist}


@torch.jit.script
def rssm_stack_states(h: List[torch.Tensor],
                      z: List[torch.Tensor],
                      z_prior: List[torch.Tensor],
                      z_post: List[torch.Tensor],
                      rnn_state: List[torch.Tensor],
                      s_embedding: List[torch.Tensor]):
    return (torch.stack(h).contiguous(),
            torch.stack(z).contiguous(),
            torch.stack(z_prior).contiguous(),
            torch.stack(z_post).contiguous(),
            torch.stack(rnn_state).contiguous(),
            torch.stack(s_embedding).contiguous())


@torch.jit.ignore
def rssm_stack_state_list(states: List[RSSMStateType]):
    return [list(x) for x in zip(*states)]


@torch.jit.script
def rssm_detach_state(h: torch.Tensor,
                      z: torch.Tensor,
                      z_prior: torch.Tensor,
                      z_post: torch.Tensor,
                      rnn_state: torch.Tensor,
                      s_embedding: torch.Tensor):
    return h.detach(), z.detach(), z_prior.detach(), z_post.detach(), rnn_state.detach(), s_embedding.detach()


@torch.jit.script
def rssm_state_seq_to_batch(h: List[torch.Tensor],
                            z: List[torch.Tensor],
                            z_prior: List[torch.Tensor],
                            z_post: List[torch.Tensor],
                            rnn_state: List[torch.Tensor],
                            s_embedding: List[torch.Tensor]):
    return (torch.concat(h, dim=0).contiguous(),
            torch.concat(z, dim=0).contiguous(),
            torch.concat(z_prior, dim=0).contiguous(),
            torch.concat(z_post, dim=0).contiguous(),
            torch.concat(rnn_state, dim=0).contiguous(),
            torch.concat(s_embedding, dim=0).contiguous())


@torch.jit.script
def rssm_state_keys() -> Tuple[str, str, str, str, str, str]:
    return 'h', 'z', 'z_prior', 'z_post', 'rnn_state', 's_embedding'


@torch.jit.ignore
def rssm_add_labels(seq: Sequence[Any, Any, Any, Any, Any, Any]) -> Dict[str, Any]:
    keys = rssm_state_keys()
    return {keys[0]: seq[0], keys[1]: seq[1], keys[2]: seq[2], keys[3]: seq[3], keys[4]: seq[4], keys[5]: seq[5]}


@torch.jit.script
def rssm_remove_labels(state: Dict[str, Any]) -> Tuple[Any, Any, Any, Any, Any, Any]:
    keys = rssm_state_keys()
    return state[keys[0]], state[keys[1]], state[keys[2]], state[keys[3]], state[keys[4]], state[keys[5]]
