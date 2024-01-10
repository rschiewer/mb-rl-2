from __future__ import annotations

from typing import Tuple, Union, List, Sequence, TypeVar, Dict, Optional, Iterable
from collections import namedtuple, OrderedDict
from functools import reduce
from math import ceil
from sys import gettrace
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import Line2D

RnnStateType = TypeVar('RnnStateType', torch.Tensor, Tuple[torch.Tensor, torch.Tensor])
_Placeholder = namedtuple('placeholder', 'device')


def stack_if_list(x: Union[torch.Tensor, List[torch.Tensor]],
                  dim: int = 0):
    if isinstance(x, list):
        x = torch.stack(x, dim=dim)
    return x


class CustomLinear(torch.nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


class CustomConv2d(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


class CustomConvTranspose2d(torch.nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


class CustomGRUCell(torch.nn.GRUCell):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_ih)
        torch.nn.init.orthogonal_(self.weight_hh)
        torch.nn.init.zeros_(self.bias_ih)
        torch.nn.init.zeros_(self.bias_hh)


class DeviceMixin:

    def __new__(cls, *args, **kwargs):
        if not issubclass(cls, torch.nn.Module):
            raise RuntimeError(f'The class {cls} can\'t use this mixin, it\'s designed for subclasses of '
                               f'torch.nn.Module only!')
        return super(DeviceMixin, cls).__new__(cls)

    @property
    def device(self: torch.nn.Module):
        # return next(iter(self.parameters())).device  # hack if this turns out to be too much of a bottleneck

        ph = _Placeholder(None)
        first_param = reduce(lambda a, b: a if a.device == b.device else ph, self.parameters())
        if type(first_param) is _Placeholder:
            raise RuntimeError(f'Model {self} has parameters on multiple devices')

        return first_param.device


class FuzzyDeviceMixin(torch.nn.Module):

    def __new__(cls, *args, **kwargs):
        if not issubclass(cls, torch.nn.Module):
            raise RuntimeError(f'The class {cls} can\'t use this mixin, it\'s designed for subclasses of '
                               f'torch.nn.Module only!')
        instance = super(FuzzyDeviceMixin, cls).__new__(cls)
        instance._device = torch.device('cpu')
        return instance

    @property
    def device(self):
        return self._device

    def _check_device(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            if device.type == 'cuda':
                device_type = 'cuda:0'
                self._device = torch.device(device_type)
            else:
                self._device = device

    def _add_cuda_id(self, arg: Union[torch.device, str]):
        if type(arg) is torch.device:
            return arg

        if arg == 'cuda':
            return 'cuda:0'
        else:
            return arg

    def to(self, *args, **kwargs):
        self._check_device(*args, **kwargs)
        return super().to(*args, **kwargs)

    def to_empty(self, *args, device: Union[str, torch.device]):
        self._check_device(device=device)
        return super.to_empty(*args, device=device)

    def cpu(self):
        self._device = torch.device('cpu')
        return super().cpu()

    def cuda(self, device: int = None):
        device = device if device else 0
        self._device = torch.device(f'cuda:{device}')
        return super().cuda(device)


class StatefulTrainingModule(torch.nn.Module):

    def __init__(self):
        super(StatefulTrainingModule, self).__init__()
        self._current_train_step = None
        self.register_forward_hook(self._incr_counter)

    def _incr_counter(self, *args):
        if self.training:
            self._current_train_step += 1

    def prepare_for_training(self,
                             _processed: List[torch.nn.Module] = None):
        if _processed is None:
            _processed = []

        # call all modules (including myself)
        for m in self.modules():
            if m in _processed:  # in case of recursion, which should not happen in normal scenarios
                continue

            # basic preparation for all stateful modules
            m._current_train_step = 0

            # check if there are custom preparations to be made
            prepare_fn = getattr(m, '_prepare_for_training', None)
            if callable(prepare_fn):
                prepare_fn()

            # remember that we're done with preparing this module
            _processed.append(m)

            # call prepare on children modules of m recursively, which should not be necessary
            if isinstance(m, StatefulTrainingModule):
                m.prepare_for_training(_processed)


class ManagedStatefulTrainingModule(torch.nn.Module):

    def __init__(self):
        super(ManagedStatefulTrainingModule, self).__init__()
        self._current_train_step = None

    def prepare_for_training(self):
        self._current_train_step = 0

    def increase_train_step(self):
        self._current_train_step += 1


# adapted from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py
class RunningMeanStd(torch.jit.ScriptModule):
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self,
                 epsilon: float = 1e-4,
                 shape: tuple = ()):
        super().__init__()
        """Tracks the mean, variance and count of values."""
        self._mean = torch.nn.Parameter(torch.zeros(*shape, dtype=torch.float64), requires_grad=False)
        self._var = torch.nn.Parameter(torch.ones(*shape, dtype=torch.float64), requires_grad=False)
        self.count = torch.nn.Parameter(torch.tensor(epsilon, dtype=torch.float64), requires_grad=False)

    @property
    def mean(self):
        return self._mean.detach()

    @property
    def var(self):
        return self._var.detach()

    @torch.no_grad()
    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        self.update(x, mask)
        return self.mean, self.var

    @torch.jit.export
    def normalize(self,
                  x: torch.Tensor,
                  mean: torch.Tensor,
                  var: torch.Tensor):
        s_orig = x.shape
        x_flat = x.reshape(-1, *self._mean.shape)
        x_flat = (x_flat - mean.unsqueeze(0)) / var.unsqueeze(0)
        x = x_flat.reshape(s_orig)
        return x

    @torch.jit.export
    @torch.no_grad()
    def update(self,
               x: torch.Tensor,
               mask: Optional[torch.Tensor] = None):
        x = x.detach()
        x = x.reshape(-1, *self._mean.shape)
        if mask is None:
            mask = torch.zeros_like(x)
        else:
            mask = mask.reshape(-1, *self._mean.shape)
        batch_mean = masked_mean(x, mask, dim=0)
        batch_var = masked_var(x, mask, dim=0)

        # if mask is None:
        #    weights = torch.ones_like(x)
        # else:
        #    weights = 1 - torch.flatten(mask, start_dim=0, end_dim=-(self._mean.ndim + 1))

        # weights = unsqueeze_right(weights, x)
        # weight_denom = weights.sum(dim=0)
        # weights = torch.where(weight_denom > 0, weights / weight_denom, 0.0)
        # batch_mean = torch.sum(x * weights, dim=0, dtype=torch.float64)
        # batch_var = torch.sum(((x - batch_mean[None, ...]) ** 2) * weights, dim=0, dtype=torch.float64)
        batch_count = torch.sum(1 - mask.flatten(start_dim=1).mean(dim=1))
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        new_mean, new_var, new_count = update_mean_var_count_from_moments(self._mean, self._var, self.count,
                                                                          batch_mean, batch_var, batch_count)
        self._mean.copy_(new_mean)
        self._var.copy_(new_var)
        self.count.copy_(new_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class RecurrentBlock(torch.nn.Module, DeviceMixin):

    def __init__(self,
                 *d_inputs: int,
                 d_hidden: int,
                 n_layers: int = 1,
                 dropout: float = 0,
                 batch_first: bool = True):
        super(RecurrentBlock, self).__init__()

        self.d_inputs = d_inputs
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.layer_list = torch.nn.LSTM(sum(d_inputs), d_hidden, num_layers=n_layers, batch_first=batch_first,
                                        dropout=dropout)

    def forward(self,
                *xs: torch.Tensor,
                h: Tuple[torch.Tensor, torch.Tensor] = None):
        d_batch = xs[0].shape[0]
        if h is None:
            h = self.gen_h_placeholder(d_batch)

        x = torch.concat(xs, dim=-1)
        x, h = self.layer_list(x, h)

        return x, h

    def gen_h_placeholder(self, d_batch: int) -> Tuple[torch.Tensor, torch.Tensor]:
        d = self.device
        return (torch.zeros(self.n_layers, d_batch, self.d_hidden, device=d),
                torch.zeros(self.n_layers, d_batch, self.d_hidden, device=d))


def build_categorical(params: torch.Tensor,
                      params_are_probs: bool = False):
    if params_are_probs:
        dist = torch.distributions.OneHotCategorical(probs=params)
    else:
        dist = torch.distributions.OneHotCategorical(logits=params)
    return dist


def sample_from_categorical(params: torch.Tensor,
                            params_are_probs: bool = False,
                            gradient: bool = True):
    if params_are_probs:
        dist = torch.distributions.OneHotCategorical(probs=params)
    else:
        dist = torch.distributions.OneHotCategorical(logits=params)

    if gradient:
        probs = torch.nn.functional.softmax(dist.probs, dim=-1)
        return dist.sample() + probs - probs.detach()
    else:
        return dist.sample()


def make_gaussian_params(params: torch.Tensor,
                         epsilon: float) -> torch.Tensor:
    assert params.shape[-1] % 2 == 0
    mu, logvar = torch.tensor_split(params, 2, dim=-1)
    sigma = torch.exp(0.5 * logvar) + epsilon  # torch.nn.functional.relu(logvar) + epsilon
    return torch.concat([mu, sigma], dim=-1)


def build_gaussian(params: torch.Tensor) -> torch.distributions.Normal:
    assert params.shape[-1] % 2 == 0
    mu, sigma = torch.tensor_split(params, 2, dim=-1)
    dist = torch.distributions.Normal(loc=mu, scale=sigma)
    return dist


def sample_from_gaussian(gaussian_params: torch.Tensor,
                         gradient: bool = True) -> torch.Tensor:
    assert gaussian_params.shape[-1] % 2 == 0
    mu, sigma = torch.tensor_split(gaussian_params, 2, dim=-1)
    dist = torch.distributions.Normal(loc=mu, scale=sigma)
    if gradient:
        return dist.rsample()
    else:
        return dist.sample()


def get_mu(gaussian_params: torch.Tensor) -> torch.Tensor:
    assert gaussian_params.shape[-1] % 2 == 0
    mu, sigma = torch.tensor_split(gaussian_params, 2, dim=-1)
    return mu


def get_sigma(gaussian_params: torch.Tensor) -> torch.Tensor:
    assert gaussian_params.shape[-1] % 2 == 0
    mu, sigma = torch.tensor_split(gaussian_params, 2, dim=-1)
    return sigma


def build_bernoulli(params: torch.Tensor) -> torch.distributions.ContinuousBernoulli:
    dist = torch.distributions.ContinuousBernoulli(logits=params)
    return dist


def make_bernoulli_params(params: torch.Tensor) -> torch.Tensor:
    # params = torch.nn.functional.sigmoid(params)
    return params


def sample_from_bernoulli(params: torch.Tensor,
                          gradient: bool = True) -> torch.Tensor:
    dist = build_bernoulli(params)
    if gradient:
        return dist.rsample()
    else:
        return dist.sample()


def bin_every_k_steps(data: torch.Tensor,
                      k: int,
                      padding_val: Union[int, float, None] = None):
    d_time, d_batch = data.shape[:2]
    d_data = data.shape[2:]
    n_macro_steps = ceil(d_time / k)
    d_padding = n_macro_steps * k - d_time

    if padding_val is None:
        padding = torch.repeat_interleave(data[-1, None], d_padding, dim=0)
    else:
        padding = torch.full((d_padding, d_batch, *d_data), fill_value=padding_val, dtype=data.dtype,
                             device=data.device)
    data_padded = torch.concat([data, padding], dim=0)
    binned = data_padded.reshape((d_time + d_padding) // k, k, d_batch, *d_data)

    # bins = []
    # for i in range(0, d_time, k):
    #    bins.append(data[:, i:i+k])
    # binned2 = torch.stack(bins, dim=1)

    return binned


def _get_act_fn(descr: str):
    if descr == 'identity':
        act_constr = torch.nn.Identity
    elif descr == 'relu':
        act_constr = torch.nn.ReLU
    elif descr == 'gelu':
        act_constr = torch.nn.GELU
    elif descr == 'elu':
        act_constr = torch.nn.ELU
    elif descr == 'silu':
        act_constr = torch.nn.SiLU
    elif descr == 'tanh':
        act_constr = torch.nn.Tanh
    elif descr == 'sigmoid':
        act_constr = torch.nn.Sigmoid
    else:
        raise ValueError(f'Unkown descr function: {descr}')
    return act_constr


_lin_layer_constr = torch.nn.Linear
_conv_layer_constr = torch.nn.Conv2d


def layers_with_activation(lws: Sequence[int], activation: str = 'relu', layer_norm: bool = False, name: str = None,
                           final_activation_function: str = None):
    assert len(lws) >= 2, f'Need at least w_in and w_out for one layer, but lws contains less than 2 elements'

    act_constr = _get_act_fn(activation)
    layers = []
    for w_in, w_out in zip(lws[:-1], lws[1:-1]):
        if layer_norm:
            layers += [_lin_layer_constr(w_in, w_out), torch.nn.LayerNorm(w_out), act_constr()]
        else:
            layers += [_lin_layer_constr(w_in, w_out), act_constr()]
    layers.append(_lin_layer_constr(lws[-2], lws[-1]))

    if final_activation_function:
        final_act_constr = _get_act_fn(final_activation_function)
        layers.append(final_act_constr())

    if name:
        layers = OrderedDict([(f'{name}_{i}', l) for i, l in enumerate(layers)])

    return layers


def conv_layers_with_activation(channels: Sequence[int], kernel_sizes: Sequence[Tuple[int, int]],
                                strides: Sequence[int], activation: str = 'relu', layer_norm: bool = False,
                                name: str = None, final_activation_function: str = None):
    assert len(channels) >= 2, f'Need at least c_in and c_out for one layer, but lws contains less than 2 elements'

    act_constr = _get_act_fn(activation)
    layers = []
    for w_in, w_out, ks, st in zip(channels[:-1], channels[1:-1], kernel_sizes, str):
        if layer_norm:
            layers += [_conv_layer_constr(in_channels=w_in, out_channels=w_out, kernel_size=ks, stride=st),
                       torch.nn.LayerNorm(w_out), act_constr()]
        else:
            layers += [_conv_layer_constr(in_channels=w_in, out_channels=w_out, kernel_size=ks, stride=st),
                       act_constr()]
    layers.append(_conv_layer_constr(in_channels=channels[-2], out_channels=channels[-1], kernel_size=kernel_sizes[-1],
                                     stride=strides[-1]))

    if final_activation_function:
        final_act_constr = _get_act_fn(final_activation_function)
        layers.append(final_act_constr())

    if name:
        layers = OrderedDict([(f'{name}_{i}', l) for i, l in enumerate(layers)])

    return layers


def get_dist_params(d: torch.distributions.Distribution):
    if isinstance(d, (torch.distributions.Normal, torch.distributions.Cauchy, torch.distributions.Gumbel,
                      torch.distributions.Laplace, torch.distributions.LogNormal)):
        return {'loc': d.loc, 'scale': d.scale}
    elif isinstance(d, torch.distributions.RelaxedOneHotCategorical):
        return {'logits': d.logits, 'temperature': d.temperature}
    elif isinstance(d, torch.distributions.ContinuousBernoulli):
        return {'logits': d.logits}
    else:
        raise RuntimeError(f'Can\'t extract parameters of the given distribution: {d}')


def detach_dist(d: torch.distributions.Distribution):
    if isinstance(d, (torch.distributions.Normal, torch.distributions.Cauchy, torch.distributions.Gumbel,
                      torch.distributions.Laplace, torch.distributions.LogNormal)):
        return type(d)(loc=d.loc.detach(), scale=d.scale.detach())
    elif isinstance(d, SquashedNormal):
        return type(d)(loc=d.loc.detach(), scale=d.scale.detach())
    elif isinstance(d, torch.distributions.RelaxedOneHotCategorical):
        return type(d)(temperature=d.temperature.detach(), logits=d.logits.detach())
    elif isinstance(d, torch.distributions.ContinuousBernoulli):
        return type(d)(probs=d.probs.detach())
    elif isinstance(d, torch.distributions.Bernoulli):
        return type(d)(probs=d.probs.detach())
    elif isinstance(d, torch.distributions.Independent):
        return torch.distributions.Independent(detach_dist(d.base_dist),
                                               reinterpreted_batch_ndims=d.reinterpreted_batch_ndims)
    elif isinstance(d, torch.distributions.OneHotCategorical):
        return type(d)(probs=d.probs.detach())
    # elif hasattr(d, 'logits'):
    #    return type(d)(logits=d.logits.detach())
    # elif hasattr(d, 'probs'):
    #    return type(d)(probs=d.probs.detach())
    else:
        raise RuntimeError(f'Can\'t detach the given distribution: {d}')


def stack_dists(dists: List[torch.distributions.Distribution]):
    cls = set([type(d) for d in dists])
    assert len(cls) == 1, 'All distributions have to share the same class'
    cls = cls.pop()

    if cls in (torch.distributions.Normal, torch.distributions.Cauchy, torch.distributions.Gumbel,
               torch.distributions.Laplace, torch.distributions.LogNormal):
        loc = torch.stack([d.loc for d in dists])
        scale = torch.stack([d.scale for d in dists])
        return torch.distributions.Normal(loc=loc, scale=scale)
    elif cls == torch.distributions.ContinuousBernoulli:
        probs = torch.stack([d.probs for d in dists])
        return torch.distributions.ContinuousBernoulli(probs=probs)
    elif cls == torch.distributions.Bernoulli:
        probs = torch.stack([d.probs for d in dists])
        return torch.distributions.Bernoulli(probs=probs)
    elif cls == SquashedNormal:
        loc = torch.stack([d.loc for d in dists])
        scale = torch.stack([d.scale for d in dists])
        return SquashedNormal(loc=loc, scale=scale)
    elif cls == torch.distributions.TransformedDistribution:
        base_dists = [d.base_dist for d in dists]
        transforms = [d.transforms for d in dists]
        for t in transforms:
            assert t == transforms[0], 'All transforms have to be equal'
        stacked_base_dists = stack_dists(base_dists)
        return torch.distributions.TransformedDistribution(transforms=transforms[0],
                                                           base_distribution=stacked_base_dists)
    elif cls == torch.distributions.RelaxedBernoulli:
        temperature = torch.stack([d.temperature for d in dists])
        probs = torch.stack([d.probs for d in dists])
        return torch.distributions.RelaxedBernoulli(temperature=temperature, probs=probs)
    elif cls == torch.distributions.RelaxedOneHotCategorical:
        temperature = torch.stack([d.temperature for d in dists])
        probs = torch.stack([d.probs for d in dists])
        return torch.distributions.RelaxedOneHotCategorical(temperature=temperature, probs=probs)
    elif cls == torch.distributions.OneHotCategorical:
        probs = torch.stack([d.probs for d in dists])
        return torch.distributions.OneHotCategorical(probs=probs)
    else:
        raise ValueError(f'Unsupported distribution class: {cls}')


def unstack_dist(dist: torch.distributions.Distribution,
                 dim: int):
    if isinstance(dist, torch.distributions.Normal):
        locs = list(dist.loc.unbind(dim))
        scales = list(dist.scale.unbind(dim))
        return [torch.distributions.Normal(loc=loc, scale=scale) for loc, scale in zip(locs, scales)]
    elif isinstance(dist, torch.distributions.RelaxedBernoulli):
        probs = list(dist.probs.unbind(dim))
        temps = list(dist.temperature.unbind(dim))
        return [torch.distributions.RelaxedBernoulli(probs=prob, temperature=temp) for prob, temp in zip(probs, temps)]
    elif isinstance(dist, torch.distributions.ContinuousBernoulli):
        probs = list(dist.probs.unbind(dim))
        return [torch.distributions.ContinuousBernoulli(probs=prob) for prob in probs]
    else:
        raise ValueError(f'Unsupported distribution class: {type(dist)}')


def concat_dists(dists: List[torch.distributions.Distribution],
                 dim: int = 0):
    cls = set([type(d) for d in dists])
    assert len(cls) == 1, 'All distributions have to share the same class'
    cls = cls.pop()

    if cls in (torch.distributions.Normal, torch.distributions.Cauchy, torch.distributions.Gumbel,
               torch.distributions.Laplace, torch.distributions.LogNormal):
        loc = torch.concat([d.loc for d in dists], dim=dim)
        scale = torch.concat([d.scale for d in dists], dim=dim)
        return torch.distributions.Normal(loc=loc, scale=scale)
    elif cls == torch.distributions.ContinuousBernoulli:
        probs = torch.concat([d.probs for d in dists], dim=dim)
        return torch.distributions.ContinuousBernoulli(probs=probs)
    elif cls == torch.distributions.Bernoulli:
        probs = torch.concat([d.probs for d in dists], dim=dim)
        return torch.distributions.Bernoulli(probs=probs)
    elif cls == torch.distributions.RelaxedOneHotCategorical:
        temperature = torch.concat([d.temperature for d in dists], dim=dim)
        probs = torch.concat([d.probs for d in dists], dim=dim)
        return torch.distributions.RelaxedOneHotCategorical(temperature=temperature, probs=probs)
    elif cls == torch.distributions.OneHotCategorical:
        probs = torch.concat([d.probs for d in dists], dim=dim)
        return torch.distributions.OneHotCategorical(probs=probs)
    else:
        raise ValueError(f'Unsupported distribution class: {cls}')


def extract_sub_distribution(d: torch.distributions.Distribution,
                             *idx: int | Sequence[int] | torch.Tensor,
                             keepdim: bool = False):
    if len(d.batch_shape) < len(idx):
        raise ValueError(f'Batch size of distribution should be smaller or equal to number of specified indices ',
                         f'but found {len(d.batch_shape)} and {len(idx)}')
    if keepdim:  # make single int indices to slices of length 1 to prevent loss of dimension
        tmp = []
        for i in idx:
            if type(i) is int:
                tmp.append(slice(i, i + 1))
            elif isinstance(i, torch.Tensor):
                tmp.append(slice(i.detach().cpu().numpy().item(), i.detach().cpu().numpy().item() + 1))
            else:
                tmp.append(i)
        idx = tmp
    if isinstance(d, torch.distributions.Normal):
        d_extracted = torch.distributions.Normal(loc=d.loc[idx], scale=d.scale[idx])
    elif isinstance(d, torch.distributions.OneHotCategorical):
        d_extracted = torch.distributions.OneHotCategorical(logits=d.logits[idx])
    elif isinstance(d, torch.distributions.RelaxedOneHotCategorical):
        d_extracted = torch.distributions.RelaxedOneHotCategorical(d.temperature, logits=d.logits[idx])
    elif isinstance(d, torch.distributions.ContinuousBernoulli):
        d_extracted = torch.distributions.ContinuousBernoulli(logits=d.logits[idx])
    elif isinstance(d, torch.distributions.RelaxedBernoulli):
        d_extracted = torch.distributions.RelaxedBernoulli(logits=d.logits[idx], temperature=d.temperature)
    else:
        raise ValueError(f'Distribution class not supported: {type(d)}')
    return d_extracted


def repeat_distribution(d: torch.distributions.Distribution,
                        repeats: Sequence):
    if len(d.batch_shape) != len(repeats):
        raise ValueError(f'repeats argument should have the same length as distributions batch_shape')
    if isinstance(d, torch.distributions.Normal):
        d_repeated = torch.distributions.Normal(loc=d.loc.repeat(*repeats), scale=d.scale.repeat(*repeats))
    elif isinstance(d, torch.distributions.OneHotCategorical):
        d_repeated = torch.distributions.OneHotCategorical(logits=d.logits.repeat(*repeats))
    elif isinstance(d, torch.distributions.RelaxedOneHotCategorical):
        d_repeated = torch.distributions.RelaxedOneHotCategorical(d.temperature, logits=d.logits.repeat(*repeats))
    elif isinstance(d, torch.distributions.ContinuousBernoulli):
        d_repeated = torch.distributions.ContinuousBernoulli(logits=d.logits.repeat(*repeats))
    else:
        raise ValueError(f'Distribution class not supported: {type(d)}')
    return d_repeated


def unpack_rnn_state(rnn_state_packed: torch.Tensor):
    if rnn_state_packed.shape[-2] == 2:
        h, c = rnn_state_packed.unbind(-2)
        h, c = h.transpose(0, 1), c.transpose(0, 1)
        h, c = h.contiguous(), c.contiguous()
        return h, c
    else:
        h = rnn_state_packed.squeeze(-2)
        h = h.transpose(0, 1)
        h = h.contiguous()
        return h


def pack_rnn_state(rnn_state: RnnStateType):
    if isinstance(rnn_state, tuple):
        return torch.stack([rnn_state[0].transpose(0, 1), rnn_state[1].transpose(0, 1)], dim=-2)
    else:
        return torch.stack([rnn_state.transpose(0, 1)], dim=-2)


def to_tensors(mem: List[Dict[str, int | float | np.single | np.double | bool | np.ndarray]],
               device: torch.device,
               dtypes: Union[List, Tuple] = None,
               padding: Union[List, Tuple, str] = None):
    assert padding is None, 'padding argument is deprecated'

    if len(mem) > 1:
        fids = reduce(lambda a, b: set(a) | set(b), mem)
    else:
        fids = set(mem[0])
    assert 'mask' not in fids, 'found forbidden field id "mask" in mem'
    if dtypes is None: dtypes = [torch.float32 for _ in range(len(fids))]
    if padding is None:
        padding = [0.0 for _ in range(len(fids))]
        repeat_last = False
    elif padding == 'repeat':
        padding = [0.0 for _ in range(len(fids))]
        repeat_last = True
    dtypes = {fid: dtype for fid, dtype in zip(fids, dtypes)}
    padding = {fid: pad for fid, pad in zip(fids, padding)}
    shapes = {fid: fval.shape[1:] for fid, fval in mem[0].items()}

    # collect trajectory data
    data = {fid: [] for fid in fids}
    lengths = {fid: [] for fid in fids}
    for traj in mem:
        assert set(traj) == fids, f'found trajectory in batch that misses fields: {set(traj)} vs. {fids}'
        for fid, fval in traj.items():
            data[fid].append(torch.from_numpy(fval))
            lengths[fid].append(len(fval))
    longest = {k: max(v) for k, v in lengths.items()}
    n_traj = len(mem)

    # prepare memory containers
    cont = {fid: torch.full((longest[fid], n_traj, *shapes[fid]), fill_value=padding[fid], dtype=dtypes[fid],
                            device=device) for fid in fids}

    # copy data
    for fid, fval in data.items():
        flens = lengths[fid]
        for i in range(n_traj):
            cont[fid][0:flens[i], i] = data[fid][i]
            if repeat_last and fid != 'a':
                cont[fid][flens[i]:, i] = cont[fid][flens[i] - 1, i].unsqueeze(0)

    # mask for longest field per trajectory (valid for o, a, r, term, trunc but not for higher level a)
    # if padding is None:
    cont['mask'] = torch.full((max(longest.values()), n_traj), fill_value=True, dtype=torch.float32, device=device)
    for i in range(n_traj):
        longest_field = max([x[i] for x in lengths.values()])
        cont['mask'][0:longest_field, i] = False
    # else:
    #    cont['mask'] = torch.full((max(longest.values()), n_traj), fill_value=False, dtype=torch.float32, device=device)

    return cont


def pad_first_timestep(o: torch.Tensor,
                       a: torch.Tensor,
                       r: torch.Tensor,
                       term: torch.Tensor,
                       trunc: torch.Tensor,
                       mask: torch.Tensor):
    a = torch.cat([torch.zeros_like(a[0]), a], dim=0)
    r = torch.cat([torch.zeros_like(r[0]), r], dim=0)
    term = torch.cat([torch.zeros_like(term[0]), term], dim=0)
    trunc = torch.cat([torch.zeros_like(trunc[0]), trunc], dim=0)
    mask = torch.cat([torch.zeros_like(mask[0]), mask], dim=0)
    return o, a, r, term, trunc, mask


# @torch.jit.script
def update_ema_modules(modules: List[Dict[str, torch.Tensor]], ema_modules: List[Dict[str, torch.Tensor]],
                       coeff: float):
    with torch.no_grad():
        for params_m, params_ema_m in zip(modules, ema_modules):
            for name, param_m in params_m.items():
                params_ema_m[name].sub_(coeff * (params_ema_m[name] - param_m))


def to_np(data_dict: Dict[str, Union[torch.Tensor, Dict]]):
    np_data_dict = {}
    for k, v in data_dict.items():
        if isinstance(v, dict):
            np_data_dict[k] = to_np(v)
        elif isinstance(v, torch.Tensor):
            np_data_dict[k] = v.detach().cpu().numpy().item()
        elif np.isscalar(v):
            np_data_dict[k] = v
        else:
            raise ValueError(f'Unsupported type: {type(k)}')
    return np_data_dict


def masked_mean(x: torch.Tensor,
                mask: torch.Tensor,
                dim: int | Tuple[int] | List[int] | None = None,
                keepdim: bool = False
                ):
    # assert x.shape[:mask.squeeze().ndim] == mask.squeeze().shape, 'Leading dimensions of x and mask mismatch'
    valid = 1 - mask.to(dtype=torch.float32)
    valid = unsqueeze_right(valid, x)
    valid = valid.expand_as(x)
    num = torch.sum(x * valid, dim=dim, keepdim=keepdim)
    denom = torch.sum(valid, dim=dim, keepdim=keepdim)
    denom = torch.where(denom == 0, 1.0, denom)
    ret = num / denom
    return ret


def masked_var(x: torch.Tensor,
               mask: torch.Tensor,
               dim: int | Tuple[int] | List[int] | None = None,
               keepdim: bool = False
               ):
    masked_x_mean = masked_mean(x, mask, dim, keepdim=True)

    repeats = [1 for _ in masked_x_mean.shape]
    if dim is None:
        dim = list(range(x.ndim))
    elif type(dim) == int:
        dim = [dim]
    for idx in dim:
        repeats[idx] = x.shape[idx]
    masked_x_mean = masked_x_mean.repeat(repeats)

    diff = (x - masked_x_mean) ** 2
    ret = masked_mean(diff, mask, dim, keepdim=keepdim)
    return ret


@torch.jit.script
def compute_mask(terminals: Union[List[torch.Tensor], torch.Tensor],
                 mode: str = 'default',
                 threshold: Optional[float] = None,
                 first_step_mask: Optional[torch.Tensor] = None,
                 gamma: float = 1.0,
                 disable: bool = True):
    with torch.no_grad():
        terminals = stack_if_list(terminals).detach()
        d_time, d_batch = terminals.shape[:2]

        # return torch.zeros_like(terminals)

        # never mask first time step except first_step_mask tells us to
        if first_step_mask is None:
            first_step_mask = torch.zeros(1, d_batch, 1, dtype=terminals.dtype, device=terminals.device)
        elif first_step_mask.ndim == 2:
            first_step_mask = first_step_mask.unsqueeze(0)  # add time dimension

        terminals = torch.concat([first_step_mask, terminals], dim=0)  # this shifts time one to the right
        terminals = terminals[:-1]  # cut last time step since we don't need it
        valid = 1.0 - terminals.to(dtype=torch.float32)  # invert to compute exponentially decreasing validity mask
        valid = torch.cumprod(valid, dim=0)

        gamma_mat = torch.cumprod(torch.full_like(valid[:-1], gamma), dim=0)  # constant discount factor on top
        gamma_mat = torch.concat([torch.ones_like(valid[0:1]), gamma_mat])
        valid = valid * gamma_mat

        mask = 1.0 - valid  # invert to turn into mask again

        if mode == 'deterministic' and threshold is not None:
            mask = torch.where(mask > threshold,
                               torch.tensor(1.0, device=terminals.device, dtype=terminals.dtype),
                               torch.tensor(0.0, device=terminals.device, dtype=terminals.dtype))
            mask = mask.to(dtype=torch.bool)

        # mask = torch.zeros_like(mask)
        # mask = torch.where(mask > 0.95,
        #                   torch.tensor(1.0, device=terminals.device, dtype=terminals.dtype),
        #                   torch.tensor(0.0, device=terminals.device, dtype=terminals.dtype))

        # if disable:
        #    mask = torch.zeros_like(mask)

        return mask.detach()


# compiled from:
# https://github.com/denisyarats/pytorch_sac/blob/master/agent/actor.py
# https://garage.readthedocs.io/en/v2020.06.2/_modules/garage/torch/distributions/tanh_normal.html#TanhNormal.entropy
class SquashedNormal(torch.distributions.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = torch.distributions.Normal(loc, scale)
        transforms = [torch.distributions.TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

    def entropy(self):
        raise NotImplementedError('There\'s no analytic expression for SquashedGaussian entropy')
        #return self.base_dist.entropy()

    #def log_prob(self, value, pre_tanh_value=None, epsilon=1e-6):
    #    if pre_tanh_value is None:
    #        pre_tanh_value = torch.log((1 + value) / (1 - value)) / 2
    #    norm_lp = self.base_dist.log_prob(pre_tanh_value).sum(dim=-1)
    #    ret = norm_lp - torch.log((1.0 - torch.tanh(value) ** 2) + epsilon).sum(dim=-1)
    #    return ret
    def log_prob(self, value):
        value = torch.clamp(value, min=-1.0 + 1e-6, max=1.0 - 1e-6)
        return super().log_prob(value)

    @staticmethod
    def _clip_but_pass_gradient(x: torch.Tensor, lower: float = 0., upper: float = 1.):
        """Clipping function that allows for gradients to flow through.

        Args:
            x (torch.Tensor): value to be clipped
            lower (float): lower bound of clipping
            upper (float): upper bound of clipping

        Returns:
            torch.Tensor: x clipped between lower and upper.

        """
        clip_up = (x > upper).float()
        clip_low = (x < lower).float()
        with torch.no_grad():
            clip = ((upper - x) * clip_up + (lower - x) * clip_low)
        return x + clip


# from https://github.com/ray-project/ray/blob/master/rllib/algorithms/dreamer/utils.py#L48
class TanhBijector(torch.distributions.Transform):
    def __init__(self):
        super().__init__()
        self.bijective = True
        self.domain = torch.distributions.constraints.real
        self.codomain = torch.distributions.constraints.interval(-1.0, 1.0)

    #def atanh(self, x):
    #    return 0.5 * torch.log((1 + x) / (1 - x))

    def sign(self):
        return 1.0

    def _call(self, x):
        return torch.tanh(x)

    def _inverse(self, y):
        #y = torch.clamp(y, min=-1.0 + 1e-5, max=1.0 - 1e-5)
        y = torch.atanh(y)
        return y

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (np.log(2) - x - torch.nn.functional.softplus(-2.0 * x))


def unsqueeze_right(to_expand: Union[np.ndarray, torch.Tensor], target: Union[np.ndarray, torch.Tensor]):
    if to_expand.ndim == target.ndim:
        return to_expand
    elif to_expand.ndim > target.ndim:
        raise ValueError('Expansion can only be done if to_expand has fewer dimensions than target, but got ',
                         f'{to_expand.shape}({to_expand.ndim}) vs {target.shape}({target.ndim})')

    dim_diff = target.ndim - to_expand.ndim
    return to_expand.reshape(*to_expand.shape, *[1 for _ in range(dim_diff)])


def stack_tensor_dicts(x: List[Dict[str, torch.Tensor]]):
    x_stacked = {k: [] for k in x[0]}
    for dist_params in x:
        for k, v in dist_params.item():
            x_stacked[k].append(v)
    x_stacked = {k: torch.stack(v) for k, v in x_stacked.items()}
    return x_stacked


def concat_tensor_dicts(x: List[Dict[str, torch.Tensor]],
                        dim: int):
    x_concat = {k: [] for k in x[0]}
    for dist_params in x:
        for k, v in dist_params.items():
            x_concat[k].append(v)
    x_concat = {k: torch.concat(v, dim=dim) for k, v in x_concat.items()}
    return x_concat


# from https://github.com/rlworkgroup/garage/blob/master/src/garage/torch/distributions/tanh_normal.py
def clip_but_pass_gradient(x: torch.Tensor,
                           lower: float = 0.0,
                           upper: float = 1.0):
    """Clipping function that allows for gradients to flow through.

    Args:
        x (torch.Tensor): value to be clipped
        lower (float): lower bound of clipping
        upper (float): upper bound of clipping

    Returns:
        torch.Tensor: x clipped between lower and upper.

    """
    clip_up = (x > upper).float()
    clip_low = (x < lower).float()
    with torch.no_grad():
        clip = ((upper - x) * clip_up + (lower - x) * clip_low)
    return x + clip


# from https://github.com/RajGhugare19/dreamerv2/blob/main/dreamerv2/utils/module.py#L1
def get_parameters(modules: Iterable[torch.nn.Module]):
    """
    Given a list of torch modules, returns a list of their parameters.
    :param modules: iterable of modules
    :returns: a list of parameters
    """
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters


# from https://github.com/RajGhugare19/dreamerv2/blob/main/dreamerv2/utils/module.py#L15
class FreezeParameters:
    def __init__(self, modules: Iterable[torch.nn.Module]):
        """
        Context manager to locally freeze gradients.
        In some cases with can speed up computation because gradients aren't calculated for these listed modules.
        example:
        ```
        with FreezeParameters([module]):
            output_tensor = module(input_tensor)
        ```
        :param modules: iterable of modules. used to call .parameters() to freeze gradients.
        """
        self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):
        for param in get_parameters(self.modules):
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]


def record_parameters(*modules: torch.nn.Module, reduction_fn: callable = None):
    if reduction_fn is None:
        def reduction_fn(x):
            return x

    params = []
    for m in modules:
        params.extend([reduction_fn(p.detach().cpu().numpy()) for p in m.parameters()])
    return params


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            mean = 0 if p.grad is None else p.grad.abs()._mean().item()
            max = 0 if p.grad is None else p.grad.abs().max().item()
            ave_grads.append(mean)
            max_grads.append(max)
    fig = plt.figure(figsize=(10, 10))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    # plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.yscale('log')
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.tight_layout()


def check_tensor(x: torch.Tensor,
                 neg_bound: float | None = None,
                 pos_bound: float | None = None):
    if torch.isnan(x).any():
        raise ValueError(f'Found NAN values in tensor')
    if torch.isinf(x).any():
        raise ValueError(f'Found inf values in tensor')
    if neg_bound is not None and torch.any(x < neg_bound):
        raise ValueError(f'Found value smaller than {neg_bound} in tensor')
    if pos_bound is not None and torch.any(x > pos_bound):
        raise ValueError(f'Found value greater than {pos_bound} in tensor')

class Moments(torch.nn.Module):

    def __init__(self, impl='mean_std', decay=0.99, max=1e8, eps=0.0, perclo=5, perchi=95):
        super().__init__()
        self.impl = impl
        self.decay = torch.tensor(decay, dtype=torch.float32)
        self.max = torch.tensor(max, dtype=torch.float32)
        self.eps = torch.tensor(eps, dtype=torch.float32)
        self.perclo = torch.tensor(perclo / 100, dtype=torch.float32)
        self.perchi = torch.tensor(perchi / 100, dtype=torch.float32)
        if self.impl == 'off':
            pass
        elif self.impl == 'mean_std':
            self.step = torch.nn.parameter.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=False)
            self.mean = torch.nn.parameter.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=False)
            self.sqrs = torch.nn.parameter.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=False)
        elif self.impl == 'min_max':
            self.low = torch.nn.parameter.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=False)
            self.high = torch.nn.parameter.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=False)
        elif self.impl == 'perc_ema':
            self.low = torch.nn.parameter.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=False)
            self.high = torch.nn.parameter.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=False)
        elif self.impl == 'perc_ema_corr':
            self.step = torch.nn.parameter.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=False)
            self.low = torch.nn.parameter.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=False)
            self.high = torch.nn.parameter.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=False)
        elif self.impl == 'mean_mag':
            self.mag = torch.nn.parameter.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=False)
        elif self.impl == 'max_mag':
            self.mag = torch.nn.parameter.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=False)
        else:
            raise NotImplementedError(self.impl)

    def __call__(self, x):
        self.update(x)
        return self.stats()

    def update(self, x):
        mean = torch.mean
        min_ = torch.min
        max_ = torch.max
        per = torch.quantile  # was percentile

        x = x.detach().to(torch.float32)
        m = self.decay

        if self.impl == 'off':
            pass
        elif self.impl == 'mean_std':
            self.step.copy_(self.step + 1)
            self.mean.copy_(m * self.mean + (1 - m) * mean(x))
            self.sqrs.copy_(m * self.sqrs + (1 - m) * mean(x * x))
        elif self.impl == 'min_max':
            low, high = min_(x), max_(x)
            self.low.copy_(m * torch.minimum(self.low, low) + (1 - m) * low)
            self.high.copy_(m * torch.maximum(self.high, high) + (1 - m) * high)
        elif self.impl == 'perc_ema':
            low, high = per(x, self.perclo), per(x, self.perchi)
            self.low.copy_(m * self.low + (1 - m) * low)
            self.high.copy_(m * self.high + (1 - m) * high)
        elif self.impl == 'perc_ema_corr':
            self.step.copy_(self.step + 1)
            low, high = per(x, self.perclo), per(x, self.perchi)
            self.low.copy_(m * self.low + (1 - m) * low)
            self.high.copy_(m * self.high + (1 - m) * high)
        elif self.impl == 'mean_mag':
            curr = mean(torch.abs(x))
            self.mag.copy_(m * self.mag + (1 - m) * curr)
        elif self.impl == 'max_mag':
            curr = max_(torch.abs(x))
            self.mag.copy_(m * torch.maximum(self.mag, curr) + (1 - m) * curr)
        else:
            raise NotImplementedError(self.impl)

    def stats(self):
        if self.impl == 'off':
            return 0.0, 1.0
        elif self.impl == 'mean_std':
            corr = 1 - self.decay ** self.step.to(torch.float32)
            mean = self.mean / corr
            var = (self.sqrs / corr) - self.mean ** 2
            std = torch.sqrt(torch.maximum(var, 1 / self.max ** 2) + self.eps)
            return mean.detach(), std.detach()
        elif self.impl == 'min_max':
            offset = self.low
            invscale = torch.maximum(1 / self.max, self.high - self.low)
            return offset.detach(), invscale.detach()
        elif self.impl == 'perc_ema':
            offset = self.low
            invscale = torch.maximum(1 / self.max, self.high - self.low)
            return offset.detach(), invscale.detach()
        elif self.impl == 'perc_ema_corr':
            corr = 1 - self.decay ** self.step.to(torch.float32)
            lo = self.low / corr
            hi = self.high / corr
            invscale = torch.maximum(1 / self.max, hi - lo)
            return lo.detach(), invscale.detach()
        elif self.impl == 'mean_mag':
            offset = torch.tensor(0.0).to(dtype=torch.float32, device=self.max.device)
            invscale = torch.maximum(1 / self.max, self.mag)
            return offset.detach(), invscale.detach()
        elif self.impl == 'max_mag':
            offset = torch.tensor(0.0).to(dtype=torch.float32, device=self.max.device)
            invscale = torch.maximum(1 / self.max, self.mag)
            return offset.detach(), invscale.detach()
        else:
            raise NotImplementedError(self.impl)
