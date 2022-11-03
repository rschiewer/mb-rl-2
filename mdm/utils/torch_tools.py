from typing import Tuple, Union, Iterable, List, Sequence, TypeVar
from enum import Enum
from collections import namedtuple, OrderedDict
from functools import reduce, wraps
from math import ceil

import torch
import torch.jit as jit


RnnStateType = TypeVar('RnnStateType', torch.Tensor, Tuple[torch.Tensor, torch.Tensor])


TensorData = TypeVar('TensorData', torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor])
_Placeholder = namedtuple('placeholder', 'device')


class Norm(Enum):
    LAYER = 0
    BATCH = 1


class DeviceMixin:

    def __new__(cls, *args, **kwargs):
        if not issubclass(cls, torch.nn.Module):
            raise RuntimeError(f'The class {cls} can\'t use this mixin, it\'s designed for subclasses of '
                               f'torch.nn.Module only!')
        return super(DeviceMixin, cls).__new__(cls)

    @property
    def device(self: torch.nn.Module):
        #return next(iter(self.parameters())).device  # hack if this turns out to be too much of a bottleneck

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


class FeedforwardBlock(torch.nn.Module, DeviceMixin):

    def __init__(self,
                 *d_inputs: int,
                 lws: Union[Iterable[int], int]):
        super(FeedforwardBlock, self).__init__()

        if type(lws) is int:
            lws = [lws]

        self.d_inputs = d_inputs
        self.lws = (sum(d_inputs), *lws)
        self.layer_list = torch.nn.ModuleList([torch.nn.Linear(lw_in, lw_out)
                                               for lw_in, lw_out in zip(self.lws, self.lws[1:])])
        self.d_output = lws[-1]

    def forward(self,
                *xs: torch.Tensor):
        x = torch.concat(xs, dim=-1)
        for l in self.layer_list[:-1]:
            x = l(x)
            x = torch.nn.functional.gelu(x)
        x = self.layer_list[-1](x)

        return x


class GaussianBlock(FeedforwardBlock):

    def __init__(self,
                 *d_inputs: int,
                 lws: Union[Iterable[int], int],
                 epsilon: float = 0.01):
        if type(lws) is int:
            lws = [lws]
        lws = (*lws[:-1], lws[-1] * 2)  # double last layer to have params for loc and scale

        super(GaussianBlock, self).__init__(*d_inputs, lws=lws)

        self.d_output = lws[-1] // 2
        self.epsilon = epsilon

        min_var = torch.pow(torch.tensor(epsilon, dtype=torch.float32), lws[-1])
        log_min_var = torch.log(min_var)
        if torch.isinf(log_min_var):
            raise ValueError(f'The minimal covariance matrix determinant of a {lws[-1]}d independent gaussian with '
                             f'epsilon={epsilon} is prone to numerical underflow, choose a larger epsilon.')

    def forward(self,
                *xs: torch.Tensor):
        x = super().forward(*xs)
        mu, logvar = torch.tensor_split(x, 2, dim=-1)
        #std = logvar.exp().pow(0.5) + 1.0
        #std = torch.log(1 + logvar.exp()) + 1e-1
        #std = torch.nn.functional.relu(logvar) + 0.01
        #std = torch.distributions.transform_to(torch.distributions.Normal.arg_constraints['scale'])(logvar) + 0.01
        std = torch.abs(logvar) + self.epsilon
        x_dist = torch.distributions.Normal(mu, std)

        return x_dist


class ContinuousBernoulliBlock(FeedforwardBlock):

    def __init__(self,
                 *d_inputs: int,
                 lws: Union[Iterable[int], int] = None):
        super(ContinuousBernoulliBlock, self).__init__(*d_inputs, lws=lws)

    def forward(self,
                *xs: torch.Tensor):
        x = super().forward(*xs)
        x = torch.sigmoid(x)

        x_dist = torch.distributions.ContinuousBernoulli(probs=x)

        return x_dist


def build_categorical(params: torch.Tensor,
                      params_are_probs: bool = False):
    if params_are_probs:
        dist = torch.distributions.OneHotCategorical(probs=params)
    else:
        dist = torch.distributions.OneHotCategorical(logits=params)
    return dist


def sample_from_categorical(params: torch.Tensor,
                            params_are_probs : bool = False,
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
    sigma = torch.exp(0.5 * logvar) + epsilon #torch.nn.functional.relu(logvar) + epsilon
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
    #params = torch.nn.functional.sigmoid(params)
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
    d_batch, d_time, d_data = data.shape
    n_macro_steps = ceil(d_time / k)
    d_padding = n_macro_steps * k - d_time

    if padding_val is None:
        last_valid = (n_macro_steps - 1) * k
        padding_val = torch.mean(data[:, last_valid:], dim=1, keepdim=True, dtype=data.dtype)
        padding = torch.repeat_interleave(padding_val, d_padding, dim=1)
    else:
        padding = torch.full((d_batch, d_padding, d_data), fill_value=padding_val, dtype=data.dtype, device=data.device)
    data_padded = torch.concat([data, padding], dim=1)
    binned = data_padded.reshape(d_batch, (d_time + d_padding) // k, k, d_data)

    #bins = []
    #for i in range(0, d_time, k):
    #    bins.append(data[:, i:i+k])
    #binned2 = torch.stack(bins, dim=1)

    return binned


def layers_with_activation(lws: Sequence[int], activation: str = 'relu', layer_norm: bool = False, name: str = None):
    if activation == 'relu':
        act_constr = torch.nn.ReLU
    elif activation == 'gelu':
        act_constr = torch.nn.GELU
    elif activation == 'elu':
        act_constr = torch.nn.ELU
    elif activation == 'tanh':
        act_constr = torch.nn.Tanh
    elif activation == 'sigmoid':
        act_constr = torch.nn.Sigmoid
    else:
        raise ValueError(f'Unkown activation function: {activation}')

    layers = []
    for w_in, w_out in zip(lws[:-1], lws[1:-1]):
        if layer_norm:
            layers += [torch.nn.Linear(w_in, w_out), torch.nn.LayerNorm(w_out), act_constr()]
        else:
            layers += [torch.nn.Linear(w_in, w_out), act_constr()]
    layers.append(torch.nn.Linear(lws[-2], lws[-1]))

    if name:
        layers = OrderedDict([(f'{name}_{i}', l) for i, l in enumerate(layers)])

    return layers


def add_time_dim(*xs: torch.Tensor,
                 batch_first: bool = True):
    i_unsqueeze = 1 if batch_first else 0
    unsqueezed = [x.unsqueeze(i_unsqueeze) if x.ndim > 1 else x.unsqueeze(0) for x in xs]
    if len(unsqueezed) == 1:
        unsqueezed = unsqueezed[0]
    return unsqueezed


def remove_time_dim(*xs: torch.Tensor,
                    batch_first: bool = True):
    i_unsqueeze = 1 if batch_first else 0
    squeezed = [x.squeeze(i_unsqueeze) for x in xs]
    if len(squeezed) == 1:
        squeezed = squeezed[0]
    return squeezed


def add_data_dim(*xs: torch.Tensor):
    unsqueezed = [x.unsqueeze(-1) for x in xs]
    if len(unsqueezed) == 1:
        unsqueezed = unsqueezed[0]
    return unsqueezed


def make_time_constant(*xs: torch.Tensor,
                       n_timesteps: int,
                       batch_first: bool = True):
    if batch_first:
        consts = [x.unsqueeze(1).expand(x.shape[0], n_timesteps, *x.shape[1:]) for x in xs]
    else:
        consts = [x.unsqueeze(0).expand(n_timesteps, *x.shape) for x in xs]
    if len(consts) == 1:
        consts = consts[0]
    return consts


def extract_sub_distribution(d: torch.distributions.Distribution, *idx: int):
    if len(d.batch_shape) < len(idx):
        raise ValueError(f'Batch size of distribution should be smaller or equal to number of specified indices ',
                         f'but found {len(d.batch_shape)} and {len(idx)}')
    if isinstance(d, torch.distributions.Normal):
        return torch.distributions.Normal(loc=d.loc[idx], scale=d.scale[idx])
    else:
        raise ValueError(f'Distribution class not supported: {type(d)}')


def reconstruction_loss(y_hat: torch.Tensor, y_true: torch.Tensor):
    l = torch.mean((y_hat - y_true) ** 2)
    return l


def kl_loss_normal(priors: List[torch.distributions.Normal],
                   posteriors: List[torch.distributions.Normal],
                   detach_posterior: bool = False):
    l = torch.zeros_like(priors[0].loc)
    for prior, posterior in zip(priors, posteriors):
        if detach_posterior:
            posterior = torch.distributions.Normal(loc=posterior.loc.detach(), scale=posterior.scale.detach())
        l += torch.distributions.kl.kl_divergence(posterior, prior)
    return torch.mean(l)


def kl_loss_bernolli(priors: List[torch.distributions.ContinuousBernoulli],
                     posteriors: List[torch.distributions.ContinuousBernoulli],
                     detach_posterior: bool = False):
    if priors[0].probs is None:
        l = torch.zeros_like(priors[0].logits)
    else:
        l = torch.zeros_like(priors[0].probs)

    for prior, posterior in zip(priors, posteriors):
        if detach_posterior:
            if posterior.probs is None:
                posterior = torch.distributions.ContinuousBernoulli(logits=posterior.logits)
            else:
                posterior = torch.distributions.ContinuousBernoulli(probs=posterior.probs)
        l += torch.distributions.kl.kl_divergence(posterior, prior)
    return torch.mean(l)


def kl_regularizer_normal(priors: List[torch.distributions.Normal]):
    uniform_gauss = torch.distributions.Normal(loc=torch.zeros_like(priors[0].loc, requires_grad=False),
                                               scale=torch.ones_like(priors[0].scale, requires_grad=False))
    l = torch.zeros_like(priors[0].loc)
    for prior in priors:
        l += torch.distributions.kl.kl_divergence(prior, uniform_gauss)
    return torch.mean(l)


def kl_regularizer_bernoulli(priors: List[torch.distributions.ContinuousBernoulli]):
    if priors[0].probs is None:
        uniform_bernoulli = torch.distributions.ContinuousBernoulli(logits=torch.ones_like(priors[0].logits))
        l = torch.zeros_like(priors[0].logits)
    else:
        probs = torch.ones_like(priors[0].probs)
        probs /= probs.sum()
        uniform_bernoulli = torch.distributions.ContinuousBernoulli(probs=probs)
        l = torch.zeros_like(priors[0].probs)

    for prior in priors:
        l += torch.distributions.kl.kl_divergence(prior, uniform_bernoulli)
    return torch.mean(l)


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
