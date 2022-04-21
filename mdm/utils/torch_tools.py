from typing import Tuple, Union, Iterable, List
from enum import Enum
from collections import namedtuple
from functools import reduce

import torch


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

    def gen_h_placeholder(self, d_batch: int):
        d = self.device
        return (torch.zeros(self.n_layers, d_batch, self.d_hidden, device=d),
                torch.zeros(self.n_layers, d_batch, self.d_hidden, device=d))


class GaussianBlock(torch.nn.Module, DeviceMixin):

    def __init__(self,
                 *d_inputs: int,
                 lws: Union[Iterable[int], int]):
        super(GaussianBlock, self).__init__()

        if type(lws) is int:
            lws = [lws]

        self.d_inputs = d_inputs
        self.lws = (sum(d_inputs), *lws[:-1], lws[-1] * 2)  # double last layer width to have params for loc and scale
        self.layer_list = torch.nn.ModuleList([torch.nn.Linear(lw_in, lw_out)
                                               for lw_in, lw_out in zip(self.lws, self.lws[1:])])

    def forward(self,
                *xs: torch.Tensor):
        x = torch.concat(xs, dim=-1)
        for l in self.layer_list[:-1]:  # only activations on inner layers
            x = l(x)
            x = torch.nn.functional.gelu(x)
        x = self.layer_list[-1](x)  # no activation on last layer

        mu, logvar = torch.tensor_split(x, 2, dim=-1)
        #std = logvar.exp().pow(0.5) + 1.0
        #std = torch.log(1 + logvar.exp()) + 1e-1
        #std = torch.nn.functional.relu(logvar) + 0.01
        #std = torch.distributions.transform_to(torch.distributions.Normal.arg_constraints['scale'])(logvar) + 0.01
        std = torch.abs(logvar) + 0.01
        x_dist = torch.distributions.Normal(mu, std)

        return x_dist


class FeedforwardBlock(torch.nn.Module, DeviceMixin):

    def __init__(self,
                 *d_inputs: int,
                 lws: Union[Iterable[int], int] = None):
        super(FeedforwardBlock, self).__init__()

        if lws is None:
            lws = []
        elif type(lws) is int:
            lws = [lws]

        self.d_inputs = d_inputs
        self.lws = (sum(d_inputs), *lws)
        self.layer_list = torch.nn.ModuleList([torch.nn.Linear(lw_in, lw_out)
                                               for lw_in, lw_out in zip(self.lws, self.lws[1:])])

    def forward(self,
                *xs: torch.Tensor):
        x = torch.concat(xs, dim=-1)
        for l in self.layer_list[:-1]:
            x = l(x)
            x = torch.nn.functional.gelu(x)
        x = self.layer_list[-1](x)

        return x


class ContinuousBernoulliBlock(FeedforwardBlock):

    def __init__(self,
                 *d_inputs: int,
                 lws: Union[Iterable[int], int] = None):
        super(ContinuousBernoulliBlock, self).__init__(*d_inputs, lws=lws)

    def forward(self,
                *xs: torch.Tensor):
        x = torch.concat(xs, dim=-1)
        for l in self.layer_list[:-1]:
            x = l(x)
            x = torch.nn.functional.gelu(x)
        x = self.layer_list[-1](x)
        x = torch.nn.functional.softmax(x)

        x_dist = torch.distributions.ContinuousBernoulli(probs=x)

        return x_dist


def add_time_dim(*xs: torch.Tensor,
                 batch_first: bool = True):
    i_unsqueeze = 1 if batch_first else 0
    unsqueezed = [x.unsqueeze(i_unsqueeze) for x in xs]
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
