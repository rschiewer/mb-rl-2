from typing import Tuple, Union, Iterable, List, Sequence, TypeVar, Dict
from enum import Enum
from collections import namedtuple, OrderedDict
from functools import reduce, wraps
from math import ceil

import torch
import torch.jit as jit

from mdm.utils.utils import SliceType, DataType


RnnStateType = TypeVar('RnnStateType', torch.Tensor, Tuple[torch.Tensor, torch.Tensor])
TensorData = TypeVar('TensorData', torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor])
TensorIndex = TypeVar('TensorIndex', int, SliceType, torch.Tensor)
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
    d_time, d_batch = data.shape[:2]
    d_data = data.shape[2:]
    n_macro_steps = ceil(d_time / k)
    d_padding = n_macro_steps * k - d_time

    if padding_val is None:
        padding = torch.repeat_interleave(data[-1, None], d_padding, dim=0)
    else:
        padding = torch.full((d_padding, d_batch, *d_data), fill_value=padding_val, dtype=data.dtype, device=data.device)
    data_padded = torch.concat([data, padding], dim=0)
    binned = data_padded.reshape((d_time + d_padding) // k, k, d_batch, *d_data)

    #bins = []
    #for i in range(0, d_time, k):
    #    bins.append(data[:, i:i+k])
    #binned2 = torch.stack(bins, dim=1)

    return binned


def layers_with_activation(lws: Sequence[int], activation: str = 'relu', layer_norm: bool = False, name: str = None,
                           final_activation_function: str = None):
    assert len(lws) >= 2, f'Need at least w_in and w_out for one layer, but lws contains less than 2 elements'

    def get_act_fn(descr: str):
        if descr == 'relu':
            act_constr = torch.nn.ReLU
        elif descr == 'gelu':
            act_constr = torch.nn.GELU
        elif descr == 'elu':
            act_constr = torch.nn.ELU
        elif descr == 'tanh':
            act_constr = torch.nn.Tanh
        elif descr == 'sigmoid':
            act_constr = torch.nn.Sigmoid
        else:
            raise ValueError(f'Unkown descr function: {descr}')
        return act_constr

    act_constr = get_act_fn(activation)

    layers = []
    for w_in, w_out in zip(lws[:-1], lws[1:-1]):
        if layer_norm:
            layers += [torch.nn.Linear(w_in, w_out), torch.nn.LayerNorm(w_out), act_constr()]
        else:
            layers += [torch.nn.Linear(w_in, w_out), act_constr()]
    layers.append(torch.nn.Linear(lws[-2], lws[-1]))

    if final_activation_function:
        final_act_constr = get_act_fn(final_activation_function)
        layers.append(final_act_constr())

    if name:
        layers = OrderedDict([(f'{name}_{i}', l) for i, l in enumerate(layers)])

    return layers


def get_dist_params(d: torch.distributions.Distribution):
    if isinstance(d, (torch.distributions.Normal, torch.distributions.Cauchy, torch.distributions.Gumbel,
                      torch.distributions.Laplace, torch.distributions.LogNormal)):
        return d.loc, d.scale
    elif isinstance(d, torch.distributions.RelaxedOneHotCategorical):
        return d.logits, d.temperature
    elif hasattr(d, 'logits'):
        return (d.logits,)
    elif hasattr(d, 'probs'):
        return (d.probs,)
    else:
        raise RuntimeError(f'Can\'t extract parameters of the given distribution: {d}')


def detach_dist(d: torch.distributions.Distribution):
    if isinstance(d, (torch.distributions.Normal, torch.distributions.Cauchy, torch.distributions.Gumbel,
                      torch.distributions.Laplace, torch.distributions.LogNormal)):
        return type(d)(loc=d.loc.detach(), scale=d.scale.detach())
    elif isinstance(d, torch.distributions.RelaxedOneHotCategorical):
        return type(d)(temperature=d.temperature.detach(), logits=d.logits.detach())
    elif hasattr(d, 'logits'):
        return type(d)(logits=d.logits.detach())
    elif hasattr(d, 'probs'):
        return type(d)(probs=d.probs.detach())
    else:
        raise RuntimeError(f'Can\'t detach the given distribution: {d}')


def add_time_dim(*xs: torch.Tensor,
                 batch_first: bool = False):
    i_unsqueeze = 1 if batch_first else 0
    unsqueezed = [x.unsqueeze(i_unsqueeze) if x.ndim > 1 else x.unsqueeze(0) for x in xs]
    if len(unsqueezed) == 1:
        unsqueezed = unsqueezed[0]
    return unsqueezed


def remove_time_dim(*xs: torch.Tensor,
                    batch_first: bool = False):
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


def extract_sub_distribution(d: torch.distributions.Distribution,
                             *idx: TensorIndex,
                             keepdim: bool = False):
    if len(d.batch_shape) < len(idx):
        raise ValueError(f'Batch size of distribution should be smaller or equal to number of specified indices ',
                         f'but found {len(d.batch_shape)} and {len(idx)}')
    if keepdim:  # make single int indices to slices of length 1 to prevent loss of dimension
        tmp = []
        for i in idx:
            if type(i) is int:
                tmp.append(slice(i, i+1))
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


def to_tensors(mem: List[Dict[str, DataType]],
               device: torch.device,
               dtypes: Sequence = None,
               padding: Sequence = None):
    if dtypes is None:
        dtypes = (torch.float32, torch.float32, torch.float32, torch.float32, torch.float32)
    if padding is None:
        padding = (0.0, 0.0, 0.0, 0.0, 0.0)
    n_trajectories = len(mem)

    # find out shapes
    s_o = mem[0]['o'].shape[1:]
    s_a = mem[0]['a'].shape[1:]

    # collect data
    o, a, r, term, trunc, lengths = [], [], [], [], [], []
    for traj in mem:
        o.append(torch.from_numpy(traj['o']))
        a.append(torch.from_numpy(traj['a']))
        r.append(torch.from_numpy(traj['r']))
        term.append(torch.from_numpy(traj['terminal']))
        trunc.append(torch.from_numpy(traj['truncated']))
        lengths.append(len(traj['o']))
    longest = max(lengths)

    # prepare memory containers
    o_torch = torch.full((n_trajectories, longest, *s_o), fill_value=padding[0], dtype=dtypes[0], device=device)
    a_torch = torch.full((n_trajectories, longest, *s_a), fill_value=padding[1], dtype=dtypes[1], device=device)
    r_torch = torch.full((n_trajectories, longest), fill_value=padding[1], dtype=dtypes[2], device=device)
    term_torch = torch.full((n_trajectories, longest), fill_value=padding[1], dtype=dtypes[3], device=device)
    trunc_torch = torch.full((n_trajectories, longest), fill_value=padding[1], dtype=dtypes[4], device=device)
    mask = torch.full_like(r_torch, True)

    # copy data
    for i in range(n_trajectories):
        o_torch[i, 0:lengths[i]] = o[i]
        a_torch[i, 0:lengths[i]] = a[i]
        r_torch[i, 0:lengths[i]] = r[i]
        term_torch[i, 0:lengths[i]] = term[i]
        trunc_torch[i, 0:lengths[i]] = trunc[i]
        mask[i, 0:lengths[i]] = False

    return o_torch, a_torch, r_torch, term_torch, trunc_torch, mask

