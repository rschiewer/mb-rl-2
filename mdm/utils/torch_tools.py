from typing import Tuple, Union, List, Sequence, TypeVar, Dict
from collections import namedtuple, OrderedDict
from functools import reduce, wraps
from math import ceil
from sys import gettrace
import os

import torch
import numpy as np

RnnStateType = TypeVar('RnnStateType', torch.Tensor, Tuple[torch.Tensor, torch.Tensor])
_Placeholder = namedtuple('placeholder', 'device')

# define torch.compile decorator depending on whether we're in debug mode or not
if gettrace() or 'PYCHARM_HOSTED' in os.environ:
    print('Debugging or running in PyCharm IDE, disabling torch.compile')


    def compile_if_not_debug(func):
        return func
else:
    print('Compiling functions with compile_if_not_debug decorator')
    torch.set_float32_matmul_precision('high')
    compile_if_not_debug = torch.compile


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
    elif descr == 'tanh':
        act_constr = torch.nn.Tanh
    elif descr == 'sigmoid':
        act_constr = torch.nn.Sigmoid
    else:
        raise ValueError(f'Unkown descr function: {descr}')
    return act_constr


def layers_with_activation(lws: Sequence[int], activation: str = 'relu', layer_norm: bool = False, name: str = None,
                           final_activation_function: str = None):
    assert len(lws) >= 2, f'Need at least w_in and w_out for one layer, but lws contains less than 2 elements'

    act_constr = _get_act_fn(activation)
    layers = []
    for w_in, w_out in zip(lws[:-1], lws[1:-1]):
        if layer_norm:
            layers += [torch.nn.Linear(w_in, w_out), torch.nn.LayerNorm(w_out), act_constr()]
        else:
            layers += [torch.nn.Linear(w_in, w_out), act_constr()]
    layers.append(torch.nn.Linear(lws[-2], lws[-1]))

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
            layers += [torch.nn.Conv2d(in_channels=w_in, out_channels=w_out, kernel_size=ks, stride=st),
                       torch.nn.LayerNorm(w_out), act_constr()]
        else:
            layers += [torch.nn.Conv2d(in_channels=w_in, out_channels=w_out, kernel_size=ks, stride=st), act_constr()]
    layers.append(torch.nn.Conv2d(in_channels=channels[-2], out_channels=channels[-1], kernel_size=kernel_sizes[-1],
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
    elif isinstance(d, torch.distributions.RelaxedOneHotCategorical):
        return type(d)(temperature=d.temperature.detach(), logits=d.logits.detach())
    elif isinstance(d, torch.distributions.ContinuousBernoulli):
        return type(d)(probs=d.probs.detach())
    elif isinstance(d, torch.distributions.Bernoulli):
        return type(d)(probs=d.probs.detach())
    elif isinstance(d, torch.distributions.Independent):
        return torch.distributions.Independent(detach_dist(d.base_dist),
                                               reinterpreted_batch_ndims=d.reinterpreted_batch_ndims)
    # elif hasattr(d, 'logits'):
    #    return type(d)(logits=d.logits.detach())
    # elif hasattr(d, 'probs'):
    #    return type(d)(probs=d.probs.detach())
    else:
        raise RuntimeError(f'Can\'t detach the given distribution: {d}')


@compile_if_not_debug
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
    else:
        raise ValueError(f'Unsupported distribution class: {cls}')


@compile_if_not_debug
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
               padding: Union[List, Tuple] = None):
    if len(mem) > 1:
        fids = reduce(lambda a, b: set(a) | set(b), mem)
    else:
        fids = set(mem[0])
    assert 'mask' not in fids, 'found forbidden field id "mask" in mem'
    if dtypes is None: dtypes = [torch.float32 for _ in range(len(fids))]
    if padding is None: padding = [0.0 for _ in range(len(fids))]
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

    # mask for longest field per trajectory (valid for o, a, r, term, trunc but not for higher level a)
    cont['mask'] = torch.full((max(longest.values()), n_traj), fill_value=True, dtype=torch.float32, device=device)
    for i in range(n_traj):
        longest_field = max([x[i] for x in lengths.values()])
        cont['mask'][0:longest_field, i] = False

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
            np_data_dict[k] = v.detach().cpu().numpy()
        else:
            raise ValueError(f'Unsupported type: {type(k)}')
    return np_data_dict
