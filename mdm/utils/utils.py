import io
import pickle
import re
import sys
import random
from enum import Enum, auto
from inspect import stack
from itertools import product
from pathlib import Path
from typing import Union, Dict, Tuple, TypeVar, Sequence, List

import gym
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import torch
import yaml
from PIL import Image

from mdm.gridworld.gridworld import Gridworld, CellType
from mdm.memory.trajectory_memory import flatten_and_unsqueeze, TrajectoryMemory

SliceType = TypeVar("SliceType", bound=Sequence)
BasicDtype = TypeVar('BasicDtype', int, float, np.single, np.double, bool)
DataType = TypeVar('DataType', int, float, np.single, np.double, bool, np.ndarray)


class DistributionType(Enum):
    NONE = auto()
    NORMAL = auto()
    CATEGORICAL = auto()


def here() -> Path:
    filename = stack()[1].filename
    parent_path = Path(filename).parent
    return parent_path


def add_to_pythonpath(relative_path: str):
    relative_path = Path(relative_path)
    caller_path = Path(stack()[1].filename).parent
    full_path = caller_path / relative_path
    sys.path.append(full_path)


def load_yaml(path: Union[str, Path]) -> Dict:
    path = Path(path)
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config


hierarchy_sep = '|'
cfg_placeholder = re.compile(r'.*?(<.+?>).*?')
float_pattern = re.compile(r'^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$')
int_pattern = re.compile(r'^[-+]?(0[xX][\dA-Fa-f]+|0[0-7]*|\d+)$')


def fill_placeholders(cfg: dict, _flattened_cfg: dict = None):
    if _flattened_cfg is None:
        _flattened_cfg = pd.json_normalize(cfg, sep=hierarchy_sep).to_dict(orient='records')[0]
    for k, v in cfg.items():
        if isinstance(v, dict):
            # re-build _flattened_cfg in case a placeholer was updated
            _flattened_cfg = pd.json_normalize(cfg, sep=hierarchy_sep).to_dict(orient='records')[0]
            fill_placeholders(v, _flattened_cfg)
        elif isinstance(v, str):
            matches = cfg_placeholder.findall(v)
            for m in matches:
                identifier = m[1:-1]
                insert_value = str(_flattened_cfg[identifier])
                new_value = cfg[k].replace(m, insert_value)
                cfg[k] = new_value
            # cast all numeric strings to their true data type
            if int_pattern.match(cfg[k]):
                cfg[k] = int(cfg[k])
            elif float_pattern.match(cfg[k]):
                cfg[k] = float(cfg[k])


def np_one_hot(x: np.array,
               n_categories: int) -> np.ndarray:
    if not np.issubdtype(x.dtype, np.integer):
        raise ValueError('Only integer arrays can be converted to one-hot encoding')

    # this solution works with masked arrays as well
    x = np.expand_dims(x, -1)
    idx = x.copy()
    x = np.repeat(x, n_categories, -1)
    mask = x.mask if isinstance(x, ma.MaskedArray) else np.zeros_like(x, dtype=bool)
    x[~mask] = 0
    np.put_along_axis(x, idx, 1, axis=-1)

    return x


def exhaustive_traversion(env, mdl, walk_distance):
    # find all possible starting locations
    free_locations = env.find_cell_type(CellType.FREE)
    agent_locations = env.find_cell_type(CellType.AGENT)
    if agent_locations.size == 2:
        agent_locations = agent_locations[np.newaxis, ...]
    free_locations = np.concatenate([free_locations, agent_locations], axis=0)
    free_locations = [tuple(loc) for loc in free_locations]

    # generate all possible action sequences of length walk_distance
    available_actions = list(range(env.action_space.n))
    action_sequences = list(product(available_actions, repeat=walk_distance))

    # find true trajectory for each action sequence
    traj_o, traj_a, traj_r, traj_term = [], [], [], []
    for loc in free_locations:
        for a_seq in action_sequences:
            env.reset()  # don't use first observation since we'll use teleport_agent
            o_mem, r_mem, term_mem = [env.teleport_agent(loc)], [0.0], [False]
            a_mem = (0,) + a_seq
            for a in a_mem[1:]:
                o, r, done, _ = env.step(a)
                o_mem.append(o)
                r_mem.append(r)
                term_mem.append(done)

            o = torch.from_numpy(np.stack(o_mem))
            a = torch.tensor(a_mem)
            r = torch.tensor(r_mem)
            term = torch.tensor(term_mem)

            o, a, r, term = o.to(mdl.device), a.to(mdl.device), r.to(mdl.device), term.to(mdl.device)
            traj_o.append(o)
            traj_a.append(a)
            traj_r.append(r)
            traj_term.append(term)
    traj_o = torch.stack(traj_o)
    traj_a = torch.stack(traj_a)
    traj_r = torch.stack(traj_r)
    traj_term = torch.stack(traj_term)
    return traj_a, traj_o, traj_r, traj_term


def infer_position(env: Gridworld,
                   macro_ss: torch.Tensor,
                   macro_terms: torch.Tensor,
                   macro_ss_list: np.ndarray,
                   loc_list: np.ndarray,
                   act_seq_list: np.ndarray):
    plot_mats = []

    for t in range(len(macro_ss)):
        macro_s = macro_ss[t].detach().cpu().numpy()
        target = np.tile(macro_s, (len(macro_ss_list), 1))
        diffs = np.mean((target - macro_ss_list) ** 2, axis=1)
        i_sorted = np.argsort(diffs)
        intensities = (- diffs[i_sorted] + diffs.max()) / np.abs(diffs.max() - diffs.min())
        loc_list_sorted = loc_list[i_sorted]
        act_sequences = act_seq_list[i_sorted]
        plot_mat = np.zeros((env.grid_h, env.grid_w))
        for loc, intensity in zip(loc_list_sorted, intensities):
            plot_mat[tuple(loc)] = intensity
        plot_mats.append(plot_mat)

    return np.stack(plot_mats)


def gen_video(frames: np.ndarray, interval: int, repeat_delay: int):
    fig, ax = plt.subplots()
    ims = []
    for t, frame in enumerate(frames):
        im = ax.imshow(frame, animated=True)
        label = ax.text(0.01, 0.01, f'{t}', transform=ax.transAxes, color='red')
        ims.append([im, label])
    ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=True, repeat_delay=repeat_delay)
    plt.show()
    return ani


def normalize_obs(obs: Union[torch.Tensor, np.ndarray],
                  env: gym.Env) -> Union[torch.Tensor, np.ndarray]:
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise ValueError('Only environments with box observations space are supported')

    denom = env.observation_space.high.astype(np.float32)
    if isinstance(obs, torch.Tensor):
        denom = torch.from_numpy(denom).to(obs.device)

    return obs / denom


def to_onehot(x: Union[torch.Tensor, np.ndarray],
              n_classes: int) -> Union[torch.Tensor, np.ndarray]:
    if (x % 1 != 0).any():
        raise ValueError('All elements in actions must be ints or castable to int without loss of precision')
    if (x > n_classes).any():
        raise ValueError('Actions contains elements that are larger than n_classes!')

    # remove redundant last dimension if present
    if x.shape[-1] == 1:
        x = x.squeeze(-1)

    if isinstance(x, torch.Tensor):
        if x.ndim <= 1:
            x = x.unsqueeze(-1)
        x = x.to(dtype=torch.int64)
        x = torch.nn.functional.one_hot(x, num_classes=n_classes)
        x = x.to(device=x.device, dtype=torch.float32)
    else:
        x = x.astype(np.int64)
        x = np_one_hot(x, n_classes)

    return x


def fig_to_img(fig, clear_fig: bool = True):
    buffer = io.BytesIO()
    fig.savefig(buffer)
    if clear_fig:
        plt.clf()
    buffer.seek(0)
    return Image.open(buffer)


def prepare_data(o: Union[np.ndarray, torch.Tensor],
                 a: Union[np.ndarray, torch.Tensor],
                 r: Union[np.ndarray, torch.Tensor],
                 terminal: Union[np.ndarray, torch.Tensor],
                 truncated: Union[np.ndarray, torch.Tensor],
                 mask: Union[np.ndarray, torch.Tensor],
                 subtrajectory_len: int = 0,
                 a_discrete: bool = False,
                 a_max: int = 0,
                 o_discrete: bool = False,
                 o_max: int = 0,
                 swap_batch_time_dim: bool = False
                 ) -> Tuple[Union[torch.tensor, np.ndarray], Union[torch.tensor, np.ndarray],
                            Union[torch.tensor, np.ndarray], Union[torch.tensor, np.ndarray],
                            Union[torch.tensor, np.ndarray], Union[torch.tensor, np.ndarray]]:
    assert o.ndim >= 2, f'Observation dim is {o.ndim}, but needs to be at least 2'
    assert a.ndim >= 2, f'Action dim is {a.ndim}, but needs to be at least 2'
    assert 2 <= r.ndim <= 3, f'Reward dim is {r.ndim}, but needs to be 2 or 3'
    assert 2 <= terminal.ndim <= 3, f'Terminal dim is {terminal.ndim}, but needs to be 2 or 3'
    assert 2 <= truncated.ndim <= 3, f'Truncated dim is {truncated.ndim}, but needs to be 2 or 3'
    assert 2 <= mask.ndim <= 3, f'Mask dim is {mask.ndim}, but needs to be 2 or 3'

    # expand data dimensions to at least one, so e.g. rewards have shape (batch, time, 1)
    if o.ndim == 2: o = o.unsqueeze(-1)
    if a.ndim == 2: a = a.unsqueeze(-1)
    if r.ndim == 2: r = r.unsqueeze(-1)
    if terminal.ndim == 2: terminal = terminal.unsqueeze(-1)
    if truncated.ndim == 2: truncated = truncated.unsqueeze(-1)
    if mask.ndim == 2: mask = mask.unsqueeze(-1)

    if o_discrete:
        o = to_onehot(o, o_max)
    if a_discrete:
        a = to_onehot(a, a_max)

    # swap batch and time axis
    if swap_batch_time_dim:
        o = o.swapaxes(0, 1)
        a = a.swapaxes(0, 1)
        r = r.swapaxes(0, 1)
        terminal = terminal.swapaxes(0, 1)
        truncated = truncated.swapaxes(0, 1)
        mask = mask.swapaxes(0, 1)

    if subtrajectory_len > 0:
        # select a block of l_segment timesteps out of all trajectories
        l_data = r.shape[0]
        t_start = random.randint(0, l_data - subtrajectory_len)
        t_end = t_start + subtrajectory_len
        o = o[t_start:t_end]
        a = a[t_start:t_end]
        r = r[t_start:t_end]
        terminal = terminal[t_start:t_end]
        truncated = truncated[t_start:t_end]
        mask = mask[t_start:t_end]

    return o, a, r, terminal, truncated, mask


def prepare_data_gridworld(s: Union[np.ndarray, torch.Tensor],
                           a: Union[np.ndarray, torch.Tensor],
                           r: Union[np.ndarray, torch.Tensor],
                           terminal: Union[np.ndarray, torch.Tensor],
                           truncated: Union[np.ndarray, torch.Tensor],
                           mask: Union[np.ndarray, torch.Tensor],
                           env: Gridworld,
                           subtrajectory_len: int = 0,
                           ) -> Tuple[Union[torch.tensor, np.ndarray], Union[torch.tensor, np.ndarray],
                            Union[torch.tensor, np.ndarray], Union[torch.tensor, np.ndarray],
                            Union[torch.tensor, np.ndarray], Union[torch.tensor, np.ndarray]]:
    s, a, r, terminal, truncated, mask = flatten_and_unsqueeze(s, a, r, terminal, truncated, mask)
    s = to_onehot(s, max(env.grid_w, env.grid_h))
    #s = normalize_obs(s, env)
    a = to_onehot(a, env.action_space.n)  # don't care about the action being [1, 0, ... ] if it's always this way

    # swap batch and time axis
    s = s.swapaxes(0, 1)
    a = a.swapaxes(0, 1)
    r = r.swapaxes(0, 1)
    terminal = terminal.swapaxes(0, 1)
    truncated = truncated.swapaxes(0, 1)
    mask = mask.swapaxes(0, 1)

    #s.masked_fill(~mask.to(torch.bool).unsqueeze(-1), 0)
    #a.masked_fill(~mask.to(torch.bool), 0)
    #r.masked_fill(~mask.to(torch.bool), 0)
    #terminal.masked_fill(~mask.to(torch.bool), 0)
    #truncated.masked_fill(~mask.to(torch.bool), 0)
    #mask[:] = 0

    if subtrajectory_len > 0:
        # select a block of l_segment timesteps out of all trajectories
        l_data = r.shape[0]
        t_start = random.randint(0, l_data - subtrajectory_len)
        t_end = t_start + subtrajectory_len
        s = s[t_start:t_end]
        a = a[t_start:t_end]
        r = r[t_start:t_end]
        terminal = terminal[t_start:t_end]
        truncated = truncated[t_start:t_end]
        mask = mask[t_start:t_end]

    return s, a, r, terminal, truncated, mask


def trajectory_uncertainty(mem: [Dict[str, List[torch.Tensor]]]):
    # get distributions
    z_dists = []
    for z_prior, z_post in zip(mem['prim_z_prior'], mem['prim_z_post']):
        z_dist = z_prior if z_post is None else z_post
        z_dists.append(z_dist)
    r_dists = mem['prim_r_dist']

    # get variances
    z_var = [d.scale for d in z_dists]
    z_var = torch.stack(z_var)
    #z_var = z_var.mean(axis=1)
    r_var = [d.scale for d in r_dists]
    r_var = torch.stack(r_var)
    #r_var = r_var.mean(axis=1)

    return z_var, r_var
    #return z_var.max(), z_var.std(), r_var.max(), r_var.std()

    fig, ax = plt.subplots(1, 2, figsize=(16, 10))
    for i in range(z_var.shape[-1]):
        ax[0].plot(z_var[:, i], label=f'avg z variance dim {i + 1}')
    ax[1].plot(r_var, label='avg r variance')
    plt.legend()
    plt.show()


def augment_train_data_random(o: torch.Tensor,
                              a: torch.Tensor,
                              r: torch.Tensor,
                              term: torch.Tensor,
                              trunc: torch.Tensor,
                              mask: torch.Tensor,
                              n_augment: int,
                              r_pessimistic: float = -10):
    o_aug = torch.randint(o.shape[-1], o[:, :n_augment].shape[:-1]).to(o.device)
    o_aug = torch.nn.functional.one_hot(o_aug, o.shape[-1])
    a_aug = torch.randint(a.shape[-1], a[:, :n_augment].shape[:-1]).to(a.device)
    a_aug = torch.nn.functional.one_hot(a_aug, a.shape[-1])
    r_aug = torch.full_like(r[:, :n_augment], r_pessimistic)
    term_aug = torch.randint_like(term[:, :n_augment], 0, 2).to(torch.float32)
    trunc_aug = torch.randint_like(trunc[:, :n_augment], 0, 2).to(torch.float32)
    mask_aug = torch.ones_like(mask[:, :n_augment])

    o = torch.concat([o, o_aug], dim=1)
    a = torch.concat([a, a_aug], dim=1)
    r = torch.concat([r, r_aug], dim=1)
    term = torch.concat([term, term_aug], dim=1)
    trunc = torch.concat([trunc, trunc_aug], dim=1)
    mask = torch.concat([mask, mask_aug], dim=1)

    return o, a, r, term, trunc, mask


def augment_data_trajectory_ends(o: torch.Tensor,
                                 a: torch.Tensor,
                                 r: torch.Tensor,
                                 term: torch.Tensor,
                                 trunc: torch.Tensor,
                                 mask: torch.Tensor,
                                 n_augment: int):
    raise NotImplementedError('not yet done')
    n_trajectories = o.shape[1]

    o_aug = torch.clone(o)
    a_aug = torch.clone(a)
    r_aug = torch.clone(r)
    term_aug = torch.clone(term)
    trunc_aug = torch.clone(trunc)
    mask_aug = torch.clone(mask)

    t_end = mask_aug.sum(dim=0)
    t_start_aug = []
    for t in t_end:
        t_start_aug.append(random.randint(0, t-1))

    for i_traj, t in enumerate(t_start_aug):
        o_aug[:, i_traj, t_start_aug:] = 0


def to_np_arrays(mem: List[Dict[str, DataType]], dtypes: Sequence = None, padding: Sequence = None):
    if dtypes is None:
        dtypes = (float, float, float, float, float)
    if padding is None:
        padding = (0.0, 0.0, 0.0, 0.0, 0.0)
    n_trajectories = len(mem)

    # find out shapes
    s_o = mem[0]['o'].shape[1:]
    s_a = mem[0]['a'].shape[1:]

    # collect data
    o, a, r, term, trunc, lengths = [], [], [], [], [], []
    for traj in mem:
        o.append(traj['o'])
        a.append(traj['a'])
        r.append(traj['r'])
        term.append(traj['terminal'])
        trunc.append(traj['truncated'])
        lengths.append(len(traj['o']))
    longest = max(lengths)

    # prepare memory containers
    o_np = np.full((n_trajectories, longest, *s_o), fill_value=padding[0], dtype=dtypes[0])
    a_np = np.full((n_trajectories, longest, *s_a), fill_value=padding[1], dtype=dtypes[1])
    r_np = np.full((n_trajectories, longest), fill_value=padding[1], dtype=dtypes[2])
    term_np = np.full((n_trajectories, longest), fill_value=padding[1], dtype=dtypes[3])
    trunc_np = np.full((n_trajectories, longest), fill_value=padding[1], dtype=dtypes[4])
    #o_mask = np.full_like(o_np, True)
    #a_mask = np.full_like(a_np, True)
    mask = np.full_like(r_np, True)

    # copy data
    for i in range(n_trajectories):
        o_np[i, 0:lengths[i]] = o[i]
        a_np[i, 0:lengths[i]] = a[i]
        r_np[i, 0:lengths[i]] = r[i]
        term_np[i, 0:lengths[i]] = term[i]
        trunc_np[i, 0:lengths[i]] = trunc[i]
        #o_mask[i, 0:lengths[i]] = False
        #a_mask[i, 0:lengths[i]] = False
        mask[i, 0:lengths[i]] = False

    # generate masked arrays
    #o_np = ma.array(o_np, mask=o_mask)
    #a_np = ma.array(a_np, mask=a_mask)
    #r_np = ma.array(r_np, mask=mask)
    #term_np = ma.array(term_np, mask=mask)
    #trunc_np = ma.array(trunc_np, mask=mask)

    return o_np, a_np, r_np, term_np, trunc_np, mask


def load_memory(path: Union[str, Path]) -> List[Dict[str, DataType]]:
    with open(path, 'rb') as f:
        mem = pickle.load(f)
    return mem


def store_memory(mem: List[Dict[str, DataType]],
                 path: Union[str, Path]) -> bool:
    with open(path, 'wb') as f:
        pickle.dump(mem, f)
    return True


def compute_returns(mem: TrajectoryMemory, gamma: float = 0.99):
    s, a, r, term, w = mem.to_np_arrays()
    disc_mat = np.cumprod(np.full_like(r.data, fill_value=gamma), axis=1)
    disc_mat = np.roll(disc_mat, 1, axis=1)
    disc_mat[:, 0] = 1
    ep_returns = np.sum(r * disc_mat, axis=1)
    for t, R in zip(mem, ep_returns):
        t['w'] = R
    mem.mark_modified()


def discrete_stats(module: torch.nn.Module, n_inputs: int, seq_len: int, n_repetitions: int = 1, module_kwargs: dict = None):
    if not module_kwargs:
        module_kwargs = {}

    device = next(module.parameters()).device

    # generate all permutations of possible inputs
    available_inputs = list(range(n_inputs))
    input_sequences = list(product(available_inputs, repeat=seq_len))
    input_sequences = torch.tensor(input_sequences).to(device)
    input_sequences = to_onehot(input_sequences, n_inputs)
    n_unique_sequences = len(input_sequences)
    input_sequences = input_sequences.repeat(n_repetitions, 1, 1)

    # query the model
    Y = module(input_sequences, **module_kwargs)
    d_out = Y.shape[-1]
    Y = Y.reshape(n_repetitions, n_unique_sequences, d_out)

    # per sequence mean and std
    Y_mean = torch.mean(Y, dim=0)
    Y_std = torch.std(Y, dim=0)

    # similarity matrix
    # note: torch tensors are row-major
    diff = torch.repeat_interleave(Y_mean, len(Y_mean), dim=0) - Y_mean.repeat(len(Y_mean), 1)
    diff = torch.sum(diff ** 2, dim=1)
    Y_mae = diff.reshape(n_unique_sequences, n_unique_sequences)

    # total mean and std
    Y_mean = torch.mean(Y)
    Y_std = torch.std(Y)

    Y_mean = Y_mean.detach().cpu().numpy()
    Y_std = Y_std.detach().cpu().numpy()
    Y_mae = Y_mae.detach().cpu().numpy()

    return Y_mean, Y_std, Y_mae


def sensitivity_analysis(module: torch.nn.Module,
                         const_input: torch.Tensor,
                         input_mask: torch.Tensor,
                         n_runs: int = 1,
                         random_range_low: torch.Tensor = -1.0,
                         random_range_high: torch.Tensor = 1.0):
    assert const_input.shape == input_mask.shape, 'const_input should have the same shape as input_mask'
    assert torch.all(torch.logical_or(input_mask == 0, input_mask == 1))

    dims = [n_runs] + [1 for _ in range(const_input.ndim)]
    const_input = torch.tile(const_input, dims=dims)
    input_mask = torch.tile(input_mask, dims=dims)

    rand_tens = torch.rand_like(const_input) * (random_range_high - random_range_low) + random_range_low
    input = torch.where(input_mask, const_input, rand_tens)
    output = module(input)

    if isinstance(output, dict):
        ret_mean = {k: v.mean(dims=0) for k, v in output.items()}
        ret_std = {k: v.std(dims=0) for k, v in output.items()}
    elif isinstance(output, tuple):
        ret_mean = [x.mean(dims=0) for x in output]
        ret_std = [x.std(dims=0) for x in output]
    else:
        ret_mean = torch.mean(output, dim=0)
        ret_std = torch.std(output, dim=0, unbiased=False)

    return ret_mean, ret_std


def random_walk_success_rate(env: gym.Env,
                             n_steps: int,
                             n_tries: int):
    success = 0
    ep_returns = []
    ep_lens = []
    for i_run in range(n_tries):
        env.reset()
        r_ep, l_ep = 0, 0
        for i_step in range(n_steps):
            a = env.action_space.sample()
            o, r, term, trunc, info = env.step(a)
            l_ep += 1
            r_ep += r
            if term:
                success += 1
                break
            if trunc:
                break
        ep_returns.append(r_ep)
        ep_lens.append(l_ep)
    success /= n_tries
    ep_returns = np.array(ep_returns)
    ep_lens = np.array(ep_lens)
    return success, ep_returns, ep_lens


