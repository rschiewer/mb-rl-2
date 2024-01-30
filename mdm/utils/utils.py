from __future__ import annotations

import io
import os
import pickle
import re
import sys
import random
import time
from inspect import stack
from pathlib import Path
from typing import Any, List
from functools import reduce

import gymnasium as gym
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import minigrid.minigrid_env
import numpy.ma as ma
import pandas as pd
import torch
import yaml
from PIL import Image

from mdm.gridworld.gridworld import Gridworld, CellType
from mdm.memory.trajectory_memory import flatten_and_unsqueeze
from mdm.models.building_blocks import *
from mdm.logging.logger import Logger, Scope
from mdm.models.rssm_cell import rssm_state_seq_to_batch, rssm_state_keys, RSSMStateType
from mdm.utils.torch_tools import unsqueeze_right, stack_if_list

DataType = TypeVar('DataType', int, float, np.single, np.double, bool, np.ndarray)


class InMemoryFile:

    def __init__(self,
                 resource: str | Path | io.BytesIO | io.FileIO,
                 name: str = '',
                 extension: str = ''):
        if isinstance(resource, (str, Path)):
            resource = str(resource)
            name_start = resource.rindex('/') + 1 if '/' in resource else 0
            ext_start = resource.rindex('.') + 1 if '.' in resource else len(resource)
            self.name = resource[name_start:ext_start].rstrip('.') if len(name) == 0 else name
            self.extension = resource[ext_start:] if len(extension) == 0 else extension
            with open(resource, 'rb') as f:
                self.buffer = io.BytesIO(f.read())
        elif isinstance(resource(io.BytesIO, io.FileIO)):
            assert extension is not None, f'File extension required for buffers'
            self.buffer = resource  # don't copy buffer here
            self.name = name
            self.extension = extension
        else:
            raise ValueError(f'Unknown resource: {resource}')

        self.name = self.name.strip()
        self.extension = self.extension.strip()

    @property
    def full_name(self):
        full_name = self.name
        if self.extension:
            full_name += f'.{self.extension}'
        return full_name

    @staticmethod
    def consume_file(path: str | Path,
                     new_name: str = '',
                     new_extension: str = ''):
        in_memory_file = InMemoryFile(path, new_name, new_extension)
        os.remove(path)
        return in_memory_file


class TempFigure:

    def __init__(self,
                 **kwargs):
        self.fig = plt.figure(**kwargs)

    def __enter__(self):
        return self.fig

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fig.clear()
        plt.close(self.fig)
        del self.fig


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


def fig_to_img(fig: plt.Figure,
               clear_fig: bool = True,
               minimize_size: bool = True,
               **fig_kwargs):
    buffer = io.BytesIO()
    fig.savefig(buffer, bbox_inches='tight', **fig_kwargs)
    # print(buffer.tell() / 1024)
    if clear_fig:
        plt.clf()
    buffer.seek(0)
    img = Image.open(buffer)
    if minimize_size:
        opt_buffer = io.BytesIO()
        img.save(opt_buffer, format='GIF', optimize=True)
        # print(opt_buffer.tell() / 1024)
        opt_buffer.seek(0)
        img = Image.open(opt_buffer)
    return img


def anim_to_gif(anim: animation.Animation,
                fps: int = 10):
    timestamp = time.time_ns()
    pid = os.getpid()
    tmp_file_name = f'.{pid}_{timestamp}_gif_anim.gif'
    anim.save(tmp_file_name, writer='pillow', fps=fps)

    with open(tmp_file_name, 'rb') as f:
        img = Image.open(f)
        img.load()
    os.remove(tmp_file_name)

    return img


def anim_to_vid(anim: animation.Animation,
                fps: int = 10,
                dpi: int = 50):
    timestamp = time.time_ns()
    pid = os.getpid()
    tmp_file_name = f'.{pid}_{timestamp}_video_anim.mp4'
    extra_args = ['-vcodec', 'libx264', '-pix_fmt', 'yuv420p']
    writer = animation.writers['ffmpeg'](fps=fps, extra_args=extra_args)
    anim.save(tmp_file_name, writer=writer, dpi=dpi)
    # print(os.path.getsize(tmp_file_name) / 1024 )

    # old version
    # anim.save(tmp_file_name, writer='ffmpeg', fps=fps, dpi=dpi)

    f = InMemoryFile.consume_file(tmp_file_name)

    return f


def join_trajectories(t1: Dict[str, np.ndarray], t2: Dict[str, np.ndarray]):
    joined = {}
    for k, v in t1.items():
        assert k in t2
        joined[k] = v + t2[k]
    return joined


def trajectories_from_simulation(model_mem: Dict[str, List[torch.Tensor]],
                                 model: 'HierarchicalRSSM' = None,
                                 level: int = 0):
    n_trajs = model_mem['o'][0].shape[0]
    if isinstance(model_mem['o'], list):  # time dimension is list
        trajs = {k: torch.stack(model_mem[k]) for k in ('o', 'a', 'r', 'terminal')}
    else:  # time dimension is tensor
        trajs = {k: model_mem[k] for k in ('o', 'a', 'r', 'terminal')}

    if level > 0:
        reconstr_goal = model.rssm_modules[level - 1].decode(trajs['o'], sample=False, reconstruct_observation=True)
        trajs['o'] = reconstr_goal['o']
        # to make reconstructed traj fit the original traj
        n_repeat = model.strides[level]
        trajs = {k: torch.repeat_interleave(v, n_repeat, dim=0) for k, v in trajs.items()}

    trajs = [{k: v[:, i].detach().cpu().numpy() for k, v in trajs.items()} for i in range(n_trajs)]
    return trajs


def prepare_data_old(o: Union[np.ndarray, torch.Tensor],
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
                     swap_batch_time_dim: bool = False):
    assert o.ndim >= 2, f'Observation dim is {o.ndim}, but needs to be at least 2'
    assert a.ndim >= 2, f'Action dim is {a.ndim}, but needs to be at least 2'
    assert 2 <= r.ndim <= 3, f'Reward dim is {r.ndim}, but needs to be 2 or 3'
    assert 2 <= terminal.ndim <= 3, f'Terminal dim is {terminal.ndim}, but needs to be 2 or 3'
    assert 2 <= truncated.ndim <= 3, f'Truncated dim is {truncated.ndim}, but needs to be 2 or 3'
    assert 2 <= mask.ndim <= 3, f'Mask dim is {mask.ndim}, but needs to be 2 or 3'

    # expand data dimensions to at least one, so e.g. rewards have shape (batch, time, 1)
    if o.ndim == 2: o = o[..., None]
    if a.ndim == 2: a = a[..., None]
    if r.ndim == 2: r = r[..., None]
    if terminal.ndim == 2: terminal = terminal[..., None]
    if truncated.ndim == 2: truncated = truncated[..., None]
    if mask.ndim == 2: mask = mask[..., None]

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

    return {'o': o, 'a': a, 'r': r, 'terminal': terminal, 'truncated': truncated, 'mask': mask}


def count_env_interactions(mem: Sequence[Dict[str, np.ndarray]]):
    # N observations means N-1 interactions, so subtract 1 from trajectory length
    n_interactions = reduce(lambda a, b: a + len(b['o']) - 1,  mem, 0)
    return n_interactions


def prepare_data(data: Dict[str, Union[torch.Tensor, np.ndarray]],
                 n_categories: Dict[str, int] | None = None,
                 swap_batch_time_dim: bool = False,
                 remove_keys: Sequence[str] | None = ()):
    if n_categories is None: n_categories = {}

    # here go all the specific requirements for individual fields in the data dict
    required_fids = {'o', 'a', 'r', 'terminal', 'truncated', 'mask'}
    assert required_fids <= data.keys(), f'Missing required fields,\nfound:\t{data.keys()},\nneed:\t{required_fids}'
    assert data['o'].ndim >= 2, f'Observation dim is {data["o"].ndim}, but needs to be at least 2'
    assert data['a'].ndim >= 2, f'Action dim is {data["a"].ndim}, but needs to be at least 2'
    assert 2 <= data['r'].ndim <= 3, f'Reward dim is {data["r"].ndim}, but needs to be 2 or 3'
    assert 2 <= data['terminal'].ndim <= 3, f'Terminal dim is {data["terminal"].ndim}, but needs to be 2 or 3'
    assert 2 <= data['truncated'].ndim <= 3, f'Truncated dim is {data["truncated"].ndim}, but needs to be 2 or 3'
    assert 2 <= data['mask'].ndim <= 3, f'Mask dim is {data["mask"].ndim}, but needs to be 2 or 3'

    # take care of expanding dimensions to at least one data dim, one-hot transformations and axis swapping
    for fid, fval in data.items():
        if fval.ndim == 2:
            data[fid] = fval[..., None]
        if fid in n_categories:
            data[fid] = to_onehot(fval, n_categories[fid])
        if swap_batch_time_dim:
            data[fid] = fval.swapaxes(0, 1)

    for k in remove_keys:
        del data[k]

    # data['time_step'] = torch.arange(0, data['o'].shape[0], device=data['o'].device)
    # data['time_step'] = data['time_step'][:, None, None].repeat(1, data['o'].shape[1], 1)

    return data


def subtrajectories(data: Dict[str, torch.Tensor],
                    length: int):
    d_t = data['o'].shape[0]
    assert length < d_t
    i_start = random.randint(0, d_t - length)
    data = {k: v[i_start:] for k, v in data.items()}
    return data


def valid_subtrajectories(data: Dict[str, torch.Tensor],
                          length: int):
    if length >= data['o'].shape[0]:
        length = data['o'].shape[0]

    n_trajs = data['o'].shape[1]
    l_trajs = (1 - data['mask']).sum(dim=0).detach().cpu().numpy().squeeze()
    i_start = np.random.randint(low=[0 for _ in range(n_trajs)], high=np.maximum(l_trajs - length, 1))
    i_matrix = np.tile(np.arange(0, length), (n_trajs, 1)) + i_start[..., None]
    i_matrix = torch.from_numpy(i_matrix).to(device=data['o'].device, dtype=torch.int64)

    ret_data = {}
    for k, v in data.items():
        v = v.swapaxes(0, 1)
        i_matr_exp = unsqueeze_right(i_matrix, v)
        i_matr_exp = i_matr_exp.repeat(1, 1, *v.shape[2:])
        v_new = torch.gather(v, dim=1, index=i_matr_exp)
        ret_data[k] = v_new.swapaxes(0, 1)

    return ret_data


def add_no_ops(data: List[Dict[str, np.ndarray]],
               chunk_length: int):
    new_data = []
    for traj in data:
        traj_end = np.logical_or(traj['terminal'], traj['truncated'])
        t_end = np.nonzero(traj_end)[0][0] + 1  # take first true terminal or truncated flag
        overhang = t_end % chunk_length
        n_pad = chunk_length - overhang
        new_traj = {}
        if n_pad > 0:
            padding_timesteps = np.random.randint(1, t_end, size=(n_pad,))  # choose n_pad random time steps for padding
            padding_timesteps = np.sort(padding_timesteps)[::-1]  # sort in reverse order
            for t_pad in padding_timesteps:
                # repeat previous observation, reward, terminal and truncated
                for x in ('o', 'r', 'terminal', 'truncated'):
                    new_traj[x] = np.concatenate([traj[x][:t_pad], traj[x][None, t_pad - 1], traj[x][t_pad:]], axis=0)
                # add zero action
                new_traj['a'] = np.concatenate([traj['a'][:t_pad], np.zeros_like(traj['a'][None, t_pad]),
                                                traj['a'][t_pad:]], axis=0)
        else:
            new_traj = {k: np.copy(v) for k, v in traj.items()}
        new_data.append(new_traj)
    return new_data


def valid_subtrajectories_unbiased_fast(data: Dict[str, torch.Tensor],
                                        length: int):
    l_max, n_trajs = data['o'].shape[:2]

    if length > l_max:
        length = l_max

    l_trajs = (1 - data['mask']).sum(dim=0).detach().cpu().numpy().squeeze()

    low = np.array([-length + 1]).repeat(n_trajs)
    i_start = np.random.randint(low=low, high=l_trajs)
    i_end = i_start + length
    # get indices that simply address their current position
    i_row, i_col_raw = np.indices((n_trajs, l_max), sparse=True)
    # these are the raw subsequence indices with potentially invalid start/endpoints before first or after last step
    i_col = i_col_raw + i_start[:, None]
    # mask negative start indices with special index -1
    i_col_masked_start = np.where(i_col < 0, -1, i_col)
    # from num of masked start indices, compute how many time steps to rotate each row to the left
    shifts = l_max - (i_col_masked_start == -1).sum(axis=1)
    # do the rotation, inspired by https://stackoverflow.com/questions/20360675/roll-rows-of-a-matrix-independently
    shifts = i_col_raw - shifts[:, None]
    i_col_shifted = i_col_masked_start[i_row, shifts]
    # mask end indices that should not be available due to natural end of a trajectory, for that
    # redirect those indices to an artificial last element that is later concatenated to the end of the buffers,
    # currently this index would cause out of bounds exception
    i_col_corrected = np.where(i_col_shifted >= np.minimum(i_end, l_trajs)[..., None], l_max, i_col_shifted)
    # cut off part of index matrix that can only contain invalid indices
    i_col_corrected = i_col_corrected[:, :length]
    # if length > l_max, i_col_corrected can contain trailing -1 indices due to shifting that have to be replaced by
    # l_max since torch.gather doesn't support negative indices
    i_col_corrected = np.where(i_col_corrected == -1, l_max, i_col_corrected)
    # transform to torch tensor
    i_matrix = torch.from_numpy(i_col_corrected).to(device=data['o'].device, dtype=torch.int64)

    ret_data = {}
    for k, v in data.items():
        if k == 'mask':
            pad = torch.ones(n_trajs, 1, *v.shape[2:], device=v.device, dtype=v.dtype)
        else:
            pad = torch.zeros(n_trajs, 1, *v.shape[2:], device=v.device, dtype=v.dtype)
        v = v.swapaxes(0, 1)
        v_padded = torch.concat([v, pad], dim=1)
        # gather doesn't support broadcasting and if the data in v is multi-dimensional, we need to broadcast the
        # indices over all data dimensions
        i_matr_exp = unsqueeze_right(i_matrix, v)  # add necessary dimensions
        i_matr_exp = i_matr_exp.repeat(1, 1, *v.shape[2:])  # broadcast over newly added dimensions
        v_new = torch.gather(v_padded, dim=1, index=i_matr_exp)
        ret_data[k] = v_new.swapaxes(0, 1)

    return ret_data


def valid_subtrajectories_unbiased(data: Dict[str, torch.Tensor],
                                   length: int):
    raise RuntimeError('Check indices in case length < l_max first')

    if length > data['o'].shape[0]:
        length = data['o'].shape[0] - 1

    n_trajs = data['o'].shape[1]
    l_trajs = (1 - data['mask']).sum(dim=0).detach().cpu().numpy().squeeze()

    i_start = np.random.randint(low=[-length + 1 for _ in range(n_trajs)], high=l_trajs)
    i_end = i_start + length
    i_start = np.clip(i_start, 0, l_trajs)
    i_end = np.clip(i_end, 0, l_trajs)
    i_start = i_start.astype(int)
    i_end = i_end.astype(int)
    i_start = torch.from_numpy(i_start)
    i_end = torch.from_numpy(i_end)
    # i_start = torch.from_numpy(i_start).to(device=data['o'].device, dtype=torch.float64)
    # i_end = torch.from_numpy(i_end).to(device=data['o'].device, dtype=torch.float64)
    # redirect invalid indices to -1, which is a zero-element we'll append to the data further down
    # NOTE: Doesn't work since we still can end up with -1 indices at the beginning of a trajectory
    # i_start = np.where(i_start < 0, -1, i_start)
    # i_end = np.where(i_end > l_trajs, -1, l_trajs)

    ret_data = {}
    for k, v in data.items():
        v = v.swapaxes(0, 1)
        if k == 'mask':
            v_new = torch.ones(n_trajs, length, *v.shape[2:], device=v.device, dtype=v.dtype)
        else:
            v_new = torch.zeros(n_trajs, length, *v.shape[2:], device=v.device, dtype=v.dtype)
        for i_traj, (i_0, i_1) in enumerate(zip(i_start, i_end)):
            v_new[i_traj, 0: i_1 - i_0] = v[i_traj, i_0: i_1]
        ret_data[k] = v_new.swapaxes(0, 1)

    return ret_data


def valid_subtrajectories_2(data: Dict[str, torch.Tensor],
                            length: int,
                            make_shorter_if_required: bool = True):
    if not make_shorter_if_required:
        assert length < data['o'].shape[0]
    else:
        length = min(data['o'].shape[0], length)

    n_trajs = data['o'].shape[1]

    ret_data = {}
    for k, v in data.items():
        if k == 'mask':
            empty = torch.ones(length, n_trajs, *v.shape[2:], device=v.device, dtype=v.dtype)
        else:
            empty = torch.zeros(length, n_trajs, *v.shape[2:], device=v.device, dtype=v.dtype)
        ret_data[k] = empty

    for i_traj in range(n_trajs):
        traj_len = (1 - data['mask'][:, i_traj]).sum().detach().cpu().numpy()
        i_start = random.randint(- length + 1, traj_len - 1)
        i_end = i_start + length
        # clamp to obtain only valid trajectories
        i_start = np.maximum(0, i_start).astype(int)
        i_end = np.minimum(i_end, traj_len).astype(int)
        for k, v in data.items():
            ret_data[k][0: i_end - i_start, i_traj] = v[i_start:i_end, i_traj]

    return ret_data


def valid_subtrajectories_debug(data: Dict[str, torch.Tensor],
                                length: int,
                                make_shorter_if_required: bool = True):
    if not make_shorter_if_required:
        assert length < data['o'].shape[0]
    else:
        length = min(data['o'].shape[0], length)

    n_trajs = data['o'].shape[1]

    ret_data = {}
    for k, v in data.items():
        if k in ['mask', 'terminal']:
            empty = torch.ones(length, n_trajs, *v.shape[2:], device=v.device, dtype=v.dtype)
        else:
            empty = torch.zeros(length, n_trajs, *v.shape[2:], device=v.device, dtype=v.dtype)
        ret_data[k] = empty

    for i_traj in range(n_trajs):
        traj_len = (1 - data['mask'][:, i_traj]).sum().detach().cpu().numpy()
        i_start = random.randint(- length + 1, traj_len - 1)
        i_end = i_start + length
        # clamp to obtain only valid trajectories
        i_start = np.maximum(0, i_start).astype(int)
        i_end = np.minimum(i_end, traj_len).astype(int)
        for k, v in data.items():
            ret_data[k][0: i_end - i_start, i_traj] = v[i_start:i_end, i_traj]

    return ret_data


"""
def subtrajectories(mem: List[Dict[str, DataType]],
                    length: int):
    for traj in mem:
        l_traj = len(traj['o'])
        i_start = random.randint(0, l_traj - length)
        i_end = i_start + length
        for k, v in traj:
            traj[k] = v[i_start, i_end]
    return mem
"""


def apply_mask(o: torch.Tensor,
               a: torch.Tensor,
               r: torch.Tensor,
               terminal: torch.Tensor,
               truncated: torch.Tensor,
               mask: torch.Tensor,
               o_pad_val: float = 0,
               a_pad_val: float = 0,
               r_pad_val: float = 0,
               terminal_pad_val: float = 0,
               truncated_pad_val: float = 0):
    boolean_mask = mask.to(torch.bool)
    boolean_o_mask = boolean_mask.reshape(*boolean_mask.shape, *[1 for _ in range(o.ndim - boolean_mask.ndim)])
    o.masked_fill_(boolean_o_mask, o_pad_val)
    a.masked_fill_(boolean_mask, a_pad_val)
    r.masked_fill_(boolean_mask, r_pad_val)
    terminal.masked_fill_(boolean_mask, terminal_pad_val)
    truncated.masked_fill_(boolean_mask, truncated_pad_val)
    mask.fill_(0)


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
    # s = normalize_obs(s, env)
    a = to_onehot(a, env.action_space.n)  # don't care about the action being [1, 0, ... ] if it's always this way

    # swap batch and time axis
    s = s.swapaxes(0, 1)
    a = a.swapaxes(0, 1)
    r = r.swapaxes(0, 1)
    terminal = terminal.swapaxes(0, 1)
    truncated = truncated.swapaxes(0, 1)
    mask = mask.swapaxes(0, 1)

    # s.masked_fill(~mask.to(torch.bool).unsqueeze(-1), 0)
    # a.masked_fill(~mask.to(torch.bool), 0)
    # r.masked_fill(~mask.to(torch.bool), 0)
    # terminal.masked_fill(~mask.to(torch.bool), 0)
    # truncated.masked_fill(~mask.to(torch.bool), 0)
    # mask[:] = 0

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
    # z_var = z_var.mean(axis=1)
    r_var = [d.scale for d in r_dists]
    r_var = torch.stack(r_var)
    # r_var = r_var.mean(axis=1)

    return z_var, r_var
    # return z_var.max(), z_var.std(), r_var.max(), r_var.std()

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
        t_start_aug.append(random.randint(0, t - 1))

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
    # o_mask = np.full_like(o_np, True)
    # a_mask = np.full_like(a_np, True)
    mask = np.full_like(r_np, True)

    # copy data
    for i in range(n_trajectories):
        o_np[i, 0:lengths[i]] = o[i]
        a_np[i, 0:lengths[i]] = a[i]
        r_np[i, 0:lengths[i]] = r[i]
        term_np[i, 0:lengths[i]] = term[i]
        trunc_np[i, 0:lengths[i]] = trunc[i]
        # o_mask[i, 0:lengths[i]] = False
        # a_mask[i, 0:lengths[i]] = False
        mask[i, 0:lengths[i]] = False

    # generate masked arrays
    # o_np = ma.array(o_np, mask=o_mask)
    # a_np = ma.array(a_np, mask=a_mask)
    # r_np = ma.array(r_np, mask=mask)
    # term_np = ma.array(term_np, mask=mask)
    # trunc_np = ma.array(trunc_np, mask=mask)

    return o_np, a_np, r_np, term_np, trunc_np, mask


def trajectory_statistics(eval_mem: List[Dict[str, np.ndarray]]):
    n_eval_trajectories = len(eval_mem)
    ep_len, success, avg_return = 0, 0, 0
    for ep in eval_mem:
        avg_return += np.stack(ep['r']).sum()
        if ep['terminal'].sum() == 1:
            ep_len += len(ep['terminal'])
            success += 1
        elif ep['terminal'].sum() > 1:
            raise RuntimeError('More than one terminal flag, there is something wrong!')
        else:
            ep_len += len(ep['terminal'])
    success /= n_eval_trajectories
    ep_len /= n_eval_trajectories
    avg_return /= n_eval_trajectories

    return {'success': success, 'ep_len': ep_len, 'avg_return': avg_return}


def load_memory(path: Union[str, Path]) -> List[Dict[str, DataType]]:
    with open(path, 'rb') as f:
        mem = pickle.load(f)
    return mem


def store_memory(mem: List[Dict[str, DataType]],
                 path: Union[str, Path]) -> bool:
    with open(path, 'wb') as f:
        pickle.dump(mem, f)
    return True


def discrete_stats(module: torch.nn.Module, n_inputs: int, seq_len: int, n_repetitions: int = 1,
                   module_kwargs: dict = None):
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


def filter_mem_state_seq_to_batch(mem: Dict[str, List[torch.Tensor]],
                                  mask: List[torch.Tensor] | torch.Tensor,
                                  i_start: int = 0,
                                  i_end: int = None):
    if i_end is None:
        i_end = len(mem['z'])

    states = {k: v[i_start: i_end] for k, v in mem.items() if k in rssm_state_keys()}
    states = rssm_state_seq_to_batch(**states)
    mask = stack_if_list(mask[i_start:i_end])
    mask = mask.reshape(mask.shape[0] * mask.shape[1], 1)

    return states, mask


def seq_to_batch(mem: Dict[str, torch.Tensor | List[torch.Tensor]],
                 i_start: int = 0,
                 i_end: int = None):
    if i_end is None:
        first_elem = next(iter(mem.values()))
        i_end = len(first_elem)

    mem = {k: stack_if_list(v[i_start: i_end]) for k, v in mem.items()}
    mem = {k: v.reshape(v.shape[0] * v.shape[1], *v.shape[2:]) for k, v in mem.items()}

    return mem


def log_params(model: torch.nn.Module,
               logger: Logger,
               scope: Scope,
               time_step: int):
    # async def _log_fn():
    max_param = sys.float_info.min
    min_param = sys.float_info.max
    for name, param in model.named_parameters():
        full_scope = scope / name
        param_np = param.detach().cpu().numpy()
        logger.log({'mean': param_np.mean(), 'std': param_np.std(), 'min': param_np.min(), 'max': param_np.max()},
                   full_scope, time_step=time_step)
        if param_np.min() < min_param:
            min_param = param_np.min()
        if param_np.max() > max_param:
            max_param = param_np.max()
    logger.log({'largest_param': max_param, 'smallest_param': min_param}, scope, time_step=time_step)

    # asyncio.run(_log_fn())


def copy_params(src: 'HierarchicalRSSM',
                dst: 'HierarchicalRSSM'):
    dst.load_state_dict(src.state_dict())
    for ag_src, ag_dst in zip(src.goal_seeking_agents, dst.goal_seeking_agents):
        ag_dst[0].load_state_dict(ag_src[0].state_dict())
    for ag_src, ag_dst in zip(src.r_max_agents, dst.r_max_agents):
        ag_dst[0].load_state_dict(ag_src[0].state_dict())


@torch.jit.ignore
def append_memory(memory: Dict[str, List[Any]],
                  **kwitems: Any):
    for k, v in kwitems.items():
        data = memory.get(k, [])
        data.append(v)
        memory[k] = data
    return memory


@torch.jit.ignore
def extend_memory(memory: Dict[str, Sequence[Any]],
                  predictions: Dict[str, Any]):
    for k, v in predictions.items():
        data = memory.get(k, [])
        data.extend(list(v))
        memory[k] = data
    return memory


def get_env_instance(env: gym.Env):
    core_env = env.unwrapped
    if getattr(core_env, 'is_vector_env', False):
        env_creating_fn = core_env.env_fns[0]
        dummy_env = env_creating_fn()
    else:
        dummy_env = gym.make(core_env.spec)
    return dummy_env


def env_class_is(env: gym.Env, other):
    base_env = get_env_instance(env)
    base_env = base_env.unwrapped
    if isinstance(other, str):
        return base_env.spec.id == other
    elif isinstance(other, gym.envs.registration.EnvSpec):
        return base_env.spec.id == other.id
    elif isinstance(other, gym.Env):
        return base_env.spec.id == other.spec.id
    else:
        return issubclass(type(base_env), other)


def numpyfy(x: torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor],
            squeeze: bool = True):
    if isinstance(x, (list, tuple)):
        x = torch.stack(list(x))
    x = x.detach().cpu().numpy()
    if squeeze and x.shape[-1] == 1:
        x = x.squeeze(axis=-1)
    return x


def prepare_env(env: gym.Env):
    # limit continuous action spaces to [-1.0, 1.0] to match the SquashedGaussian of the agent, we could also use a
    # clip action wrapper and not restrict the agent's actions
    if isinstance(env.action_space, gym.spaces.Box):
        env = gym.wrappers.RescaleAction(env, min_action=-1.0, max_action=1.0)

    # Flatten observation dicts
    if isinstance(env.unwrapped, minigrid.minigrid_env.MiniGridEnv):
        #env = minigrid.wrappers.FullyObsWrapper(env)
        env = minigrid.wrappers.ImgObsWrapper(env)
        #env = minigrid.wrappers.FlatObsWrapper(env)
        env = gym.wrappers.FlattenObservation(env)
    elif isinstance(env.observation_space, gym.spaces.dict.Dict):
        env = gym.wrappers.FlattenObservation(env)
    return env


def infer_action_info(env: gym.Env):
    if isinstance(env.action_space, gym.spaces.Box):
        assert len(env.action_space.shape) == 1
        d_a = env.action_space.shape[0]
        is_discrete = False
    elif isinstance(env.action_space, gym.spaces.Discrete):
        d_a = env.action_space.n
        is_discrete = True
    else:
        raise RuntimeError(f'Unsupported action space {env.action_space} of environment {env}.')
    return d_a, is_discrete


def list_of_tuples_to_tuple_of_lists(list_of_tpls: List[Any]):
    state = tuple([list(x) for x in zip(*list_of_tpls)])
    return state


def list_of_dicts_to_dict_of_lists(list_of_dicts: List[Dict[Any, Any]]):
    ret = {k: [] for k in list_of_dicts[0].keys()}
    for i,x in enumerate(list_of_dicts):
        try:
            for k, v in x.items():
                ret[k].append(v)
        except KeyError:
            print(f'Unexpected key {k} in {i}-th list element found.',
                  f'Expected keys are: {list_of_dicts[0].keys()}')
    return ret


def store_model_params(model, model_opt, path, logger, *, store_locally, upload):
    # make sure the save directory exists and infer final path for the file
    if not os.path.exists(path):
        os.makedirs(path)
    final_path = Path(path) / f'{logger.run_id}.ptmdl'

    # collect state dicts
    state_dicts = {'HRSSM': model.state_dict(), 'HRSSM_opt': model_opt.state_dict()}

    for i_agent, agent in enumerate(model.r_max_agents + model.goal_seeking_agents):
        state_dicts[f'agent_{i_agent}'] = agent[0].state_dict()  # agent state dict
        for i_opt, opt in enumerate(agent[1].values()):
            state_dicts[f'agent_{i_agent}_opt_{i_opt}'] = opt.state_dict()  # agent optimizer state dicts

    # store weights of all model components
    torch.save(state_dicts, final_path)

    if upload:
        model_weights = InMemoryFile(final_path, name='final_weights')
        logger.start_session()
        logger.log_file(model_weights, Scope.DATA() / 'weights')

        # clear last checkpoint file

    if not store_locally:
        os.remove(final_path)


def load_model_params(model, model_opt, path, run_id, api_token, project, force_reload=False):
    final_path = Path(path) / f'{run_id}.ptmdl'

    if force_reload and os.path.exists(final_path):
        os.remove(final_path)

    if not os.path.exists(final_path):
        if not os.path.exists(os.path.dirname(final_path)):
            os.makedirs(os.path.dirname(final_path))
        import neptune
        from neptune.exceptions import RunNotFound
        print('loading model parameters from run database')
        try:
            tmp_run = neptune.init_run(api_token=api_token, project=project, with_id=run_id, mode='read-only')
            tmp_run[f'{Scope.DATA()}/weights/final_weights'].download(destination=str(final_path))
            tmp_run.stop()
        except RunNotFound:
            raise RuntimeError(f'Tried to load parameters from run {run_id} to {final_path} but run doesn\'t',
                               'exist in database')

    state_dicts = torch.load(final_path)

    model.load_state_dict(state_dicts['HRSSM'])
    model_opt.load_state_dict(state_dicts['HRSSM_opt'])

    for i_agent, agent in enumerate(model.r_max_agents + model.goal_seeking_agents):
        agent[0].load_state_dict(state_dicts[f'agent_{i_agent}'])
        for i_opt, opt in enumerate(agent[1].values()):
            opt.load_state_dict(state_dicts[f'agent_{i_agent}_opt_{i_opt}'])