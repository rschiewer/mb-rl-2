from inspect import stack
from pathlib import Path
from typing import Union, Dict, Tuple, List
from itertools import product, chain, repeat
import sys
import re

import gym
import numpy as np
import yaml
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

from mdm.gridworld.gridworld import Gridworld, CellType
from mdm.models.multiscale_model import MultiscaleDynamicsModel
from mdm.memory.trajectory_memory import flatten_and_unsqueeze
from mdm.training.gym_driver import GymStepDriver


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

    x = np.squeeze(x, axis=-1)  # remove possible redundant 1-dim data dimension
    x_onehot = np.zeros((*x.shape, n_categories))
    x = x[..., np.newaxis]  # make sure x_onehot and x have same number of dimensions
    np.put_along_axis(x_onehot, x, 1, axis=-1)  # use x as index array for x_onehot and put 1 at respective indices

    return x_onehot


def gen_macro_state_map(env: Gridworld,
                        mdl: MultiscaleDynamicsModel,
                        n_trials: int,
                        available_actions: List[int] = None):
    # find all possible action sequences
    available_actions = list(range(env.action_space.n)) if available_actions is None else available_actions
    action_sequences = list(product(available_actions, repeat=mdl.macro_step_size))
    n_unique_sequences = len(action_sequences)

    # find all possible starting locations
    free_locations = env.find_cell_type(CellType.FREE)
    agent_locations = env.find_cell_type(CellType.AGENT)
    if agent_locations.size == 2:
        agent_locations = agent_locations[np.newaxis, ...]
    free_locations = np.concatenate([free_locations, agent_locations], axis=0)
    free_locations = [tuple(loc) for loc in free_locations]

    # find all possible destination locations
    groundtruth_s_final = {}
    for loc in free_locations:
        groundtruth_s_final[loc] = {}
        for a_seq in action_sequences:
            s_ = env.reset()  # TODO: check!
            env.teleport_agent(loc)
            for a in a_seq:
                s_, r, done, _ = env.step(a)
                if done: break
            groundtruth_s_final[loc][a_seq] = tuple(s_)

    #for s_final in groundtruth_s_final:
    #    assert(tuple(s_final) in free_locations)

    # convert to tensors
    action_sequences_torch = [torch.tensor(s).to(mdl.device) for s in action_sequences]
    action_sequences_torch = torch.stack(action_sequences_torch)
    action_sequences_torch = torch.nn.functional.one_hot(action_sequences_torch, num_classes=mdl.d_action)
    ss_start = torch.tensor(free_locations).to(mdl.device)
    ss_start = normalize_obs(ss_start, env)

    # do rollouts
    macro_s_init_history = {}
    for n in range(n_trials):
        for s_start, loc in zip(ss_start, free_locations):
            # prepare
            s_start = torch.tile(s_start, dims=(n_unique_sequences, 1))
            s_start = s_start.unsqueeze(1)
            zero_macro_s = torch.zeros(n_unique_sequences, mdl.d_macro_state, device=mdl.device)
            zero_macro_a = torch.zeros(n_unique_sequences, mdl.d_macro_action, device=mdl.device)

            # predict
            pred_prim = mdl.rollout_single_step(s_start, action_sequences_torch)
            pred_abstr = mdl.macro_next_posterior(zero_macro_s, zero_macro_a, pred_prim['h'])
            macro_ss_next = pred_abstr['macro_s_next'].detach().cpu().numpy()

            # store
            loc_hash = macro_s_init_history.get(loc, {})
            for a_seq, macro_s_next in zip(action_sequences, macro_ss_next):
                macro_s_next_list = loc_hash.get(a_seq, [])
                macro_s_next_list.append(macro_s_next)
                loc_hash[a_seq] = macro_s_next_list
            macro_s_init_history[loc] = loc_hash

    # compute statistics
    macro_s_init_mean = {pos: 0 for pos in free_locations}
    macro_s_init_std = {pos: 0 for pos in free_locations}
    for pos in free_locations:
        macro_s_all_actions = np.concatenate([np.stack(trials) for trials in macro_s_init_history[pos].values()])
        macro_s_init_mean[pos] = np.mean(macro_s_all_actions, axis=0)
        macro_s_init_std[pos] = np.std(macro_s_all_actions, axis=0)

    return macro_s_init_mean, macro_s_init_std, macro_s_init_history


def transform_macro_s_init_history(macro_s_init_history):
    result_macro_ss, result_locs, result_act_sequences = [], [], []
    for loc, data in macro_s_init_history.items():
        for a_seq in data.keys():
            macro_s = macro_s_init_history[loc][a_seq]
            if len(macro_s) > 1:
                macro_s = np.mean(macro_s, axis=0)
            result_macro_ss.append(macro_s)
            result_locs.append(loc)
            result_act_sequences.append(a_seq)

    return np.stack(result_macro_ss), np.stack(result_locs), np.stack(result_act_sequences)


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
    fig = plt.figure()
    ims = []
    for frame in frames:
        im = plt.imshow(frame, animated=True)
        ims.append([im])
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

    return obs / denom - 0.5


def to_onehot(actions: Union[torch.Tensor, np.ndarray],
              n_classes: int) -> Union[torch.Tensor, np.ndarray]:
    if (actions % 1 != 0).any():
        raise ValueError('All elements in actions must be ints or castable to int without loss of precision')

    if isinstance(actions, torch.Tensor):
        actions = actions.to(dtype=torch.int64)
        actions = torch.nn.functional.one_hot(actions, num_classes=n_classes)
        actions = actions.to(device=actions.device, dtype=torch.float32)
    else:
        actions = actions.astype(np.int64)
        actions = np_one_hot(actions, n_classes)

    return actions


def prepare_data(s: Union[np.ndarray, torch.Tensor],
                 a: Union[np.ndarray, torch.Tensor],
                 r: Union[np.ndarray, torch.Tensor],
                 terminal: Union[np.ndarray, torch.Tensor],
                 env: gym.Env) -> Tuple[Union[torch.tensor, np.ndarray], Union[torch.tensor, np.ndarray],
                                        Union[torch.tensor, np.ndarray], Union[torch.tensor, np.ndarray]]:
    s, a, r, terminal = flatten_and_unsqueeze(s, a, r, terminal)
    s = normalize_obs(s, env)
    a = a[:, :-1]  # last timestep is padding in any case, so omit it because we need one less action than o, r, term
    a = to_onehot(a, env.action_space.n)

    # since r and terminal were padded with one element anyway, rotate it to the front and make it zero
    if isinstance(r, torch.Tensor):
        r = torch.roll(r, shifts=1, dims=1)
    else:
        r = np.roll(r, shift=1, axis=1)
    r[:, 0] = 0
    if isinstance(terminal, torch.Tensor):
        terminal = torch.roll(terminal, shifts=1, dims=1)
    else:
        terminal = np.roll(terminal, shift=1, axis=1)
    terminal[:, 0] = 0

    return s, a, r, terminal
