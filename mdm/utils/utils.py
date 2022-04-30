from inspect import stack
from pathlib import Path
from typing import Union, Dict, Tuple, List
from itertools import product
import sys
import time

import gym
import numpy as np
import yaml
import torch

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
    available_actions = list(range(env.action_space.n)) if available_actions is None else available_actions
    action_sequences = list(product(available_actions, repeat=mdl.macro_step_size)) * n_trials
    free_locations = env.find_cell_type(CellType.FREE)
    n_locations = len(free_locations)

    groundtruth_s_final = []
    for a_seq in action_sequences:
        for loc in free_locations:
            env.reset()
            env.teleport_agent(loc)
            for a in a_seq:
                s_, r, done, _ = env.step(a)
                if done: break
            groundtruth_s_final.append(s_)
    groundtruth_s_final = np.stack(groundtruth_s_final)

    for s_final in groundtruth_s_final:
        assert(tuple(s_final) in free_locations)

    # convert to tensors
    action_sequences = [torch.tensor(s).to(mdl.device) for s in action_sequences]
    s_start = torch.from_numpy(free_locations).to(mdl.device)
    s_start = s_start.unsqueeze(1)  # add time dimension
    s_start = normalize_obs(s_start, env)

    macro_s_init_history = []
    for a_seq in action_sequences:
        a_seq_batch = torch.tile(a_seq, dims=(n_locations, 1))  # copy same starting observation along batch
        a_seq_batch = torch.nn.functional.one_hot(a_seq_batch, num_classes=mdl.d_action)
        predictions_ss = mdl.rollout_single_step(s_start, a_seq_batch)
        zero_macro_s = torch.zeros(n_locations, mdl.d_macro_state, device=mdl.device)
        zero_macro_a = torch.zeros(n_locations, mdl.d_macro_action, device=mdl.device)
        predictions_ms = mdl.macro_next_posterior(zero_macro_s, zero_macro_a, predictions_ss['h'])
        macro_s_next_post, macro_s_next_post_dist, macro_r_next_post, macro_r_next_post_dist = predictions_ms
        macro_s_init_history.append(macro_s_next_post.detach().cpu().numpy())

    macro_s_init_history = np.stack(macro_s_init_history)
    macro_s_init_history = macro_s_init_history.reshape(len(action_sequences) * n_locations, mdl.d_macro_state)
    #macro_s_init_mean = macro_s_init_history.mean(axis=0)
    #macro_s_init_mean = (macro_s_init_mean + np.abs(macro_s_init_mean.min(axis=0))) / (macro_s_init_mean.max(axis=0)
    #                                                                                   - macro_s_init_mean.min(axis=0))
    #return free_locations, macro_s_init_mean

    macro_s_init_mean = {tuple(pos): [] for pos in free_locations}
    macro_s_init_std = {tuple(pos): [] for pos in free_locations}
    for s_final, macro_s_init in zip(groundtruth_s_final, macro_s_init_history):
        macro_s_init_mean[tuple(s_final)].append(macro_s_init)
    for k, v in macro_s_init_mean.items():
        if len(v):
            macro_s_init_mean[k] = np.mean(v, axis=0)
            macro_s_init_std[k] = np.std(v, axis=0)
        else:
            macro_s_init_mean[k] = np.zeros(mdl.d_macro_state)
            macro_s_init_std[k] = np.zeros(mdl.d_macro_state)


    #macro_s_init_mean = np.stack([macro_s_init_mean[tuple(loc)] for loc in free_locations])
    #macro_s_init_std = np.stack([macro_s_init_std[tuple(loc)] for loc in free_locations])

    return macro_s_init_mean, macro_s_init_std


def normalize_obs(obs: Union[torch.Tensor, np.ndarray],
                  env: gym.Env) -> Union[torch.Tensor, np.ndarray]:
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise ValueError('Only environments with box observations space are supported')

    denom = env.observation_space.high.astype(np.float32)
    if isinstance(obs, torch.Tensor):
        denom = torch.from_numpy(denom).to(obs.device)

    return obs / denom - 0.5


def one_hot_actions(actions: Union[torch.Tensor, np.ndarray],
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
    a = one_hot_actions(a, env.action_space.n)
    return s, a, r, terminal
