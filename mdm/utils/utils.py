from inspect import stack
from pathlib import Path
from typing import Union, Dict, Tuple, List
from itertools import product, chain, repeat
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


def infer_position(macro_s: np.ndarray, macro_ss_lookup: np.ndarray, positions_lookup: np.ndarray):
    n_macro_ss, d_macro_s = macro_ss_lookup.shape
    target = np.tile(macro_s, (n_macro_ss, 1))
    diff = np.mean(np.abs(target - macro_ss_lookup), axis=-1)
    idxs = np.argsort(diff)
    return positions_lookup[idxs], diff[idxs]



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
    a = to_onehot(a, env.action_space.n)
    return s, a, r, terminal
