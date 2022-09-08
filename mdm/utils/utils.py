from inspect import stack
from pathlib import Path
from typing import Union, Dict, Tuple, List, Optional
from itertools import product, chain, repeat
import sys
import re

import gym
import numpy as np
import numpy.ma as ma
import yaml
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

from mdm.gridworld.gridworld import Gridworld, CellType
from mdm.models.multiscale_model_mk2 import MultiscaleDynamicsModelMK2
from mdm.memory.trajectory_memory import flatten_and_unsqueeze
from mdm.utils.torch_tools import add_time_dim, get_mu, get_sigma, unpack_rnn_state, pack_rnn_state


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

    #x = np.squeeze(x, axis=-1)  # remove possible redundant 1-dim data dimension
    x_onehot = np.zeros((*x.shape, n_categories))
    x = x[..., np.newaxis]  # make sure x_onehot and x have same number of dimensions
    np.put_along_axis(x_onehot, x, 1, axis=-1)  # use x as index array for x_onehot and put 1 at respective indices

    return x_onehot


def gen_macro_state_map(env: Gridworld,
                        mdl: MultiscaleDynamicsModelMK2,
                        n_trials: int,
                        available_actions: List[int] = None):
    # find all possible action sequences
    available_actions = list(range(env.action_space.n)) if available_actions is None else available_actions
    action_sequences = list(product(available_actions, repeat=mdl.abstract_step_size))
    n_unique_sequences = len(action_sequences)

    # find all possible starting locations
    free_locations = env.find_cell_type(CellType.FREE)
    agent_locations = env.find_cell_type(CellType.AGENT)
    if agent_locations.size == 2:
        agent_locations = agent_locations[np.newaxis, ...]
    free_locations = np.concatenate([free_locations, agent_locations], axis=0)
    free_locations = [tuple(loc) for loc in free_locations]

    # find all possible destination locations
    # TODO: rewrite this to collect init data like in init_s_prim
    # CAUTION: mind that initial aciton must be zero
    # CAUTION: if done, add zero padding to memories
    init_chunk_groundtruth = {}
    for loc in free_locations:
        init_chunk_groundtruth[loc] = {}
        for a_seq in action_sequences:
            env.teleport_agent(loc)
            o_mem, r_mem, term_mem = [env.reset()], [0.0], [False]
            a_mem = [0] + a_seq
            for a in a_mem[1:]:
                o, r, done, _ = env.step(a)
                o_mem.append(tuple(o))
                r_mem.append(r)
                term_mem.append(done)
                if done: break
            init_chunk_groundtruth[loc][a_seq] = {'o': o_mem, 'a': a_mem, 'r': r_mem, 'term': term_mem}

    #for s_final in init_chunk_groundtruth:
    #    assert(tuple(s_final) in free_locations)

    # convert to tensors
    #a_sequences = [torch.tensor(o).to(mdl.device) for o in action_sequences]
    #a_sequences = torch.stack(a_sequences)
    a_sequences = torch.tensor(action_sequences).to(mdl.device)
    a_sequences = to_onehot(a_sequences, mdl.d_action)
    abstr_a = mdl.abstract_action_model(a_sequences)
    abstr_a = add_time_dim(abstr_a)
    o_start = torch.tensor(free_locations).to(mdl.device)
    o_start = normalize_obs(o_start, env)

    # do rollouts
    abstr_s_init_history = {}
    for n in range(n_trials):
        for o, loc in zip(o_start, free_locations):
            # prepare
            o = torch.tile(o, dims=(n_unique_sequences, 1))
            o = o.unsqueeze(1)

            # predict
            mem, pred_prim_final = mdl.rollout_primitive(a=a_sequences, o=o)
            mem = mdl.pack_mem(mem)
            abstr_r_target = mem['prim_r'].sum(dim=1, keepdim=True)
            abstr_term_target = mem['prim_term'].max(dim=1, keepdim=True).values
            ctx_low_level = add_time_dim(mdl.fuse_state(pred_prim_final['o'], pred_prim_final['rnn_state']))
            _, pred_abstr_final = mdl.rollout_abstract(a=abstr_a, o_target=ctx_low_level,
                                                       r_target=abstr_r_target, term_target=abstr_term_target,
                                                       use_posterior=True)
            #pred_abstr_final = mdl.macro_next_posterior(zero_abstr_s, zero_abstr_a, pred_prim['h'])
            abstr_s_next_batch = pred_abstr_final['o'].detach().cpu().numpy()

            # store
            loc_hash = abstr_s_init_history.get(loc, {})
            for a_seq, abstr_s_next in zip(action_sequences, abstr_s_next_batch):
                abstr_s_next_list = loc_hash.get(a_seq, [])
                abstr_s_next_list.append(abstr_s_next)
                loc_hash[a_seq] = abstr_s_next_list
            abstr_s_init_history[loc] = loc_hash

    # compute statistics
    macro_s_init_mean = {pos: 0 for pos in free_locations}
    macro_s_init_std = {pos: 0 for pos in free_locations}
    for pos in free_locations:
        macro_s_all_actions = np.concatenate([np.stack(trials) for trials in abstr_s_init_history[pos].values()])
        macro_s_init_mean[pos] = np.mean(macro_s_all_actions, axis=0)
        macro_s_init_std[pos] = np.std(macro_s_all_actions, axis=0)

    return macro_s_init_mean, macro_s_init_std, abstr_s_init_history


def gen_prim_rnn_state_map(env: Gridworld,
                           mdl: MultiscaleDynamicsModelMK2,
                           walk_distance: int):
    traj_a, traj_o, traj_r, traj_term = exhaustive_traversion(env, mdl, walk_distance)
    o_start, a_start, r_start, term_start = prepare_data(traj_o, traj_a, traj_r, traj_term, env)
    mem, prim_current = mdl.rollout_primitive(a=a_start, o=o_start, r=r_start, term=term_start,
                                              n_posterior_steps=-1, sample=False)
    mem = mdl.pack_mem(mem)

    rnn_state_mean = mem['prim_rnn_state'][:, -1, -1, 0].detach().cpu().numpy()
    final_pos = traj_o[:, -1].detach().cpu().numpy()

    map_mean = np.zeros((mdl.primitive_model.d_hidden, env.grid_h, env.grid_w), dtype=np.float32)
    map_std = np.zeros_like(map_mean)

    for pos in final_pos:
        masked_idx = np.all(final_pos == pos, axis=1, keepdims=True)  # row-wise and
        masked_idx = np.logical_not(masked_idx)
        masked_idx = np.repeat(masked_idx, mdl.primitive_model.d_hidden, axis=1)
        valid_trajectories_mean = ma.MaskedArray(rnn_state_mean, mask=masked_idx)
        _s_mean = valid_trajectories_mean.mean(axis=0)
        _s_std = valid_trajectories_mean.std(axis=0)
        map_mean[:, pos[0], pos[1]] = _s_mean
        map_std[:, pos[0], pos[1]] = _s_std

    return map_mean, map_std


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


def gen_prim_state_map(env: Gridworld,
                       mdl: MultiscaleDynamicsModelMK2,
                       walk_distance: int):
    traj_a, traj_o, traj_r, traj_term = exhaustive_traversion(env, mdl, walk_distance)
    o_start, a_start, r_start, term_start = prepare_data(traj_o, traj_a, traj_r, traj_term, env)
    mem, prim_current = mdl.rollout_primitive(a=a_start, o=o_start, r=r_start, term=term_start,
                                              n_posterior_steps=-1, sample=False)
    mem = mdl.pack_mem(mem)

    #prim_s_final = mem['prim_s'][:, -1]
    #prim_rnn_state_final = unpack_rnn_state(mem['prim_rnn_state'][:, -1])
    #prim_rnn_state_final = mdl.filter_rnn_state(prim_rnn_state_final)
    #x = torch.concat([prim_s_final, prim_rnn_state_final], dim=-1)
    #prim_s_enc = mdl.prim_state_enc(x)
    #s_mean = prim_s_enc.detach().cpu().numpy()
    #s_std = np.zeros_like(s_mean)

    s_mean = mem['prim_s'][:, -1].detach().cpu().numpy()
    s_std = np.zeros_like(s_mean)

    #s_mean = get_mu(mem['prim_s_post'][:, -1]).detach().cpu().numpy()
    #s_std = get_sigma(mem['prim_s_post'][:, -1]).detach().cpu().numpy()
    final_pos = traj_o[:, -1].detach().cpu().numpy()

    map_mean = np.zeros((s_mean.shape[1], env.grid_h, env.grid_w), dtype=np.float32)
    map_std = np.zeros_like(map_mean)

    for pos in final_pos:
        masked_idx = np.all(final_pos == pos, axis=1, keepdims=True)  # row-wise and
        masked_idx = np.logical_not(masked_idx)
        masked_idx = np.repeat(masked_idx, s_mean.shape[1], axis=1)
        valid_trajectories_mean = ma.MaskedArray(s_mean, mask=masked_idx)
        valid_trajectories_std = ma.MaskedArray(s_std, mask=masked_idx)
        _s_mean = valid_trajectories_mean.mean(axis=0)
        _s_std = valid_trajectories_mean.std(axis=0)
        map_mean[:, pos[0], pos[1]] = _s_mean
        map_std[:, pos[0], pos[1]] = _s_std

    return map_mean, map_std

    ## remove time dimension and get distribution parameters
    #s_mean = get_mu(mem['prim_s_post'][:, -1]).detach().cpu().numpy()
    #s_std = get_sigma(mem['prim_s_post'][:, -1]).detach().cpu().numpy()
    #
    #map_mean = np.zeros((mdl.primitive_model.d_state, env.grid_h, env.grid_w), dtype=np.float32)
    #map_std = np.zeros_like(map_mean)
    #
    #for loc, _s_mean, _s_std in zip(free_locations, s_mean, s_std):
    #    map_mean[:, loc[0], loc[1]] = _s_mean
    #    map_std[:, loc[0], loc[1]] = _s_std
    #
    #return map_mean, map_std


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


def visualize_plan(trajectory_history: Dict[str, torch.Tensor], env: Gridworld, mdl: MultiscaleDynamicsModelMK2):
    assert trajectory_history['abstr_o'].shape[0] == 1, 'Batch size of trajectory history should be 1'

    map_mean, map_std = gen_prim_state_map(env, mdl)
    map_flat = map_mean.reshape((mdl.primitive_model.d_state, -1))
    map_flat = np.transpose(map_flat, (1, 0))
    total_positions = map_mean.shape[1] * map_mean.shape[2]
    similarity_maps = []

    for predicted_prim_s in trajectory_history['abstr_o'][0]:  # remove batch size
        predicted_prim_s = predicted_prim_s.detach().cpu().numpy()
        predicted_prim_s_batch = predicted_prim_s[None, ...].repeat(total_positions, axis=0)
        mse = np.mean((predicted_prim_s_batch - map_flat) ** 2, axis=1)
        similarity_maps.append(mse.reshape(map_mean.shape[1], map_mean.shape[2]))

    similarity_maps = np.stack(similarity_maps)  # time is first dimension now
    gen_video(similarity_maps, 500, 3)
    return similarity_maps


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

    return obs / denom - 0.5


def to_onehot(actions: Union[torch.Tensor, np.ndarray],
              n_classes: int) -> Union[torch.Tensor, np.ndarray]:
    if (actions % 1 != 0).any():
        raise ValueError('All elements in actions must be ints or castable to int without loss of precision')

    actions = actions.squeeze(-1)
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
    a = to_onehot(a, env.action_space.n)  # don't care about the action being [1, 0, ... ] if it's always this way
    return s, a, r, terminal


def discrete_stats(module: torch.nn.Module, n_inputs: int, seq_len: int, n_repetitions: int = 1):
    device = next(module.parameters()).device

    # generate all permutations of possible inputs
    available_inputs = list(range(n_inputs))
    input_sequences = list(product(available_inputs, repeat=seq_len))
    input_sequences = torch.tensor(input_sequences).to(device)
    input_sequences = to_onehot(input_sequences, n_inputs)
    n_unique_sequences = len(input_sequences)
    input_sequences = input_sequences.repeat(n_repetitions, 1, 1)

    # query the model
    Y = module(input_sequences)
    d_out = Y.shape[-1]
    Y = Y.reshape(n_unique_sequences, n_repetitions, d_out)

    # per sequence mean and std
    Y_mean = torch.mean(Y, dim=1)
    Y_std = torch.std(Y, dim=1)

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

