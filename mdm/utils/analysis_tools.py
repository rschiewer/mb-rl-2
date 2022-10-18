from itertools import product
from typing import List, Dict

import numpy as np
import torch
from numpy import ma as ma

from mdm.gridworld.gridworld import Gridworld, CellType
from mdm.models.multiscale_model_mk2 import MultiscaleDynamicsModelMK2
from mdm.utils.torch_tools import add_time_dim, unpack_rnn_state, get_mu
from mdm.utils.utils import to_onehot, normalize_obs, exhaustive_traversion, prepare_data, gen_video


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

    map_mean = np.zeros((mdl.primitive_model.d_h, env.grid_h, env.grid_w), dtype=np.float32)
    map_std = np.zeros_like(map_mean)

    for pos in final_pos:
        masked_idx = np.all(final_pos == pos, axis=1, keepdims=True)  # row-wise and
        masked_idx = np.logical_not(masked_idx)
        masked_idx = np.repeat(masked_idx, mdl.primitive_model.d_h, axis=1)
        valid_trajectories_mean = ma.MaskedArray(rnn_state_mean, mask=masked_idx)
        _s_mean = valid_trajectories_mean.mean(axis=0)
        _s_std = valid_trajectories_mean.std(axis=0)
        map_mean[:, pos[0], pos[1]] = _s_mean
        map_std[:, pos[0], pos[1]] = _s_std

    return map_mean, map_std


def gen_value_map_prim(env: Gridworld,
                       mdl: MultiscaleDynamicsModelMK2,
                       walk_distance: int,
                       posterior_steps: int = -1,
                       quantity: str = 's'):
    traj_a, traj_o, traj_r, traj_term = exhaustive_traversion(env, mdl, walk_distance)
    o_start, a_start, r_start, term_start = prepare_data(traj_o, traj_a, traj_r, traj_term, env)
    mem, prim_current = mdl.rollout_primitive(a=a_start, o=o_start, r=r_start, term=term_start,
                                              n_posterior_steps=posterior_steps, sample=False)
    mem = mdl.pack_mem(mem)

    #prim_s_final = mem['prim_s'][:, -1]
    #prim_rnn_state_final = unpack_rnn_state(mem['prim_rnn_state'][:, -1])
    #prim_rnn_state_final = mdl.filter_rnn_state(prim_rnn_state_final)
    #x = torch.concat([prim_s_final, prim_rnn_state_final], dim=-1)
    #prim_s_enc = mdl.prim_state_enc(x)
    #quant_mean = prim_s_enc.detach().cpu().numpy()
    #quant_std = np.zeros_like(quant_mean)

    key = 'prim_' + quantity
    if quantity == 'rnn_state':
        quant_mean = mdl.filter_rnn_state(unpack_rnn_state(mem['prim_rnn_state'][:, -1]))
        quant_mean = quant_mean.detach().cpu().numpy()
    elif quantity.endswith(('prior', 'post', 'dist')):
        quant_mean = get_mu(mem[key][:, -1]).detach().cpu().numpy()
    else:
        quant_mean = mem[key][:, -1].detach().cpu().numpy()
    quant_std = np.zeros_like(quant_mean)

    #quant_mean = get_mu(mem['prim_s_post'][:, -1]).detach().cpu().numpy()
    #quant_std = get_sigma(mem['prim_s_post'][:, -1]).detach().cpu().numpy()
    final_pos = traj_o[:, -1].detach().cpu().numpy()

    map_mean = np.zeros((quant_mean.shape[1], env.grid_h, env.grid_w), dtype=np.float32)
    map_std = np.zeros_like(map_mean)

    for pos in final_pos:
        masked_idx = np.all(final_pos == pos, axis=1, keepdims=True)  # row-wise and
        masked_idx = np.logical_not(masked_idx)
        masked_idx = np.repeat(masked_idx, quant_mean.shape[1], axis=1)
        valid_trajectories_mean = ma.MaskedArray(quant_mean, mask=masked_idx)
        valid_trajectories_std = ma.MaskedArray(quant_std, mask=masked_idx)
        _s_mean = valid_trajectories_mean.mean(axis=0)
        _s_std = valid_trajectories_mean.std(axis=0)
        map_mean[:, pos[0], pos[1]] = _s_mean
        map_std[:, pos[0], pos[1]] = _s_std

    return map_mean, map_std

    ## remove time dimension and get distribution parameters
    #quant_mean = get_mu(mem['prim_s_post'][:, -1]).detach().cpu().numpy()
    #quant_std = get_sigma(mem['prim_s_post'][:, -1]).detach().cpu().numpy()
    #
    #map_mean = np.zeros((mdl.primitive_model.d_state, env.grid_h, env.grid_w), dtype=np.float32)
    #map_std = np.zeros_like(map_mean)
    #
    #for loc, _s_mean, _s_std in zip(free_locations, quant_mean, quant_std):
    #    map_mean[:, loc[0], loc[1]] = _s_mean
    #    map_std[:, loc[0], loc[1]] = _s_std
    #
    #return map_mean, map_std


def visualize_plan(trajectory_history: Dict[str, torch.Tensor], env: Gridworld, mdl: MultiscaleDynamicsModelMK2):
    assert trajectory_history['abstr_o'].shape[0] == 1, 'Batch size of trajectory history should be 1'

    map_mean, map_std = gen_value_map_prim(env, mdl)
    map_flat = map_mean.reshape((mdl.primitive_model.d_z, -1))
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


def primitive_action_maps(env: Gridworld, model: MultiscaleDynamicsModelMK2):
    device = next(model.parameters()).device
    n_actions = model.d_action
    seq_len = model.abstract_step_size

    # generate all permutations of possible inputs
    available_inputs = list(range(n_actions))
    input_sequences = list(product(available_inputs, repeat=seq_len))
    descriptions = [' '.join([env.action_descriptions[a] for a in seq]) for seq in input_sequences]
    input_sequences = torch.tensor(input_sequences).to(device)
    input_sequences = to_onehot(input_sequences, n_actions)
    abstract_actions = model.abstract_action_model(input_sequences)
    return descriptions, abstract_actions.detach().cpu().numpy()


def infer_primitive_actions(trajectory_history, action_descriptions, canonical_abstr_a):
    n_canonical_abstr_a = canonical_abstr_a.shape[0]
    plan_description = []
    for abstr_a in trajectory_history['abstr_a'][0]:
        abstr_a = abstr_a.detach().cpu().numpy()
        abstr_a = np.tile(abstr_a, reps=(n_canonical_abstr_a, 1))
        closest = np.sum((abstr_a - canonical_abstr_a) ** 2, axis=-1).argmin()
        plan_description.append(action_descriptions[closest])
    plan_description = ' | '.join(plan_description)
    return plan_description
