from itertools import product
from typing import List, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from numpy import ma as ma

from mdm.gridworld.gridworld import Gridworld, CellType
from mdm.models.multiscale_model_mk2 import MultiscaleDynamicsModelMK2
from mdm.utils.torch_tools import unpack_rnn_state, get_mu
from mdm.utils.utils import to_onehot, normalize_obs, exhaustive_traversion, prepare_data_gridworld, gen_video


def gen_prim_rnn_state_map(env: Gridworld,
                           mdl: MultiscaleDynamicsModelMK2,
                           walk_distance: int):
    traj_a, traj_o, traj_r, traj_term = exhaustive_traversion(env, mdl, walk_distance)
    o_start, a_start, r_start, term_start = prepare_data_gridworld(traj_o, traj_a, traj_r, traj_term, env)
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
    o_start, a_start, r_start, term_start = prepare_data_gridworld(traj_o, traj_a, traj_r, traj_term, env)
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
    n_actions = model.primitive_model.d_action
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


def plot_plan(env: Gridworld, traj_history: Dict[str, torch.Tensor], n_plot_steps: int):
    obs = traj_history['abstr_o'][0, :n_plot_steps]
    obs = obs.detach().cpu().numpy() + 0.5
    obs[:, 0] *= (env.grid_h - 1)
    obs[:, 1] *= (env.grid_w - 1)

    actions = traj_history['prim_a'][0, :n_plot_steps]
    actions = actions.detach().cpu().numpy().argmax(axis=-1)

    rewards = traj_history['abstr_r'][0, :n_plot_steps].squeeze()
    rewards = rewards.detach().cpu().numpy()

    terminals = traj_history['abstr_term'][0, :n_plot_steps].squeeze()
    terminals = terminals.detach().cpu().numpy()

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    ax[0].matshow(env.grid)
    for i, (y, x) in enumerate(obs):
        c = 'cyan' if i <= env.current_ep_time else 'red'
        dy, dx = env.move_offset[actions[i]]
        ax[0].text(x, y, str(i), color=c, fontsize=12, ha='center', va='center')
        ax[0].arrow(x + dx * 0.2, y + dy * 0.2, dx * 0.2, dy * 0.2, color=c)

    ax[1].grid(True)
    ax[1].plot(rewards, alpha=0.7, linewidth=2, label='reward')
    ax[1].plot(terminals, alpha=0.7, label='terminal')
    ax[1].plot(rewards * np.cumprod(1 - terminals), alpha=0.7, label='discounted rewards')

    plt.legend()
    plt.show()


def plot_trajectory_stats(mem, bins: int):
    lengths, actions, rewards, terminals, truncateds = [], [], [], [], []
    successful = 0

    for t in mem:
        lengths.append(len(t['o']))
        actions.extend(t['a'])
        rewards.extend(t['r'])
        terminals.extend([t.astype(int) for t in t['terminal']])
        truncateds.extend([t.astype(int) for t in t['truncated']])
        successful += t['terminal'][-1]

    lengths = np.array(lengths)
    actions = np.array(actions)
    rewards = np.array(rewards)
    terminals = np.array(terminals)
    truncateds = np.array(truncateds)

    terminal_false, terminal_true = np.bincount(terminals, minlength=2) / len(terminals)
    truncated_false, truncated_true = np.bincount(truncateds, minlength=2) / len(truncateds)
    successful_false, successful_true = 1 - successful/len(mem), successful/len(mem)

    print('start plotting...')

    fig, ax = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'Per Timestep Statistics over {len(mem)} Trajectories')

    ax.flat[0].set_title('actions')
    sns.histplot(actions, ax=ax.flat[0], bins=bins)

    ax.flat[1].set_title('rewards')
    sns.histplot(rewards, ax=ax.flat[1], bins=bins)

    ax.flat[2].set_title('episode lengths')
    sns.histplot(lengths, ax=ax.flat[2], bins=bins)

    ax.flat[3].set_title('truncated flags')
    ax.flat[3].pie([truncated_true, truncated_false], labels=['true', 'false'], autopct='%1.1f%%')

    ax.flat[4].set_title('terminal flags')
    ax.flat[4].pie([terminal_true, terminal_false], labels=['true', 'false'], autopct='%1.1f%%')

    ax.flat[5].set_title('successful episodes')
    ax.flat[5].pie([successful_true, successful_false], labels=['true', 'false'], autopct='%1.1f%%')
    plt.show()