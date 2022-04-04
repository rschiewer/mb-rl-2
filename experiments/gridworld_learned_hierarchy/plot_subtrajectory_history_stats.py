from math import ceil

import torch
import numpy as np
import matplotlib.pyplot as plt

from mdm.gridworld.gridworld import Gridworld
from mdm.models.multiscale_model import MultiscaleDynamicsModel
from mdm.utils.utils import here
from mdm.memory.trajectory_memory import TrajectoryMemory
from mdm.training.offline_rl_driver import OfflineRLDriver
from mdm.memory.trajectory_memory import flatten_and_unsqueeze


def find_suitable_trajectory(mem: TrajectoryMemory, min_traj_len: int, shuffle: bool = True):
    if shuffle:
        mem = mem.shuffle()
    for traj in mem:
        if len(traj['s']) > min_traj_len:
            return traj


def plot_hidden_state_variation(n_locations: int, action_sequences: dict, seq_histories: dict, locations: list):
    fig, ax = plt.subplots(n_locations, len(action_sequences), figsize=(16, 10))
    # handle special cases
    if n_locations == 1: ax = ax[None, ...]
    if len(action_sequences) == 1: ax = ax[..., None]

    fig.suptitle('LSTM hidden state variation w.r.t. different action sequences')
    for i_loc in range(n_locations):
        for i_seq, (seq_descr, seq_data) in enumerate(seq_histories.items()):
            curr_run = seq_data[i_loc]
            h_mean = curr_run.mean(dim=0)
            h_std = curr_run.std(dim=0)

            x = range(len(h_mean))
            y = h_mean.detach().cpu().numpy()
            y_err = h_std.detach().cpu().numpy()
            ax[i_loc, i_seq].plot(x, y, label='mean')
            ax[i_loc, i_seq].plot(x, y_err, label='std')

    # column labels
    for i_ax, seq_descr in enumerate(action_sequences.keys()):
        ax[0, i_ax].title.set_text(seq_descr)
    # row labels
    for i_row, loc in enumerate(locations):
        ax[i_row, 0].set_ylabel(f'{loc}', rotation=0, size='large')

    plt.legend()
    plt.show()


def plot_hidden_state_distances(n_locations: int, action_sequences: dict, seq_histories: dict, locations: list):
    #n_cols = ceil(n_locations / 4)
    fig, ax = plt.subplots(n_locations, 1, figsize=(16, 10))
    ax = ax[..., None]
    # handle special cases
    if n_locations == 1: ax = ax[None, ...]
    #if n_cols == 1: ax = ax[..., None]

    fig.suptitle('LSTM hidden state distances')
    for i_loc in range(n_locations):
        mean_mat = torch.stack([seq_data[i_loc].mean(dim=0)
                                for seq_data in seq_histories.values()], dim=0)
        mean_mat = torch.nn.functional.normalize(mean_mat, p=2.0, dim=1)
        dist_mat = torch.matmul(mean_mat, mean_mat.T)
        dist_mat /= dist_mat.max()
        dist_mat = dist_mat.detach().cpu().numpy()
        ax[i_loc, 0].matshow(dist_mat)

    # row labels
    for i_row, loc in enumerate(locations):
        ax[i_row, 0].set_ylabel(f'{loc}', rotation=0, size='large')

    #plt.legend()
    plt.show()


if __name__ == '__main__':
    env = Gridworld.from_cleartext(here() / '../../mdm/gridworld/8x8_v0.mapdata')
    mdl: MultiscaleDynamicsModel = torch.load(here() / 'model.ptmdl')
    mem = TrajectoryMemory.load(here() / 'gridworld_train.samples')
    driver = OfflineRLDriver(mem)

    # define parameters
    n_evals = 50
    n_locations = 10
    action_sequences = {
        'UP': [0 for _ in range(mdl.abstract_step_size)],
        'RIGHT': [1 for _ in range(mdl.abstract_step_size)],
        'DOWN': [2 for _ in range(mdl.abstract_step_size)],
        'LEFT': [3 for _ in range(mdl.abstract_step_size)],
    }

    action_sequences = {k: torch.tensor(v,device=mdl.device) for k, v in action_sequences.items()}  # convert to tensor
    seq_histories = {k: [] for k in action_sequences.keys()}
    locations = []

    # generate data
    for i_loc in range(n_locations):
        traj = find_suitable_trajectory(mem, mdl.abstract_step_size, True)
        s_start = traj['s'][0]
        locations.append(s_start)

        s_start = torch.from_numpy(s_start).to(mdl.device)  # to torch tensor
        s_start = s_start.unsqueeze(0).unsqueeze(0)  # add time and batch dimensions
        s_start = s_start.float() / torch.tensor((env.grid_h - 1, env.grid_w - 1), device=mdl.device)  # normalize

        for seq_descr, a_seq in action_sequences.items():
            s_start_batch = torch.tile(s_start, dims=(n_evals, 1, 1))  # copy same starting observation along batch
            a_seq_batch = torch.tile(a_seq, dims=(n_evals, 1))  # copy same starting observation along batch
            a_seq_batch = torch.nn.functional.one_hot(a_seq_batch, num_classes=mdl.d_action)
            s, s_dist, r, r_dist, h = mdl.rollout_single_step(s_start_batch, a_seq_batch)
            seq_histories[seq_descr].append(mdl._flatten_h(h))

    #plot_hidden_state_variation(n_locations, action_sequences, seq_histories, locations)
    plot_hidden_state_distances(n_locations, action_sequences, seq_histories, locations)


