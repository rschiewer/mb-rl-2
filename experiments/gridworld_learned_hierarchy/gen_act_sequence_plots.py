from math import ceil
from random import shuffle

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
from sklearn.cluster import KMeans

from mdm.gridworld.gridworld import Gridworld
from mdm.models.multiscale_model import MultiscaleDynamicsModel
from mdm.utils.utils import here
from mdm.memory.trajectory_memory import TrajectoryMemory
from mdm.training.offline_rl_driver import OfflineRLDriver


def find_suitable_trajectory(mem: TrajectoryMemory, min_traj_len: int, shuffle: bool = True):
    if shuffle:
        mem = mem.shuffle()
    for traj in mem:
        if len(traj['s']) > min_traj_len:
            return traj


def mse_per_location(title: str, quantity_history: dict, s_history: dict, env_h: int, env_w: int, max_n_cols: int = 5):
    n_act_sequences = len(quantity_history.keys())
    seq_descr = list(quantity_history.keys())
    quantity_history = list(quantity_history.values())
    quantity_history = np.stack(quantity_history)  # shape: (seq, location, h)
    quantity_history = quantity_history.transpose([1, 0, 2])  # shape: (location, seq, h)
    s_history = list(s_history.values())
    s_history = np.stack(s_history)
    s_history = s_history.transpose([1, 0, 2, 3])  # shape: (location, seq, timestep, state)
    start_locations = np.round(s_history[:, 0, 0] * np.array([env_h - 1, env_w - 1]))

    seq_len = s_history.shape[2]
    move_distances = np.sqrt((np.diff(s_history, axis=-2) ** 2).sum(axis=-1))
    threshold = np.sqrt(1/env_h ** 2 + 1/env_w ** 2) * 0.5
    successful = (move_distances < threshold).sum(axis=-1) < seq_len - 1

    mse_mats = []
    for location in quantity_history:
        MSEs = []
        for seq_h in location:
            mse = np.sum((location - np.tile(seq_h, (n_act_sequences, 1))) ** 2, axis=-1)
            MSEs.append(mse)
        mse_mat = np.stack(MSEs, axis=0)
        mse_mats.append(mse_mat)

    n_rows = ceil(len(mse_mats) / max_n_cols)
    fig, axes = plt.subplots(n_rows, max_n_cols, figsize=(16, 10))
    ticks = np.arange(0, n_act_sequences, 1.0)
    labels = [''.join([word[0] for word in descr.split('_')]) for descr in seq_descr]

    for mse_mat, ax, start_location, success_in_location in zip(mse_mats, axes.flat, start_locations, successful):
        if False in success_in_location:
            mat_labels = [l if success else l + '*' for l, success in zip(labels, success_in_location)]
        else:
            mat_labels = labels
        im = ax.matshow(mse_mat)
        ax.title.set_text(start_location)
        ax.set_xticks(ticks)
        ax.set_xticklabels(mat_labels, rotation=90)
        ax.xaxis.set_ticks_position('bottom')
        ax.set_yticks(ticks)
        ax.set_yticklabels(mat_labels)
        ax.grid(False)

    plt.subplots_adjust(top=0.9, wspace=0.2, hspace=0.5)
    fig.colorbar(im, ax=axes.flat, shrink=0.95)
    fig.suptitle(title)
    plt.show()

    return mse_mats


def cluster_h(h_history):
    n_act_sequences = len(h_history.keys())
    h_history = list(h_history.values())
    n_locations = len(h_history[0])
    h_history = np.stack(h_history)  # shape: (seq, location, h)
    labels_gt = np.zeros((n_act_sequences, n_locations, 1)) + np.arange(0, n_act_sequences)[:, None, None]

    h_history = h_history.reshape((-1, h_history.shape[-1]))  # shape: (seq * location, h)
    labels_gt = labels_gt.flatten()

    # shuffle
    indices = list(range(len(h_history)))
    shuffle(indices)
    h_history = h_history[indices]
    labels_gt = labels_gt[indices]

    kmeans = KMeans(n_clusters=n_act_sequences).fit(h_history)
    labels_assigned = kmeans.labels_

    #n_rows = ceil(n_act_sequences / max_n_cols)
    #fig, axes = plt.subplots(n_rows, max_n_cols, figsize=(16, 10))
    #for cur_class, ax in enumerate(axes.flat):
    #    gt_class_members = labels_gt == cur_class
    #    ax.hist(labels_assigned[gt_class_members])
    #    ax.set_xlim([0, n_act_sequences])
    #plt.show()

    fig = plt.figure(figsize=(16, 10))
    for cur_class in range(n_act_sequences):
        cluster_members = labels_assigned == cur_class
        plt.hist(labels_gt[cluster_members], label=cur_class)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    env = Gridworld.from_cleartext(here() / '../../mdm/gridworld/8x8_v0.mapdata')
    mdl: MultiscaleDynamicsModel = torch.load(here() / 'model.ptmdl')
    mem = TrajectoryMemory.load(here() / 'gridworld_train.samples')
    driver = OfflineRLDriver(mem)

    # define parameters
    n_locations = 9
    action_sequences = {
        'UP_UP_UP': [0, 0, 0],
        'UP_UP_RIGHT': [0, 0, 1],
        'UP_RIGHT_RIGHT': [0, 1, 1],
        'RIGHT_RIGHT_RIGHT': [1, 1, 1],
        'RIGHT_RIGHT_DOWN': [1, 1, 2],
        'RIGHT_DOWN_DOWN': [1, 2, 2],
        'DOWN_DOWN_DOWN': [2, 2, 2],
        'DOWN_DOWN_LEFT': [2, 2, 3],
        'DOWN_LEFT_LEFT': [2, 3, 3],
        'LEFT_LEFT_LEFT': [3, 3, 3],
        'LEFT_LEFT_UP': [3, 3, 0],
        'LEFT_UP_UP': [3, 0, 0],
    }
    mode = 'various'

    #action_sequences = {
    #    'UP_UP_RIGHT': [0, 0, 1],
    #    'UP_RIGHT_UP': [0, 1, 0],
    #    'RIGHT_UP_UP': [1, 0, 0],
    #}
    #mode = 'similar'

    action_sequences = {k: torch.tensor(v,device=mdl.device) for k, v in action_sequences.items()}  # convert to tensor
    h_history = {k: None for k in action_sequences.keys()}
    s_history = {k: None for k in action_sequences.keys()}
    macro_s_start_history = {k: None for k in action_sequences.keys()}
    macro_s_prior_history = {k: None for k in action_sequences.keys()}
    macro_s_posterior_history = {k: None for k in action_sequences.keys()}

    # generate data
    s_start = []
    for i_loc in range(n_locations):
        traj = find_suitable_trajectory(mem, mdl.abstract_step_size, True)
        s_start.append(traj['s'][0])
    s_start = np.stack(s_start, axis=0)

    s_start = torch.from_numpy(s_start).to(mdl.device)  # to torch tensor
    s_start = s_start.unsqueeze(1)  # add time dimension
    s_start = s_start.float() / torch.tensor((env.grid_h - 1, env.grid_w - 1), device=mdl.device)  # normalize

    for seq_descr, a_seq in action_sequences.items():
        a_seq_batch = torch.tile(a_seq, dims=(n_locations, 1))  # copy same starting observation along batch
        a_seq_batch = torch.nn.functional.one_hot(a_seq_batch, num_classes=mdl.d_action)
        s, s_dist, r, r_dist, h = mdl.rollout_single_step(s_start, a_seq_batch)
        h_history[seq_descr] = h#mdl.filter_single_step_model_history(h)
        s_history[seq_descr] = torch.concat([s_start, s], dim=1)

    for seq_descr, h in h_history.items():
        zero_macro_s = torch.zeros(n_locations, mdl.d_macro_state, device=mdl.device)
        zero_macro_a = torch.zeros(n_locations, mdl.d_macro_action, device=mdl.device)
        predictions = mdl.macro_next_posterior(zero_macro_s, zero_macro_a, h)
        macro_s_next_post, macro_s_next_post_dist, macro_r_next_post, macro_r_next_post_dist = predictions
        macro_s_start_history[seq_descr] = macro_s_next_post

    for seq_descr, a_seq in action_sequences.items():
        macro_s_start = macro_s_start_history[seq_descr].unsqueeze(1)
        a_seq_batch = torch.tile(a_seq, dims=(n_locations, 1))  # copy same starting observation along batch
        a_seq_batch = torch.nn.functional.one_hot(a_seq_batch, num_classes=mdl.d_action).to(dtype=torch.float32)
        macro_a = mdl.macro_action_model(a_seq_batch).unsqueeze(1)

        # prior
        predictions = mdl.rollout_abstract(macro_s_start, macro_a)
        macro_s_prior, macro_s_prior_dist, macro_r_prior, macro_r_prior_dist = predictions
        macro_s_prior_history[seq_descr] = macro_s_prior.squeeze(1)

        # posterior
        predictions = mdl.macro_next_posterior(macro_s_start.squeeze(1), macro_a.squeeze(1), h_history[seq_descr])
        macro_s_post, macro_s_post_dist, macro_r_post, macro_r_post_dist = predictions
        macro_s_posterior_history[seq_descr] = macro_s_post

    h_history = {k: mdl.filter_single_step_model_history(v).detach().cpu().numpy() for k, v in h_history.items()}
    s_history = {k: v.detach().cpu().numpy() for k, v in s_history.items()}
    macro_s_start_history = {k: v.detach().cpu().numpy() for k, v in macro_s_start_history.items()}
    macro_s_prior_history = {k: v.detach().cpu().numpy() for k, v in macro_s_prior_history.items()}
    macro_s_posterior_history = {k: v.detach().cpu().numpy() for k, v in macro_s_posterior_history.items()}

    mse_per_location(f'h {mode} sequences', h_history, s_history, env.grid_h, env.grid_w, 3)
    mse_per_location(f'macro_s_init {mode} sequences', macro_s_start_history, s_history, env.grid_h, env.grid_w, 3)
    mse_per_location(f'macro_s_prior {mode} sequences', macro_s_prior_history, s_history, env.grid_h, env.grid_w, 3)
    mse_per_location(f'macro_s_post {mode} sequences', macro_s_posterior_history, s_history, env.grid_h, env.grid_w, 3)
    macro_s_prior_history.update({k + '_P': v for k, v in macro_s_posterior_history.items()})
    mse_per_location(f'macro_s_prior_post_comp {mode} sequences', macro_s_prior_history, s_history, env.grid_h, env.grid_w, 3)
    cluster_h(h_history)

    
