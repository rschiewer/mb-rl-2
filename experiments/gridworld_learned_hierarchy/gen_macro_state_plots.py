from math import ceil
from random import shuffle
from itertools import product

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from mdm.gridworld.gridworld import Gridworld, CellType
from mdm.models.multiscale_model import MultiscaleDynamicsModel
from mdm.utils.utils import here
from mdm.memory.trajectory_memory import TrajectoryMemory
from mdm.training.offline_rl_driver import OfflineRLDriver
from mdm.memory.trajectory_memory import flatten_and_unsqueeze


def gen_macro_state_map(env: Gridworld, mdl: MultiscaleDynamicsModel, n_trials: int):
    available_actions = list(range(env.action_space.n))
    action_sequences = list(product(available_actions, repeat=mdl.macro_step_size)) * n_trials
    free_locations = env.find_cell_type(CellType.FREE)
    n_locations = len(free_locations)

    # convert to tensors
    action_sequences = [torch.tensor(s).to(mdl.device) for s in action_sequences]
    s_start = torch.from_numpy(free_locations).to(mdl.device)
    s_start = s_start.unsqueeze(1)  # add time dimension
    s_start = s_start.float() / torch.tensor((env.grid_h - 1, env.grid_w - 1), device=mdl.device)  # normalize

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
    macro_s_init_mean = macro_s_init_history.mean(axis=0)
    macro_s_init_mean = (macro_s_init_mean + np.abs(macro_s_init_mean.min(axis=0))) / (macro_s_init_mean.max(axis=0)
                                                                                       - macro_s_init_mean.min(axis=0))
    return free_locations, macro_s_init_mean


if __name__ == '__main__':
    env = Gridworld.from_cleartext(here() / '../../mdm/gridworld/8x8_v1.mapdata')
    mdl: MultiscaleDynamicsModel = torch.load(here() / 'model.ptmdl')
    n_trials = 50

    free_locations, macro_s_init_mean = gen_macro_state_map(env, mdl, n_trials)

    plot_mats = np.ones((env.grid_h, env.grid_w, mdl.d_macro_state)) * macro_s_init_mean.min(axis=0)
    for loc, data in zip(free_locations, macro_s_init_mean):
        plot_mats[tuple(loc)] = data
    plot_mats = np.transpose(plot_mats, [2, 0, 1])

    max_n_cols = 5
    n_rows = ceil(len(plot_mats) / max_n_cols)
    fig, axes = plt.subplots(n_rows, max_n_cols, figsize=(16, 10))
    for i_comp, (mat, ax) in enumerate(zip(plot_mats, axes.flat)):
        ax.grid(False)
        ax.matshow(mat)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.title.set_text(f'Component {i_comp}')
    fig.suptitle('macro_s_init inidvidual components averaged over all action sequences')
    plt.show()