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
from mdm.utils.utils import here, gen_macro_state_map
from mdm.memory.trajectory_memory import TrajectoryMemory
from mdm.training.offline_rl_driver import OfflineRLDriver
from mdm.memory.trajectory_memory import flatten_and_unsqueeze

if __name__ == '__main__':
    env = Gridworld.from_cleartext(here() / '../../mdm/gridworld/8x8_v1.mapdata')
    mdl: MultiscaleDynamicsModel = torch.load(here() / 'model.ptmdl')
    n_trials = 10

    free_locations, macro_s_init_mean, macro_s_init_std = gen_macro_state_map(env, mdl, n_trials)
    #macro_s_init_mean = macro_s_init_std

    plot_mats = np.ones((env.grid_h, env.grid_w, mdl.d_macro_state)) * macro_s_init_mean.min(axis=0)
    for loc, data in zip(free_locations, macro_s_init_mean):
        plot_mats[tuple(loc)] = data
    plot_mats = np.transpose(plot_mats, [2, 0, 1])

    max_n_cols = 4
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