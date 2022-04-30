from math import ceil
from random import shuffle
from itertools import product

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns; sns.set_theme()
from mpl_toolkits.axes_grid1 import ImageGrid
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

    available_actions = [1]
    macro_s_init_mean, macro_s_init_std = gen_macro_state_map(env, mdl, n_trials)
    #macro_s_init_mean = macro_s_init_std

    plot_mats = np.zeros((env.grid_h, env.grid_w, mdl.d_macro_state))
    for loc, data in macro_s_init_mean.items():
        plot_mats[tuple(loc)] = data
    plot_mats = np.transpose(plot_mats, [2, 0, 1])
    min_val, max_val = np.amin(plot_mats), np.amax(plot_mats)

    max_n_cols = 2
    n_rows = ceil(len(plot_mats) / max_n_cols)
    fig, axes = plt.subplots(n_rows, max_n_cols, figsize=(16, 10))
    for i_comp, (mat, ax) in enumerate(zip(plot_mats, axes.flat)):
        ax.grid(False)
        im = ax.matshow(mat, vmin=min_val, vmax=max_val)
        ax.set_axis_off()
        ax.title.set_text(f'Component {i_comp}')
    fig.suptitle('macro_s_init inidvidual components averaged over all action sequences')
    #plt.tight_layout()
    fig.colorbar(im, ax=axes.ravel().tolist())
    #cax, kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
    #plt.colorbar(im, cax=cax)
    plt.show()

    #fig = plt.figure(figsize=(16, 10))
    #grid = ImageGrid(fig, 111, nrows_ncols=(n_rows, max_n_cols), axes_pad=0.15, share_all=True, cbar_location='right',
    #                 cbar_mode='single', cbar_size='10%', cbar_pad=0.45)
#
#    for i_comp, (mat, ax) in enumerate(zip(plot_mats, grid)):
#        ax.grid(False)
#        im = ax.matshow(mat)
#        ax.set_axis_off()
#        ax.title.set_text(f'Component {i_comp}')
#    fig.suptitle('macro_s_init inidvidual components averaged over all action sequences')
#    plt.show()
