import argparse

from math import ceil
from random import shuffle
from itertools import product

import torch
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import seaborn as sns; sns.set_theme()
from mpl_toolkits.axes_grid1 import ImageGrid

from mdm.gridworld.gridworld import Gridworld, CellType
from mdm.models.multiscale_model_mk2 import MultiscaleDynamicsModelMK2
from mdm.utils.utils import here, gen_macro_state_map, load_yaml, gen_prim_rnn_state_map
from mdm.memory.trajectory_memory import TrajectoryMemory
from mdm.training.offline_rl_driver import OfflineRLDriver
from mdm.memory.trajectory_memory import flatten_and_unsqueeze


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Provide neptune run_id for loading the correct model')
    parser.add_argument('id', type=str)
    parser.add_argument('-log', action='store_true')
    args = parser.parse_args()

    cfg = load_yaml(here() / 'model_cfg.yaml')
    neptune_cfg = load_yaml(here() / cfg['neptune_cfg'])

    if len(args.id) == 0:
        model_path = f'{cfg["final_model_path"]}.ptmdl'
    else:
        model_path = f'{cfg["final_model_path"]}_{args.id}.ptmdl'

    env = Gridworld.from_cleartext(here() / '../../mdm/gridworld/8x8_v0.mapdata')
    mdl: MultiscaleDynamicsModelMK2 = torch.load(here() / model_path)

    maps_s_mean, maps_s_std = [], []
    for n_step in range(5):
        _map_s_mean, _map_s_std = gen_prim_rnn_state_map(env, mdl, n_step)
        maps_s_mean.append(_map_s_mean)
        maps_s_std.append(_map_s_std)
    maps_s_mean = np.stack(maps_s_mean, axis=0)
    maps_s_std = np.stack(maps_s_std, axis=0)

    #v_min = np.minimum(maps_s_mean.min(), maps_s_std.min())
    #v_max = np.maximum(maps_s_mean.max(), maps_s_std.max())
    all_values = np.concatenate([maps_s_mean, maps_s_std])
    v_min = all_values.min()
    v_max = all_values.max()

    max_n_cols = min(maps_s_mean.shape[1], 4)
    n_rows = ceil(maps_s_mean.shape[1] / max_n_cols)
    fig = plt.figure( figsize=(20, 20))
    gs = gridspec.GridSpec(n_rows, max_n_cols, wspace=0.2, hspace=0.2, figure=fig, left=0.01, right=0.99, bottom=0.01,
                           top=0.99)
    subfig_axes = []
    for i, subfig in enumerate(gs):
        subfig_grid = gridspec.GridSpecFromSubplotSpec(1, 2, hspace=0.1, subplot_spec=subfig)
        axes_0 = fig.add_subplot(subfig_grid[0, 0])
        axes_1 = fig.add_subplot(subfig_grid[0, 1])
        axes_0.grid(False)
        axes_1.grid(False)
        axes_0.set_title('mean')
        axes_1.set_title('std')
        axes_0.axis('off')
        axes_1.axis('off')
        subfig_axes.append([axes_0, axes_1])

    artists = []
    for t, (frames_mean, frames_std) in enumerate(zip(maps_s_mean, maps_s_std)):
        current_frame = []
        for axes, frame_mean, frame_std in zip(subfig_axes, frames_mean, frames_std):
            im_0 = axes[0].matshow(frame_mean, vmin=v_min, vmax=v_max)
            im_1 = axes[1].matshow(frame_std, vmin=v_min, vmax=v_max)
            label = axes[0].text(0.01, 0.01, f'{t}', transform=axes[0].transAxes, color='green')
            current_frame += [im_0, im_1, label]
            if t == 0:
                fig.colorbar(im_0, ax=axes)
        artists.append(current_frame)

    ani = animation.ArtistAnimation(fig, artists, interval=1000, blit=True, repeat_delay=0)
    plt.tight_layout()
    plt.show()
    #writer = animation.PillowWriter(fps=10)
    #ani.save(f'prim_state_plot_{args.id}.gif', writer=writer)







