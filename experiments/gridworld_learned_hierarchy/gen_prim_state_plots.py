import argparse

from math import ceil
from random import shuffle
from itertools import product

import torch
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import seaborn as sns; sns.set_theme()
from mpl_toolkits.axes_grid1 import ImageGrid

from mdm.gridworld.gridworld import Gridworld, CellType
from mdm.models.multiscale_model_mk2 import MultiscaleDynamicsModelMK2
from mdm.utils.utils import here, gen_macro_state_map, load_yaml, gen_prim_state_map
from mdm.memory.trajectory_memory import TrajectoryMemory
from mdm.training.offline_rl_driver import OfflineRLDriver
from mdm.memory.trajectory_memory import flatten_and_unsqueeze


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Provide neptune run_id for loading the correct model')
    parser.add_argument('id', type=str, nargs=1)
    parser.add_argument('-log', action='store_true')
    args = parser.parse_args()

    cfg = load_yaml(here() / 'model_cfg.yaml')
    neptune_cfg = load_yaml(here() / cfg['neptune_cfg'])

    if len(args.id) == 0:
        model_path = f'{cfg["final_model_path"]}.ptmdl'
    else:
        model_path = f'{cfg["final_model_path"]}_{args.id[0]}.ptmdl'

    env = Gridworld.from_cleartext(here() / '../../mdm/gridworld/8x8_v0.mapdata')
    mdl: MultiscaleDynamicsModelMK2 = torch.load(here() / model_path)

    maps_s_mean, maps_s_std = [], []
    for n_step in range(2):
        _map_s_mean, _map_s_std = gen_prim_state_map(env, mdl, n_step)
        maps_s_mean.append(_map_s_mean)
        maps_s_std.append(_map_s_std)
    maps_s_mean = np.stack(maps_s_mean, axis=1)
    maps_s_std = np.stack(maps_s_std, axis=1)

    all_values = np.concatenate([maps_s_mean, maps_s_std])
    v_min = all_values.min()
    v_max = all_values.max()

    max_n_cols = min(len(maps_s_mean), 4)
    n_rows = ceil(len(maps_s_mean) / max_n_cols)
    fig = plt.figure(constrained_layout=True, figsize=(20, 20))
    subfigs = fig.subfigures(n_rows, max_n_cols, wspace=0.1, hspace=0.2)
    animations = []
    for i, (subfig, _map_s_mean, _map_s_std) in enumerate(zip(subfigs.flat, maps_s_mean, maps_s_std)):
        subfig.suptitle(f'Component {i}')
        axes = subfig.subplots(1, 2)
        axes[0].grid(False)
        axes[1].grid(False)
        axes[0].set_title('mean')
        axes[1].set_title('std')
        axes[0].axis('off')
        axes[1].axis('off')

        ims = []
        for t, (frame_mean, frame_std) in enumerate(zip(_map_s_mean, _map_s_std)):
            im_0 = axes[0].matshow(frame_mean, vmin=v_min, vmax=v_max)
            im_1 = axes[1].matshow(frame_std, vmin=v_min, vmax=v_max)
            label = axes[0].text(0.01, 0.01, f'{t}', transform=axes[0].transAxes, color='green')
            ims.append([im_0, im_1, label])
        subfig.colorbar(im_0, ax=axes.ravel().tolist())
        ani = animation.ArtistAnimation(subfig, ims, interval=200, blit=True, repeat_delay=0)
        animations.append(ani)
        #subfig.set_facecolor('0.8')
    writer = animation.FFMpegWriter(fps=60)
    plt.show()







