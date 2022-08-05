import argparse

from math import ceil
from random import shuffle
from itertools import product

import torch
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
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

    map_s_mean, map_s_std = gen_prim_state_map(env, mdl)

    all_values = np.concatenate([map_s_mean, map_s_std])
    v_min = all_values.min()
    v_max = all_values.max()

    max_n_cols = min(len(map_s_mean), 4)
    n_rows = ceil(len(map_s_mean) / max_n_cols)
    fig = plt.figure(constrained_layout=True, figsize=(20, 20))
    subfigs = fig.subfigures(n_rows, max_n_cols, wspace=0.1, hspace=0.2)
    for i, (subfig, _map_s_mean, _map_s_std) in enumerate(zip(subfigs.flat, map_s_mean, map_s_std)):
        axes = subfig.subplots(1, 2)
        axes[0].matshow(_map_s_mean, vmin=v_min, vmax=v_max)
        im = axes[1].matshow(_map_s_std, vmin=v_min, vmax=v_max)
        subfig.suptitle(f'Component {i}')
        axes[0].grid(False)
        axes[1].grid(False)
        axes[0].set_title('mean')
        axes[1].set_title('std')
        axes[0].axis('off')
        axes[1].axis('off')
        subfig.colorbar(im, ax=axes.ravel().tolist())
        #subfig.set_facecolor('0.8')
    plt.show()







