import argparse
from itertools import product
from math import ceil

import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

from mdm.gridworld.gridworld import Gridworld, CellType
from mdm.models.multiscale_model_mk2 import MultiscaleDynamicsModelMK2
from mdm.memory.trajectory_memory import TrajectoryMemory
from mdm.training.offline_rl_driver import OfflineRLDriver
from mdm.utils.utils import here, load_yaml


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Provide neptune run_id for loading the correct model')
    parser.add_argument('id', type=str, nargs=1)
    parser.add_argument('-log', action='store_true')
    args = parser.parse_args()

    cfg = load_yaml(here() / 'model_mk2.yaml')
    neptune_cfg = load_yaml(here() / cfg['neptune_cfg'])

    if len(args.id) == 0:
        model_path = f'{cfg["final_model_path"]}.ptmdl'
    else:
        model_path = f'{cfg["final_model_path"]}_{args.id[0]}.ptmdl'

    env = Gridworld.from_cleartext(here() / '../../mdm/gridworld/8x8_v0.mapdata')
    mdl: MultiscaleDynamicsModelMK2 = torch.load(here() / model_path)

    n_trials = 10

    available_actions = list(range(env.action_space.n))
    action_sequences = list(product(available_actions, repeat=mdl.abstract_step_size))
    n_unique_sequences = len(action_sequences)
    action_sequences *= n_trials
    action_sequences = [torch.tensor(s) for s in action_sequences]
    action_sequences = torch.stack(action_sequences, dim=0).to(mdl.device)
    action_sequences = torch.nn.functional.one_hot(action_sequences, num_classes=mdl.d_action).to(dtype=torch.float32)

    abstr_a = mdl.abstract_action_model(action_sequences)
    abstr_a = abstr_a.detach().cpu().numpy()
    #histogram_x, histogram_y = np.unique(macro_actions, return_counts=True)

    abstr_a = abstr_a.reshape(n_trials, n_unique_sequences, mdl.d_abstract_action)
    abstr_a = abstr_a.transpose(1, 0, 2)  # bring sequence index to front
    abstr_a_mean = abstr_a.mean(axis=1)
    abstr_a_var = abstr_a.std(axis=1)

    max_cols = 5
    n_rows = ceil(n_unique_sequences / max_cols)
    fig, axes = plt.subplots(n_rows, max_cols, figsize=(20, 14))
    for ax, a_mean, a_var in zip(axes.flat, abstr_a_mean, abstr_a_var):
        ax.bar(list(range(mdl.d_abstract_action)), a_mean, yerr=a_var)
    plt.tight_layout()
    plt.show()

    mse = np.zeros((len(abstr_a_mean), len(abstr_a_mean)))
    for i in range(len(abstr_a_mean)):
        for j in range(i, len(abstr_a_mean)):
            diff = np.sum((abstr_a_mean[i] - abstr_a_mean[j]) ** 2)
            mse[i, j] = diff

    mse /= mse.max()

    plt.matshow(mse)
    plt.colorbar()
    plt.show()

    #plt.bar(histogram_x, histogram_y)
    #plt.show()

    #plt.scatter(list(range(len(macro_actions))), macro_actions_mean)
    #plt.show()

    #plt.scatter(list(range(len(macro_actions))), macro_actions_var)
    #plt.show()