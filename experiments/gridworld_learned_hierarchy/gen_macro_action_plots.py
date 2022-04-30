from itertools import product
from math import ceil

import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

from mdm.gridworld.gridworld import Gridworld, CellType
from mdm.models.multiscale_model import MultiscaleDynamicsModel
from mdm.memory.trajectory_memory import TrajectoryMemory
from mdm.training.offline_rl_driver import OfflineRLDriver
from mdm.utils.utils import here


if __name__ == '__main__':
    env = Gridworld.from_cleartext(here() / '../../mdm/gridworld/8x8_v0.mapdata')
    mdl: MultiscaleDynamicsModel = torch.load(here() / 'model.ptmdl')
    mem = TrajectoryMemory.load(here() / 'gridworld_train.samples')
    driver = OfflineRLDriver(mem)

    n_trials = 20

    available_actions = list(range(env.action_space.n))
    action_sequences = list(product(available_actions, repeat=mdl.macro_step_size))
    n_unique_sequences = len(action_sequences)
    action_sequences *= n_trials
    action_sequences = [torch.tensor(s) for s in action_sequences]
    action_sequences = torch.stack(action_sequences, dim=0).to(mdl.device)
    action_sequences = torch.nn.functional.one_hot(action_sequences, num_classes=mdl.d_action).to(dtype=torch.float32)

    macro_actions = mdl.macro_action_model(action_sequences)
    macro_actions = macro_actions.detach().cpu().numpy()
    #histogram_x, histogram_y = np.unique(macro_actions, return_counts=True)

    macro_actions = macro_actions.reshape(n_trials, n_unique_sequences, mdl.d_macro_action)
    macro_actions = macro_actions.transpose(1, 0, 2)  # bring sequence index to front
    macro_actions_mean = macro_actions.mean(axis=1)
    macro_actions_var = macro_actions.std(axis=1)

    max_cols = 5
    n_rows = ceil(n_unique_sequences / max_cols)
    fig, axes = plt.subplots(n_rows, max_cols, figsize=(20, 14))
    for ax, macro_a_mean, macro_a_std in zip(axes.flat, macro_actions_mean, macro_actions_var):
        ax.bar(list(range(mdl.d_macro_action)), macro_a_mean, yerr=macro_a_std)
    plt.tight_layout()
    plt.show()

    mse = np.zeros((len(macro_actions_mean), len(macro_actions_mean)))
    for i in range(len(macro_actions_mean)):
        for j in range(i, len(macro_actions_mean)):
            diff = np.sum((macro_actions_mean[i] - macro_actions_mean[j]) ** 2)
            mse[i, j] = diff

    mse /= mse.max()

    plt.matshow(mse)
    plt.show()

    #plt.bar(histogram_x, histogram_y)
    #plt.show()

    #plt.scatter(list(range(len(macro_actions))), macro_actions_mean)
    #plt.show()

    #plt.scatter(list(range(len(macro_actions))), macro_actions_var)
    #plt.show()