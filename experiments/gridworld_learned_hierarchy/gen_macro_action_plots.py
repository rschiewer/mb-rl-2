from itertools import product

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

    n_trials = 50

    available_actions = list(range(env.action_space.n))
    action_sequences = list(product(available_actions, repeat=mdl.macro_step_size))
    n_unique_sequences = len(action_sequences)
    action_sequences *= n_trials
    action_sequences = [torch.tensor(s) for s in action_sequences]
    action_sequences = torch.stack(action_sequences, dim=0).to(mdl.device)
    action_sequences = torch.nn.functional.one_hot(action_sequences, num_classes=mdl.d_action).to(dtype=torch.float32)

    macro_actions = mdl.macro_action_model(action_sequences)
    macro_actions = macro_actions.detach().cpu().numpy().argmax(axis=-1)
    histogram_x, histogram_y = np.unique(macro_actions, return_counts=True)

    macro_actions = macro_actions.reshape(n_trials, n_unique_sequences)
    macro_actions = macro_actions.transpose(1, 0)
    macro_actions_var = macro_actions.std(axis=1)

    plt.bar(histogram_x, histogram_y)
    plt.show()

    plt.scatter(list(range(len(macro_actions))), macro_actions_var)
    plt.show()
