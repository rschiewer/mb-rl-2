import pickle

from tqdm import tqdm
import numpy as np

from mdm.gridworld.gridworld import Gridworld, FullyObservableGridworld
from mdm.memory.trajectory_memory import TrajectoryMemory
from mdm.utils.utils import here, load_yaml, load_memory
from mdm.utils.analysis_tools import plot_trajectory_stats


if __name__ == '__main__':
    cfg = load_yaml(here() / 'model_cfg.yaml')

    map_version = 'v0'
    env = Gridworld.from_cleartext(here() / f'../../mdm/gridworld/8x8_{map_version}.mapdata')
    #env = FullyObservableGridworld(env)

    train_mem = load_memory(here() / cfg['train_samples'])
    test_mem = load_memory(here() / cfg['test_samples'])

    plot_trajectory_stats(train_mem, bins=20)
    plot_trajectory_stats(test_mem, bins=20)

    for traj in tqdm(train_mem, desc='Checking train samples'):
        s, a, r, terminal, truncated = traj.values()
        env.enact_sequence(s, a, r, terminal, truncated, False, 0)

    for traj in tqdm(test_mem, desc='Checking test samples'):
        s, a, r, terminal, truncated = traj.values()
        env.enact_sequence(s, a, r, terminal, truncated, False, 0)


