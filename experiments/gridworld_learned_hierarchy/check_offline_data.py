import pickle

from tqdm import tqdm
import numpy as np

from mdm.gridworld.gridworld import Gridworld, FullyObservableGridworld
from mdm.memory.trajectory_memory import TrajectoryMemory
from mdm.utils.utils import here, load_yaml
from mdm.utils.analysis_tools import plot_trajectory_stats


if __name__ == '__main__':
    cfg = load_yaml(here() / 'model_cfg.yaml')

    map_version = 'v0'
    env = Gridworld.from_cleartext(here() / f'../../mdm/gridworld/8x8_{map_version}.mapdata')
    #env = FullyObservableGridworld(env)

    with open(here() / cfg['train_samples'], 'rb') as f:
        train_mem = pickle.load(f)
    with open(here() / cfg['test_samples'], 'rb') as f:
        test_mem = pickle.load(f)

    plot_trajectory_stats(train_mem, bins=20)
    plot_trajectory_stats(test_mem, bins=20)

    for traj in tqdm(train_mem, desc='Checking train samples'):
        s, a, r, terminal, truncated = traj.values()
        env.enact_sequence(s, a, r, terminal, truncated, False, 0)

    for traj in tqdm(test_mem, desc='Checking test samples'):
        s, a, r, terminal, truncated = traj.values()
        env.enact_sequence(s, a, r, terminal, truncated, False, 0)


