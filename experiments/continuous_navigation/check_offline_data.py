import pickle

from tqdm import tqdm
import numpy as np
import gym
import gym_nav2d

from mdm.memory.trajectory_memory import TrajectoryMemory
from mdm.utils.utils import here, load_yaml, load_memory
from mdm.utils.analysis_tools import plot_trajectory_stats


if __name__ == '__main__':
    cfg = load_yaml(here() / 'model_cfg.yaml')

    map_version = 'VeryEasy'
    env = gym.make(f'gym_nav2d:nav2d{map_version}-v0')

    train_mem = load_memory(here() / cfg['train_samples'])
    test_mem = load_memory(here() / cfg['test_samples'])

    plot_trajectory_stats(train_mem, bins=20)
    plot_trajectory_stats(test_mem, bins=20)
