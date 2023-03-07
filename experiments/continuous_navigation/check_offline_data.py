import pickle

from tqdm import tqdm
import numpy as np
import gym
import gym_nav2d

from mdm.memory.trajectory_memory import TrajectoryMemory
from mdm.utils.utils import here, load_yaml, load_memory, random_walk_success_rate
from mdm.utils.analysis_tools import plot_trajectory_stats


if __name__ == '__main__':
    cfg = load_yaml(here() / 'cfg_simple_rssm_train.yaml')
    planning_cfg = load_yaml(here() / 'cfg_rssm_plan.yaml')

    map_version = 'VeryEasy'
    env = gym.make(f'gym_nav2d:nav2d{map_version}-v0')

    success, r_ep, l_ep = random_walk_success_rate(env, 50, planning_cfg['n_rollouts'][0])
    print(f'Random walk statistics with planning parameters:')
    print(f'Expected initial success rate: {success}')
    print(f'Expected initial average return: {r_ep.mean()}')
    print(f'Expected initial average episode length: {l_ep.mean()}')

    train_mem = load_memory(here() / cfg['train_samples'])
    test_mem = load_memory(here() / cfg['test_samples'])

    plot_trajectory_stats(train_mem, bins=20)
    plot_trajectory_stats(test_mem, bins=20)
