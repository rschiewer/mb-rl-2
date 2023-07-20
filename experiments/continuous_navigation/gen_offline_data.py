import random
import math
import pickle
from multiprocessing import Pool

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import gym
import gym_nav2d

from mdm.training.gym_driver import GymEpisodeDriver
from mdm.memory.trajectory_memory import TrajectoryMemory
from mdm.gridworld.gridworld import Gridworld, CellType, FullyObservableGridworld
from mdm.utils.utils import here, to_np_arrays, store_memory
from mdm.utils.gym_wrappers import CacheLastStepEnv
from mdm.policies.expert_policies import nav2d_expert_policy
from mdm.utils.gym_nav2d_tools import visualize_trajectory

if __name__ == '__main__':
    map_version = 'EasySparse'
    env = gym.make(f'gym_nav2d:nav2d{map_version}-v0')
    env = CacheLastStepEnv(env)
    n_episodes_train = 5000
    perc_test = 0.1
    expert_trajectories = 0.0

    train_mem = []
    if expert_trajectories > 0:
        def collect_policy(*args):
            return nav2d_expert_policy(*args)
    else:
        def collect_policy(*args):
            return env.action_space.sample()

    collect_driver = GymEpisodeDriver(env, collect_policy)
    collect_driver.interact(round(n_episodes_train * expert_trajectories), train_mem)

    rand_driver = GymEpisodeDriver(env, lambda *args: env.action_space.sample())
    rand_driver.interact(n_episodes_train - len(train_mem), train_mem)

    #fig, ani = visualize_trajectory(train_mem[0])
    #plt.show()
    #quit()

    n_episodes_test = round(len(train_mem) * perc_test)
    random.shuffle(train_mem)
    train_mem = train_mem[n_episodes_test:]
    test_mem = train_mem[:n_episodes_test]

    expert_mem = []
    expert_driver = GymEpisodeDriver(env, nav2d_expert_policy)
    expert_driver.interact(3000, expert_mem)

    store_memory(train_mem, here() / f'nav2d_{map_version}_train.samples')
    store_memory(test_mem, here() / f'nav2d_{map_version}_test.samples')
    store_memory(expert_mem, here() / f'nav2d_{map_version}_expert.samples')


