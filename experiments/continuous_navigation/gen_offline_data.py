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


def expert_collect_policy(env: CacheLastStepEnv):
    o = env.last_o
    agent_pos = o[:2]
    goal_pos = o[2:4]
    distance = o[4]
    adjacent = (goal_pos[1] - agent_pos[1])
    disjacent = (goal_pos[0] - agent_pos[0])
    angle = math.atan2(adjacent, disjacent) + math.pi * 1.5

    dist_a = 1.0 if distance > 0.05 else 0.1
    if angle > 2 * math.pi:
        angle -= 2 * math.pi

    angle_a = angle / (2 * math.pi) * 2 - 1
    angle_a += (np.random.random() - 0.5) * 0.5
    angle_a = np.clip(angle_a, -1.0, 1.0)
    a = np.array([angle_a, dist_a], dtype=np.float32)  # a = env.action_space.sample()

    """
    plt.clf()
    plt.scatter(agent_pos[0], agent_pos[1], c='red')
    plt.scatter(goal_pos[0], goal_pos[1], c='green')
    plt.xlim((-1, 1))
    plt.ylim((-1, 1))
    plt.ion()
    plt.pause(0.001)
    plt.show()
    """

    return a


if __name__ == '__main__':
    map_version = 'Easy'
    env = gym.make(f'gym_nav2d:nav2d{map_version}-v0')
    env = CacheLastStepEnv(env)
    n_episodes_train = 5000
    perc_test = 0.1
    expert_trajectories = 0.0

    train_mem = []
    if expert_trajectories > 0:
        def collect_policy(*args):
            return expert_collect_policy(*args)
    else:
        def collect_policy(*args):
            return env.action_space.sample()

    collect_driver = GymEpisodeDriver(env, collect_policy)
    collect_driver.interact(round(n_episodes_train * expert_trajectories), True, train_mem)

    rand_driver = GymEpisodeDriver(env, lambda *args: env.action_space.sample())
    rand_driver.interact(n_episodes_train - len(train_mem), True, train_mem)

    n_episodes_test = round(len(train_mem) * perc_test)
    random.shuffle(train_mem)
    train_mem = train_mem[n_episodes_test:]
    test_mem = train_mem[:n_episodes_test]

    expert_mem = []
    expert_driver = GymEpisodeDriver(env, expert_collect_policy)
    expert_driver.interact(3000, True, expert_mem)

    store_memory(train_mem, here() / f'nav2d_{map_version}_train.samples')
    store_memory(test_mem, here() / f'nav2d_{map_version}_test.samples')
    store_memory(expert_mem, here() / f'nav2d_{map_version}_expert.samples')


