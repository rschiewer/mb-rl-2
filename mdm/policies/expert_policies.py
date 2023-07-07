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
from mdm.utils.gym_wrappers import CacheLastStepEnv, CacheLastStepVecEnv
from mdm.utils.utils import visualize_trajectory


def nav2d_expert_policy_old(env: CacheLastStepEnv):
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


def nav2d_expert_policy(env: CacheLastStepEnv | CacheLastStepVecEnv):

    if isinstance(env, CacheLastStepEnv):
        o = env.last_o
        agent_pos = o[:2]
        goal_pos = o[2:4]
        distance = o[4]

        noise = (np.random.random(2) - 0.5) * 0.5
        direction_vec = np.clip((goal_pos - agent_pos) / np.linalg.norm(goal_pos - agent_pos) + noise, -1.0, 1.0)
        a = direction_vec.astype(np.float32)

        """
        plt.clf()
        plt.scatter(agent_pos[0], agent_pos[1], c='red')
        plt.scatter(goal_pos[0], goal_pos[1], c='green')
        plt.xlim((-1, 1))
        plt.ylim((-1, 1))
        plt.ion()
        plt.pause(0.005)
        plt.show()
        """

    else:
        o = env.last_o
        agent_pos = o[:, :2]
        goal_pos = o[:, 2:4]
        distance = o[:, 4]
        d_batch = agent_pos.shape[0]

        noise = (np.random.random((d_batch, 2)) - 0.5) * 0.5
        direction_vec = np.clip((goal_pos - agent_pos) / np.linalg.norm(goal_pos - agent_pos) + noise, -1.0, 1.0)
        a = direction_vec.astype(np.float32)

    return a


"""
def expert_collect_policy(env: CacheLastStepVecEnv):
    o = env.last_o
    agent_pos = o[:, :2]
    goal_pos = o[:, 2:4]
    distance = o[:, 4]
    adjacent = (goal_pos[:, 1] - agent_pos[:, 1])
    disjacent = (goal_pos[:, 0] - agent_pos[:, 0])
    angle = np.arctan2(adjacent, disjacent) + math.pi * 1.5

    dist_a = np.where(distance > 0.05, 1.0, 0.1)
    angle = np.where(angle > 2 * math.pi, angle - 2 * math.pi, angle)
    angle_a = angle / (2 * math.pi) * 2 - 1
    angle_a += (np.random.random() - 0.5) * 0.5
    angle_a = np.clip(angle_a, -1.0, 1.0)
    a = np.stack([angle_a, dist_a], axis=1).astype(np.float32)

    return a
"""

