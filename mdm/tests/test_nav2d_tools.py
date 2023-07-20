import gym
import gym_nav2d
import numpy as np
import torch
import matplotlib.pyplot as plt

from mdm.utils.gym_nav2d_tools import *

env = gym.make('gym_nav2d:nav2dEasySparse-v0')

trajectories = gen_regular_grid_trajectories(env, 10, 10, 0.5)

for i_traj, traj in enumerate(trajectories):
    coords = np.array(traj['o'])
    plt.scatter(coords[:, 0], coords[:, 1], label=str(i_traj))

plt.legend()
plt.show()


