import numpy as np
import gymnasium as gym
from gymnasium_robotics.envs.maze.maze_v4 import MazeEnv
import matplotlib.pyplot as plt

from mdm.utils.utils import env_class_is, get_env_instance


def visualize_env(env: gym.Env, canvas: plt.Figure | plt.Axes, **kwargs):
    if env_class_is(env, MazeEnv):
        plot_maze_env(env=env, canvas=canvas, **kwargs)


def plot_maze_env(env, canvas: plt.Axes = None, observations: np.ndarray = None, goal_observations: np.ndarray = None):
    assert env_class_is(env, MazeEnv)
    if canvas is None:
        plt_target = plt
    else:
        plt_target = canvas
    instance = get_env_instance(env).unwrapped
    x_center = instance.maze.x_map_center
    y_center = instance.maze.y_map_center
    length = instance.maze.map_length
    width = instance.maze.map_width
    start_locations = instance.maze.unique_reset_locations
    goal_locations = instance.maze.unique_goal_locations
    maze_map = instance.maze.maze_map
    # remove maze_map start and goal locations
    for row in maze_map:
        for x in range(len(row)):
            if type(row[x]) is str:
                row[x] = 0
    maze_map = np.array(maze_map)
    left = - width / 2
    right = width / 2
    bottom = - length / 2
    top = length / 2
    plt_target.imshow(maze_map, extent=(left, right, bottom, top))
    for loc in start_locations:
        plt_target.scatter(loc[0], loc[1], marker='.', c='red', s=500)
        plt_target.scatter(loc[0], loc[1], marker='$S$', c='white', s=45)
    for loc in goal_locations:
        plt_target.scatter(loc[0], loc[1], marker='.', c='green', s=500)
        plt_target.scatter(loc[0], loc[1], marker='$R$', c='white', s=45)

    # plot individual trajectories
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    goal_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S']
    if observations is not None:
        for i_traj in range(observations.shape[1]):
            c = colors[i_traj % len(colors)]
            for t in range(observations.shape[0]):
                plt_target.scatter(observations[t, i_traj, 4], observations[t, i_traj, 5], marker=f'${t}$', c=c, s=25)
    if goal_observations is not None:
        for i_traj in range(goal_observations.shape[1]):
            c = colors[i_traj % len(colors)]
            for t in range(goal_observations.shape[0]):
                plt_target.scatter(goal_observations[t, i_traj, 4] - 0.002, goal_observations[t, i_traj, 5] - 0.002,
                                   marker=f'${goal_names[t]}$', c='black', s=25)
                plt_target.scatter(goal_observations[t, i_traj, 4], goal_observations[t, i_traj, 5],
                                   marker=f'${goal_names[t]}$', c=c, s=25)
