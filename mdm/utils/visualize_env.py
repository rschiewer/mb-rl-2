import numpy as np
import gymnasium as gym
from gym_nav2d.envs import Nav2dEnv
from gymnasium_robotics.envs.maze.maze_v4 import MazeEnv
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse

from mdm.utils.utils import env_class_is, get_env_instance


def visualize_env(env: gym.Env, canvas: plt.Figure | plt.Axes, **kwargs):
    if canvas is None:
        canvas = plt.figure().add_subplot()
    elif isinstance(canvas, plt.Figure):
        canvas = canvas.add_subplot()

    if env_class_is(env, MazeEnv):
        plot_maze_env(env=env, canvas=canvas, **kwargs)
    elif env_class_is(env, Nav2dEnv):
        plot_nav2d_env(env=env, canvas=canvas, **kwargs)

#def draw_timestep(canvas: plt.Axes, x: float, y:float, i_t: int, size: float = 1, fontweight: int = 300):
#    canvas.add_patch(Ellipse((x, y), width=0.05 * size, height=0.05 * size, fill=None))
#    canvas.text(x, y, f'{t + 1}', c=c,
#                horizontalalignment='center', verticalalignment='center_baseline', fontweight=fontweight)

def plot_maze_env(env: gym.Env,
                  canvas: plt.Axes,
                  observations: np.ndarray = None,
                  goal_observations: np.ndarray = None,
                  n_trajs: int = 1):
    assert env_class_is(env, MazeEnv)

    # draw maze itself
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
    canvas.imshow(maze_map, extent=(left, right, bottom, top))
    for loc in start_locations:
        canvas.scatter(loc[0], loc[1], marker='.', c='red', s=500)
        canvas.scatter(loc[0], loc[1], marker='$S$', c='white', s=45)
    for loc in goal_locations:
        canvas.scatter(loc[0], loc[1], marker='.', c='green', s=500)
        canvas.scatter(loc[0], loc[1], marker='$R$', c='white', s=45)

    # plot individual trajectories
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    if observations is not None:
        observations = observations[:, :n_trajs]
        for i_traj in range(observations.shape[1]):
            c = colors[i_traj % len(colors)]
            for t in range(observations.shape[0]):
                if t == 0:
                    canvas.scatter(observations[t, i_traj, 4], observations[t, i_traj, 5], marker=r'$\bigcirc$', c=c,
                                   s=90)
                canvas.scatter(observations[t, i_traj, 4], observations[t, i_traj, 5], marker=f'${t + 1}$', c=c, s=25)
    if goal_observations is not None:
        goal_observations = goal_observations[:, :n_trajs]
        for i_traj in range(goal_observations.shape[1]):
            c = colors[i_traj % len(colors)]
            for t in range(goal_observations.shape[0]):
                canvas.scatter(goal_observations[t, i_traj, 0], goal_observations[t, i_traj, 1],
                               marker='h', c=c, s=80)
                canvas.scatter(goal_observations[t, i_traj, 0], goal_observations[t, i_traj, 1],
                               marker=f'${t + 1}$', c='white', s=25)


def plot_nav2d_env(env: gym.Env,
                   canvas: plt.Axes,
                   observations: np.ndarray = None,
                   goal_observations: np.ndarray = None,
                   n_trajs: int = 1):
    assert env_class_is(env, Nav2dEnv)

    # draw environment and borders
    canvas.set_xlim([-1.1, 1.1])
    canvas.set_ylim([-1.1, 1.1])
    canvas.add_patch(Rectangle((-1.0, -1.0), 2.0, 2.0, fill=None, linestyle='--'))

    # plot individual trajectories
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    if observations is not None:
        observations = observations[:, :n_trajs]
        for i_traj in range(observations.shape[1]):
            c = colors[i_traj % len(colors)]
            # plot individual time steps
            for t in range(observations.shape[0]):
                fontweight = 900 if t == 0 else 300
                x, y = observations[t, i_traj, :2]
                border = Ellipse((x, y), width=0.05, height=0.05, fill=True, color='white', zorder=i_traj+t*0.01)
                canvas.add_patch(border)
                border = Ellipse((x, y), width=0.05, height=0.05, fill=False, color=c, zorder=i_traj+(t+0.25)*0.01)
                canvas.add_patch(border)
                canvas.text(x, y, f'{t + 1}', c=c, horizontalalignment='center', verticalalignment='center_baseline',
                            fontweight=fontweight, fontsize='small', zorder=i_traj+(t+0.5)*0.01)
            # plot reward location for completeness
            reward_location = observations[0, i_traj, 2:4]
            canvas.scatter(reward_location[0], reward_location[1], marker='x', c=c, s=50, zorder=i_traj+(t+0.75)*0.01)
    if goal_observations is not None:
        goal_observations = goal_observations[:, :n_trajs]
        for i_traj in range(goal_observations.shape[1]):
            c = colors[i_traj % len(colors)]
            for t in range(goal_observations.shape[0]):
                x, y = goal_observations[t, i_traj, :2]
                canvas.scatter(x, y, marker='h', c=c, s=200, zorder=n_trajs+t*0.01)
                canvas.text(x, y, f'{t + 1}', c='white', horizontalalignment='center',
                            verticalalignment='center_baseline', fontweight=900, fontsize='medium',
                            zorder=n_trajs+(t+0.5)*0.01)
