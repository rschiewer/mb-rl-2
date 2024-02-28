import colorsys

import numpy as np
import gymnasium as gym
from gym_nav2d.envs import Nav2dEnv
from gymnasium_robotics.envs.maze.maze_v4 import MazeEnv
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
from matplotlib.colors import to_rgb
from minigrid.minigrid_env import MiniGridEnv

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
    #elif env_class_is(env, MiniGridEnv):
    #    plot_minigrid_env(env=env, canvas=canvas, **kwargs)


# def draw_timestep(canvas: plt.Axes, x: float, y:float, i_t: int, size: float = 1, fontweight: int = 300):
#    canvas.add_patch(Ellipse((x, y), width=0.05 * size, height=0.05 * size, fill=None))
#    canvas.text(x, y, f'{t + 1}', c=c,
#                horizontalalignment='center', verticalalignment='center_baseline', fontweight=fontweight)

def plot_maze_env(env: gym.Env,
                  canvas: plt.Axes,
                  observations: np.ndarray = None,
                  goal_observations: np.ndarray = None,
                  rewards: np.ndarray = None,
                  terminals: np.ndarray = None,
                  draw_background: bool = True,
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

    if draw_background:
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
        # select only the observations we want to plot
        observations = observations[:, :n_trajs]
        # use terminals if available to belnd out observations past terminal state
        if terminals is None:
            terminals = np.zeros_like(observations)
        else:
            terminals = terminals[:, :n_trajs]

        # precompute alpa value from terminals using the mask calculation technique
        terminals = np.concatenate([np.zeros_like(terminals[0:1]), terminals[:-1]])
        alpha = np.cumprod(1 - terminals, axis=0)

        for i_traj in range(observations.shape[1]):
            c = colors[i_traj % len(colors)]
            for t in range(observations.shape[0]):
                if t == 0:
                    canvas.scatter(observations[t, i_traj, 4], observations[t, i_traj, 5], marker=r'$\bigcirc$', c=c,
                                   s=90)
                canvas.text(observations[t, i_traj, 4], observations[t, i_traj, 5], f'${t + 1}$', c=c, alpha=alpha[t, i_traj])
    if goal_observations is not None:
        goal_observations = goal_observations[:, :n_trajs]
        for i_traj in range(goal_observations.shape[1]):
            c = colors[i_traj % len(colors)]
            for t in range(goal_observations.shape[0]):
                canvas.scatter(goal_observations[t, i_traj, 4], goal_observations[t, i_traj, 5],
                               marker='h', c=c, s=80)
                canvas.scatter(goal_observations[t, i_traj, 4], goal_observations[t, i_traj, 5],
                               marker=f'${t + 1}$', c='white', s=25)


def plot_nav2d_env(env: gym.Env,
                   canvas: plt.Axes,
                   observations: np.ndarray = None,
                   goal_observations: np.ndarray = None,
                   rewards: np.ndarray = None,
                   terminals: np.array = None,
                   n_trajs: int = 1):
    assert env_class_is(env, Nav2dEnv)

    # draw environment and borders
    canvas.set_xlim([-1.1, 1.1])
    canvas.set_ylim([-1.1, 1.1])
    canvas.add_patch(Rectangle((-1.0, -1.0), 2.0, 2.0, fill=None, linestyle='--'))

    # plot individual trajectories
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    zo = Counter()
    if observations is not None:
        observations = observations[:, :n_trajs]

        if terminals is not None:
            terminals = terminals[:, :n_trajs]
            saturation = 1 - np.cumsum(terminals, axis=0)
            saturation = np.clip(saturation, 0.1, 1.0)
        else:
            saturation = np.ones(observations.shape[:2], dtype=np.float32)

        if rewards is not None:
            rewards = rewards[:, :n_trajs]

        for i_traj in range(observations.shape[1]):
            c = colors[i_traj % len(colors)]
            # plot individual time steps
            for t in range(observations.shape[0]):
                fontweight = 900 if t == 0 else 300
                x, y = observations[t, i_traj, :2]

                # make terminal time steps desaturated
                c_rgb = to_rgb(c)
                c_hsv = colorsys.rgb_to_hsv(*c_rgb)
                c = colorsys.hsv_to_rgb(c_hsv[0], c_hsv[1] * saturation[t, i_traj], c_hsv[2])

                # draw step reward if available
                if rewards is not None:
                    reward_rect = Rectangle((x + 0.01, y + 0.01), 0.075, 0.04, fill=True, facecolor=c, edgecolor=c,
                                            zorder=zo())
                    canvas.add_patch(reward_rect)
                    canvas.text(x + 0.05, y + 0.028, f'{rewards[t, i_traj]:1.2f}', c='white',
                                horizontalalignment='center', verticalalignment='center_baseline',
                                fontweight=800, fontsize='xx-small', zorder=zo())

                # draw time step
                step_dot = Ellipse((x, y), width=0.05, height=0.05, fill=True, facecolor='white', edgecolor=c,
                                   zorder=zo())
                canvas.add_patch(step_dot)
                canvas.text(x, y, f'{t + 1}', c=c, horizontalalignment='center', verticalalignment='center_baseline',
                            fontweight=fontweight, fontsize='small', zorder=zo())

            # plot reward location for completeness
            reward_locations = observations[:, i_traj, 2:4]
            canvas.scatter(reward_locations[:, 0], reward_locations[:, 1], marker='x', c=c, s=50, zorder=zo())
    if goal_observations is not None:
        goal_observations = goal_observations[:, :n_trajs]
        for i_traj in range(goal_observations.shape[1]):
            c = colors[i_traj % len(colors)]
            for t in range(goal_observations.shape[0]):
                x, y = goal_observations[t, i_traj, :2]
                canvas.scatter(x, y, marker='h', c=c, s=200, zorder=zo())
                canvas.text(x, y, f'{t + 1}', c='white', horizontalalignment='center',
                            verticalalignment='center_baseline', fontweight=900, fontsize='medium', zorder=zo())


def plot_minigrid_env(env: gym.Env,
                      canvas: plt.Axes,
                      observations: np.ndarray,
                      n_trajs: int = 1):
    pass


def plot_halfcheetah(env: gym.Env,
                     canvas: plt.Axes,
                     observations: np.ndarray,
                     n_trajs: int = 1):
    # get relevant observation elements
    # 0: z-cooord of front tip
    # 1: angle of front tip
    # 2:
    pass


class Counter:

    def __init__(self, start=0, increment=1):
        self.current = start
        self.increment = increment

    def __call__(self, *args, **kwargs):
        retval = self.current
        self.current += self.increment
        return retval
