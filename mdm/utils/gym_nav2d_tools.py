from math import floor
from typing import Dict

import gym
import gym_nav2d
import numpy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt, animation as animation


def gen_regular_grid_trajectories(env: gym.Env, trajs_vert: int, trajs_horiz: int, step_size: float, move_dir: int = 0):
    env_width = env.len_court_x
    env_height = env.len_court_y
    step_width = env.max_step_size * step_size
    traj_horiz_margin = env_width / (trajs_horiz + 1)
    traj_vert_margin = env_height / (trajs_vert + 1)
    goal_pos = [env.goal_x, env.goal_y]

    start_pos_bot_to_top = [[i * traj_horiz_margin, 0.0] for i in range(1, trajs_horiz + 1)]  # start pos for bottom to top trajectories
    n_actions_bot_to_top = floor(env_height // step_width)
    action_bot_to_top = np.array([0.0, 1.0], dtype=env.action_space.dtype)
    start_pos_left_to_right = [[0.0, i * traj_vert_margin] for i in range(1, trajs_vert + 1)]  # start pos for left to right trajectories
    n_actions_left_to_right = floor(env_width // step_width)
    action_left_to_right = np.array([1.0, 0.0], dtype=env.action_space.dtype)

    #plt.scatter(np.array(start_pos_bot_to_top)[:, 0], np.array(start_pos_bot_to_top)[:, 1])
    #plt.scatter(np.array(start_pos_left_to_right)[:, 0], np.array(start_pos_left_to_right)[:, 1])
    #plt.xlim(-5, 260)
    #plt.ylim(-5, 260)
    #plt.show()

    trajectories = []
    for start_pos in start_pos_bot_to_top:
        env.reset()
        env.teleport_agent(*start_pos)
        o_init = env.get_current_obs()
        a_init = np.zeros_like(env.action_space.sample())
        trajectory = {'o': [o_init], 'a': [a_init], 'r': [0.0], 'terminal': [0.0], 'truncated': [0.0]}
        a = action_bot_to_top
        for _ in range(n_actions_bot_to_top):
            o, r, term, trunc, info = env.step(a)
            trajectory['o'].append(o)
            trajectory['a'].append(a)
            trajectory['r'].append(r)
            trajectory['terminal'].append(term)
            trajectory['truncated'].append(term)
        trajectories.append(trajectory)
    for start_pos in start_pos_left_to_right:
        env.reset()
        env.teleport_agent(*start_pos)
        o_init = env.get_current_obs()
        a_init = np.zeros_like(env.action_space.sample())
        trajectory = {'o': [o_init], 'a': [a_init], 'r': [0.0], 'terminal': [0.0], 'truncated': [0.0]}
        a = action_left_to_right
        for _ in range(n_actions_left_to_right):
            o, r, term, trunc, info = env.step(a)
            trajectory['o'].append(o)
            trajectory['a'].append(a)
            trajectory['r'].append(r)
            trajectory['terminal'].append(term)
            trajectory['truncated'].append(term)
        trajectories.append(trajectory)

    trajectories = [{k: np.stack(v) for k, v in traj.items()} for traj in trajectories]

    return trajectories


def visualize_overlaid_trajectories(*trajectories: Dict[str, np.ndarray], figure: plt.Figure = None):
    n_steps = set([t['o'].shape[0] for t in trajectories])
    assert len(n_steps) == 1, f'All provided trajectories must have the same length, but found {n_steps}!'
    n_steps = n_steps.pop()

    fig, ax = plt.subplots(2, 2, figsize=(6, 6), num=figure)
    plt.tight_layout()
    color_cycle = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    ax[0, 0].set_title('Observation')
    ax[1, 0].set_title('Reward')
    ax[1, 1].set_title('Terminal Flag')
    ax[0, 0].set_xlim(-1, 1)
    ax[0, 0].set_ylim(-1, 1)

    positions = []
    goals = []
    for trajectory in trajectories:
        c = next(color_cycle)
        pos = ax[0, 0].scatter(x=trajectory['o'][0, 0], y=trajectory['o'][0, 1], marker='o', c=c)  # init pos agent
        goal = ax[0, 0].scatter(x=trajectory['o'][0, 2], y=trajectory['o'][0, 3], marker='x', c=c)  # init pos goal
        ax[1, 0].plot(trajectory['r'])
        ax[1, 1].plot(trajectory['terminal'])
        positions.append(pos)
        goals.append(goal)

    time_marker_r = ax[1, 0].axvline(x=0, color='gray', linestyle='dotted')
    time_marker_terminal = ax[1, 1].axvline(x=0, color='gray', linestyle='dotted')

    def animate_r(i):
        for i_traj, trajectory in enumerate(trajectories):
            positions[i_traj].set_offsets([trajectory['o'][i, 0:2]])
            goals[i_traj].set_offsets([trajectory['o'][i, 2:4]])
        # draw action
        # angle, stepwidth = trajectory['a'][i]
        # dx, dy = np.arccos(angle) * stepwidth, np.arcsin(angle) * stepwidth
        # ax[0, 0].arrow(x=trajectory['o'][i, 0], y=trajectory['o'][i, 1], dx=dx, dy=dy)
        # draw time markers
        # act.set_offsets([dx, dy])
        time_marker_r.set_xdata(i)
        time_marker_terminal.set_xdata(i)
        return *positions, *goals, time_marker_r, time_marker_terminal

    ani = animation.FuncAnimation(fig, animate_r, frames=n_steps, interval=100, blit=True)
    return fig, ani


def visualize_trajectory(trajectory: Dict[str, np.ndarray]):
    n_steps = trajectory['o'].shape[0]

    fig, ax = plt.subplots(2, 2, figsize=(6, 6))
    color_cycle = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    # headings and preparations
    ax[0, 0].set_title('Observation')
    ax[1, 0].set_title('Reward')
    ax[1, 1].set_title('Terminal Flag')
    ax[0, 0].set_xlim(-1, 1)
    ax[0, 0].set_ylim(-1, 1)

    # plotting
    c = next(color_cycle)
    pos = ax[0, 0].scatter(x=trajectory['o'][0, 0], y=trajectory['o'][0, 1], marker='o', c=c)  # init pos agent
    goal = ax[0, 0].scatter(x=trajectory['o'][0, 2], y=trajectory['o'][0, 3], marker='x', c=c)  # init pos goal
    # angle, stepwidth = trajectory['a'][0]
    # dx, dy = np.arccos(angle) * stepwidth, np.arcsin(angle) * stepwidth
    # act = ax[0, 0].arrow(x=trajectory['o'][0, 0], y=trajectory['o'][0, 1], dx=dx, dy=dy)  # init action
    ax[1, 0].plot(trajectory['r'])
    ax[1, 1].plot(trajectory['terminal'])

    time_marker_r = ax[1, 0].axvline(x=0, color='gray', linestyle='dotted')
    time_marker_terminal = ax[1, 1].axvline(x=0, color='gray', linestyle='dotted')

    def animate_r(i):
        pos.set_offsets([trajectory['o'][i, 0:2]])
        goal.set_offsets([trajectory['o'][i, 2:4]])
        # draw action
        # angle, stepwidth = trajectory['a'][i]
        # dx, dy = np.arccos(angle) * stepwidth, np.arcsin(angle) * stepwidth
        # ax[0, 0].arrow(x=trajectory['o'][i, 0], y=trajectory['o'][i, 1], dx=dx, dy=dy)
        # draw time markers
        # act.set_offsets([dx, dy])
        time_marker_r.set_xdata(i)
        time_marker_terminal.set_xdata(i)
        return pos, goal, time_marker_r, time_marker_terminal

    ani = animation.FuncAnimation(fig, animate_r, frames=n_steps, interval=100, blit=True)
    return fig, ani
