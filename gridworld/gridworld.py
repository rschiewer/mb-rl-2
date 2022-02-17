from pathlib import Path
from enum import IntEnum
from typing import Union, Iterable
from tkinter import Canvas, Tk

import gym
import numpy as np


class CellType(IntEnum):
    FREE = 0
    AGENT = 1
    WALL = 2
    REWARD = 3


class Gridworld(gym.Env):

    # up, right, down, left
    move_offset = ((-1, 0), (0, 1), (1, 0), (0, -1))
    colors = {
        CellType.AGENT: '#00FF00',
        CellType.REWARD: '#FF0000',
        CellType.FREE: '#FFFFFF',
        CellType.WALL: '#505050'
    }

    def __init__(self, grid_h: int,
                 grid_w: int,
                 time_limit: int,
                 reward: float,
                 step_reward: float,
                 discount: float):
        super(Gridworld, self).__init__()

        if time_limit <= 0:
            raise ValueError('Time limit must be larger than zero')

        self._grid = np.zeros((grid_h, grid_w), dtype=np.int8)
        self._init_grid = np.zeros((grid_h, grid_w), dtype=np.int8)
        self._grid.flags.writeable = False
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.time_limit = time_limit
        self.reward = reward
        self.step_reward = step_reward
        self.discount = discount
        self._current_ep_time = 0

        self.action_space = gym.spaces.Discrete(n=4)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(2,), dtype=np.int64)

        self.canvas = None
        self._tk_master = None
        self._canvas_w = None
        self._canvas_h = None
        self._cell_w = None
        self._cell_h = None

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, new_grid):
        self._grid.flags.writeable = True
        self._grid[:] = new_grid[:]
        self._grid.flags.writeable = False
        self._init_grid[:] = new_grid[:]

    def add_object(self, pos: Iterable[int], object_type: CellType):
        self._grid.flags.writeable = True
        self._grid[tuple(pos)] = object_type
        self._grid.flags.writeable = False
        self._init_grid[:] = self._grid

    @staticmethod
    def from_cleartext(path: Union[str, Path]):
        args = {}
        map_data_section = -1
        grid = None
        lookup_table = {'.': CellType.FREE, 'a': CellType.AGENT, 'r': CellType.REWARD, 'x': CellType.WALL}

        with open(Path(path), 'r') as map_data_file:
            for i_l, line in enumerate(map_data_file):
                line_clean = ''.join(line.split())  # removes tabs, newlines etc.
                if grid is None:
                    k, v = line_clean.split(':')
                    if k == 'map_data':
                        map_data_section = i_l + 1
                        grid = np.zeros((args['grid_h'], args['grid_w']), dtype=np.int8)
                    elif k == 'grid_h' or k == 'grid_w' or k == 'time_limit':
                        args[k] = int(v)
                    else:
                        args[k] = float(v)
                else:
                    for i_c, column in enumerate(line_clean):
                        grid[i_l - map_data_section][i_c] = lookup_table[column.lower()]
            gridworld = Gridworld(**args)
            gridworld.grid = grid
            return gridworld

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f'Action {action} is not a member of this environment\'s action space')

        self._grid.flags.writeable = True

        pos_agent = self._get_agent_cell()
        dest_pos = pos_agent + self.move_offset[action]
        dest_pos_clipped = np.clip(dest_pos, (0, 0), (self.grid_h - 1, self.grid_w - 1))
        dest_type = self._grid[tuple(dest_pos_clipped)]
        #print(f'dest_pos: {dest_pos}')
        #print(f'dest_type: {dest_type}')

        reward = 0
        done = False
        info = {}

        if dest_type == CellType.FREE:
            self._grid[tuple(pos_agent)] = CellType.FREE
            self._grid[tuple(dest_pos_clipped)] = CellType.AGENT
        elif dest_type == CellType.REWARD:
            reward = self.reward
            done = True

        self._grid.flags.writeable = False

        self._current_ep_time += 1
        if self._current_ep_time == self.time_limit - 1:
            done = True

        return self._get_agent_cell(), reward, done, info

    def reset(self):
        self._grid.flags.writeable = True
        self._grid[:] = self._init_grid[:]
        self._grid.flags.writeable = False
        self._current_ep_time = 0
        return self._get_agent_cell()

    def render(self, mode="human"):
        if self.canvas is None:
            self._tk_master = Tk()
            self._canvas_h, self._canvas_w = 300, 300
            self._cell_h, self._cell_w = round(self._canvas_h / self.grid_h), round(self._canvas_w / self.grid_w)
            self.canvas = Canvas(self._tk_master, width=self._canvas_w, height=self._canvas_h)
            self.canvas.pack()

        for y in range(self.grid_h):
            for x in range(self.grid_w):
                x0, y0 = self._cell_w * x, self._cell_h * y
                x1, y1 = x0 + self._cell_w, y0 + self._cell_h
                color = self.colors[self._grid[y, x]]
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color)
                self._tk_master.update()


    def _get_agent_cell(self):
        agent_pos = np.argwhere(self._grid == CellType.AGENT).squeeze()
        if agent_pos.shape != (2,):
            raise RuntimeError('No agent found in gridworld, please add agent to grid before calling step()')
        return agent_pos
