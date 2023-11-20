from pathlib import Path
from enum import IntEnum
from typing import Union, Iterable, Sequence
from tkinter import Canvas, Tk, Toplevel
from time import sleep

import gymnasium as gym
import numpy as np


class CellType(IntEnum):
    FREE = 0
    AGENT = 1
    WALL = 2
    REWARD = 3


class Gridworld(gym.Env):

    # up, right, down, left
    move_offset = ((-1, 0), (0, 1), (1, 0), (0, -1))
    action_descriptions = ['up', 'right', 'down', 'left']
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
                 discount: float,
                 fully_observable: bool = False):
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
        self.current_ep_time = 0

        self.action_space = gym.spaces.Discrete(n=4)
        self.observation_space = gym.spaces.Box(low=np.array([0, 0]), high=np.array([grid_h - 1, grid_w - 1]),
                                                shape=(2,), dtype=np.int64)

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

        pos_agent = self.find_cell_type(CellType.AGENT)
        dest_pos = pos_agent + self.move_offset[action]
        dest_pos_clipped = np.clip(dest_pos, (0, 0), (self.grid_h - 1, self.grid_w - 1))
        dest_type = self._grid[tuple(dest_pos_clipped)]
        #print(f'dest_pos: {dest_pos}')
        #print(f'dest_type: {dest_type}')

        reward = self.step_reward
        terminated, truncated = False, False
        info = {}

        if dest_type == CellType.FREE or dest_type == CellType.REWARD:
            self._grid[tuple(pos_agent)] = CellType.FREE
            self._grid[tuple(dest_pos_clipped)] = CellType.AGENT
            if dest_type == CellType.REWARD:
                reward += self.reward
                terminated = True

        self._grid.flags.writeable = False

        self.current_ep_time += 1
        if self.current_ep_time == self.time_limit - 1:
            truncated = True

        return self.find_cell_type(CellType.AGENT), reward, terminated, truncated, info

    def teleport_agent(self, pos: Sequence[int]):
        if len(pos) != 2:
            raise ValueError('pos should be sequence of length 2')

        agent_pos = self.find_cell_type(CellType.AGENT)
        if len(agent_pos) != 2:
            raise RuntimeError('Agent position is ambiguous or unknown, did you call reset() first?')

        if self._grid[tuple(pos)] != CellType.FREE and self._grid[tuple(pos)] != CellType.AGENT:
            raise ValueError('Destination position must be unoccupied')

        self._grid.flags.writeable = True
        self._grid[tuple(agent_pos)] = CellType.FREE
        self._grid[tuple(pos)] = CellType.AGENT
        self._grid.flags.writeable = False

        return self.find_cell_type(CellType.AGENT)

    def reset(self, seed: int = None, options: dict = None):
        self._grid.flags.writeable = True
        self._grid[:] = self._init_grid[:]

        agent_pos = self.find_cell_type(CellType.AGENT)
        if agent_pos.size == 0:  # no start position given, pick free cell at random
            candidate_positions = self.find_cell_type(CellType.FREE)
            start_pos = np.random.default_rng().choice(candidate_positions)
            self._grid[tuple(start_pos)] = CellType.AGENT
        elif agent_pos.size > 2:  # multiple possible start positions given, pick one
            candidate_positions = agent_pos
            for pos in candidate_positions:
                self._grid[tuple(pos)] = CellType.FREE
            start_pos = np.random.default_rng().choice(candidate_positions)
            self._grid[tuple(start_pos)] = CellType.AGENT

        self._grid.flags.writeable = False
        self.current_ep_time = 0
        return self.find_cell_type(CellType.AGENT), {}

    def render(self, mode="human"):
        if self.canvas is None:
            self._tk_master = Tk()
            self._canvas_h, self._canvas_w = 400, 400
            self.canvas = Canvas(self._tk_master, width=self._canvas_w, height=self._canvas_h)
            self.canvas.pack(fill='both', expand=True)

            def close_handler():
                self.canvas = None
                Tk.destroy(self._tk_master)

            self._tk_master.protocol('WM_DELETE_WINDOW', close_handler)

        self._cell_h, self._cell_w = round(self._canvas_h / self.grid_h), round(self._canvas_w / self.grid_w)

        self.canvas.delete('all')
        for y in range(self.grid_h):
            for x in range(self.grid_w):
                x0, y0 = self._cell_w * x, self._cell_h * y
                x1, y1 = x0 + self._cell_w, y0 + self._cell_h
                color = self.colors[self._grid[y, x]]
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color)
        self._tk_master.update()

    def enact_sequence(self,
                       states: Sequence,
                       actions: Sequence,
                       rewards: Sequence,
                       terminals: Sequence,
                       truncateds: Sequence,
                       render: bool = True,
                       t_sleep: int = 0):
        # place agent at correct starting point
        self.reset()
        self._grid.flags.writeable = True
        self._grid[self._grid == CellType.AGENT] = CellType.FREE
        self._grid[tuple(states[0])] = CellType.AGENT
        self._grid.flags.writeable = False

        for i_t, (s_seq, a_seq, r_seq, term_seq, trunc_seq) in enumerate(zip(states[1:], actions, rewards, terminals,
                                                                             truncateds)):
            if render:
                self.render()
            sleep(t_sleep)

            s, r, term, trunc, info = self.step(a_seq)

            if s[0] != s_seq[0] or s[1] != s_seq[1]:
                raise RuntimeError(f'State of provided sequence and generated state differ in step {i_t}, '
                                   f'sequence state: {s_seq}, generated state: {s}.')
            if r != r_seq:
                raise RuntimeError(f'Reward of provided sequence and generated reward differ in step {i_t}, '
                                   f'sequence reward: {r_seq}, generated reward: {r}.')
            if term != term_seq:
                raise RuntimeError(f'Terminal flag of provided sequence and generated terminal flag differ in step '
                                   f'{i_t}, sequence terminal flag: {term_seq}, generated terminal flag: {term}.')
            if trunc != trunc_seq:
                raise RuntimeError(f'Truncated flag of provided sequence and generated truncsted flag differ in step '
                                   f'{i_t}, sequence truncated flag: {trunc_seq}, generated truncated flag: {trunc}.')

    def render_alternative(self, mode="human"):
        if self.canvas is None:
            self._tk_master = Tk()
            self._canvas_h, self._canvas_w = 400, 400
            self.canvas = GridworldGUI(self._tk_master, 400, 400)

        self.canvas.draw(self._grid)

    def find_cell_type(self,
                       type: CellType) -> np.ndarray:
        """
        Returns positions of cells having type :type: in (y, x) coordinate format.
        :param type: Desired cell type
        :return: numpy array containing the coordinates, 1st dimension is over cells, 2nd dimension over yx coordinates
        """
        cells = np.argwhere(self._grid == type).squeeze()
        return cells


class FullyObservableGridworld(gym.ObservationWrapper):

    def __init__(self, env: Gridworld):
        super(FullyObservableGridworld, self).__init__(env)

        low = int(min(CellType))
        high = int(max(CellType))
        shape = (env.grid_h * env.grid_w, )
        self.observation_space = gym.spaces.Box(low, high, shape=shape, dtype=np.uint8)

    def observation(self, observation):
        return self.env.grid.flatten()


class ResetWrapper(gym.Wrapper):
    
    def __init__(self, env: gym.Env):
        super(ResetWrapper, self).__init__(env)

    def reset(self, **kwargs):
        init_o = super(ResetWrapper, self).reset(**kwargs)
        init_r = 0.0
        init_term = False
        return init_o, init_r, init_term, {}


class NormalizedObsGridworld(gym.ObservationWrapper):

    def __init__(self, env: gym.Env):
        super(NormalizedObsGridworld, self).__init__(env)

        low = np.array([-1, -1], dtype=np.float32)
        high = np.array([1, 1], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        self._orig_high = env.observation_space.high
        self._orig_low = env.observation_space.low

    def observation(self, observation):
        raise NotImplementedError('TODO')


class GridworldGUI(Toplevel):

    colors = {
        CellType.AGENT: '#00FF00',
        CellType.REWARD: '#FF0000',
        CellType.FREE: '#FFFFFF',
        CellType.WALL: '#505050'
    }

    def __init__(self, master, width, height):
        Toplevel.__init__(self, master)
        self.title('Gridworld')
        self.resizable(True, True)
        self.geometry(f'{width}x{height}')
        self.canvas = Canvas(self, width=width, height=height)
        self.canvas.pack(fill='both', expand=True)

    def callback(self): pass

    def draw(self, grid: np.ndarray):
        grid_h, grid_w = grid.shape
        cell_h, cell_w = round(self.canvas.winfo_height() / grid_h), round(self.canvas.winfo_width() / grid_w)
        for y in range(grid_h):
            for x in range(grid_w):
                x0, y0 = cell_w * x, cell_h * y
                x1, y1 = x0 + cell_w, y0 + cell_h
                color = self.colors[grid[y, x]]
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color)
        self.update()
