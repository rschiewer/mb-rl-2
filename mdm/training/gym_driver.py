from typing import Callable, List, Dict, TypeVar

import gym
import numpy as np
from tqdm import tqdm

from mdm.training.driver import Driver
from mdm.memory.trajectory_memory import TrajectoryMemory
from mdm.utils.utils import compute_returns


DataType = TypeVar('DataType', np.ndarray, int, float, bool)


class GymEpisodeDriver(Driver):

    def __init__(self,
                 env: gym.Env,
                 policy: Callable,
                 observation_postprocessing: Callable = None):
        super(GymEpisodeDriver, self).__init__()

        self.env = env
        self.policy = policy

        if observation_postprocessing is None:
            self._process_obs = lambda x: x
            self._process_step = lambda x: x
        else:
            self._process_obs = observation_postprocessing
            self._process_step = lambda x: (self._process_obs(x[0]), x[1], x[2], x[3], x[4])  # o, r, term, trunc, info

    def interact(self,
                 n_episodes: int,
                 progress_bar: bool = False,
                 mem: List[Dict[str, DataType]] = None,
                 **kwargs) -> List[Dict[str, DataType]]:
        if mem is None:
            mem = []

        ep_iter = range(n_episodes)
        if progress_bar:
            ep_iter = tqdm(ep_iter, desc='Collecting Samples')

        for i_ep in ep_iter:
            traj_o, traj_a, traj_r, traj_term, traj_trunc = [], [], [], [], []

            o, _ = self.env.reset()
            traj_o.append(self._process_obs(o))
            traj_a.append(np.zeros_like(self.env.action_space.sample()))  # by convention, make (a_0, r_0, t_0) = 0
            traj_r.append(0)
            traj_term.append(False)
            traj_trunc.append(False)

            terminal = False
            truncated = False
            while not terminal and not truncated:
                a = self.policy(traj_o[-1], traj_r[-1], traj_term[-1], i_ep)
                o_, r, terminal, truncated, info = self._process_step(self.env.step(a))

                traj_o.append(o_)
                traj_a.append(a)
                traj_r.append(r)
                traj_term.append(terminal)
                traj_trunc.append(truncated)

            traj = {'o': np.array(traj_o), 'a': np.array(traj_a), 'r': np.array(traj_r),
                    'terminal': np.array(traj_term), 'truncated': np.array(traj_trunc)}
            mem.append(traj)
            #mem.push(np.array(traj_o), np.array(traj_a), np.array(traj_r), np.array(traj_term))

        #compute_returns(mem, 0.99)
        return mem


class GymStepDriver(GymEpisodeDriver):

    def interact(self,
                 n_steps: int,
                 progress_bar: bool = False,
                 **kwargs) -> TrajectoryMemory:
        mem = TrajectoryMemory()

        step_iter = range(n_steps)
        if progress_bar:
            step_iter = tqdm(step_iter, desc='Collecting Samples')

        traj_s, traj_a, traj_r, traj_terminal, s = None, None, None, None, None
        for i_t in step_iter:
            if s is None:
                traj_s, traj_a, traj_r, traj_terminal = [], [], [], []
                s = self._process_obs(self.env.reset())

            a = self.policy(s)
            s_, r, terminal, info = self._process_step(self.env.step(a))

            traj_s.append(s)
            traj_a.append(a)
            traj_r.append(r)
            traj_terminal.append(terminal)

            if terminal:
                s = None
                traj_s.append(s_)
                mem.push(np.array(traj_s), np.array(traj_a), np.array(traj_r), np.array(traj_terminal))
            else:
                s = s_

        return mem