from typing import Callable

import gym
import numpy as np
from tqdm import tqdm

from mdm.training.driver import Driver
from mdm.memory.trajectory_memory import TrajectoryMemory


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
            self._process_step = lambda x: (self._process_obs(x[0]), x[1], x[2], x[3])  # o, r, term, info

    def interact(self,
                 n_episodes: int,
                 progress_bar: bool = False,
                 **kwargs) -> TrajectoryMemory:
        mem = TrajectoryMemory()

        ep_iter = range(n_episodes)
        if progress_bar:
            ep_iter = tqdm(ep_iter, desc='Collecting Samples')

        for i_ep in ep_iter:
            traj_s, traj_a, traj_r, traj_terminal = [], [], [], []
            s = self._process_obs(self.env.reset())

            terminal = False
            while not terminal:
                a = self.policy(s)
                s_, r, terminal, info = self._process_step(self.env.step(a))

                traj_s.append(s)
                traj_a.append(a)
                traj_r.append(r)
                traj_terminal.append(terminal)

                s = s_
            traj_s.append(s)  # final observation

            mem.push(np.array(traj_s), np.array(traj_a), np.array(traj_r), np.array(traj_terminal))

        return mem

