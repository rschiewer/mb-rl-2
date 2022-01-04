from typing import Any, Callable

import gym
import torch
import numpy as np

from training.driver import Driver


class GymEpisodeDriver(Driver):

    def __init__(self, env: gym.Env, policy: Callable, observation_postprocessing: Callable = None):
        super(GymEpisodeDriver, self).__init__()

        self.env = env
        self.policy = policy

        if observation_postprocessing is None:
            self._process_obs = lambda x: x
            self._process_step = lambda x: x
        else:
            self._process_obs = observation_postprocessing
            self._process_step = lambda x: (self._process_obs(x[0]), x[1], x[2], x[3])  # o, r, term, info

    def interact(self, num_episodes: int) -> Any:
        mem = []
        for i_ep in range(num_episodes):
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

            mem.append({'s': np.array(traj_s), 'a': np.array(traj_a), 'r': np.array(traj_r),
                              'terminal': np.array(traj_terminal)})

        return mem

