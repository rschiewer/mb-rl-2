from typing import Tuple

import gym
from gym.core import ObsType
import numpy as np


class ManagedEnv(gym.Wrapper):

    def __init__(self, env: gym.Env):
        super(ManagedEnv, self).__init__(env)
        self._curr_timestep = 0

    def is_first_step(self):
        return self._curr_timestep == 0

    def step(self, a):
        if self._curr_timestep == 0:  # by convention, make (a_0, r_0, term_0, trunc_0) = 0
            o = self.env.reset()
            a = np.zeros_like(self.env.action_space.sample())
            r = 0.0
            term = False
            trunc = False
            info = {}
        else:
            o, r, term, trunc, info = self.env.step(a)
            if term or trunc:
                self._curr_timestep = 0
            else:
                self._curr_timestep += 1
        return o, a, r, term, trunc, info

    def reset(self, **kwargs) -> Tuple[ObsType, dict]:
        self._curr_timestep = 0
        return self.env.reset()


