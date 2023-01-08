from typing import Tuple

import gym
from gym.core import ObsType, ActType
import numpy as np


class ManagedEnv(gym.Wrapper):

    def __init__(self, env: gym.Env):
        super(ManagedEnv, self).__init__(env)
        self._curr_timestep = 0

    def is_first_step(self):
        return self._curr_timestep == 0

    def next_step(self, a):
        o, r, term, trunc, info = self.env.step(a)
        if term or trunc:
            self._curr_timestep = 0
        else:
            self._curr_timestep += 1
        return o, a, r, term, trunc, info

    def restart(self, **kwargs):
        self._curr_timestep = 0
        # by convention, make (a_0, r_0, term_0, trunc_0) = 0
        o, info = self.env.reset(**kwargs)
        a = np.zeros_like(self.env.action_space.sample())
        r = 0.0
        term = False
        trunc = False
        return o, a, r, term, trunc, info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        raise NotImplementedError('Please use the next_step() method which has a differnt signature')

    def reset(self, **kwargs) -> Tuple[ObsType, dict]:
        raise NotImplementedError('Please use the restart() method which has a differnt signature')
