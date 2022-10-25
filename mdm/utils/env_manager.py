import gym
import numpy as np


class EnvManager(gym.Wrapper):

    def __init__(self, env: gym.Env):
        super(EnvManager, self).__init__(env)
        self._curr_timestep = 0

    def is_first_step(self):
        return self._curr_timestep == 0

    def interact(self, a):
        if self._curr_timestep == 0:  # by convention, make (a_0, r_0, t_0) = 0
            o = self.env.reset()
            a = np.zeros_like(self.env.action_space.sample())
            r = 0.0
            term = False
            info = {}
        else:
            o, r, term, info = self.env.step(a)
            if term:
                self._curr_timestep = 0
            else:
                self._curr_timestep += 1
        return o, a, r, term, info


