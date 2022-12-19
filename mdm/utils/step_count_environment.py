from typing import Tuple, Optional

import gym
from gym.core import ObsType, ActType


class StepCountEnv(gym.Wrapper):

    def __init__(self,
                 env: gym.Env):
        super(StepCountEnv, self).__init__(env)
        self.current_step = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        self.current_step = 0
        return super(StepCountEnv, self).reset(seed=seed, options=options)

    def step(self,
             action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        o, r, term, trunc, info = self.env.step(action)
        if term or trunc:
            self.current_step = 0
        else:
            self.current_step += 1
        return o, r, term, trunc, info