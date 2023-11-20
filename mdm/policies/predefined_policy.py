import gymnasium as gym

from mdm.policies.policy import Policy


class PredefinedPolicy(Policy):

    def __init__(self,
                 action_list: gym.core.ActType):
        self.action_list = action_list
        self.current_timestep = 0
        self.max_timestep = action_list.shape[0]

    def __call__(self, *args):
        if self.current_timestep < self.max_timestep:
            a = self.action_list[self.current_timestep]
            self.current_timestep += 1
            return a
        else:
            raise StopIteration(f'No more actions available')
