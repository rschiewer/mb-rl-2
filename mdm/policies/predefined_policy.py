import gym


class PredefinedPolicy:

    def __init__(self,
                 env: gym.Env,
                 action_list: gym.core.ActType):
        self.env = env
        self.action_list = action_list
        self.current_timestep = 0
        self.max_timestep = action_list.shape[0]

    def __call__(self, *args):
        if self.current_timestep < self.max_timestep:
            a = self.action_list[self.current_timestep]
            self.current_timestep += 1
            return {'a': a}
        else:
            raise StopIteration(f'No more actions available')
