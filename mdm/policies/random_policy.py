import gym


class RandomPolicy:

    def __init__(self, env: gym.Env):
        self.env = env

    def __call__(self, *args):
        return {'a': self.env.action_space.sample()}
