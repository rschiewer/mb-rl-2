import gymnasium as gym


from mdm.policies.policy import Policy


class RandomPolicy(Policy):

    def __init__(self, env: gym.Env):
        self.env = env

    def __call__(self, *args):
        return self.env.action_space.sample()
