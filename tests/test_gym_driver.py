import unittest
from typing import Iterable

import gym
import numpy as np

from training.gym_driver import GymEpisodeDriver


class GymEpisodeDriverTest(unittest.TestCase):

    @staticmethod
    def _run_env(env: gym.Env, actions: Iterable = None):
        trajectory = {'s': [], 'a':[], 'r': [], 'terminal': []}

        observation = env.reset()
        trajectory['s'].append(observation)

        if actions is None:
            get_act = lambda: env.action_space.sample()
        else:
            a_iter = iter(actions)
            get_act = lambda: next(a_iter)

        while True:
            action = get_act()
            observation, reward, done, info = env.step(action)

            trajectory['s'].append(observation)
            trajectory['a'].append(action)
            trajectory['r'].append(reward)
            trajectory['terminal'].append(done)

            if done:
                break

        return {k: np.array(v) for k, v in trajectory.items()}

    def setUp(self) -> None:
        self.rand_seed = 42
        self.batch_sizes = [1, 10]
        self.num_batches = 3
        self.envs = [gym.make('CartPole-v0'), gym.make('MountainCar-v0'), gym.make('PongNoFrameskip-v4')]

    def test_get_batch(self):
        for batch_size in self.batch_sizes:
            for env in self.envs:
                env.seed(self.rand_seed + batch_size)
                np.random.seed(self.rand_seed + batch_size)
                collector = GymEpisodeDriver(env, lambda s: env.action_space.sample())

                trajectories = []
                for _ in range(self.num_batches):
                    trajectories.extend(collector.interact(batch_size))
                self.assertEqual(len(trajectories), batch_size * self.num_batches)

                # now reproduce samples with real environment (only test deterministic envs)
                env.seed(self.rand_seed + batch_size)
                np.random.seed(self.rand_seed + batch_size)
                for traj in trajectories:
                    rerun_traj = self._run_env(env, traj['a'])
                    for orig, rerun in zip(traj.values(), rerun_traj.values()):
                        diff = np.sum(np.abs(orig.astype(np.float32) - rerun.astype(np.float32)))
                        self.assertTrue(np.isclose(diff, 0))


if __name__ == '__main__':
    unittest.main()
