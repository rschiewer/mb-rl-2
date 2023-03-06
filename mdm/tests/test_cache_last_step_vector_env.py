import unittest

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np

from mdm.utils.gym_wrappers import CacheLastStepVecEnv
from mdm.training.gym_driver import collect_data


class TestCacheLastStepVecEnv(unittest.TestCase):

    def test_toy_text_manual(self):
        n_envs = 10

        map_layouts = [generate_random_map(size=8) for _ in range(n_envs)]
        env_fns = [lambda: gym.make('FrozenLake-v1', is_slippery=False, desc=l) for l in map_layouts]
        my_own_envs = [fn() for fn in env_fns]

        vec_env = gym.vector.AsyncVectorEnv(env_fns)
        vec_env = CacheLastStepVecEnv(vec_env)

        actions = []
        observations = []
        rewards = []
        terminals = []
        truncateds = []
        vec_env.reset()  # ignore first obervation
        while not vec_env.all_envs_done:
            a = vec_env.action_space.sample()
            vec_env.step(a)

            actions.append(vec_env.last_a)
            observations.append(vec_env.last_o)
            rewards.append(vec_env.last_r)
            terminals.append(vec_env.last_term)
            truncateds.append(vec_env.last_trunc)

        actions = np.stack(actions)
        observations = np.stack(observations)
        rewards = np.stack(rewards)
        terminals = np.stack(terminals)
        truncateds = np.stack(truncateds)

        for i_env in range(n_envs):
            env = my_own_envs[i_env]
            env_a = actions[:, i_env]
            env_o = observations[:, i_env]
            env_r = rewards[:, i_env]
            env_term = terminals[:, i_env]
            env_trunc = truncateds[:, i_env]

            env.reset()
            for t, a in enumerate(env_a):
                o, r, term, trunc, info = env.step(a)
                self.assertTrue(np.isclose(o, env_o[t]))
                self.assertTrue(np.isclose(r, env_r[t]))
                self.assertTrue(np.isclose(term, env_term[t]))
                self.assertTrue(np.isclose(trunc, env_trunc[t]))
                if term or trunc:
                    break

    def test_toy_text_collect_data(self):
        n_envs = 10

        map_layouts = [generate_random_map(size=8) for _ in range(n_envs)]
        env_fns = [lambda: gym.make('FrozenLake-v1', is_slippery=False, desc=l) for l in map_layouts]
        my_own_envs = [fn() for fn in env_fns]

        vec_env = gym.vector.AsyncVectorEnv(env_fns)
        vec_env = CacheLastStepVecEnv(vec_env)

        mem = collect_data(vec_env, -1, lambda a, b, c, d, e: vec_env.action_space.sample())

        for i_env in range(n_envs):
            env = my_own_envs[i_env]
            env_a = mem[i_env]['a']
            env_o = mem[i_env]['o']
            env_r = mem[i_env]['r']
            env_term = mem[i_env]['terminal']
            env_trunc = mem[i_env]['truncated']

            env.reset()
            for t, a in enumerate(env_a):
                o, r, term, trunc, info = env.step(a)
                self.assertTrue(np.isclose(o, env_o[t]))
                self.assertTrue(np.isclose(r, env_r[t]))
                self.assertTrue(np.isclose(term, env_term[t]))
                self.assertTrue(np.isclose(trunc, env_trunc[t]))
                if term or trunc:
                    break

if __name__ == '__main__':
    unittest.main()
