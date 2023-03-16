import unittest
import math

import gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
import gym_nav2d

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

        mem = collect_data(vec_env, -1, lambda x: vec_env.action_space.sample())

        for i_env in range(n_envs):
            env = my_own_envs[i_env]
            env_a = mem[i_env]['a']
            env_o = mem[i_env]['o']
            env_r = mem[i_env]['r']
            env_term = mem[i_env]['terminal']
            env_trunc = mem[i_env]['truncated']

            # check if first step padding was applied correctly
            self.assertTrue(np.isclose(env_a[0].sum(), 0))
            self.assertTrue(np.isclose(env_r[0], 0))
            self.assertEqual(env_term[0], False)
            self.assertEqual(env_trunc[0], False)

            # remove padding for comparing with unwrapped environments
            env_a = env_a[1:]
            init_o, env_o = env_o[0], env_o[1:]
            env_r = env_r[1:]
            env_term = env_term[1:]
            env_trunc = env_trunc[1:]

            # now compare every single time step from collect_data to what the individual environment produces
            o, info = env.reset()
            self.assertTrue(np.isclose(init_o, o).all())
            for t, a in enumerate(env_a):
                o, r, term, trunc, info = env.step(a)
                self.assertTrue(np.isclose(o, env_o[t]).all())
                self.assertTrue(np.isclose(r, env_r[t]))
                self.assertEqual(term, env_term[t])
                self.assertEqual(trunc, env_trunc[t])
                if term or trunc:
                    self.assertEqual(t + 1, len(env_a))  # make sure that all actions have been used
                    break

    def test_toy_text_collect_data_limited_steps(self):
        n_envs = 10
        n_steps = 10

        map_layouts = [generate_random_map(size=8) for _ in range(n_envs)]
        env_fns = [lambda: gym.make('FrozenLake-v1', is_slippery=False, desc=l) for l in map_layouts]
        my_own_envs = [fn() for fn in env_fns]

        vec_env = gym.vector.AsyncVectorEnv(env_fns)
        vec_env = CacheLastStepVecEnv(vec_env)

        mem = collect_data(vec_env, n_steps, lambda x: vec_env.action_space.sample())

        for i_env in range(n_envs):
            env = my_own_envs[i_env]
            env_a = mem[i_env]['a']
            env_o = mem[i_env]['o']
            env_r = mem[i_env]['r']
            env_term = mem[i_env]['terminal']
            env_trunc = mem[i_env]['truncated']

            # check if first step padding was applied correctly
            self.assertTrue(np.isclose(env_a[0].sum(), 0))
            self.assertTrue(np.isclose(env_r[0], 0))
            self.assertEqual(env_term[0], False)
            self.assertEqual(env_trunc[0], False)

            # remove padding for comparing with unwrapped environments
            env_a = env_a[1:]
            init_o, env_o = env_o[0], env_o[1:]
            env_r = env_r[1:]
            env_term = env_term[1:]
            env_trunc = env_trunc[1:]

            # now compare every single time step from collect_data to what the individual environment produces
            o, info = env.reset()
            self.assertTrue(np.isclose(init_o, o).all())
            for t, a in enumerate(env_a):
                o, r, term, trunc, info = env.step(a)
                self.assertTrue(np.isclose(o, env_o[t]).all())
                self.assertTrue(np.isclose(r, env_r[t]))
                self.assertEqual(term, env_term[t])
                self.assertEqual(trunc, env_trunc[t])
                if term or trunc:
                    self.assertEqual(t + 1, len(env_a))  # make sure that all actions have been used
                    break

    def test_nav_2d_env_expert_policy(self):
        n_envs = 10

        def expert_policy(env: CacheLastStepVecEnv):
            o = env.last_o
            agent_pos = o[:, :2]
            goal_pos = o[:, 2:4]
            distance = o[:, 4]
            adjacent = (goal_pos[:, 1] - agent_pos[:, 1])
            disjacent = (goal_pos[:, 0] - agent_pos[:, 0])
            angle = np.arctan2(adjacent, disjacent) + math.pi * 1.5

            dist_a = np.where(distance > 0.05, 1.0, 0.1)
            angle = np.where(angle > 2 * math.pi, angle - 2 * math.pi, angle)

            angle_a = angle / (2 * math.pi) * 2 - 1
            angle_a = np.clip(angle_a, -1.0, 1.0)
            a = np.stack([angle_a, dist_a]).transpose()
            a = a.astype(np.float32)

            return a

        env_fns = [lambda: gym.make('gym_nav2d:nav2dVeryEasy-v0') for _ in range(n_envs)]
        my_own_envs = [fn() for fn in env_fns]

        vec_env = gym.vector.AsyncVectorEnv(env_fns)
        vec_env = CacheLastStepVecEnv(vec_env)

        mem = collect_data(vec_env, -1, lambda o: expert_policy(o))

        for i_env in range(n_envs):
            env = my_own_envs[i_env]
            env_a = mem[i_env]['a']
            env_o = mem[i_env]['o']
            env_r = mem[i_env]['r']
            env_term = mem[i_env]['terminal']
            env_trunc = mem[i_env]['truncated']

            # check if first step padding was applied correctly
            self.assertTrue(np.isclose(env_a[0].sum(), 0))
            self.assertTrue(np.isclose(env_r[0], 0))
            self.assertEqual(env_term[0], False)
            self.assertEqual(env_trunc[0], False)

            # remove padding for comparing with unwrapped environments
            env_a = env_a[1:]
            init_o, env_o = env_o[0], env_o[1:]
            env_r = env_r[1:]
            env_term = env_term[1:]
            env_trunc = env_trunc[1:]

            # now compare every single time step from collect_data to what the individual environment produces
            o, info = env.reset()
            self.assertTrue(np.isclose(init_o, o).all())
            for t, a in enumerate(env_a):
                o, r, term, trunc, info = env.step(a)
                self.assertTrue(np.isclose(o, env_o[t]).all())
                self.assertTrue(np.isclose(r, env_r[t]))
                self.assertEqual(term, env_term[t])
                self.assertEqual(trunc, env_trunc[t])
                if term or trunc:
                    self.assertEqual(t + 1, len(env_a))  # make sure that all actions have been used
                    break


if __name__ == '__main__':
    unittest.main()
