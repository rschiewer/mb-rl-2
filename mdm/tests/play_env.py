import time

import gymnasium as gym
import gym_nav2d
import numpy as np
from mdm.policies.expert_policies import get_expert_policy
from mdm.utils.gym_wrappers import CacheLastStepEnv

env_name = 'gym_nav2d:nav2dEasySparse-v0'
env = CacheLastStepEnv(gym.make(env_name, render_mode='human'))
policy = get_expert_policy(env_name, lambda: env.action_space.sample())

env.reset()
env.render()
for _ in range(50):
    time.sleep(0.2)
    a = policy(env)
    o, r, term, trunc, info = env.step(a)
    env.render()
    print(f'obs: {o}, reward: {r}, term: {term}, trunc: {trunc}, env: {env.is_reward_collected()}')
    if term or trunc:
        break

