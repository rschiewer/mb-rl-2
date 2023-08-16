import sys
from typing import Callable, List, Dict, TypeVar, Union

import gymnasium as gym
from gymnasium.core import ActType
import numpy as np
from tqdm import tqdm

from mdm.training.driver import Driver
from mdm.policies.random_policy import RandomPolicy
from mdm.utils.gym_wrappers import CacheLastStepEnv, CacheLastStepVecEnv, CacheLastStepVecEnvPool

DataType = TypeVar('DataType', np.ndarray, int, float, bool)


class GymEpisodeDriver(Driver):

    def __init__(self,
                 env: CacheLastStepEnv | CacheLastStepVecEnv,
                 policy: Callable):
        super(GymEpisodeDriver, self).__init__()
        if not isinstance(env, (CacheLastStepEnv, CacheLastStepVecEnv, CacheLastStepVecEnvPool)):
            raise ValueError(f'Provided environment needs to be wrapped in {CacheLastStepEnv.__class__} ',
                             f'or {CacheLastStepVecEnv.__class__}')
        self.env = env
        self.policy = policy

    def interact(self,
                 n_episodes: int,
                 mem: List[Dict[str, DataType]] = None,
                 seed: Union[int, List[int]] = None,
                 **kwargs) -> List[Dict[str, DataType]]:
        if mem is None:
            mem = []

        ep_iter = tqdm(range(n_episodes), desc='Collecting Samples')
        if type(seed) is int:
            seed = [seed for _ in range(n_episodes)]
        elif seed is None:
            seed = [None for _ in range(n_episodes)]

        while ep_iter.n < n_episodes:
            trajectories = collect_data(self.env, -1, self.policy)
            mem.extend(trajectories)
            ep_iter.n += len(trajectories)
            ep_iter.refresh()

        #for i_ep in ep_iter:
        #    traj_o, traj_a, traj_r, traj_term, traj_trunc, traj_mask = [], [], [], [], [], []
        #    act_in_env(self.env, self.policy, -1, traj_o, traj_a, traj_r, traj_term, traj_trunc, traj_mask,
        #               seed=seed[i_ep])
        #    traj = {'o': np.array(traj_o), 'a': np.array(traj_a), 'r': np.array(traj_r),
        #            'terminal': np.array(traj_term), 'truncated': np.array(traj_trunc)}
        #    mem.append(traj)

        return mem


def collect_data(env: Union[CacheLastStepEnv, CacheLastStepVecEnv], n_steps: int, policy: callable = None,
                 seed: int = None, options: dict = None):
    if not isinstance(env, (CacheLastStepEnv, CacheLastStepVecEnv, CacheLastStepVecEnvPool)):
        print(f'Normal gym envs need to be wrapped in {CacheLastStepEnv.__class__.__name__} and ',
              f'vectorized environments need to be wrapped in {CacheLastStepVecEnv.__class__.__name__}',
              f' or {CacheLastStepVecEnvPool.__class__.__name__}')

    if policy is None:
        policy = RandomPolicy(env)

    traj_o, traj_a, traj_r, traj_term, traj_trunc, traj_mask = [], [], [], [], [], []
    if isinstance(env, CacheLastStepEnv):
        act_in_env(env, policy, n_steps, traj_o, traj_a, traj_r, traj_term, traj_trunc, traj_mask, seed=seed,
                   options=options)
    elif isinstance(env, (CacheLastStepVecEnv, CacheLastStepVecEnvPool)):
        act_in_vector_env(env, policy, n_steps, traj_o, traj_a, traj_r, traj_term, traj_trunc, traj_mask, seed=seed,
                          options=options)

    o = np.stack(traj_o)
    a = np.stack(traj_a)
    r = np.stack(traj_r)
    term = np.stack(traj_term)
    trunc = np.stack(traj_trunc)
    mask = np.stack(traj_mask)

    l_trajs = (~mask).sum(axis=0)
    if isinstance(env, CacheLastStepEnv):
        o = o[:, None]
        a = a[:, None]
        r = r[:, None]
        term = term[:, None]
        trunc = trunc[:, None]
        l_trajs = [l_trajs]
        n_trajs = 1
    elif isinstance(env, (CacheLastStepVecEnv, CacheLastStepVecEnvPool)):
        n_trajs = env.num_envs

    mem = []
    for i_traj in range(n_trajs):
        l_traj = l_trajs[i_traj]
        traj = {'o': o[:l_traj, i_traj], 'a': a[:l_traj, i_traj], 'r': r[:l_traj, i_traj],
                'terminal': term[:l_traj, i_traj], 'truncated': trunc[:l_traj, i_traj]}
        mem.append(traj)

    #print(f'Collected {len(mem)} trajectories with lengths between {np.min(l_trajs)} and {np.max(l_trajs)} steps')

    return mem


def act_in_vector_env(env: CacheLastStepVecEnv,
                      policy: callable,
                      n_steps: int,
                      traj_o: List[DataType],
                      traj_a: List[ActType],
                      traj_r: List[DataType],
                      traj_term: List[DataType],
                      traj_trunc: List[DataType],
                      traj_mask: List[DataType],
                      seed: int = None,
                      options: dict = None,
                      pad_data: bool = True):
    if n_steps == -1:
        n_steps = sys.maxsize.real

    if env.current_step == 0:
        n_steps -= 1
        _, _ = env.reset(seed=seed, options=options)
        traj_o.append(env.last_o)
        traj_mask.append(env.envs_done.copy())
        if pad_data:  # by convention, make (a_0, r_0, t_0) = 0
            traj_a.append(env.last_a)
            traj_r.append(env.last_r)
            traj_term.append(env.last_term)
            traj_trunc.append(env.last_trunc)

    all_actions_done = True
    for t in range(n_steps):
        if env.all_envs_done:
            all_actions_done = False
            break

        a = policy(env)
        env.step(a)

        traj_o.append(env.last_o)
        traj_a.append(env.last_a)
        traj_r.append(env.last_r)
        traj_term.append(env.last_term)
        traj_trunc.append(env.last_trunc)
        traj_mask.append(env.envs_done.copy())

    # roll the mask one to the right to prevent it from masking the final reward
    last_step = traj_mask.pop(-1)
    traj_mask.insert(0, np.full_like(last_step, False))
    return all_actions_done


def act_in_env(env: CacheLastStepEnv,
               policy: callable,
               n_steps: int,
               traj_o: List[DataType],
               traj_a: List[ActType],
               traj_r: List[DataType],
               traj_term: List[DataType],
               traj_trunc: List[DataType],
               traj_mask: List[DataType],
               seed: int = None,
               options: dict = None,
               pad_data: bool = True):
    assert env.current_step == 0 or (len(traj_o) > 0 and len(traj_a) > 0 and len(traj_r) > 0 and len(traj_term) > 0
                                     and len(traj_trunc) > 0), 'env must either be in step 0 or last step info must ' \
                                                               'be provided'
    if n_steps == -1:
        n_steps = sys.maxsize.real

    env_done = False
    if env.current_step == 0:
        n_steps -= 1
        _, _ = env.reset(seed=seed, options=options)
        traj_o.append(env.last_o)
        if pad_data:  # by convention, make (a_0, r_0, t_0) = 0
            env_done = env.last_term or env.last_trunc
            traj_a.append(env.last_a)
            traj_r.append(env.last_r)
            traj_term.append(env.last_term)
            traj_trunc.append(env.last_trunc)
            traj_mask.append(env_done)

    all_actions_done = True
    for t in range(n_steps):
        a = policy(env)
        env.step(a)
        env_done = env.last_term or env.last_trunc or env_done

        traj_o.append(env.last_o)
        traj_a.append(env.last_a)
        traj_r.append(env.last_r)
        traj_term.append(env.last_term)
        traj_trunc.append(env.last_trunc)
        traj_mask.append(env.last_term or env.last_trunc)

        if env.last_term or env.last_trunc:
            all_actions_done = False
            break
    # roll the mask one to the right to prevent it from masking the final reward
    traj_mask.pop(-1)
    traj_mask.insert(0, False)
    return all_actions_done