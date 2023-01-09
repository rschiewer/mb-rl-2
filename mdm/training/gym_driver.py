import sys
from typing import Callable, List, Dict, TypeVar, Union

import gym
from gym.core import ActType
import numpy as np
from tqdm import tqdm

from mdm.training.driver import Driver
from mdm.memory.trajectory_memory import TrajectoryMemory
from mdm.utils.gym_wrappers import CacheLastStepEnv, CacheLastStepVecEnv


DataType = TypeVar('DataType', np.ndarray, int, float, bool)


class GymEpisodeDriver(Driver):

    def __init__(self,
                 env: CacheLastStepEnv,
                 policy: Callable):
        super(GymEpisodeDriver, self).__init__()
        if not isinstance(env, CacheLastStepEnv):
            raise ValueError(f'Provided environment needs to be wrapped in {CacheLastStepEnv.__class__.__name__}')
        self.env = env
        self.policy = policy

    def interact(self,
                 n_episodes: int,
                 progress_bar: bool = False,
                 mem: List[Dict[str, DataType]] = None,
                 seed: Union[int, List[int]] = None,
                 **kwargs) -> List[Dict[str, DataType]]:
        if mem is None:
            mem = []

        ep_iter = range(n_episodes)
        if progress_bar:
            ep_iter = tqdm(ep_iter, desc='Collecting Samples')
        if type(seed) is int:
            seed = [seed for _ in range(n_episodes)]
        elif seed is None:
            seed = [None for _ in range(n_episodes)]

        for i_ep in ep_iter:
            traj_o, traj_a, traj_r, traj_term, traj_trunc = [], [], [], [], []
            act_in_env(self.env, self.policy, -1, traj_o, traj_a, traj_r, traj_term, traj_trunc,
                       seed=seed[i_ep])
            traj = {'o': np.array(traj_o), 'a': np.array(traj_a), 'r': np.array(traj_r),
                    'terminal': np.array(traj_term), 'truncated': np.array(traj_trunc)}
            mem.append(traj)

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
                      pad_data: bool = True):
    if n_steps == -1:
        n_steps = sys.maxsize.real

    if env.current_step == 0:
        o, _ = env.reset(seed=seed)
        traj_o.append(env.last_o)
        traj_mask.append(env.envs_done.copy())
        if pad_data:  # by convention, make (a_0, r_0, t_0) = 0
            traj_a.append(env.last_a)
            traj_r.append(env.last_r)
            traj_term.append(env.last_term)
            traj_trunc.append(env.last_trunc)

    for t in range(n_steps):
        a = policy(env.last_o, env.last_r, env.last_term, env.last_trunc, env.current_step == 0)
        o_, r, terminal, truncated, info = env.step(a)

        traj_o.append(o_)
        traj_a.append(a)
        traj_r.append(r)
        traj_term.append(terminal)
        traj_trunc.append(truncated)
        traj_mask.append(env.envs_done.copy())

        if env.all_envs_done:
            return False
    return True


def act_in_env(env: CacheLastStepEnv,
               policy: callable,
               n_steps: int,
               traj_o: List[DataType],
               traj_a: List[ActType],
               traj_r: List[DataType],
               traj_term: List[DataType],
               traj_trunc: List[DataType],
               seed: int = None,
               pad_data: bool = True):
    assert env.current_step == 0 or (len(traj_o) > 0 and len(traj_a) > 0 and len(traj_r) > 0 and len(traj_term) > 0
                                     and len(traj_trunc) > 0), 'env must either be in step 0 or last step info must ' \
                                                               'be provided'
    if n_steps == -1:
        n_steps = sys.maxsize.real

    if env.current_step == 0:
        o, _ = env.reset(seed=seed)
        traj_o.append(env.last_o)
        if pad_data:  # by convention, make (a_0, r_0, t_0) = 0
            traj_a.append(env.last_a)
            traj_r.append(env.last_r)
            traj_term.append(env.last_term)
            traj_trunc.append(env.last_trunc)

    for t in range(n_steps):
        a = policy(env.last_o, env.last_r, env.last_term, env.last_trunc, env.current_step == 0)
        o_, r, terminal, truncated, info = env.step(a)

        traj_o.append(o_)
        traj_a.append(a)
        traj_r.append(r)
        traj_term.append(terminal)
        traj_trunc.append(truncated)

        if terminal or truncated:
            return False
    return True


class GymStepDriver(GymEpisodeDriver):

    def interact(self,
                 n_steps: int = -1,
                 mem: List[Dict[str, DataType]] = None,
                 **kwargs) -> List[Dict[str, DataType]]:
        if mem is None:
            mem = []
        if n_steps == -1:
            n_steps = sys.maxsize.real

        traj_o, traj_a, traj_r, traj_term, traj_trunc = [], [], [], [], []
        o, _ = self.env.reset()
        traj_o.append(self._process_obs(o))
        traj_a.append(np.zeros_like(self.env.action_space.sample()))  # by convention, make (a_0, r_0, t_0) = 0
        traj_r.append(0.0)
        traj_term.append(False)
        traj_trunc.append(False)

        terminal = False
        truncated = False
        for t in range(n_steps):
            a = self.policy(traj_o[-1], traj_r[-1], traj_term[-1], traj_trunc[-1], )
            o_, r, terminal, truncated, info = self._process_step(self.env.step(a))

            traj_o.append(o_)
            traj_a.append(a)
            traj_r.append(r)
            traj_term.append(terminal)
            traj_trunc.append(truncated)

            if terminal or truncated:
                break

        traj = {'o': np.array(traj_o), 'a': np.array(traj_a), 'r': np.array(traj_r),
                'terminal': np.array(traj_term), 'truncated': np.array(traj_trunc)}

        return mem