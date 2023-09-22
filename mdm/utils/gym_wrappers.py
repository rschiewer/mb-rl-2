from typing import Tuple, Optional, Union, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType, ActType
import envpool
from envpool.python.gym_envpool import GymEnvPoolMeta
from envpool.python.gymnasium_envpool import GymnasiumEnvPoolMeta
from gymnasium.vector.utils import write_to_shared_memory
import sys

from mdm.utils.torch_tools import unsqueeze_right


def check_action(action: np.ndarray, env: gym.Env):
    if np.isnan(action).any():
        raise ValueError('NAN actions found!')
    if np.isinf(action).any():
        raise ValueError('inf actions found!')
    if not env.action_space.contains(action):
        raise ValueError(f'Invalid action {action} for action space {env.action_space}')


class StepCountEnv(gym.Wrapper):

    def __init__(self,
                 env: gym.Env):
        super(StepCountEnv, self).__init__(env)
        self.current_step = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        self.current_step = 0
        return super(StepCountEnv, self).reset(seed=seed, options=options)

    def step(self,
             action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        o, r, term, trunc, info = self.env.step(action)
        if term or trunc:
            self.current_step = 0
        else:
            self.current_step += 1
        return o, r, term, trunc, info


class CacheLastStepEnv(StepCountEnv):

    def __init__(self,
                 env: gym.Env):
        super(CacheLastStepEnv, self).__init__(env)
        self.last_o = None
        self.last_a = None
        self.last_r = None
        self.last_term = None
        self.last_trunc = None
        self.last_info = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        o, info = self.env.reset(seed=seed, options=options)
        self.last_o = o
        self.last_a = np.zeros_like(self.action_space.sample())
        self.last_r = 0.0
        self.last_term = False
        self.last_trunc = False
        self.last_info = info
        return o, info

    def step(self,
             action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        o, r, term, trunc, info = self.env.step(action)
        self.last_o = o
        self.last_a = action
        self.last_r = r
        self.last_term = term
        self.last_trunc = trunc
        self.last_info = info
        return o, r, term, trunc, info


class CacheLastStepVecEnv(gym.Wrapper):

    def __init__(self,
                 env: gym.vector.VectorEnv):
        super(CacheLastStepVecEnv, self).__init__(env)
        self.envs_done = np.full((self.unwrapped.num_envs,), False)
        self.last_o = None
        self.last_a = None
        self.last_r = None
        self.last_term = None
        self.last_trunc = None
        self.last_info = None
        self.current_step = 0

    @property
    def all_envs_done(self):
        return self.envs_done.all()

    def reset(self,
              *,
              seed: Optional[Union[int, List[int]]] = None,
              options: Optional[dict] = None):
        self.envs_done[:] = False
        o, infos = self.env.reset(seed=seed, options=options)
        self.last_o = o
        self.last_a = np.zeros_like(self.env.action_space.sample())
        self.last_r = np.full((self.unwrapped.num_envs,), 0.0)
        self.last_term = np.full((self.unwrapped.num_envs,), False)
        self.last_trunc = np.full((self.unwrapped.num_envs,), False)
        self.last_info = infos
        self.current_step = 0
        return o, infos

    def step(self,
             actions):
        if self.envs_done.all():
            raise RuntimeError('All envs are already done, call reset()')
        check_action(actions, self.env)  # invalid actions in vectorized envs cause hard to understand errors

        o, r, term, trunc, infos = self.env.step(actions)
        self.last_o = np.where(unsqueeze_right(self.envs_done, o), np.zeros_like(o), o)
        self.last_a = np.where(unsqueeze_right(self.envs_done, actions), np.zeros_like(actions), actions)
        self.last_r = np.where(self.envs_done, np.zeros_like(r), r)
        self.last_term = np.where(self.envs_done, np.zeros_like(term), term)
        self.last_trunc = np.where(self.envs_done, np.zeros_like(trunc), trunc)
        self.last_info = infos

        done_now = np.bitwise_or(term, trunc)
        #if done_now.any():
        #    for i_env, final_obs_available in enumerate(infos['_final_observation']):
        #        if final_obs_available:
        #            self.last_o[i_env] = infos['final_observation'][i_env]
        #    # mask = np.bitwise_and(infos['_final_observation'], self.envs_done)
        #    # mask = expand_shape_right(mask, o)
        #    # self.last_o = np.where(~mask, expand_shape_right(infos['final_observation'], o), o)  # TODO: check this
        self.envs_done = np.bitwise_or(self.envs_done, done_now)

        if self.envs_done.all():
            self.current_step = 0
        else:
            self.current_step += 1

        return None, None, None, None, None


class CacheLastStepVecEnvPool:

    def __init__(self,
                 env: GymEnvPoolMeta):
        self.unwrapped = env
        self.envs_done = np.full((len(env.all_env_ids),), False)
        self.last_o = None
        self.last_a = None
        self.last_r = None
        self.last_term = None
        self.last_trunc = None
        self.last_info = None
        self.current_step = 0

        # calculate action space of batched envs
        single_space = self.unwrapped.action_space
        if not isinstance(single_space, gym.spaces.Box):
            raise ValueError('Only box action spaces are currently supported')

        shape = single_space.shape
        ndim = len(shape)
        lows = np.tile(single_space.low[None, ...], reps=(self.num_envs, *[1 for _ in range(ndim)]))
        highs = np.tile(single_space.high[None, ...], reps=(self.num_envs, *[1 for _ in range(ndim)]))
        space = gym.spaces.Box(low=lows, high=highs, dtype=single_space.dtype)
        self.action_space = space

    @property
    def num_envs(self):
        return len(self.unwrapped.all_env_ids)

    @property
    def all_envs_done(self):
        return self.envs_done.all()

    def reset(self,
              *,
              seed: Optional[Union[int, List[int]]] = None,
              options: Optional[dict] = None):
        self.envs_done[:] = False
        o, infos = self.unwrapped.reset()
        self.last_o = o
        self.last_a = np.zeros_like(self.action_space.sample())
        self.last_r = np.full((self.num_envs,), 0.0)
        self.last_term = np.full((self.num_envs,), False)
        self.last_trunc = np.full((self.num_envs,), False)
        self.last_info = infos
        self.current_step = 0
        return o, infos

    def step(self,
             actions):
        if self.envs_done.all():
            raise RuntimeError('All envs are already done, call reset()')

        o, r, term, trunc, infos = self.unwrapped.step(actions)
        self.last_o = np.where(unsqueeze_right(self.envs_done, o), np.zeros_like(o), o)
        self.last_a = np.where(unsqueeze_right(self.envs_done, actions), np.zeros_like(actions), actions)
        self.last_r = np.where(self.envs_done, np.zeros_like(r), r)
        self.last_term = np.where(self.envs_done, np.zeros_like(term), term)
        self.last_trunc = np.where(self.envs_done, np.zeros_like(trunc), trunc)
        self.last_info = infos

        done_now = np.bitwise_or(term, trunc)
        self.envs_done = np.bitwise_or(self.envs_done, done_now)

        if self.envs_done.all():
            self.current_step = 0
        else:
            self.current_step += 1

        # return o, r, term, trunc, infos
        return None, None, None, None, None


def vec_env_worker_no_auto_reset(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
    assert shared_memory is not None
    env = env_fn()
    observation_space = env.observation_space
    parent_pipe.close()
    try:
        episode_terminated = True
        episode_truncated = True
        while True:
            command, data = pipe.recv()
            if command == "reset":
                observation, info = env.reset(**data)
                episode_terminated = False
                episode_truncated = False
                write_to_shared_memory(observation_space, index, observation, shared_memory)
                pipe.send(((None, info), True))
            elif command == "step":
                if episode_terminated or episode_truncated:  # give zero o, r and repeat final term, trunc flags
                    observation = np.zeros(env.observation_space.shape, dtype=env.observation_space.dtype)
                    reward = 0
                    terminated = episode_terminated
                    truncated = episode_truncated
                    info = {'padding': True}
                else:
                    observation, reward, terminated, truncated, info = env.step(data)

                if terminated or truncated:
                    info["final_observation"] = observation
                    info["final_info"] = info
                    episode_terminated = terminated
                    episode_truncated = truncated

                write_to_shared_memory(observation_space, index, observation, shared_memory)
                pipe.send(((None, reward, terminated, truncated, info), True))
            elif command == "seed":
                env.seed(data)
                pipe.send((None, True))
            elif command == "close":
                pipe.send((None, True))
                break
            elif command == "_call":
                name, args, kwargs = data
                if name in ["reset", "step", "seed", "close"]:
                    raise ValueError(
                        f"Trying to call function `{name}` with "
                        f"`_call`. Use `{name}` directly instead."
                    )
                function = getattr(env, name)
                if callable(function):
                    pipe.send((function(*args, **kwargs), True))
                else:
                    pipe.send((function, True))
            elif command == "_setattr":
                name, value = data
                setattr(env, name, value)
                pipe.send((None, True))
            elif command == "_check_spaces":
                pipe.send(
                    ((data[0] == observation_space, data[1] == env.action_space), True)
                )
            else:
                raise RuntimeError(
                    f"Received unknown command `{command}`. Must "
                    "be one of {`reset`, `step`, `seed`, `close`, `_call`, "
                    "`_setattr`, `_check_spaces`}."
                )
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()
