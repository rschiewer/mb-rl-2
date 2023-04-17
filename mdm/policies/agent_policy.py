from typing import Dict, Union, Sequence

import torch
import numpy as np

from mdm.policies.actor_critic_agent import ActorCriticAgent
from mdm.policies.policy import Policy
from mdm.models.hierarchical_rssm import HierarchicalRSSM
from mdm.utils.torch_tools import to_tensors
from mdm.utils.utils import prepare_data
from mdm.utils.gym_wrappers import CacheLastStepEnv, CacheLastStepVecEnv


class AgentPolicy(Policy):

    def __init__(self,
                 agent: ActorCriticAgent):
        super().__init__()
        self.agent = agent

    def __call__(self, env):
        last_o = torch.from_numpy(env.last_o).to(self.agent.device)
        a_dist, a, v = self.agent(last_o)
        return a.detach().cpu().numpy()


class LatentAgentPolicy(Policy):

    def __init__(self,
                 agent: ActorCriticAgent,
                 model: HierarchicalRSSM,
                 init_data: Dict[str, torch.Tensor] = None,
                 use_ema_modules: bool = False):
        super().__init__()
        assert np.prod(agent.d_o) == model.rssm_modules[agent.level].d_z

        self.agent = agent
        self.model = model
        self._use_ema_modules = use_ema_modules

        if init_data is not None:
            mem, env_state = model.observe(o=init_data['o'], a=init_data['a'], r=init_data['r'],
                                           terminal=init_data['terminal'], level=agent.level,
                                           use_ema_modules=use_ema_modules)
            # mem, env_state = model(o=init_data['o'], a=init_data['a'], r=init_data['r'], terminal=init_data['terminal'],
            #                       level=agent.level, use_ema_modules=use_ema_modules)
            self._current_env_state = env_state
        else:
            self._current_env_state = None

    def __call__(self, env: Union[CacheLastStepEnv, CacheLastStepVecEnv]):
        device = self.agent.device

        if env.current_step == 0:
            if isinstance(env, CacheLastStepVecEnv):
                d_batch = env.last_o.shape[0]
            else:
                d_batch = 1
            self._current_env_state = self.model.rssm_modules[self.agent.level].init_state(d_batch, device)

        # get data and add time dim
        o = torch.from_numpy(env.last_o).unsqueeze(0).to(device=device, dtype=torch.float32)
        a = torch.from_numpy(env.last_a).unsqueeze(0).to(device=device, dtype=torch.float32)
        r = torch.from_numpy(env.last_r).unsqueeze(0).to(device=device, dtype=torch.float32)
        terminal = torch.from_numpy(env.last_term).unsqueeze(0).to(device=device, dtype=torch.float32)
        truncated = torch.from_numpy(env.last_trunc).unsqueeze(0).to(device=device, dtype=torch.float32)
        if isinstance(env, CacheLastStepEnv):  # add batch dim if unbatched env
            o, a, r, terminal, truncated = [x.unsqueeze(1) for x in (o, a, r, terminal, truncated)]

        env_data = {'o': o, 'a': a, 'r': r, 'terminal': terminal, 'truncated': truncated, 'mask': torch.empty_like(r)}
        env_data = prepare_data(env_data)
        # mem, self._current_env_state = self.model(o=env_data['o'], a=env_data['a'], r=env_data['r'],
        #                                          terminal=env_data['terminal'], start_state=self._current_env_state,
        #                                          level=self.agent.level, use_ema_modules=self._use_ema_modules)
        mem, self._current_env_state = self.model.observe(o=env_data['o'], a=env_data['a'], r=env_data['r'],
                                                          terminal=env_data['terminal'],
                                                          start_state=self._current_env_state,
                                                          level=self.agent.level, use_ema_modules=self._use_ema_modules)
        a_dist, a, v = self.agent(mem[self.agent.observation_key][-1])
        return a.detach().cpu().numpy()


class HierarchicalLatentAgentPolicy(Policy):
    """
    1.  Observe first env data time step from env initialization
    2.  From lower to higher level:
    3.    Use current level r_max agent to make decision and collect new time step from env
    4.    If not enough time steps to escalate to next level, GOTO 3
    5.  Use r_max agent on highest level to make decision
    6.  From higher to lower level:
    7.    While above level goals are available:
    8.      Use current level goal_seeking agent to find proposed goal from above
    9.      Store achieved model latent states in every step
    10.   Filter out every k-th step as goals for lower level
    11. Execute lowest level actions in real world
    """

    def __init__(self,
                 model: HierarchicalRSSM,
                 use_ema_modules: bool = False):
        super().__init__()

        self.model = model
        self._use_ema_modules = use_ema_modules
        self._current_env_states = [None for _ in range(model.levels)]
        self._env_data_below_cache = [{} for _ in range(model.levels)]
        self._last_step_cache = [{} for _ in range(model.levels)]
        self._act_cache = [[] for _ in range(model.levels)]
        self._action_queue = []
        self._next_state_update = model.strides

    def _reset(self):
        self._current_env_states = [None for _ in range(self.model.levels)]
        self._env_data_below_cache = [{} for _ in range(self.model.levels)]
        self._last_step_cache = [{} for _ in range(self.model.levels)]
        self._act_cache = [[] for _ in range(self.model.levels)]
        self._next_state_update = self.model.strides

    def _prep_step(self,
                   env: Union[CacheLastStepEnv, CacheLastStepVecEnv]):
        device = self.model.device
        # get data and add time dim
        o = torch.from_numpy(env.last_o).unsqueeze(0).to(device=device, dtype=torch.float32)
        a = torch.from_numpy(env.last_a).unsqueeze(0).to(device=device, dtype=torch.float32)
        r = torch.from_numpy(env.last_r).unsqueeze(0).to(device=device, dtype=torch.float32)
        terminal = torch.from_numpy(env.last_term).unsqueeze(0).to(device=device, dtype=torch.float32)
        truncated = torch.from_numpy(env.last_trunc).unsqueeze(0).to(device=device, dtype=torch.float32)
        if isinstance(env, CacheLastStepEnv):  # add batch dim if unbatched env
            o, a, r, terminal, truncated = [x.unsqueeze(1) for x in (o, a, r, terminal, truncated)]
        env_data = {'o': o, 'a': a, 'r': r, 'terminal': terminal, 'truncated': truncated, 'mask': torch.empty_like(r)}
        env_data = prepare_data(env_data)
        return env_data

    def _check_state_update(self):
        for i_lvl in range(self.model.levels):
            self._next_state_update[i_lvl] -= 1
            if self._next_state_update[i_lvl] > 0:
                continue  # no need to update level yet

            # prepare data from level below for this level's model to digest
            env_data_below = {k: torch.stack(v) for k, v in self._env_data_below_cache[i_lvl].items()}
            actions = self._act_cache[i_lvl].pop(0).unsqueeze(0)  # take oldest action from cache
            data_filtered = {k: self.model.upwards_filters[i_lvl][k](env_data_below[k]) for k in ('o', 'r', 'terminal')}
            data_filtered['a'] = actions  # we don't want filtered actions from lower level, but original ones from this

            # memorize the latest inputs the model has seen as they are needed for the agent during planning

            state = self._current_env_states[i_lvl]
            mem, new_state = self.model.observe(**data_filtered, start_state=state, level=i_lvl,
                                              use_ema_modules=self._use_ema_modules)
            self._last_step_cache[i_lvl] = mem
            self._current_env_states[i_lvl] = new_state

            # store updated state in cache for upper level
            if i_lvl < self.model.i_top:
                data = self._env_data_below_cache[i_lvl + 1].get('o', [])
                data += mem[self.model.links[i_lvl]]
                self._env_data_below_cache[i_lvl + 1]['o'] = data
                for k in ('r', 'terminal'):
                    data = self._env_data_below_cache[i_lvl + 1].get(k, [])
                    data += mem[k]
                    self._env_data_below_cache[i_lvl + 1][k] = data

            # reset counter and clear caches
            self._next_state_update[i_lvl] = self.model.strides[i_lvl]
            #self._act_cache[i_lvl] = []
            self._env_data_below_cache[i_lvl] = {}

    def _replan(self):
        # find highest planning level
        i_highest = sum([state is not None for state in self._current_env_states]) - 1

        # 1: plan with r_max agent on highest level available
        # 2: use all agents below to go to intermediate goals

        # find out if we're in warmup phase and need to plan more than one step ahead
        if i_highest == 0:
            n_plan_steps = self.model.strides[i_highest + 1] - 1  # first initial zero action
        elif i_highest < self.model.i_top:
            n_plan_steps = self.model.strides[i_highest + 1]
        else:
            n_plan_steps = 1

        # highest level
        hist, state = self._last_step_cache[i_highest], self._current_env_states[i_highest]
        goal_mem, _, lowest_agent = self.model.simulate(n_steps=n_plan_steps, before_history=hist, start_state=state,
                                                        level=i_highest)
        self._act_cache[i_highest] = lowest_agent['a']

        for i_lvl in reversed(range(1, i_highest + 1)):
            hist, state = self._last_step_cache[i_lvl], self._current_env_states[i_lvl]
            goal_mem, lowest_agent = self._follow_plan(i_lvl, goal_mem, hist, state)
            self._act_cache[i_lvl - 1] = lowest_agent['a']

        self._action_queue += lowest_agent['a']

    def _follow_plan(self,
                     lvl_current: int,
                     mem_current: Dict[str, torch.Tensor],
                     last_step_below: Dict[str, torch.Tensor],
                     state_below: Dict[str, torch.Tensor]):
        lvl_below = lvl_current - 1
        chunk_size = self.model.strides[lvl_current]
        # TODO: currently uses only first chunk for warmup, could be varied during training for better results
        n_warmup_chunks = 1
        agent_mem_below = {}
        mem_below = {}
        for goal in mem_current['o']:
            mem_below, state_below, agent_mem_below = self.model.simulate(n_steps=chunk_size,
                                                                          before_history=last_step_below,
                                                                          start_state=state_below, agent_goal=goal,
                                                                          level=lvl_below, memory=mem_below,
                                                                          agent_memory=agent_mem_below)
        return mem_below, agent_mem_below

    def __call__(self,
                 env: Union[CacheLastStepEnv, CacheLastStepVecEnv]):
        if env.current_step == 0:
            self._reset()
            d_batch = env.last_a.shape[0]
            self._act_cache = [[torch.zeros((d_batch, rssm.d_a), device=self.model.device, dtype=torch.float32)] for rssm in self.model.rssm_modules]

        # first step: store current env ground truth data to cache
        self._env_data_below_cache[0] = {k: v.unbind(0) for k, v in self._prep_step(env).items()}
        self._check_state_update()  # update individual level's states if necessary

        # check if re-planning is needed
        if len(self._action_queue) == 0:
            self._replan()

        action = self._action_queue.pop(0)
        return action.detach().cpu().numpy()
