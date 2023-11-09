import itertools
from typing import Dict, Union, Sequence

import torch
import numpy as np

from mdm.policies.actor_critic_agent import ActorCriticAgent
from mdm.policies.policy import Policy
from mdm.models.hierarchical_rssm import HierarchicalRSSM
from mdm.utils.utils import prepare_data
from mdm.utils.torch_tools import unsqueeze_right
from mdm.utils.gym_wrappers import CacheLastStepEnv, CacheLastStepVecEnv, CacheLastStepVecEnvPool


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
                 explore: bool = False,
                 init_data: Dict[str, torch.Tensor] = None):
        super().__init__()
        #assert np.prod(agent.d_o) == model.rssm_modules[agent.level].d_z_smpl

        self.agent = agent
        self.model = model
        self.explore = explore

        if init_data is not None:
            mem, env_state = model.observe(o=init_data['o'], a=init_data['a'], r=init_data['r'],
                                           terminal=init_data['terminal'], level=agent.level,
                                           use_ema_modules=agent.use_slow_world_model)
            self._current_env_state = env_state
        else:
            self._current_env_state = None

    @torch.no_grad()
    def __call__(self, env: Union[CacheLastStepEnv, CacheLastStepVecEnv]):
        device = self.model.current_device
        self.model.eval()
        self.agent.eval()

        if env.current_step == 0:
            if isinstance(env, (CacheLastStepVecEnv, CacheLastStepVecEnvPool)):
                d_batch = env.last_o.shape[0]
            else:
                d_batch = 1
            self._current_env_state = self.model.rssm_modules[self.agent.level].init_state(d_batch, device)

        # get data and add time dim
        o = torch.from_numpy(env.last_o).unsqueeze(0).to(device=device, dtype=torch.float32)
        a = torch.from_numpy(env.last_a).unsqueeze(0).to(device=device, dtype=torch.float32)
        r = torch.from_numpy(np.array(env.last_r)).unsqueeze(0).to(device=device, dtype=torch.float32)
        terminal = torch.from_numpy(np.array(env.last_term)).unsqueeze(0).to(device=device, dtype=torch.float32)
        truncated = torch.from_numpy(np.array(env.last_trunc)).unsqueeze(0).to(device=device, dtype=torch.float32)
        if isinstance(env, CacheLastStepEnv):  # add batch dim if unbatched env
            o, a, r, terminal, truncated = [x.unsqueeze(1) for x in (o, a, r, terminal, truncated)]
        # prepare data
        env_data = {'o': o, 'a': a, 'r': r, 'terminal': terminal, 'truncated': truncated, 'mask': torch.zeros_like(r)}
        env_data = prepare_data(env_data)

        for k, v in env_data.items():
            if torch.isnan(v).any():
                raise RuntimeError(f'Invalid NAN input for key {k} in step {env.current_step}: {v}')
            if torch.isinf(v).any():
                raise RuntimeError(f'Invalid inf input for key {k} in step {env.current_step}: {v}')

        # digest new groundtruth data in level 0 model
        _, _, self._current_env_state = self.model.forward_static(trajectory=env_data,
                                                                  start_state=self._current_env_state,
                                                                  level=self.agent.level, n_steps=1, n_warmup=1,
                                                                  sample_state=False, sample_output=False,
                                                                  use_ema_modules=self.agent.use_slow_world_model)

        if torch.isnan(self._current_env_state[0]).any():
            raise RuntimeError(f'Invalid NAN state in step {env.current_step}: {self._current_env_state[0]}')
        if torch.isinf(self._current_env_state[0]).any():
            raise RuntimeError(f'Invalid inf state in step {env.current_step}: {self._current_env_state[0]}')

        agent_o = self.agent.fuse_o_with_goal(self._current_env_state)
        a_dist, a, = self.agent(agent_o, sample=self.explore, disable_exploration=not self.explore)

        if torch.isnan(a).any():
            raise RuntimeError(f'Invalid NAN action: {a}')
        if torch.isinf(a).any():
            raise RuntimeError(f'Invalid inf action: {a}')

        if isinstance(env, CacheLastStepEnv):
            a = a[0]

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
                 explore: bool = False):
        super().__init__()

        self.model = model
        self.explore = explore
        self._grounded_env_states = [None for _ in range(model.levels)]
        self._env_data_below_cache = [self._empty_cache() for _ in range(model.levels)]
        self._act_cache = [[] for _ in range(model.levels)]
        self._action_queue = []
        self._next_state_update = model.strides
        self.flight_record = [{'z': [], 'o': [], 'a': [], 'r': [], 'terminal': [], 'time_step': [], 'z_post': []}
                              for _ in range(model.levels)]

        use_slow_world_model = []
        o_key = []
        for agent in itertools.chain.from_iterable([model.r_max_agents, model.goal_seeking_agents]):
            if agent is not None:
                use_slow_world_model.append(agent[0].use_slow_world_model)
                o_key.append(agent[0].observation_type)

        use_slow_world_model = set(use_slow_world_model)
        o_key = set(o_key)
        assert len(use_slow_world_model) == 1, "all agents should either use the slow world model or the fast one"
        assert len(o_key) == 1, "all agents should use the same observation key"
        self._use_ema_modules = use_slow_world_model.pop()

    @staticmethod
    def _empty_cache():
        return {'o': [], 'r': [], 'terminal': [], 'time_step': []}

    def _reset(self,
               d_batch: int):
        self._grounded_env_states = [rssm.init_state(d_batch, self.model.device) for rssm in self.model.rssm_modules]
        self._env_data_below_cache = [self._empty_cache() for _ in range(self.model.levels)]
        self._act_cache = [[] for _ in range(self.model.levels)]
        self._next_state_update = self.model.strides
        self._action_queue = []

    @torch.no_grad()
    def _prep_step(self,
                   env: Union[CacheLastStepEnv, CacheLastStepVecEnv]):
        device = self.model.device
        # get data and add time dim
        o = torch.from_numpy(np.array(env.last_o)).unsqueeze(0).to(device=device, dtype=torch.float32)
        a = torch.from_numpy(np.array(env.last_a)).unsqueeze(0).to(device=device, dtype=torch.float32)
        r = torch.from_numpy(np.array(env.last_r)).unsqueeze(0).to(device=device, dtype=torch.float32)
        terminal = torch.from_numpy(np.array(env.last_term)).unsqueeze(0).to(device=device, dtype=torch.float32)
        truncated = torch.from_numpy(np.array(env.last_trunc)).unsqueeze(0).to(device=device, dtype=torch.float32)
        if isinstance(env, CacheLastStepEnv):  # add batch dim if unbatched env
            o, a, r, terminal, truncated = [x.unsqueeze(1) for x in (o, a, r, terminal, truncated)]
        env_data = {'o': o, 'a': a, 'r': r, 'terminal': terminal, 'truncated': truncated, 'mask': torch.empty_like(r)}
        env_data = prepare_data(env_data, remove_keys=['a', 'truncated', 'mask'])
        env_data = {k: list(v.unbind(0)) for k, v in env_data.items()}
        return env_data

    @torch.no_grad()
    def _check_state_update(self,
                            env: Union[CacheLastStepEnv, CacheLastStepVecEnv]):
        for i_lvl in range(self.model.levels):
            self._next_state_update[i_lvl] -= 1
            if self._next_state_update[i_lvl] > 0:
                continue  # no need to update level yet

            # prepare data from level below for this level's model to digest
            # env_data_below = {k: torch.stack(v) for k, v in self._env_data_below_cache[i_lvl].items()}
            # actions = self._act_cache[i_lvl].pop(0).unsqueeze(0)  # take oldest action from cache
            # data_filtered = {k: self.model.upwards_filters[i_lvl][k](env_data_below[k]) for k in ('o', 'r', 'terminal')}
            # data_filtered['a'] = actions  # we don't want filtered actions from lower level, but original ones from this

            n_steps = self.model.strides[i_lvl]
            data_filtered = self.model.filter_up(o=self._env_data_below_cache[i_lvl]['o'],
                                                 r=self._env_data_below_cache[i_lvl]['r'],
                                                 terminal=self._env_data_below_cache[i_lvl]['terminal'],
                                                 level=i_lvl, n_steps=n_steps,
                                                 respect_terminal_flag=True)
            data_filtered['a'] = self._act_cache[i_lvl].pop(0).unsqueeze(0)  # take oldest action from cache

            # memorize the latest inputs the model has seen as they are needed for the agent during planning
            state = self._grounded_env_states[i_lvl]
            mem, _, new_state = self.model.forward_static(data_filtered, start_state=state, level=i_lvl, n_steps=-1,
                                                          n_warmup=-1, sample_state=False, sample_output=False,
                                                          use_ema_modules=self._use_ema_modules)
            self._grounded_env_states[i_lvl] = new_state

            # bookkeeping
            self.flight_record[i_lvl]['o'].append(data_filtered['o'][0])
            self.flight_record[i_lvl]['z'].append(new_state[0])
            self.flight_record[i_lvl]['z_post'].append(mem['z_post'][0])
            self.flight_record[i_lvl]['a'].append(data_filtered['a'][0])
            self.flight_record[i_lvl]['r'].append(mem['r'][0])
            self.flight_record[i_lvl]['terminal'].append(mem['terminal'][0])
            timestep = torch.tensor(env.current_step, dtype=torch.float32, device=state[0].device)

            if isinstance(env, CacheLastStepVecEnv):
                timestep = torch.tile(timestep[None, ...], [env.unwrapped.num_envs, 1])
            self.flight_record[i_lvl]['time_step'].append(timestep)

            # store updated state in cache for upper level
            if i_lvl < self.model.i_top:
                o_key = self.model.r_max_agents[i_lvl + 1][0].observation_type
                self._env_data_below_cache[i_lvl + 1]['o'] += mem[o_key]
                self._env_data_below_cache[i_lvl + 1]['r'] += mem['r']
                self._env_data_below_cache[i_lvl + 1]['terminal'] += mem['terminal']
                self._env_data_below_cache[i_lvl + 1]['time_step'].append(timestep)

            # reset counter and clear caches
            self._next_state_update[i_lvl] = self.model.strides[i_lvl]
            self._env_data_below_cache[i_lvl] = self._empty_cache()

    @torch.no_grad()
    def _replan(self):
        # find highest planning level
        i_highest = sum([state is not None for state in self._grounded_env_states]) - 1

        # 1: plan with r_max agent on highest level available
        # 2: use all agents below to go to intermediate goals

        # find out if we're in warmup phase and need to plan more than one step ahead
        # if i_highest < self.model.i_top:
        #    n_plan_steps = self.model.strides[i_highest + 1] - 1  # first initial zero action
        # elif i_highest < self.model.i_top:
        #    n_plan_steps = self.model.strides[i_highest + 1]
        # else:
        #    n_plan_steps = 1  # we're not in warmup phase anymore, only plan a single step ahead on highest level

        # one r_max step on highest level
        state = self._grounded_env_states[i_highest]
        agent = self.model.r_max_agents[i_highest][0]
        simulation = agent.act_in_sim(env_start_state=state, sim_env=self.model, n_steps=1, sample_actions=self.explore,
                                      sample_states=False, disable_exploration=not self.explore,
                                      reconstruct=i_highest > 0)
        self._act_cache[i_highest] += simulation['agent']['a']

        if i_highest > 0:
            goals_from_above = simulation['model']['o']
            for i_lvl in reversed(range(0, i_highest)):
                state = self._grounded_env_states[i_lvl]
                agent = self.model.goal_seeking_agents[i_lvl][0]
                n_steps = self.model.strides[i_lvl + 1]
                new_goals = []
                for goal in goals_from_above:
                    simulation = agent.act_in_sim(env_start_state=state, sim_env=self.model, n_steps=n_steps, goal=goal,
                                                  sample_actions=False, disable_exploration=True,  # never explore here
                                                  sample_states=False, reconstruct=i_lvl > 0)
                    state = simulation['model_state']
                    self._act_cache[i_lvl] += simulation['agent']['a']
                    if i_lvl > 0:
                        new_goals += simulation['model']['o']
                goals_from_above = new_goals

        self._action_queue += self._act_cache[0]

        """
        simulation = agent.act_in_sim(env_state=state, sim_env=self.model, n_steps=1, sample_actions=True,
                                      sample_model=True, disable_exploration=True, reconstruct=i_highest > 0)
        self._act_cache[i_highest] += simulation['agent']['a']

        if i_highest > 0:
            goals_from_above = simulation['model']['o']
            is_terminal_chunk = simulation['model']['terminal']
            for i_lvl in reversed(range(0, i_highest)):
                state = self._grounded_env_states[i_lvl]
                g_agent = self.model.goal_seeking_agents[i_lvl][0]
                r_agent = self.model.r_max_agents[i_lvl][0]
                n_steps = self.model.strides[i_lvl + 1]
                new_goals = []
                for term_chunk, goal in zip(is_terminal_chunk, goals_from_above):
                    g_sim = g_agent.act_in_sim(env_state=state, sim_env=self.model, n_steps=n_steps, goal=goal,
                                               sample_actions=True, disable_exploration=True,
                                               sample_model=True, reconstruct=i_lvl > 0)
                    r_sim = r_agent.act_in_sim(env_state=state, sim_env=self.model, n_steps=n_steps,
                                               sample_actions=True, disable_exploration=True,
                                               sample_model=True, reconstruct=i_lvl > 0)

                    term_chunk = term_chunk.to(torch.bool)
                    state['z'] = torch.where(term_chunk, r_sim['model_state']['z'], g_sim['model_state']['z'])
                    r_sim_rnn_states = pack_rnn_state(r_sim['model_state']['rnn_state'])
                    g_sim_rnn_states = pack_rnn_state(g_sim['model_state']['rnn_state'])
                    term_chunk_expanded = unsqueeze_right(term_chunk, r_sim_rnn_states)
                    state['rnn_state'] = unpack_rnn_state(torch.where(term_chunk_expanded,
                                                                      r_sim_rnn_states, g_sim_rnn_states))
                    a_tmp = [torch.where(term_chunk, a_r_sim, a_g_sim) for
                             a_r_sim, a_g_sim in zip(r_sim['agent']['a'], g_sim['agent']['a'])]
                    self._act_cache[i_lvl] += a_tmp
                    if i_lvl > 0:
                        goals_tmp = [torch.where(term_chunk, g_r_sim, g_g_sim) for
                                     g_r_sim, g_g_sim in zip(r_sim['model']['o'], g_sim['model']['o'])]
                        new_goals += goals_tmp
                goals_from_above = new_goals

        self._action_queue += self._act_cache[0]
        """

    @torch.no_grad()
    def __call__(self,
                 env: Union[CacheLastStepEnv, CacheLastStepVecEnv]):
        self.model.eval()
        for agent in self.model.r_max_agents + self.model.goal_seeking_agents:
            if agent is not None:
                agent[0].eval()

        if isinstance(env, (CacheLastStepVecEnv, CacheLastStepVecEnvPool)):
            d_batch = env.last_o.shape[0]
        else:
            d_batch = 1
        if env.current_step == 0:
            self._reset(d_batch)
            # store default zero actions as last performend action
            self._act_cache = [[torch.zeros((d_batch, rssm.d_a), device=self.model.device, dtype=torch.float32)] for
                               rssm in self.model.rssm_modules]

        # TODO: are the correct state time steps recorded?

        with torch.no_grad():
            # first step: store current env ground truth data to cache
            self._env_data_below_cache[0] = self._prep_step(env)
            self._check_state_update(env)  # update individual level's states if necessary

            # check if re-planning is needed
            if len(self._action_queue) == 0:
                self._replan()

            action = self._action_queue.pop(0)

        if isinstance(env, CacheLastStepEnv):  # remove batch dimension if it's not a vector env
            action = action[0]
        return action.detach().cpu().numpy()
