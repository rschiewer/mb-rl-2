import itertools
import random
from typing import Dict, Union, Sequence

import torch
import numpy as np

from mdm.policies.actor_critic_agent import ActorCriticAgent
from mdm.policies.policy import Policy
from mdm.models.hierarchical_rssm import HierarchicalRSSM
from mdm.utils.utils import prepare_data, append_memory, extend_memory
from mdm.utils.torch_tools import unsqueeze_right, compute_mask
from mdm.utils.gym_wrappers import CacheLastStepEnv, CacheLastStepVecEnv


class LatentAgentPolicy(Policy):

    def __init__(self,
                 agent: ActorCriticAgent,
                 model: HierarchicalRSSM,
                 exploration_noise: float = 0.0,
                 stochastic: bool = False,
                 init_data: Dict[str, torch.Tensor] = None):
        super().__init__()
        # assert np.prod(agent.d_o) == model.rssm_modules[agent.level].d_z_smpl

        self.agent = agent
        self.model = model
        self.exploration_noise = exploration_noise
        self.stochastic = stochastic

        if init_data is not None:
            mem, env_state = model.observe(o=init_data['o'], a=init_data['a'], r=init_data['r'],
                                           terminal=init_data['terminal'], level=agent.level,
                                           use_ema_modules=agent.use_slow_world_model)
            self._current_env_state = env_state
        else:
            self._current_env_state = None

        self.sample_world_model = True

        if agent.discrete_actions:
            self.one_hot_keys = {'a': agent.d_a}
        else:
            self.one_hot_keys = {}

    def reset(self):
        self._current_env_state = None

    @torch.no_grad()
    def __call__(self, env: Union[CacheLastStepEnv, CacheLastStepVecEnv]):
        device = self.model.current_device
        self.model.eval()
        self.agent.eval()

        if env.current_step == 0:
            if isinstance(env, (CacheLastStepVecEnv)):
                d_batch = env.last_o.shape[0]
            else:
                d_batch = 1
            self._current_env_state = self.model.rssm_modules[self.agent.level].init_state(d_batch, device)

        # get data and add time dim
        o = torch.from_numpy(np.array(env.last_o)).unsqueeze(0).to(device=device, dtype=torch.float32)
        a = torch.from_numpy(np.array(env.last_a)).unsqueeze(0).to(device=device, dtype=torch.float32)
        r = torch.from_numpy(np.array(env.last_r)).unsqueeze(0).to(device=device, dtype=torch.float32)
        terminal = torch.from_numpy(np.array(env.last_term)).unsqueeze(0).to(device=device, dtype=torch.float32)
        truncated = torch.from_numpy(np.array(env.last_trunc)).unsqueeze(0).to(device=device, dtype=torch.float32)
        if isinstance(env, CacheLastStepEnv):  # add batch dim if unbatched env
            o, a, r, terminal, truncated = [x.unsqueeze(1) for x in (o, a, r, terminal, truncated)]
        # prepare data
        env_data = {'o': o, 'a': a, 'r': r, 'terminal': terminal, 'truncated': truncated, 'mask': torch.zeros_like(r)}
        env_data = prepare_data(env_data, n_categories=self.one_hot_keys)

        for k, v in env_data.items():
            if torch.isnan(v).any():
                raise RuntimeError(f'Invalid NAN input for key {k} in step {env.current_step}: {v}')
            if torch.isinf(v).any():
                raise RuntimeError(f'Invalid inf input for key {k} in step {env.current_step}: {v}')

        # digest new groundtruth data in level 0 model
        _, self._current_env_state = self.model.forward_static(trajectory=env_data,
                                                               start_state=self._current_env_state,
                                                               level=self.agent.level, n_steps=1, n_warmup=1,
                                                               sample_state=self.sample_world_model,
                                                               sample_output=False, reconstruct=False,
                                                               use_ema_modules=self.agent.use_slow_world_model)

        if torch.isnan(self._current_env_state[0]).any():
            raise RuntimeError(f'Invalid NAN state in step {env.current_step}: {self._current_env_state[0]}')
        if torch.isinf(self._current_env_state[0]).any():
            raise RuntimeError(f'Invalid inf state in step {env.current_step}: {self._current_env_state[0]}')

        agent_o = self.agent.o_from_state(self._current_env_state)
        a_dist, a, = self.agent(agent_o, sample=self.stochastic, expl_noise=self.exploration_noise)

        if torch.isnan(a).any():
            raise RuntimeError(f'Invalid NAN action: {a}')
        if torch.isinf(a).any():
            raise RuntimeError(f'Invalid inf action: {a}')

        # transform one-hot actions to int index format
        if 'a' in self.one_hot_keys:
            a = torch.argmax(a, dim=-1)

        if isinstance(env, CacheLastStepEnv):  # remove batch dimension if it's not a vector env
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
                 exploration_noise: float = 0.0,
                 stochastic: bool = False):
        super().__init__()
        chunk_length_offset = 0

        self.model = model
        self.exploration_noise = exploration_noise
        self.stochastic = stochastic
        self.grounded_env_states = [None for _ in range(model.levels)]
        self.env_data_below_cache = [self._empty_cache() for _ in range(model.levels)]
        self.act_cache = [[] for _ in range(model.levels)]
        self.action_queue = []
        self.chunk_lengths = tuple([x + chunk_length_offset for x in model.strides])
        self.next_state_update = list(self.chunk_lengths)
        self.level_active = [None for _ in range(model.levels)]
        self.flight_record = [{} for _ in range(model.levels)]
        self.planning_record = [{} for _ in range(model.levels)]
        self.action_history = [[] for _ in range(model.levels)]
        self.chunk_history = []

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
        self.sample_world_model = True

        if model.r_max_agents[0][0].discrete_actions:
            self.one_hot_keys = {'a': model.r_max_agents[0][0].d_a}
        else:
            self.one_hot_keys = {}

        self.reset()

    @staticmethod
    def _empty_cache():
        return {'o': [], 'a': [], 'r': [], 'terminal': [], 'time_step': []}

    def reset(self):
        self.grounded_env_states = [None for _ in self.model.rssm_modules]
        self.env_data_below_cache = [self._empty_cache() for _ in self.model.rssm_modules]
        self.act_cache = [[] for _ in self.model.rssm_modules]
        self.next_state_update = list(self.chunk_lengths)
        self.level_active = [False for _ in self.model.rssm_modules]
        self.flight_record = [{'z': [], 'o': [], 'a': [], 'r': [], 'terminal': [], 'time_step': [], 'z_post': []}
                              for _ in self.model.rssm_modules]
        self.planning_record = [{'o': [], 'a': [], 'r': [], 'terminal': [], 'time_step': []}
                                for _ in self.model.rssm_modules]
        self.action_queue = []
        self.chunk_history = []
        self.action_history = [[] for _ in self.model.rssm_modules]

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
        env_data = prepare_data(env_data, remove_keys=['truncated', 'mask'], n_categories=self.one_hot_keys)
        env_data = {k: list(v.unbind(0)) for k, v in env_data.items()}
        return env_data

    @torch.no_grad()
    def _check_state_update(self,
                            env: Union[CacheLastStepEnv, CacheLastStepVecEnv]):
        for i_lvl in range(self.model.levels):
            self.next_state_update[i_lvl] -= 1
            if self.next_state_update[i_lvl] > 0:
                continue  # no need to update level yet

            # if an env has finished, don't care about upfiltering and state updating, so ignore mask
            respect_mask = False

            # TODO: we can now use the real ground truth data to update states on all levels
            # NOTE: I think this is already done below as prediction data is exchanged with filtered up data
            n_steps = self.model.strides[i_lvl]
            data_filtered = self.model.filter_up(o=self.env_data_below_cache[i_lvl]['o'],
                                                 a=self.env_data_below_cache[i_lvl]['a'],
                                                 r=self.env_data_below_cache[i_lvl]['r'],
                                                 terminal=self.env_data_below_cache[i_lvl]['terminal'],
                                                 level=i_lvl, n_steps=n_steps,
                                                 respect_mask=respect_mask,
                                                 sample_action_autoencoder=False)

            # remove data used for this update step
            # self.env_data_below_cache[i_lvl]['o'] = self.env_data_below_cache[i_lvl]['o'][n_steps:]
            # self.env_data_below_cache[i_lvl]['a'] = self.env_data_below_cache[i_lvl]['a'][n_steps:]
            # self.env_data_below_cache[i_lvl]['r'] = self.env_data_below_cache[i_lvl]['r'][n_steps:]
            # self.env_data_below_cache[i_lvl]['terminal'] = self.env_data_below_cache[i_lvl]['terminal'][n_steps:]

            # take real actions from this level instead of the ones from action autoencoder if they are available
            # if len(self.act_cache[i_lvl]) > 0:
            #    data_filtered['a'] = self.act_cache[i_lvl].pop(0).unsqueeze(0)  # take oldest action from cache
            # else:
            #    self.action_history[i_lvl].append(data_filtered['a'][0])  # add filtered up action + remove time dim
            if len(self.act_cache[i_lvl]) > 0:
                self.act_cache[i_lvl].pop(0)  # remove oldest action from action cache

            # memorize the latest inputs the model has seen as they are needed for the agent during planning
            state = self.grounded_env_states[i_lvl]
            mem, new_state = self.model.forward_static(data_filtered, start_state=state, level=i_lvl, n_steps=1,
                                                       n_warmup=1, sample_state=self.sample_world_model,
                                                       reconstruct=True, sample_output=False,
                                                       use_ema_modules=self._use_ema_modules)
            self.grounded_env_states[i_lvl] = new_state
            self.level_active[i_lvl] = True

            # bookkeeping
            timestep = torch.tensor(env.current_step, dtype=torch.float32, device=mem['r'][0].device)
            if isinstance(env, CacheLastStepVecEnv):
                timestep = torch.tile(timestep[None, ...], [env.unwrapped.num_envs, 1])
            for k, v in mem.items():
                mem[k] = v[0]  # remove redundant list wrapper since we always only do one step
            for k, v in data_filtered.items():
                mem[k] = v[0]  # replace reconstructed quantities with more correct ground truth data where possible
            append_memory(self.flight_record[i_lvl], **mem, timestep=timestep)

            # store updated state in cache for upper level
            if i_lvl < self.model.i_top:
                o_key = self.model.r_max_agents[i_lvl + 1][0].observation_type
                self.env_data_below_cache[i_lvl + 1]['o'].append(mem[o_key])
                # note: we replaced the model outputs in mem with filtered up ground truth data further above
                self.env_data_below_cache[i_lvl + 1]['a'].append(mem['a'])
                self.env_data_below_cache[i_lvl + 1]['r'].append(mem['r'])
                self.env_data_below_cache[i_lvl + 1]['terminal'].append(mem['terminal'])
                # self.env_data_below_cache[i_lvl + 1]['time_step'].append(timestep)

            # remove data used for this update step
            self.env_data_below_cache[i_lvl]['o'] = self.env_data_below_cache[i_lvl]['o'][n_steps:]
            self.env_data_below_cache[i_lvl]['a'] = self.env_data_below_cache[i_lvl]['a'][n_steps:]
            self.env_data_below_cache[i_lvl]['r'] = self.env_data_below_cache[i_lvl]['r'][n_steps:]
            self.env_data_below_cache[i_lvl]['terminal'] = self.env_data_below_cache[i_lvl]['terminal'][n_steps:]

            # reset counter
            self.next_state_update[i_lvl] = self.model.strides[i_lvl]

    @torch.no_grad()
    def _replan(self):
        # find highest planning level
        i_highest = sum(self.level_active) - 1

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
        state = self.grounded_env_states[i_highest]
        agent = self.model.r_max_agents[i_highest][0]
        simulation = agent.act_in_sim(env_start_state=state, sim_env=self.model, n_steps=1,
                                      sample_actions=self.stochastic, sample_states=self.sample_world_model,
                                      expl_noise=self.exploration_noise, reconstruct=True)
        a_agent = simulation['agent']['a'][1:]  # remove frst action from record as it's padding
        model_sim = {k: v[1:] for k, v in simulation['model'].items()}  # remove first time step as it's the input
        extend_memory(self.planning_record[i_highest], model_sim)

        assert len(a_agent) == 1

        self.act_cache[i_highest] += a_agent
        self.action_history[i_highest] += a_agent

        # act with goal seeking agents
        if i_highest > 0:
            goals_from_above = simulation['model']['o'][1:]  # remove first goal as it's for current state
            chunk_id = random.randint(0, 100000)
            for i_lvl in reversed(range(0, i_highest)):
                state = self.grounded_env_states[i_lvl]
                agent = self.model.goal_seeking_agents[i_lvl][0]
                n_steps = self.chunk_lengths[i_lvl + 1]
                new_goals = []
                for goal in goals_from_above:
                    simulation = agent.act_in_sim(env_start_state=state, sim_env=self.model, n_steps=n_steps, goal=goal,
                                                  sample_actions=self.stochastic, expl_noise=0.0,
                                                  sample_states=self.sample_world_model, reconstruct=True)
                    # simulation['agent']['a'] = [torch.zeros_like(x) for x in simulation['agent']['a']]
                    state = simulation['model_state']
                    a_agent = simulation['agent']['a'][1:]  # remove frst action from record as it's padding
                    self.act_cache[i_lvl] += a_agent
                    self.action_history[i_lvl] += a_agent
                    self.chunk_history.extend([chunk_id for _ in a_agent])
                    # store planning results to record for later
                    model_sim = {k: v[1:] for k, v in simulation['model'].items()}
                    extend_memory(self.planning_record[i_lvl], model_sim)
                    if i_lvl > 0:
                        new_goals += simulation['model']['o'][1:]
                goals_from_above = new_goals
        else:
            self.chunk_history.append(-1)

        self.action_queue += self.act_cache[0]
        """
        # act with action autoencoder
        if i_highest > 0:
            agent_a = torch.stack(simulation['agent']['a'])
            a = self.model.upwards_filters[1]['a'].decode_det(agent_a)
            #a = torch.permute(a, (0, 2, 1, 3))
            #a = a.reshape(a.shape[0] * a.shape[1], a.shape[2], a.shape[3])
            self.action_queue += list(a.unbind(0))
        else:
            self.action_queue += self.act_cache[0]
        """

        """
        # act with identity high level actions
        if i_highest > 0:
            a_lower = self.model.actions_down(simulation['agent']['a'][0], i_highest)
            #lower_level_steps = self.model.strides[i_highest]
            #a_t = simulation['agent']['a'][0]
            #a_lower = a_t.reshape(a_t.shape[0], lower_level_steps, -1)
            #a_lower = torch.permute(a_lower, (1, 0, 2))
            self.act_cache[0] += list(a_lower.unbind(0))

        self.action_queue += self.act_cache[0]
        """

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
        # if env.current_step > 0 and self.grounded_env_states[0] is None:
        #    raise RuntimeError(f'Env is already in step {env.current_step} but there\'s no recorded previous state')

        self.model.eval()
        for agent in self.model.r_max_agents + self.model.goal_seeking_agents:
            if agent is not None:
                agent[0].eval()

        # if isinstance(env, (CacheLastStepVecEnv)):
        #    d_batch = env.last_o.shape[0]
        # else:
        #    d_batch = 1
        # if env.current_step == 0:
        #    self.reset(d_batch)

        # first step: store current env ground truth data to cache
        self.env_data_below_cache[0] = self._prep_step(env)
        self._check_state_update(env)  # update individual level's states if necessary

        # check if re-planning is needed
        if len(self.action_queue) == 0:
            self._replan()

        action = self.action_queue.pop(0)

        # check if we have one-hot actions, in that case we need to transform actions back to int index format
        if 'a' in self.one_hot_keys:
            action = torch.argmax(action, dim=-1)

        if isinstance(env, CacheLastStepEnv):  # remove batch dimension if it's not a vector env
            action = action[0]
        return action.detach().cpu().numpy()


def extract_plan(policy: HierarchicalLatentAgentPolicy):
    model = policy.model
    planning_record = policy.planning_record
    flight_record = policy.flight_record

    o = [[] for _ in policy.model.rssm_modules]
    r = [[] for _ in policy.model.rssm_modules]
    terminal = [[] for _ in policy.model.rssm_modules]

    # lowest level has reconstructed ground truth data, take this first
    env_o = torch.stack(flight_record[0]['o'])
    env_r = torch.stack(flight_record[0]['r'])
    env_terminal = torch.stack(flight_record[0]['terminal'])

    for i_lvl, record in enumerate(planning_record):
        o[i_lvl] = torch.stack(record['o'])
        # decode goal to observation if we're at an higher level
        if i_lvl > 0:
            i_current_lvl = i_lvl
            while i_current_lvl > 0:
                o[i_lvl] = model.rssm_modules[i_current_lvl - 1].decode(o[i_lvl], sample=False,
                                                                        reconstruct_observation=True)['o']
                i_current_lvl -= 1
        r[i_lvl] = torch.stack(record['r'])
        terminal[i_lvl] = torch.stack(record['terminal'])

    return [{'o': o_, 'r': r_, 'terminal': terminal_} for o_, r_, terminal_ in zip(o, r, terminal)]


def extract_train_data(policy: HierarchicalLatentAgentPolicy,
                       level: int):
    flight_record = policy.flight_record

    if not flight_record[level]:  # not enough lower level steps to record any steps on this level
        return []

    trajectory = {k: flight_record[level - 1][k] for k in ('o', 'a', 'r', 'terminal')}
    trajectory = {k: torch.stack(v) for k, v in trajectory.items()}
    abstr_trajectory = {k: flight_record[level][k] for k in ('o', 'a', 'r', 'terminal')}

    # if trajectory ended within chunk, we need to manually create last step of abstract trajectory
    if trajectory['a'].shape[0] % policy.model.strides[level] != 0:
        states_below = torch.stack(flight_record[level - 1]['s_embedding'])
        filtered = policy.model.filter_up(o=states_below, r=trajectory['r'], terminal=trajectory['terminal'],
                                          level=level)

        last_abstr_o = filtered['o'][-1]
        last_abstr_a = policy.action_history[level][-1]
        last_abstr_r = filtered['r'][-1]
        last_abstr_term = filtered['terminal'][-1]

        abstr_trajectory['o'].append(last_abstr_o)
        abstr_trajectory['a'].append(last_abstr_a)
        abstr_trajectory['r'].append(last_abstr_r)
        abstr_trajectory['terminal'].append(last_abstr_term)

    abstr_trajectory = {f'{k}_abstract': torch.stack(v) for k, v in abstr_trajectory.items()}

    # if trajectory['o'].shape[0] % policy.model.strides[level] == 0:
    # the last lower level step might have been added after the last action, so an abstract action can be made
    # after the env ended and the abstract goal from the last abstract action will never be pursued by the gsa
    # which never yields any trajectory data, making the last abstract action useless
    #    actions = actions[:-1]
    #    abstr_trajectory = {k: v[:-1] for k, v in abstr_trajectory.items()}

    abstract_trajectory = {**trajectory, **abstr_trajectory}

    # move to host memory and remove redundant batch dimension
    assert abstract_trajectory['o'].shape[1] == 1, 'Batch dimension of 1 expected!'
    abstract_trajectory = {k: v.detach().cpu().numpy()[:, 0] for k, v in abstract_trajectory.items()}

    return [abstract_trajectory]
