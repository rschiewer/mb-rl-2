from __future__ import annotations

import copy
from itertools import chain
import random
import sys
from collections import OrderedDict

import torch
import torch.distributions as torchd
from torch.distributions import kl_divergence
from torch.nn import ModuleList, ModuleDict
import matplotlib.pyplot as plt

from mdm.models.building_blocks import *
from mdm.models.building_blocks import RSSMCell
from mdm.models.dynamics_model import DynamicsModel
from mdm.policies.actor_critic_agent import ActorCriticAgent
from mdm.utils.torch_tools import FuzzyDeviceMixin, compile_if_not_debug, detach_dist, update_ema_modules, stack_dists
from mdm.utils.utils import unsqueeze_right, rssm_states_seq_to_batch


class HierarchicalRSSM(DynamicsModel, FuzzyDeviceMixin):
    _filter_names = ('o', 'a', 'r', 'terminal', 'mask')

    def __init__(self,
                 rssm_modules: Sequence[RSSMCell],
                 links: Sequence[str],  # links associate output from one lvl below with inputs on this lvl
                 upwards_filters: Sequence[Dict[str, UpwardsFilter]],
                 warmup_steps: Sequence[Union[int, str]],
                 kl_betas: Sequence[float],
                 kl_reg_betas: Sequence[float],
                 r_max_agents: Sequence[ActorCriticAgent] = (None,),
                 goal_seeking_agents: Sequence[ActorCriticAgent] = (None,),
                 ema_regularization: float = 0.0,
                 ema_coeff: float = 0.99,
                 ema_update_interval: int = sys.maxsize):
        super(HierarchicalRSSM, self).__init__()

        assert len(links) == len(rssm_modules) - 1
        assert len(upwards_filters) == len(rssm_modules) - 1
        for filters in upwards_filters:
            window_sizes = set([f.window_size for f in filters.values()])
            assert len(window_sizes) == 1

        lvl_k_link = 'o'  # only here to make the loop in eval_step() method work
        lvl_0_filters = {k: IdentityUpwardsFilter() for k in self._filter_names}
        for level in upwards_filters:
            assert level.keys() <= set(self._filter_names), (f'Allowed filter names: {self._filter_names}, found'
                                                             f'filter names: {level.keys()}')
            window_size = level['o'].window_size
            mask_and_action_filters = {'mask': MinUpwardsFilter(window_size), 'a': ConstUpwardsFilter(window_size, 0)}
            level.update(mask_and_action_filters)
        upwards_filters = [ModuleDict(lvl_0_filters)] + [ModuleDict(x) for x in upwards_filters]

        # generate EMA shadow copies of the RSSMs
        self._ema_rssm_modules = ModuleList([copy.deepcopy(m) for m in rssm_modules])
        for i_mod, ema_mod in enumerate(self._ema_rssm_modules):  # disable gradient computation in ema shadow copies
            for i_param, param in enumerate(ema_mod.parameters()):
                param.detach_()

        # rssm_modules = [torch.compile(m) for m in rssm_modules]
        self.rssm_modules = ModuleList(list(rssm_modules))
        self.links = (*links, lvl_k_link)
        self.upwards_filters = ModuleList(upwards_filters)
        self.warmup_steps = tuple(warmup_steps)
        self.kl_betas = tuple(kl_betas)
        self.kl_reg_betas = tuple(kl_reg_betas)
        self.r_max_agents = tuple(r_max_agents)  # no module list to shield the agents from any pytorch functions
        self.goal_seeking_agents = tuple(goal_seeking_agents)
        self.ema_regularization = ema_regularization
        self.ema_coeff = ema_coeff
        self.ema_update_interval = ema_update_interval
        self.dbg_timestep = 0

    @property
    def levels(self) -> int:
        return len(self.rssm_modules)

    @property
    def strides(self) -> List[int]:
        return [filters['o'].window_size for filters in self.upwards_filters]

    @property
    def i_top(self) -> int:
        return self.levels - 1

    def reset_debug_counter(self):
        self.dbg_timestep = 0

    """
    def observe(self,
                o: torch.Tensor,
                a: torch.Tensor,
                r: torch.Tensor,
                terminal: torch.Tensor,
                level: int = 0,
                memory: Dict[str, torch.Tensor] | None = None,
                start_state: Dict[str, torch.Tensor] | None = None,
                sample_state: bool = True,
                sample_output: bool = True,
                reconstruct: bool = True,
                use_ema_modules: bool = False):
        assert o.shape[0] == a.shape[0] == r.shape[0] == terminal.shape[0], 'Only complete trajectories are supported'

        n_steps, d_batch = a.shape[:2]
        device = self.device
        mdl = self._ema_rssm_modules[level] if use_ema_modules else self.rssm_modules[level]
        memory = {} if memory is None else memory
        state = mdl.init_state(d_batch, device) if start_state is None else start_state

        # absorb given ground truth data to warmup the model
        for t in range(n_steps):
            o_t, a_t, r_t, term_t = o[t], a[t], r[t], terminal[t]
            pred, state = mdl(a=a_t, o=o_t, r=r_t, terminal=term_t, last_state=state,
                              use_posterior=True, sample_state=sample_state, sample_output=sample_output,
                              reconstruct=reconstruct)

            for k, v in {**pred, **state}.items():
                data = memory.get(k, [])
                data.append(v)
                memory[k] = data

        return memory, state

    def imagine(self,
                a: torch.Tensor,
                start_state: Dict[str, torch.Tensor],
                level: int = 0,
                memory: Dict[str, torch.Tensor] | None = None,
                sample_state: bool = True,
                sample_output: bool = True,
                reconstruct: bool = True,
                use_ema_modules: bool = False):
        n_steps, d_batch = a.shape[:2]
        mdl = self._ema_rssm_modules[level] if use_ema_modules else self.rssm_modules[level]
        memory = {} if memory is None else memory
        state = start_state

        # imagine a trajectory RSSMCell's prior and given actions
        for t in range(n_steps):
            a_t = a[t]
            pred, state = mdl(a=a_t, last_state=state, use_posterior=False, sample_state=sample_state,
                              sample_output=sample_output, reconstruct=reconstruct)

            for k, v in {**pred, **state}.items():
                data = memory.get(k, [])
                data.append(v)
                memory[k] = data

        return memory, state

    
    def observe_2(self,
                  o: torch.Tensor,
                  a: torch.Tensor,
                  r: torch.Tensor,
                  terminal: torch.Tensor,
                  level: int = 0,
                  memory: Dict[str, torch.Tensor] | None = None,
                  memory_other: Dict[str, torch.Tensor] | None = None,
                  start_state: Dict[str, torch.Tensor] | None = None,
                  sample_state: bool = True,
                  sample_output: bool = True,
                  reconstruct: bool = True,
                  use_ema_modules: bool = False):
        assert o.shape[0] == a.shape[0] == r.shape[0] == terminal.shape[0], 'Only complete trajectories are supported'

        n_steps, d_batch = a.shape[:2]
        mdl = self._ema_rssm_modules[level] if use_ema_modules else self.rssm_modules[level]
        mdl_other = self.rssm_modules[level] if use_ema_modules else self._ema_rssm_modules[level]
        memory = {} if memory is None else memory
        memory_other = {} if memory_other is None else memory_other

        # absorb given ground truth data to warmup the model
        state = start_state
        for t in range(n_steps):
            o_t, a_t, r_t, term_t = o[t], a[t], r[t], terminal[t]
            pred, next_state = mdl(a=a_t, o=o_t, r=r_t, terminal=term_t, last_state=state,
                                   use_posterior=True, sample_state=sample_state, sample_output=sample_output,
                                   reconstruct=reconstruct)
            pred_other, next_state_other = mdl_other(a=a_t, o=o_t, r=r_t, terminal=term_t,
                                                     last_state=state, use_posterior=True, sample_state=sample_state,
                                                     sample_output=sample_output, reconstruct=reconstruct)

            for k, v in {**pred, **state}.items():
                data = memory.get(k, [])
                data.append(v)
                memory[k] = data
            for k, v in {**pred_other, **next_state_other}.items():
                data = memory_other.get(k, [])
                data.append(v)
                memory_other[k] = data

            state = next_state

        return memory, memory_other, state

    def imagine_2(self,
                  a: torch.Tensor,
                  start_state: Dict[str, torch.Tensor],
                  level: int = 0,
                  memory: Dict[str, torch.Tensor] | None = None,
                  memory_other: Dict[str, torch.Tensor] | None = None,
                  sample_state: bool = True,
                  sample_output: bool = True,
                  reconstruct: bool = True,
                  use_ema_modules: bool = False):
        n_steps, d_batch = a.shape[:2]
        mdl = self._ema_rssm_modules[level] if use_ema_modules else self.rssm_modules[level]
        mdl_other = self.rssm_modules[level] if use_ema_modules else self._ema_rssm_modules[level]
        memory = {} if memory is None else memory
        memory_other = {} if memory_other is None else memory_other

        # imagine a trajectory RSSMCell's prior and given actions
        state = start_state
        for t in range(n_steps):
            a_t = a[t]
            pred, next_state = mdl(a=a_t, last_state=state, use_posterior=False, sample_state=sample_state,
                                   sample_output=sample_output, reconstruct=reconstruct)
            pred_other, next_state_other = mdl_other(a=a_t, last_state=state, use_posterior=False,
                                                     sample_state=sample_state, sample_output=sample_output,
                                                     reconstruct=reconstruct)

            for k, v in {**pred, **state}.items():
                data = memory.get(k, [])
                data.append(v)
                memory[k] = data
            for k, v in {**pred_other, **next_state_other}.items():
                data = memory_other.get(k, [])
                data.append(v)
                memory_other[k] = data

            state = next_state

        return memory, memory_other, state
        
    def simulate(self,
                 n_steps: int,
                 start_state: Dict[str, torch.Tensor],
                 agent_goal: torch.Tensor | torchd.Distribution | None = None,
                 level: int = 0,
                 memory: Dict[str, torch.Tensor] | None = None,
                 agent_memory: Dict[str, torch.Tensor] | None = None,
                 sample_state: bool = True,
                 sample_output: bool = True,
                 reconstruct: bool = True,
                 use_ema_modules: bool = False,
                 agent_training: bool = False):
        mdl = self._ema_rssm_modules[level] if use_ema_modules else self.rssm_modules[level]
        mdl_other = self.rssm_modules[level] if use_ema_modules else self._ema_rssm_modules[level]
        agent_memory = {} if agent_memory is None else agent_memory
        memory = {} if memory is None else memory
        state = start_state

        if n_steps == 0:
            return memory, start_state, {}

        if agent_goal is None:
            agent = self.r_max_agents[level][0]
        else:
            agent = self.goal_seeking_agents[level][0]

        if agent_training:
            def agent_decide(_step):
                agent_o = agent.preproc_o(_step, agent_goal)
                a_dist, a, v = agent(agent_o)
                ema_a_dist, ema_a, ema_v = agent(agent_o, use_ema_modules=True)
                return {'a_dist': a_dist, 'a': a, 'v': v, 'ema_a_dist': ema_a_dist, 'ema_a': ema_a, 'ema_v': ema_v}
        else:
            def agent_decide(_step):
                agent_o = agent.preproc_o(_step, agent_goal)
                a_dist, a, v = agent(agent_o)
                a_dist, a, v = detach_dist(a_dist), a.detach(), v.detach()
                ema_a_dist, ema_a, ema_v = None, None, None
                return {'a_dist': a_dist, 'a': a, 'v': v, 'ema_a_dist': ema_a_dist, 'ema_a': ema_a, 'ema_v': ema_v}

        for t in range(n_steps):
            pred_agent = agent_decide(state)
            current, next_state = mdl(a=pred_agent['a'], last_state=state, use_posterior=False,
                                      sample_state=sample_state, sample_output=sample_output, reconstruct=reconstruct)
            current.update(next_state)
            if agent_training:
                # this is a hack, other model gets for every step last_state of normal model, so it can't deviate a lot
                current_other, next_state_other = mdl_other(a=pred_agent['a'], last_state=state, use_posterior=False,
                                                            sample_state=sample_state, sample_output=sample_output,
                                                            reconstruct=reconstruct)
                current_other.update(next_state_other)
                novelty = kl_divergence(detach_dist(current['z_dist']), current_other['z_dist']).mean(dim=-1)
            else:
                novelty = torch.tensor(0, device=self.device, dtype=torch.float32)

            # add missing quantities to agent memory
            pred_agent['model_novelty'] = novelty
            pred_agent['r'] = agent.build_step_reward(current, agent_goal)
            pred_agent['terminal'] = current['terminal']

            for k, v in current.items():
                data = memory.get(k, [])
                data.append(v)
                memory[k] = data

            for k, v in pred_agent.items():
                data = agent_memory.get(k, [])
                data.append(v)
                agent_memory[k] = data

            state = next_state

        return memory, state, agent_memory
    """

    def forward_dynamic(self,
                        start_state: Dict[str, torch.Tensor],
                        start_state_below: Dict[str, torch.Tensor],
                        level: int,
                        n_steps: int,
                        n_warmup: int = 0,
                        memory: Dict[str, torch.Tensor] | None = None,
                        memory_other: Dict[str, torch.Tensor] | None = None,
                        memory_targets: Dict[str, torch.Tensor] | None = None,
                        sample_state: bool = True,
                        sample_output: bool = True,
                        reconstruct: bool = True,
                        use_ema_modules: bool = False,
                        start_time_step: torch.Tensor | None = None):
        assert level > 0, 'Not intended for level 0 model, use forward_static()'

        mdl = self._ema_rssm_modules[level] if use_ema_modules else self.rssm_modules[level]
        mdl_other = self.rssm_modules[level] if use_ema_modules else self._ema_rssm_modules[level]
        memory = {} if memory is None else memory
        memory_other = {} if memory_other is None else memory_other
        memory_targets = {} if memory_targets is None else memory_targets

        if n_warmup < 0:
            n_warmup = n_steps

        agent = self.r_max_agents[level][0]
        assert agent.observation_key == 'z', 'only latent agent supported'
        lvl_below = level - 1
        chunk_size = self.strides[level]

        state = start_state
        state_below = start_state_below
        for t in range(n_steps):
            _, a_t, _ = agent(state[agent.observation_key], sample=True)
            a_t = a_t.detach()  # prevent gradients from flowing through agent back into world model
            pred, next_state = mdl(a=a_t, last_state=state, use_posterior=False, sample_state=sample_state,
                                   sample_output=sample_output, reconstruct=reconstruct)
            # get simulated ground truth for this step from level below
            simulation = self.goal_seeking_agents[lvl_below][0].act_in_sim(env_state=state_below, sim_env=self,
                                                                           n_steps=chunk_size, goal=pred['o'],
                                                                           sample_actions=True, sample_model=True,
                                                                           reconstruct=True)
            simulated_ground_truth = self.filter_up(o=simulation['model']['z'], r=simulation['model']['r'],
                                                    terminal=simulation['model']['terminal'], level=level,
                                                    n_steps=chunk_size)
            simulated_ground_truth = {k: v.squeeze(0) for k, v in simulated_ground_truth.items()}  # remove time dim

            # augment reward with how reachable the goal was for lower level
            # simulated_ground_truth['r'] += 0.01 * simulation['agent']['r'][-1]

            if t < n_warmup:  # if still in warmup, re-do last step with simulated ground truth and use posterior
                pred, next_state = mdl(a=a_t, o=simulated_ground_truth['o'], last_state=state,
                                       use_posterior=True, sample_state=sample_state, sample_output=sample_output,
                                       reconstruct=reconstruct)
                pred_other, next_state_other = mdl_other(a=a_t, o=simulated_ground_truth['o'],
                                                         last_state=state, use_posterior=True,
                                                         sample_state=sample_state,
                                                         sample_output=sample_output, reconstruct=reconstruct)
            else:  # do ema model prediction in any case for model regularization
                pred_other, next_state_other = mdl_other(a=a_t, last_state=state, use_posterior=False,
                                                         sample_state=sample_state,
                                                         sample_output=sample_output, reconstruct=True)

            # update memories
            for k, v in {**pred, **next_state}.items():
                data = memory.get(k, [])
                data.append(v)
                memory[k] = data
            for k, v in {**pred_other, **next_state_other}.items():
                data = memory_other.get(k, [])
                data.append(v)
                memory_other[k] = data
            for k, v in simulated_ground_truth.items():
                data = memory_targets.get(k, [])
                data.append(v)
                memory_targets[k] = data

            #if start_time_step is not None:
            #    data = memory.get('time_step', [])
            #    data.append(start_time_step + (t + 1) * self.strides[level])
            #    memory['time_step'] = data

            # important: update model states
            state = next_state
            state_below = simulation['model_state']

        memory_targets = {k: torch.stack(v) for k, v in memory_targets.items()}  # to fit loss calculation scheme

        return memory, memory_other, memory_targets, state

    def forward_static(self,
                       trajectory: Dict[str, torch.Tensor],
                       level: int = 0,
                       n_steps: int = -1,
                       n_warmup: int = -1,
                       start_state: Dict[str, torch.Tensor] = None,
                       memory: Dict[str, torch.Tensor] | None = None,
                       memory_other: Dict[str, torch.Tensor] | None = None,
                       sample_state: bool = True,
                       sample_output: bool = True,
                       reconstruct: bool = True,
                       use_ema_modules: bool = False):
        mdl = self._ema_rssm_modules[level] if use_ema_modules else self.rssm_modules[level]
        mdl_other = self.rssm_modules[level] if use_ema_modules else self._ema_rssm_modules[level]
        memory = {} if memory is None else memory
        memory_other = {} if memory_other is None else memory_other
        o, a = trajectory['o'], trajectory['a']

        if n_steps < 0:
            n_steps = a.shape[0]
        else:
            assert n_steps <= a.shape[0], f'Not enough actions available ({a.shape[0]}) to go {n_steps} steps'

        if n_warmup < 0:
            n_warmup = n_steps

        state = start_state
        for t in range(n_steps):
            if t < n_warmup:
                o_t = o[t]
                use_posterior = True
            else:
                o_t = None
                use_posterior = False
            a_t = a[t]

            pred, next_state = mdl(a=a_t, o=o_t, last_state=state, use_posterior=use_posterior,
                                   sample_state=sample_state, sample_output=sample_output, reconstruct=reconstruct)
            pred_other, next_state_other = mdl_other(a=a_t, o=o_t, last_state=state, use_posterior=use_posterior,
                                                     sample_state=sample_state, sample_output=sample_output,
                                                     reconstruct=reconstruct)

            for k, v in {**pred, **next_state}.items():
                data = memory.get(k, [])
                data.append(v)
                memory[k] = data
            for k, v in {**pred_other, **next_state_other}.items():
                data = memory_other.get(k, [])
                data.append(v)
                memory_other[k] = data

            # if 'time_step' in next_state:
            #    ts = memory.get('time_step', [])
            #    ts.append(next_state['time_step'])
            #    memory['time_step'] = ts
            #if 'time_step' in trajectory:
            #    ts = memory.get('time_step', [])
            #    ts.append(trajectory['time_step'][t])
            #    memory['time_step'] = ts

            state = next_state

        return memory, memory_other, state

    def filter_up(self,
                  o: List[torch.Tensor],
                  r: List[torch.Tensor],
                  terminal: List[torch.Tensor],
                  level: int,
                  n_steps: int = -1,
                  time_step: List[torch.Tensor] | None = None,
                  **kwargs):
        if n_steps == -1:
            n_steps = len(o)
        filters = self.upwards_filters[level]
        trajectory_below_processed = {'o': torch.stack(o[:n_steps]),
                                      'r': torch.stack(r[:n_steps]),
                                      'terminal': torch.stack(terminal[:n_steps])}

        simulated_ground_truth = {k: filters[k](v).detach() for k, v in trajectory_below_processed.items()}

        #if time_step is not None:
        #    ts = torch.stack(time_step[:n_steps])
        #    simulated_ground_truth['time_step'] = PickOneUpwardsFilter(self.strides[level], -1)(ts)

        return simulated_ground_truth

    def ground_level(self,
                     trajectory_below: Dict[str, List[torch.Tensor]],
                     level: int,
                     start_state: Dict[str, torch.Tensor] = None,
                     memory: Dict[str, torch.Tensor] | None = None,
                     memory_other: Dict[str, torch.Tensor] | None = None,
                     memory_targets: Dict[str, torch.Tensor] | None = None,
                     sample_state: bool = True,
                     sample_output: bool = True,
                     reconstruct: bool = True,
                     use_ema_modules: bool = False):
        assert level > 0, 'Level 0 grounding is done automatically in forward_static() method'

        memory_targets = {} if memory_targets is None else memory_targets
        n_steps = self.strides[level]
        assert n_steps <= len(trajectory_below['z']), f'Not enough below level time steps to ground level {level}'
        simulated_ground_truth = self.filter_up(o=trajectory_below['z'], r=trajectory_below['r'],
                                                terminal=trajectory_below['terminal'], level=level,
                                                #time_step=trajectory_below['time_step'],
                                                n_steps=n_steps)
        d_batch = trajectory_below['o'][0].shape[0]
        device = trajectory_below['o'][0].device
        simulated_ground_truth['a'] = self.rssm_modules[level].zero_a(d_batch, device).unsqueeze(0)  # one dummy action

        for k, v in simulated_ground_truth.items():
            assert v.shape[0] == 1, f'k is too long: {v.shape[0]}'

        mem, mem_other, state = self.forward_static(simulated_ground_truth, n_steps=1, n_warmup=1, level=level,
                                                    memory=memory, memory_other=memory_other, start_state=start_state,
                                                    sample_state=sample_state, sample_output=sample_output,
                                                    reconstruct=reconstruct, use_ema_modules=use_ema_modules)

        simulated_ground_truth = {k: v.squeeze(0) for k, v in simulated_ground_truth.items()}  # remove time dim
        for k, v in simulated_ground_truth.items():
            data = memory_targets.get(k, [])
            data.append(v)
            memory_targets[k] = data

        # reconstruct state of model below at time step that corresponds to one step by current level
        state_below = {k: trajectory_below[k][n_steps - 1] for k in state.keys()}  # TODO: check this
        return mem, mem_other, memory_targets, state, state_below

    def forward_all_levels(self,
                           ground_truth_trajectory: Dict[str, torch.Tensor],
                           warmup_steps: List[int],
                           model_steps: List[int],
                           model_state: List[Dict[str, torch.Tensor]] | None = None,
                           sample_state: bool = True,
                           sample_output: bool = True,
                           reconstruct: bool = True,
                           dynamic: bool = True):
        memory = [None for _ in range(self.levels)]
        memory_ema = [None for _ in range(self.levels)]
        model_state = [None for _ in range(self.levels)] if model_state is None else model_state
        targets = [ground_truth_trajectory] + [None for _ in range(self.levels - 1)]

        # lvl 0 is special and gets static ground truth trajectories
        memory[0], memory_ema[0], model_state[0] = self.forward_static(ground_truth_trajectory,
                                                                       n_steps=model_steps[0],
                                                                       n_warmup=warmup_steps[0], level=0,
                                                                       start_state=model_state[0],
                                                                       sample_state=sample_state,
                                                                       sample_output=sample_output,
                                                                       reconstruct=reconstruct)
        # all other levels are only grounded with the first k steps from below and can then do what they want
        for l in range(1, self.levels):
            n_warmup = warmup_steps[l] - 1
            n_steps = model_steps[l] - 1
            # if dynamic:
            grounded = self.ground_level(trajectory_below=memory[l - 1], level=l, memory=memory[l],
                                         memory_other=memory_ema[l], memory_targets=targets[l],
                                         start_state=model_state[l],
                                         sample_state=sample_state,
                                         sample_output=sample_output,
                                         reconstruct=reconstruct)
            memory[l], memory_ema[l], targets[l], model_state[l], state_below = grounded
            simulation = self.forward_dynamic(start_state=model_state[l], start_state_below=state_below, level=l,
                                              n_steps=n_steps, n_warmup=n_warmup, memory=memory[l],
                                              memory_other=memory_ema[l], memory_targets=targets[l],
                                              sample_state=sample_state,
                                              sample_output=sample_output,
                                              reconstruct=reconstruct,)
                                              #start_time_step=memory[l]['time_step'][-1])
            memory[l], memory_ema[l], targets[l], model_state[l] = simulation
            # else:
            #    sim_gt_traj = self.filter_up(o=memory[l - 1]['z'], r=memory[l - 1]['r'],
            #                                 terminal=memory[l - 1]['terminal'], level=l)
            #    sim_gt_traj['a'] = torch.zeros((*sim_gt_traj['o'].shape[:2], self.rssm_modules[l].d_a),
            #                                   device=self.device, dtype=torch.float32)
            #    simulation = self.forward_static(sim_gt_traj, level=l, n_steps=-1, n_warmup=warmup_steps[l],
            #                                     start_state=model_state[l])
            #    memory[l], memory_ema[l], model_state[l] = simulation
            #    targets[l] = sim_gt_traj
        return memory, memory_ema, targets

    """
    def learn_states(self,
                     ground_truth_trajectory: Dict[str, torch.Tensor],
                     warmup_steps: List[int],
                     model_state: List[Dict[str, torch.Tensor]] | None = None):
        memory = [None for _ in range(self.levels)]
        memory_ema = [None for _ in range(self.levels)]
        model_state = [None for _ in range(self.levels)] if model_state is None else model_state
        targets = [None for _ in range(self.levels)]

        last_level = {'z': list(ground_truth_trajectory['o'].unbind(0)),
                      'r': list(ground_truth_trajectory['r'].unbind(0)),
                      'terminal': list(ground_truth_trajectory['terminal'].unbind(0))}
        for l in range(self.levels):
            sim_gt_traj = self.filter_up(o=last_level['z'], r=last_level['r'], terminal=last_level['terminal'], level=l)
            sim_gt_traj['a'] = torch.zeros((*sim_gt_traj['o'].shape[:2], self.rssm_modules[l].d_a),
                                           device=self.device, dtype=torch.float32)
            memory[l], memory_ema[l], model_state[l] = self.forward_static(sim_gt_traj, level=l, n_steps=-1,
                                                                           n_warmup=-1,
                                                                           start_state=model_state[l], memoryless=True)
            targets[l] = sim_gt_traj
            last_level = memory[l]

        return memory, memory_ema, targets
    """

    def _train_step(self,
                    training_data: Dict[str, torch.Tensor],
                    optimizer: torch.optim.Optimizer,
                    **kwargs):
        optimizer.zero_grad(set_to_none=True)
        losses_tf, pred_tf = self.eval_step(training_data, force_warmup=[-1 for _ in self.rssm_modules], **kwargs)
        # losses_one, pred_one = self.eval_step(training_data, force_warmup=[1 for _ in self.rssm_modules], **kwargs)
        # losses_wu, pred_wu = self.eval_step(training_data, **kwargs)

        """
        TODO:
        * es wird in jedem Zeitschritt der state NACH anwendung der Aktion gespeichert
        * d.h. der erste all-zero state wird nicht gespeichert und die richtige Aktionssequenz für einen start_state
          fängt einen Zeitschritt später an
        * d.h. die Agenten werden niemals auf dem ersten state als start_state trainiert
        * d.h. latent overshooting wird niemals vom ersten state an trainiert, wobei 0 schritt l.o. immer trainiert wrid
        * sinnvolle Änderung: forward_static() und forward_dynamic() so verändern, dass immer der state des aktuellen
          Zeitschritts gespeichert wird
        * das zieht Änderungen an anderen Stellen im Code nach sich, dies muss geprüft werden!
           * ground_level()
           * train_model()
        """

        n_lo = [10, 5]
        for l in range(self.levels):
            offset = n_lo[l]
            start_state = rssm_states_seq_to_batch(pred_tf[l], self.rssm_modules[l], i_end=-offset)
            actions = torch.stack(pred_tf[l]['a'])  # make tensor (time x batch x d_a)
            terminals = torch.stack(pred_tf[l]['terminal'])
            #time_steps = torch.stack(pred_tf[l]['time_step'])
            action_windows, terminal_windows, time_step_windows, z_post_windows = [], [], [], []
            # select for every start state the next n_lo actions
            # NOTE: since the rssm states are always recorded after an action was applied, correct actions and terminal
            # flags for a start state at time step t start from t+1
            for t in range(1, actions.shape[0] - offset + 1):
                action_windows.append(actions[t: t + offset])
                terminal_windows.append(terminals[t: t + offset])
                #time_step_windows.append(time_steps[t: t + offset])
                z_post_windows.append(stack_dists(pred_tf[l]['z_post'][t: t + offset]))
            actions = torch.concat(action_windows, dim=1)  # concat all windows along batch dimension
            terminals = torch.concat(terminal_windows, dim=1)
            #time_steps = torch.concat(time_step_windows, dim=1)
            mask = self.compute_mask(terminals)
            #trajectory = {'a': actions, 'o': None, 'r': None, 'terminal': None, 'time_step': time_steps}
            trajectory = {'a': actions, 'o': None, 'r': None, 'terminal': None}
            pred_lo_lvl, _, _ = self.forward_static(trajectory, level=l, start_state=start_state, n_warmup=0,
                                                    reconstruct=False)
            z_post = concat_dists(z_post_windows, dim=1)
            z_prior = stack_dists(pred_lo_lvl['z_prior'])
            kl_0 = torch.mean(torch.distributions.kl_divergence(detach_dist(z_post), z_prior) * (1 - mask))
            kl_1 = torch.mean(torch.distributions.kl_divergence(z_post, detach_dist(z_prior)) * (1 - mask))
            kl = (0.8 * kl_0 + 0.2 * kl_1) / offset
            losses_tf[f'kl_latent_overshooting_{l}'] = self.kl_betas[l] * kl
            losses_tf['total'] += kl

        losses = losses_tf

        for k, v in losses.items():
            if torch.isnan(v).any() or torch.isinf(v).any():
                raise RuntimeError(f'Invalid loss detected: {k}, {v}')

        # losses = {}
        # for k in losses_tf:
        #    losses[k] = (losses_tf[k] + losses_wu[k] + losses_one[k]) / 3
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()
        # update ema modules
        if self._current_train_step % self.ema_update_interval == 0:
            with torch.no_grad():
                rssm_params = chain.from_iterable([m.parameters() for m in self.rssm_modules])
                ema_params = chain.from_iterable([m.parameters() for m in self._ema_rssm_modules])
                for param, ema_param in zip(rssm_params, ema_params):
                    ema_param[:] = self.ema_coeff * ema_param + (1 - self.ema_coeff) * param

        return losses, pred_tf

    @staticmethod
    @compile_if_not_debug
    def compute_mask(terminals: torch.Tensor, threshold: float | None = None):
        d_time = terminals.shape[0]

        mask = torch.zeros_like(terminals)
        for t in range(1, d_time):
            mask[t] = torch.maximum(mask[t - 1], terminals[t - 1])

        if threshold is not None:
            mask = torch.where(mask > threshold, 1.0, 0.0)

        return mask.detach()

    def _eval_step(self,
                   training_data: Dict[str, torch.Tensor],
                   **kwargs):
        model_steps = kwargs.pop('model_steps', None)
        if model_steps is None:
            raise RuntimeError('Please specify how many steps the model should run using the model_steps kwarg')

        warmup_steps = kwargs.pop('force_warmup', self.warmup_steps)
        warmup_steps = self.maybe_sample_warmup_steps(training_data, model_steps, warmup_steps)

        # learn_states = kwargs.get('learn_states', False)

        # filter out the keys we need for the model, keep the rest for loss calculation
        #ground_truth_trajectories = {k: v for k, v in training_data.items() if
        #                             k in ('o', 'a', 'r', 'terminal', 'time_step')}
        ground_truth_trajectories = {k: v for k, v in training_data.items() if k in ('o', 'a', 'r', 'terminal')}

        # forward model
        # if learn_states:
        #    pred, pred_ema, targets = self.learn_states(ground_truth_trajectory=ground_truth_trajectories,
        #                                                warmup_steps=warmup_steps)
        # else:
        pred, pred_ema, targets = self.forward_all_levels(ground_truth_trajectory=ground_truth_trajectories,
                                                          warmup_steps=warmup_steps, model_steps=model_steps, **kwargs)

        # average losses and calculate masks
        losses = {}
        for i_lvl in range(self.levels):
            mask_lvl = self.compute_mask(targets[i_lvl]['terminal'])
            # fig, ax = plt.subplots(1, 2, figsize=(10, 10))
            # fig.suptitle(f'Level {i_lvl}')
            # ax[0].matshow(mask_lvl[:, 0:50].detach().cpu().numpy().squeeze(-1).transpose())
            # ax[0].set_title('Mask')
            # ax[1].matshow(targets[i_lvl]['terminal'][:, 0:50].detach().cpu().numpy().squeeze(-1).transpose())
            # ax[1].set_title('Terminal Flags')
            # plt.show()
            loss_level = self.calc_loss(pred[i_lvl], pred_ema[i_lvl], targets[i_lvl], mask_lvl, self.kl_betas[i_lvl],
                                        self.kl_reg_betas[i_lvl])
            loss_level = {k + f'_{i_lvl}': v for k, v in loss_level.items()}
            losses.update(loss_level)
        losses['total'] = torch.stack([v for k, v in losses.items() if k.startswith('total')]).mean()

        return losses, pred

    def maybe_sample_warmup_steps(self,
                                  training_data: Dict[str, torch.Tensor],
                                  model_steps: Tuple[str | int],
                                  warmup_steps: Tuple[str | int]):
        warmup_steps_sampled = []

        if warmup_steps[0] == 'rand':
            if model_steps[0] == -1:
                max_steps = training_data['o'].shape[0]
            else:
                max_steps = min(training_data['o'].shape[0], model_steps[0])
            wu = random.randint(1, max_steps - 1)  # do at least one step after warmup
        else:
            wu = warmup_steps[0]
        warmup_steps_sampled.append(wu)

        for l in range(1, self.levels):
            if warmup_steps[l] == 'rand':
                wu = random.randint(1, model_steps[l] - 1)  # do at least one step after warmup
            else:
                wu = warmup_steps[l]
            warmup_steps_sampled.append(wu)

        return warmup_steps_sampled

    def calc_loss(self,
                  pred: Dict[str, List[Union[torch.Tensor, torch.distributions.Distribution]]],
                  pred_ema: Dict[str, List[Union[torch.Tensor, torch.distributions.Distribution]]],
                  targets: Dict[str, torch.Tensor],
                  mask: torch.Tensor,
                  kl_beta: float,
                  kl_reg_beta: float = 0.0):
        mask = 1 - mask  # use mask to multiply irrelevant steps with zero

        assert (mask >= 0.0).all(), 'Negative mask values detected!'

        # raise NotImplementedError('choose loss')
        rec_o = self._neg_log_prob(pred['o_dist'], targets['o'], mask)
        rec_r = self._neg_log_prob(pred['r_dist'], targets['r'], mask)
        rec_term = self._neg_log_prob(pred['terminal_dist'], targets['terminal'], mask)
        # rec_o = self._mse(pred['o'], targets['o'], mask)
        # rec_r = self._mse(pred['r'], targets['r'], mask)
        # rec_term = self._mse(pred['terminal'], targets['terminal'], mask)
        # kl_z = self._kl_div(pred['z_post'], pred['z_prior'], mask)
        kl_z = (0.8 * self._kl_div(pred['z_post'], pred['z_prior'], mask, detach_ps=True)
                + 0.2 * self._kl_div( pred['z_post'], pred['z_prior'], mask, detach_qs=True))
        # torch.tensor(0.0, dtype=torch.float32, device=self.device)  # self._kl_reg(pred['z_post'], mask)
        kl_reg_z = self._kl_reg(pred['z_post'], mask)
        contrastive_z = torch.tensor(0.0,
                                     device=self.device)  # self.ema_regularization * self._contrastive_loss(pred['z'], mask)

        mae_o = self._mae(pred['o'], targets['o'], mask)
        mae_r = self._mae(pred['r'], targets['r'], mask)
        mae_term = self._mae(pred['terminal'], targets['terminal'], mask)

        total = rec_o + rec_r + rec_term + kl_z * kl_beta + kl_reg_z * kl_reg_beta + contrastive_z
        loss = {'total': total, 'o': rec_o, 'r': rec_r, 'term': rec_term, 'kl_z': kl_z, 'kl_reg_z': kl_reg_z,
                'monitoring_o': mae_o, 'monitoring_r': mae_r, 'monitoring_term': mae_term,
                'contrastive_z': contrastive_z}

        if self.ema_regularization:
            cons_o = self._kl_div(pred['o_dist'], pred_ema['o_dist'], mask, detach_qs=True)
            cons_r = self._kl_div(pred['r_dist'], pred_ema['r_dist'], mask, detach_qs=True)
            cons_term = self._kl_div(pred['terminal_dist'], pred_ema['terminal_dist'], mask, detach_qs=True)
            cons_z_prior = self._kl_div(pred['z_prior'], pred_ema['z_prior'], mask, detach_qs=True)
            cons_z_post = self._kl_div(pred['z_post'], pred_ema['z_post'], mask, detach_qs=True)
            loss['ema_reg'] = self.ema_regularization * (cons_o + cons_r + cons_term + cons_z_prior + cons_z_post)
            loss['total'] += loss['ema_reg']

        return loss

    @staticmethod
    @compile_if_not_debug
    def _contrastive_loss(x: List[torch.Tensor],
                          mask: torch.Tensor):
        x = torch.stack(x)
        # d_time, d_batch = x.shape[:2]
        # t_offset = random.randint(1, d_time - 1)
        # b_offset = random.randint(1, d_batch - 1)
        t_offset = 1
        b_offset = 0
        mask = unsqueeze_right(mask, x)
        diff = x - x.roll(shifts=[t_offset, b_offset], dims=[0, 1])
        cont_loss = torch.mean(torch.maximum(torch.tensor(0.0, device=x.device), 1.0 - (diff ** 2) * mask))
        return cont_loss

    @staticmethod
    @compile_if_not_debug
    def _neg_log_prob(distributions: List[torch.distributions.Distribution],
                      x_target: torch.Tensor,
                      mask: torch.Tensor):
        if isinstance(distributions[0], torch.distributions.RelaxedOneHotCategorical):
            # smooth out targets a bit to avoid inf/nan log probs with RelaxedOneHotCategorical
            x_target = torch.abs(x_target - 1e-5)
            x_target /= x_target.sum(dim=-1, keepdim=True)
        mask = unsqueeze_right(mask, x_target)
        neg_log_prob = [-d.log_prob(x) * m for d, x, m in zip(distributions, x_target, mask)]
        neg_log_prob = torch.stack(neg_log_prob, dim=0).mean()
        return neg_log_prob

    @staticmethod
    # should not be compiled since it causes constant re-compilation for some reason
    def _kl_div(ps: List[torch.distributions.Distribution],
                qs: List[torch.distributions.Distribution],
                mask: torch.Tensor,
                detach_ps: bool = False,
                detach_qs: bool = False):
        if detach_ps:
            ps = [detach_dist(p) if p is not None else None for p in ps]
        if detach_qs:
            qs = [detach_dist(q) if q is not None else None for q in qs]
        kl = [kl_divergence(p, q) * m for p, q, m in zip(ps, qs, mask) if None not in (p, q)]
        kl = torch.stack(kl, dim=0).mean()
        # if len(kl) > 1:
        #    kl = kl[1:].mean()  # ignore first prior since it's totally uninformed
        # else:
        #    kl = torch.tensor(0, dtype=torch.float32)
        return kl

    @staticmethod
    @compile_if_not_debug
    def _kl_reg(ps: List[torch.distributions.Distribution],
                mask: torch.Tensor):
        if isinstance(ps[0], torch.distributions.Normal):
            reg_dist = torch.distributions.Normal(loc=torch.zeros_like(ps[0].loc), scale=torch.ones_like(ps[0].scale))
        elif isinstance(ps[0], torch.distributions.ContinuousBernoulli):
            reg_dist = torch.distributions.ContinuousBernoulli(probs=torch.full_like(ps[0].probs, 0.5))
        elif isinstance(ps[0], torch.distributions.OneHotCategorical):
            reg_dist = torch.distributions.OneHotCategorical(logits=torch.ones_like(ps[0].logits))
        else:
            raise ValueError(f'No regularization distribution for distribution {ps[0]} found')

        kl_reg = [kl_divergence(p, reg_dist) * m for p, m in zip(ps, mask) if p is not None]
        kl_reg = torch.stack(kl_reg, dim=0).mean()
        return kl_reg

    @staticmethod
    @compile_if_not_debug
    def _mae(y_hats: List[torch.Tensor], ys: torch.Tensor, mask: torch.Tensor):
        y_hats = torch.stack(y_hats, dim=0)
        mask = mask.reshape(*mask.shape + (1,) * (y_hats.ndim - mask.ndim))  # append size 1 dimensions for broadcasting
        return torch.mean(torch.abs(ys - y_hats) * mask)

    @staticmethod
    @compile_if_not_debug
    def _mse(y_hats: List[torch.Tensor],
             ys: torch.Tensor,
             mask: torch.Tensor):
        mask = unsqueeze_right(mask, ys)
        y_hats = torch.stack(y_hats)
        return torch.mean(((y_hats - ys) ** 2) * mask)
