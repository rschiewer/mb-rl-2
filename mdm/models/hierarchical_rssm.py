from __future__ import annotations

import copy
from itertools import chain
import random
import sys

import torch.distributions as torchd
from torch.nn import ModuleList, ModuleDict

from mdm.models.building_blocks import *
from mdm.models.dynamics_model import DynamicsModel
from mdm.models.rssm_cell import RSSMCell, rssm_stack_states, rssm_detach_state, \
    rssm_state_keys, rssm_add_labels, rssm_remove_labels
from mdm.policies.actor_critic_agent import ActorCriticAgent
from mdm.utils.torch_tools import *
from mdm.utils.utils import filter_mem_state_seq_to_batch, fig_to_img, append_memory, extend_memory, TempFigure, \
    list_of_tuples_to_tuple_of_lists
from mdm.logging.logger import GlobalLogger, Scope
from mdm.utils.gym_nav2d_tools import *
from mdm.models.vae import VAE


class HierarchicalRSSM(DynamicsModel, FuzzyDeviceMixin):
    _filter_names = ('o', 'a', 'r', 'terminal', 'mask')

    def __init__(self,
                 rssm_modules: Sequence[RSSMCell],
                 links: Sequence[str],  # links associate output from one lvl below with inputs on this lvl
                 upwards_filters: Sequence[Dict[str, UpwardsFilter]],
                 warmup_steps: Sequence[Union[int, str]],
                 kl_betas: Sequence[float],
                 kl_reg_betas: Sequence[float],
                 r_max_agents: List[ActorCriticAgent] = (None,),
                 goal_seeking_agents: List[ActorCriticAgent] = (None,),
                 latent_overshooting: Sequence | bool = False,
                 ema_regularization: float = 0.0,
                 ema_coeff: float = 0.99,
                 ema_update_interval: int = sys.maxsize,
                 temporal_activation_regularization: int = 0,
                 kl_balance: int | bool = False):
        super(HierarchicalRSSM, self).__init__()

        assert len(links) == len(rssm_modules) - 1
        assert len(upwards_filters) == len(rssm_modules) - 1
        for filters in upwards_filters:
            window_sizes = set([f.window_size for f in filters.values()])
            assert len(window_sizes) == 1

        lvl_k_link = 'o'  # only here to make the loop in eval_step() method work
        lvl_0_filters = {k: IdentityUpwardsFilter() for k in self._filter_names}
        for i_level, level in enumerate(upwards_filters):
            assert level.keys() <= set(self._filter_names), (f'Allowed filter names: {self._filter_names}, found'
                                                             f'filter names: {level.keys()}')
            d_a_below = rssm_modules[i_level].d_a
            d_a_above = rssm_modules[i_level + 1].d_a
            window_size = level['o'].window_size
            level['a'] = AutoencodingUpwardsFilter(s_x_orig=(window_size, d_a_below), d_x_enc=d_a_above,
                                                   window_size=window_size, encoder_lws=[100, 100, 100],
                                                   decoder_lws=[100, 100, 100],
                                                   activation='relu', layer_norm=True, epsilon=0.1,
                                                   beta=0.01, reg_sigma=1.0)
            # window_size = level['o'].window_size
            # mask_and_action_filters = {'mask': MinUpwardsFilter(window_size), 'a': ConstUpwardsFilter(window_size, 0)}
            # level.update(mask_and_action_filters)
        upwards_filters = [ModuleDict(lvl_0_filters)] + [ModuleDict(x) for x in upwards_filters]

        # generate EMA shadow copies of the RSSMs
        self._ema_rssm_modules = ModuleList([copy.deepcopy(m) for m in rssm_modules])
        for i_mod, ema_mod in enumerate(self._ema_rssm_modules):  # disable gradient computation in ema shadow copies
            for i_param, param in enumerate(ema_mod.parameters()):
                param.requires_grad = False

        # transform to torchscript
        # for mod in rssm_modules:
        #    mod.o_decoder = torch.jit.script(mod.o_decoder)
        #    mod.r_decoder = torch.jit.script(mod.r_decoder)
        #    mod.term_decoder = torch.jit.script(mod.term_decoder)
        # rssm_modules = [torch.jit.script(m) for m in rssm_modules]
        # upwards_filters = [ModuleDict({k: torch.jit.script(flt) for k, flt in flt_lvl.items()})
        #                   for flt_lvl in upwards_filters]
        # for i, agent in enumerate(r_max_agents):
        #    r_max_agents[i] = torch.jit.script(agent[0]), agent[1], agent[2]
        # for i, agent in enumerate(goal_seeking_agents):
        #    goal_seeking_agents[i] = torch.jit.script(agent[0]), agent[1], agent[2]

        # store constructor args in member variables
        self.rssm_modules = ModuleList(rssm_modules)
        self.links = (*links, lvl_k_link)
        self.upwards_filters = ModuleList(upwards_filters)
        self.warmup_steps = tuple(warmup_steps)
        self.kl_betas = tuple(kl_betas)
        self.kl_reg_betas = tuple(kl_reg_betas)
        self.r_max_agents = tuple(r_max_agents)  # no module list to prevent agent parameter being part of .parameters()
        self.goal_seeking_agents = tuple(goal_seeking_agents)
        self.latent_overshooting = latent_overshooting if latent_overshooting else []
        self.ema_regularization = ema_regularization
        self.ema_coeff = ema_coeff
        self.ema_update_interval = ema_update_interval
        self.temporal_activation_regularization = temporal_activation_regularization
        self.kl_balance = kl_balance
        self.pessimism_coeff = 0.005
        self.dbg_timestep = 0
        # self._avg_z_distance = torch.nn.parameter.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=False)
        self.avg_chunk_dist_early = ModuleList([RunningMeanStd(shape=(mod.d_z,)) for mod in self.rssm_modules[:-1]])
        self.avg_chunk_dist_mid = ModuleList([RunningMeanStd(shape=(mod.d_z,)) for mod in self.rssm_modules[:-1]])
        self.avg_chunk_dist_late = ModuleList([RunningMeanStd(shape=(mod.d_z,)) for mod in self.rssm_modules[:-1]])

        # if self.levels > 1:
        #    self.act_enc = SquashedGaussianEncoder(s_x_orig=(self.strides[1], self.rssm_modules[0].d_a),
        #                                           d_x_encoded=self.rssm_modules[1].d_a, lws=[64, 64, 64],
        #                                           activation='relu', epsilon=0.1, layer_norm=True)
        #    self.act_dec = MLPDecoder(s_x_orig=(self.strides[1], self.rssm_modules[0].d_a),
        #                              d_x_encoded=self.rssm_modules[1].d_a, lws=[64, 64, 64], activation='relu',
        #                              layer_norm=True, final_activation='tanh')

        self.goal_autoencoder = VAE(d_x=rssm_modules[0].d_s_embedding, d_z=10, latent_dist='normal',
                                    encoder_lws=[128, 64, 64], decoder_lws=[64, 64, 128], activation='relu',
                                    n_latent_categories=0, layer_norm=True, beta=0.5)

    @property
    def current_device(self):
        d = next(self.parameters()).device
        return d

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

    @torch.jit.ignore
    def forward_dynamic_old(self,
                            start_state: RSSMStateType,
                            start_state_below: RSSMStateType,
                            level: int,
                            n_steps: int,
                            n_warmup: int = 0,
                            memory: Optional[Dict[str, torch.Tensor]] = None,
                            memory_other: Optional[Dict[str, torch.Tensor]] = None,
                            memory_targets: Optional[Dict[str, torch.Tensor]] = None,
                            memory_states_below: Optional[Dict[str, torch.Tensor]] = None,
                            sample_state: bool = True,
                            sample_output: bool = True,
                            reconstruct: bool = True,
                            use_ema_modules: bool = False,
                            start_time_step: Optional[torch.Tensor] = None):
        assert level > 0, 'Not intended for level 0 model, use forward_static()'

        mdl = self._ema_rssm_modules[level] if use_ema_modules else self.rssm_modules[level]
        mdl_other = self.rssm_modules[level] if use_ema_modules else self._ema_rssm_modules[level]
        memory = {} if memory is None else memory
        memory_other = {} if memory_other is None else memory_other
        memory_targets = {} if memory_targets is None else memory_targets
        memory_states_below = {} if memory_states_below is None else memory_states_below

        if n_warmup < 0:
            n_warmup = n_steps

        agent = self.r_max_agents[level][0]
        lvl_below = level - 1
        lower_level_steps = self.strides[level]  # give goal seeking agent some slack to achieve goals

        state = start_state
        state_below = start_state_below
        goal_rewards, distance_rewards, reachability_rewards, terminals = [], [], [], []
        this_run_a, this_run_o = [], []
        for t in range(n_steps):
            # get action from agent
            with torch.no_grad():
                agent_o = agent.o_from_state(state)
                _, a_t, = agent(agent_o, sample=True, disable_exploration=False)
                a_t = a_t.detach()  # prevent gradients from flowing through agent back into world model
                this_run_a.append(a_t)

            # get next model state
            next_state = mdl(a=a_t, last_state=state, use_posterior=False, sample_state=sample_state)
            pred = mdl.decode(next_state[-1], sample=sample_output, reconstruct_observation=reconstruct)

            # get simulated ground truth for this step from level below
            # if a chunk starts in an invalid trajectory part (beyond terminal state), this is filtered out later
            with torch.no_grad():
                simulation = self.goal_seeking_agents[lvl_below][0].act_in_sim(env_start_state=state_below,
                                                                               sim_env=self,
                                                                               n_steps=lower_level_steps,
                                                                               goal=pred['o'],
                                                                               sample_actions=True,
                                                                               sample_states=True,
                                                                               disable_exploration=True,
                                                                               reconstruct=True)
                simulated_ground_truth = self.filter_up(o=simulation['model'][self.links[lvl_below]],
                                                        r=simulation['model']['r'],
                                                        terminal=simulation['model']['terminal'], level=level,
                                                        n_steps=lower_level_steps, respect_mask=True,
                                                        window_size=lower_level_steps)
                # remove time dim
                simulated_ground_truth = {k: v.squeeze(0) for k, v in simulated_ground_truth.items()}

            # punishment for unreachable states
            with torch.no_grad():
                end_state_repr_below = simulation['model']['s_embedding'][-1]
                distance_reward = torch.mean((state_below[5] - end_state_repr_below) ** 2, dim=-1,
                                             keepdim=True).detach()
                reachability_coeff = torch.exp(- 100 * simulation['agent']['r'][-1] ** 2).detach()
                # simulated_ground_truth['r'] += distance_reward
                # simulated_ground_truth['r'] *= reachability_coeff

            goal_rewards.append(simulation['agent']['r'][-1])
            reachability_rewards.append(reachability_coeff)
            distance_rewards.append(distance_reward)
            terminals.append(simulated_ground_truth['terminal'])

            # TODO: sample_output changed to fixed false to reduce variance of training targets
            if t < n_warmup:  # if still in warmup, re-do last step with simulated ground truth and use posterior
                o_enc = mdl.o_encoder(simulated_ground_truth['o'])
                this_run_o.append(o_enc)
                next_state = mdl(a=a_t, o_enc=o_enc, last_state=state,
                                 use_posterior=True, sample_state=sample_state)
                pred = mdl.decode(next_state[-1], sample=sample_output, reconstruct_observation=reconstruct)

            append_memory(memory, **pred, **rssm_add_labels(next_state), a=a_t)
            append_memory(memory_targets, **simulated_ground_truth)
            append_memory(memory_states_below, **rssm_add_labels(simulation['model_state']))
            # TODO: hack until proper ema model querying is implemented
            append_memory(memory_other, **rssm_add_labels(next_state))

            # important: update model states
            state = next_state
            state_below = simulation['model_state']

        # run other model in one pass as now actions and groundtruth observations are known
        # a_ema = torch.stack(this_run_a)
        # o_ema = torch.stack(this_run_o) if len(this_run_o) > 0 else None
        # _, state_mem_other = mdl_other.scan(a_ema, o_ema, start_state, n_warmup, sample_state)
        # extend_memory(memory_other, rssm_add_labels(rssm_stack_state_list(state_mem_other)))

        if GlobalLogger.can_log('simulated_ground_truth_goal_distance', self._current_train_step):
            sim_ground_truth_r = torch.stack(memory_targets['r'][1:]).mean(dim=1).detach().cpu().numpy()
            goal_rewards = torch.stack(goal_rewards).mean(dim=1).detach().cpu().numpy().squeeze()
            reachability_rewards = torch.stack(reachability_rewards).mean(dim=1).detach().cpu().numpy().squeeze()
            distance_rewards = torch.stack(distance_rewards).mean(dim=1).detach().cpu().numpy().squeeze()
            terminals = torch.stack(terminals).mean(dim=1).detach().cpu().numpy().squeeze()
            fig = plt.figure(dpi=60)
            plt.plot(sim_ground_truth_r, label='sim ground truth r', marker='o')
            plt.plot(goal_rewards, label='agent goal reward', marker='o')
            plt.plot(reachability_rewards, label='reachability reward', marker='o')
            plt.plot(distance_rewards, label='distance reward', marker='o')
            plt.plot(terminals, label='terminal flags', marker='o')
            plt.suptitle(f'L{level} Model + Goal Seeking Agent')
            plt.legend()
            plt.tight_layout()
            GlobalLogger.logger.log_plot(fig_to_img(fig), Scope.TRAIN() / f'model/{level}/goal_reward',
                                         self._current_train_step)
            plt.close(fig)
            del fig

        memory_targets = {k: torch.stack(v) for k, v in memory_targets.items()}  # to fit loss calculation scheme

        # DEBUGGING

        # action std
        # print(f'using agent: {torch.stack(this_run_a).detach().cpu().numpy().reshape(-1, 2).mean(axis=0)} | {torch.stack(this_run_a).detach().cpu().numpy().reshape(-1, 2).std(axis=0)}')

        # start state std
        # start_state_std = agent.o_from_state(start_state).std(axis=0).detach().cpu().numpy().mean()
        # below_start_state_std = agent.o_from_state(start_state_below).std(axis=0).detach().cpu().numpy().mean()
        # print(f'start_state: {start_state_std}, below_start_state: {below_start_state_std}')

        return memory, memory_other, memory_targets, state, memory_states_below

    @torch.jit.export
    def forward_static(self,
                       trajectory: Dict[str, torch.Tensor],
                       start_state: Dict[str, torch.Tensor],
                       level: int = 0,
                       n_steps: int = -1,
                       n_warmup: int = -1,
                       memory: Optional[Dict[str, torch.Tensor]] = None,
                       memory_other: Optional[Dict[str, torch.Tensor]] = None,
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

        if n_warmup > 0:
            o = mdl.o_encoder(o)

        model_state_mem = mdl.scan(a, o, start_state, n_warmup, sample_state)
        # _, model_state_mem_other = mdl_other.scan(a, o, start_state, n_warmup, sample_state)

        s_embed = torch.stack([state[-1] for state in model_state_mem])
        pred = mdl.decode(s_embed, sample=sample_output, reconstruct_observation=reconstruct)
        pred = {k: v.unbind(0) for k, v in pred.items()}
        pred.update(rssm_add_labels(list_of_tuples_to_tuple_of_lists(model_state_mem)))
        pred['a'] = a.unbind(0)
        extend_memory(memory, pred)
        # TODO: hack until proper ema model querying is implemented
        extend_memory(memory_other, rssm_add_labels(list_of_tuples_to_tuple_of_lists(model_state_mem)))
        # extend_memory(memory_other, rssm_add_labels(rssm_stack_state_list(model_state_mem_other)))

        state = model_state_mem[-1]

        return memory, memory_other, state

    def action_alignment(self,
                         trajectory: Dict[str, torch.Tensor | List[torch.Tensor]],
                         level: int):
        assert self.links[level - 1] == 's_embedding'
        gsa = self.goal_seeking_agents[level - 1][0]
        n_steps = self.strides[level]
        a_upper = torch.stack(trajectory['a'][:-1])
        d_time, d_batch = a_upper.shape[:2]
        a_upper = a_upper.reshape(-1, a_upper.shape[-1])

        # fold time into batch dimension to perform all gsa chunks at the same time
        start_s_embedding = torch.concat(trajectory['o'][:-1])
        # super sketchy, reverse-engineer needed z_sample and rnn_state
        start_rnn_state = start_s_embedding[:, :self.rssm_modules[level - 1].d_h].unsqueeze(1)  # add layer dim
        start_z_smpl = start_s_embedding[:, self.rssm_modules[level - 1].d_h:]
        goal_s_embedding = torch.concat(trajectory['o'][1:])
        # generate state dummy and fill in necessary info
        s_start = self.rssm_modules[level - 1].init_state(start_s_embedding.shape[0], self.device)
        s_start = (s_start[0], start_z_smpl.detach(), s_start[2], s_start[3], start_rnn_state.detach(), s_start[5])

        simulaion = gsa.act_in_sim(env_start_state=s_start, sim_env=self, n_steps=n_steps, goal=goal_s_embedding)
        a_lower = torch.stack(simulaion['agent']['a'])
        a_enc = self.actions_up(a_lower, level)[0]  # remove redundant time dim

        # max dot product similarity
        upper_norm = torch.linalg.vector_norm(a_upper, dim=-1, keepdim=True)
        enc_norm = torch.linalg.vector_norm(a_enc, dim=-1, keepdim=True)
        norm = torch.maximum(upper_norm, enc_norm).detach()
        a_upper = a_upper / upper_norm
        a_enc = a_enc / enc_norm
        similarity = torch.linalg.vecdot(a_upper, a_enc.detach(), dim=-1).unsqueeze(-1)
        similarity = similarity.reshape(d_time, d_batch, 1)
        # print(similarity.mean())
        return similarity

    def actions_up(self,
                   a: torch.Tensor | List[torch.Tensor],
                   level: int,
                   mask: torch.Tensor | None = None):
        a = stack_if_list(a)
        # a_reshaped = UpwardsFilter(self.strides[level])(a, mask=mask)
        # a_reshaped = torch.permute(a_reshaped, (0, 2, 1, 3))
        # a_reshaped = torch.flatten(a_reshaped, start_dim=2)
        a_reshaped = UpwardsFilter(self.strides[level])(a, mask=mask)
        a_reshaped = torch.permute(a_reshaped, (0, 2, 1, 3))
        _, a_enc = self.act_enc(a_reshaped, sample=False)
        return a_enc

    def actions_down(self,
                     a: torch.Tensor | List[torch.Tensor],
                     level: int):
        # a = stack_if_list(a)
        # lower_level_steps = self.strides[level]
        # a_lower = a.reshape(a.shape[0], lower_level_steps, -1)
        # a_lower = torch.permute(a_lower, (1, 0, 2))
        _, a_lower = self.act_dec(a)
        a_lower = torch.permute(a_lower, (1, 0, 2))
        return a_lower

    @torch.no_grad()
    def filter_up(self,
                  o: List[torch.Tensor] | torch.Tensor | None = None,
                  a: List[torch.Tensor] | torch.Tensor | None = None,
                  r: List[torch.Tensor] | torch.Tensor | None = None,
                  terminal: List[torch.Tensor] | torch.Tensor | None = None,
                  mask: List[torch.Tensor] | torch.Tensor | None = None,
                  level: int = None,
                  n_steps: int = -1,
                  respect_mask: bool = True,
                  window_size: int | None = None,
                  **kwargs):
        assert level is not None

        if n_steps == -1:
            n_steps = len(o)
        flt = self.upwards_filters[level]

        if respect_mask:
            assert mask is not None
            assert torch.allclose(torch.round(mask), mask), 'Mask seems to contain values other than 1.0 and 0.0'
        else:
            mask = None

        simulated_ground_truth = {}
        if o is not None:
            simulated_ground_truth['o'] = flt['o'](stack_if_list(o[:n_steps]), mask=mask,
                                                   window_size=window_size).detach()
        if a is not None:
            simulated_ground_truth['a'] = flt['a'](stack_if_list(a[:n_steps]), mask=mask,
                                                   window_size=window_size).detach()
            # a_orig = stack_if_list(a[:n_steps])
            # if level > 0:
            #    simulated_ground_truth['a'] = self.actions_up(a_orig, level, mask=mask).detach()
            # else:
            #    simulated_ground_truth['a'] = a_orig.detach()

        if r is not None:
            simulated_ground_truth['r'] = flt['r'](stack_if_list(r[:n_steps]), mask=mask,
                                                   window_size=window_size).detach()
        if terminal is not None:
            simulated_ground_truth['terminal'] = flt['terminal'](stack_if_list(terminal[:n_steps]), mask=mask,
                                                                 window_size=window_size).detach()
        if mask is not None:
            simulated_ground_truth['mask'] = flt['mask'](stack_if_list(mask[:n_steps]), mask=None,
                                                         window_size=window_size).detach()

        return simulated_ground_truth

    def ground_level(self,
                     trajectory_below: Dict[str, List[torch.Tensor]],
                     level: int,
                     start_state: Dict[str, torch.Tensor],
                     memory: Dict[str, torch.Tensor] | None = None,
                     memory_other: Dict[str, torch.Tensor] | None = None,
                     memory_targets: Dict[str, torch.Tensor] | None = None,
                     memory_states_below: Dict[str, torch.Tensor] | None = None,
                     sample_state: bool = True,
                     sample_output: bool = True,
                     reconstruct: bool = True,
                     use_ema_modules: bool = False):
        assert level > 0, 'Level 0 grounding is done automatically in forward_static() method'

        memory_targets = {} if memory_targets is None else memory_targets
        memory_states_below = {} if memory_states_below is None else memory_states_below
        n_steps = self.strides[level]
        assert n_steps <= len(trajectory_below['z']), f'Not enough below level time steps to ground level {level}'

        if self.links[level - 1] == 'z':
            o_below = trajectory_below['z']
        elif self.links[level - 1] == 's':
            o_below = trajectory_below['s']
        else:
            o_below = trajectory_below['h']

        simulated_ground_truth = self.filter_up(o=o_below, a=trajectory_below['a'], r=trajectory_below['r'],
                                                terminal=trajectory_below['terminal'], level=level,
                                                respect_mask=False,
                                                n_steps=n_steps)
        d_batch = trajectory_below['o'][0].shape[0]
        device = trajectory_below['o'][0].device
        # simulated_ground_truth['a'] = self.rssm_modules[level].zero_a(d_batch, device).unsqueeze(0)  # dummy action

        for k, v in simulated_ground_truth.items():
            assert v.shape[0] == 1, f'k is too long: {v.shape[0]}'

        mem, mem_other, state = self.forward_static(simulated_ground_truth, start_state=start_state, level=level,
                                                    n_steps=1, n_warmup=1, memory=memory, memory_other=memory_other,
                                                    sample_state=sample_state, sample_output=False,
                                                    reconstruct=reconstruct, use_ema_modules=use_ema_modules)

        simulated_ground_truth = {k: v.squeeze(0) for k, v in simulated_ground_truth.items()}  # remove time dim

        append_memory(memory_targets, **simulated_ground_truth)

        # reconstruct state of model below at time step that corresponds to one step by current level
        state_below = {k: trajectory_below[k][n_steps - 1] for k in rssm_state_keys()}
        append_memory(memory_states_below, **state_below)
        state_below = rssm_remove_labels(state_below)

        avg_state = mem['s'][0].detach().cpu().numpy().mean(axis=0)
        std_state = mem['s'][0].detach().cpu().numpy().std(axis=0)

        return mem, mem_other, memory_targets, state, state_below, memory_states_below

    def forward_all_levels(self,
                           ground_truth_trajectory: Dict[str, torch.Tensor],
                           warmup_steps: List[int],
                           model_steps: List[int],
                           model_state: List[Dict[str, torch.Tensor]] | None = None,
                           sample_state: bool = True,
                           sample_output: bool = True,
                           reconstruct: bool = True,
                           dynamic: bool = True):
        d_batch = ground_truth_trajectory['o'].shape[1]
        memory = [{} for _ in range(self.levels)]
        memory_ema = [{} for _ in range(self.levels)]
        if model_state is None:
            model_state = [rssm.init_state(d_batch, self.device) for rssm in self.rssm_modules]

        targets = [ground_truth_trajectory] + [{} for _ in range(self.levels - 1)]

        # lvl 0 is special and gets static ground truth trajectories
        memory[0], memory_ema[0], model_state[0] = self.forward_static(ground_truth_trajectory,
                                                                       start_state=model_state[0], level=0,
                                                                       n_steps=model_steps[0], n_warmup=warmup_steps[0],
                                                                       sample_state=sample_state,
                                                                       sample_output=sample_output,
                                                                       reconstruct=reconstruct)
        # all other levels are only grounded with the first k steps from below and can then do what they want
        for l in range(1, self.levels):
            filtered_trajectory = self.filter_up(o=memory[l - 1]['s_embedding'], a=targets[l - 1]['a'],
                                                 r=targets[l - 1]['r'], terminal=targets[l - 1]['terminal'],
                                                 mask=targets[l - 1]['mask'], level=l, respect_mask=True)
            # first time step of trajectory is always a zero action, terminal, reward and only an observation to
            # ground the model since there is no way to decide for an action before getting the first observation
            # filtered_trajectory['a'][0] = 0
            # filtered_trajectory['r'][0] = 0
            # filtered_trajectory['terminal'][0] = 0
            memory[l], memory_ema[l], model_state[l] = self.forward_static(filtered_trajectory,
                                                                           start_state=model_state[l],
                                                                           level=l,
                                                                           n_steps=-1,  # always go all steps
                                                                           n_warmup=warmup_steps[l],
                                                                           sample_state=sample_state,
                                                                           sample_output=sample_output,
                                                                           reconstruct=reconstruct)
            targets[l] = filtered_trajectory

        return memory, memory_ema, targets

    def pessimistic_loss(self, memory, targets, level):
        start_state, start_state_mask = filter_mem_state_seq_to_batch(memory[level], targets[level]['mask'])
        start_state = rssm_detach_state(*start_state)
        rma, _ = self.r_max_agents[level]
        world = self.rssm_modules[level]
        mem = {}

        # do rollout from ground truth states, freeze agent parameters as we want to only update the model
        with FreezeParameters([rma]):
            current_state = start_state
            for t in range(10):
                agent_o = rma.o_from_state(current_state)
                _, a = rma(agent_o, sample=True, explore=True)
                current_state = world(a=a, last_state=current_state)
                pred = world.decode(current_state[-1], sample=False, reconstruct_observation=True)
                append_memory(mem, **rssm_add_labels(current_state), **pred, a=a)

            # compute eqn (4) from https://arxiv.org/abs/2204.12581
            o = torch.stack(mem['s_embedding'])
            r = torch.stack(mem['r'])
            terminal = torch.stack(mem['terminal'])
            # take pessimistic perspective and use highest available state value estimate
            v = torch.maximum(rma.critic_net(o), rma.ema_critic_net(o))
            mask = compute_mask(mem['terminal'], first_step_mask=start_state_mask)
            pessimistic_loss = r[:-1] + 0.99 * v[1:]
            #pessimistic_loss = calc_lambda_returns(r[1:], terminal[1:], v[:-1], v[-1], 0.99, 0.95)
            pessimistic_loss = self.pessimism_coeff * masked_mean(pessimistic_loss, mask[:-1])
            # pessimistic_loss = 0.001 * masked_mean(v, mask)
        return pessimistic_loss

    def agent_model_exploration(self,
                                model_predictions: List[Dict[str, List[torch.Tensor]]],
                                targets: List[Dict[str, torch.Tensor]],
                                level: int):
        # posterior start states for current level are in current level's predictions
        # posterior start states for level below are in predictions one level below and just have to be filtered
        start_state, start_state_mask = filter_mem_state_seq_to_batch(model_predictions[level],
                                                                      targets[level]['terminal'])
        start_state = rssm_detach_state(*start_state)
        start_state_below = {k: torch.stack(v) for k, v in model_predictions[level - 1].items() if
                             k in rssm_state_keys()}
        # TODO: use masks here
        # TODO: is that correct?
        # TODO: the original start states are in the targets, we don't need to re-calculate them
        start_state_below = {k: self.upwards_filters[level]['o'](v) for k, v in start_state_below.items()}
        start_state_below = {k: list(v.unbind(0)) for k, v in start_state_below.items()}  # need lists
        start_state_below, _ = filter_mem_state_seq_to_batch(start_state_below,
                                                             model_predictions[level - 1]['terminal'])
        start_state_below = rssm_detach_state(*start_state_below)

        mem, mem_other, mem_targets, mem_below, state = self.forward_dynamic_action_autoenc(start_state,
                                                                                            start_state_below,
                                                                                            level=level, n_steps=10,
                                                                                            sample_output=True,
                                                                                            reconstruct=True,
                                                                                            sample_state=True)

        return mem, mem_targets, mem_below

    def forward_all_levels_old(self,
                               ground_truth_trajectory: Dict[str, torch.Tensor],
                               warmup_steps: List[int],
                               model_steps: List[int],
                               model_state: List[Dict[str, torch.Tensor]] | None = None,
                               sample_state: bool = True,
                               sample_output: bool = True,
                               reconstruct: bool = True,
                               dynamic: bool = True):
        d_batch = ground_truth_trajectory['o'].shape[1]
        memory = [None for _ in range(self.levels)]
        memory_ema = [None for _ in range(self.levels)]
        memory_states_below = [None for _ in range(self.levels)]
        if model_state is None:
            model_state = [rssm.init_state(d_batch, self.device) for rssm in self.rssm_modules]
        targets = [ground_truth_trajectory] + [None for _ in range(self.levels - 1)]

        # lvl 0 is special and gets static ground truth trajectories
        memory[0], memory_ema[0], model_state[0] = self.forward_static(ground_truth_trajectory,
                                                                       start_state=model_state[0], level=0,
                                                                       n_steps=model_steps[0], n_warmup=warmup_steps[0],
                                                                       sample_state=sample_state,
                                                                       sample_output=sample_output,
                                                                       reconstruct=reconstruct)
        # all other levels are only grounded with the first k steps from below and can then do what they want
        for l in range(1, self.levels):
            n_warmup = warmup_steps[l] - 1
            n_steps = model_steps[l] - 1
            grounded = self.ground_level(trajectory_below=memory[l - 1], level=l, memory=memory[l],
                                         memory_other=memory_ema[l], memory_targets=targets[l],
                                         memory_states_below=memory_states_below[l],
                                         start_state=model_state[l],
                                         sample_state=sample_state,
                                         sample_output=sample_output,
                                         reconstruct=reconstruct)
            memory[l], memory_ema[l], targets[l], model_state[l], state_below, memory_states_below[l] = grounded
            simulation = self.forward_dynamic_old(start_state=model_state[l], start_state_below=state_below, level=l,
                                                  n_steps=n_steps, n_warmup=n_warmup, memory=memory[l],
                                                  memory_other=memory_ema[l], memory_targets=targets[l],
                                                  memory_states_below=memory_states_below[l], sample_state=sample_state,
                                                  sample_output=sample_output, reconstruct=reconstruct, )
            memory[l], memory_ema[l], targets[l], model_state[l], memory_states_below[l] = simulation
        return memory, memory_ema, targets, memory_states_below

    def _train_step(self,
                    training_data: Dict[str, torch.Tensor],
                    optimizer: torch.optim.Optimizer,
                    **kwargs):
        logger = kwargs.pop('logger', None)
        optimizer.zero_grad(set_to_none=True)
        losses, pred, targets = self.eval_step(training_data, force_warmup=[-1 for _ in self.rssm_modules],
                                               **kwargs)

        for k, v in losses.items():
            if torch.isnan(v).any() or torch.isinf(v).any():
                raise RuntimeError(f'Invalid loss detected {k}: {v}')

        losses['total'].backward()
        # plot_grad_flow(self.named_parameters())
        # plt.show()

        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        optimizer.step()

        if self._current_train_step % self.ema_update_interval == 0:
            with torch.no_grad():
                rssm_params = chain.from_iterable([m.parameters() for m in self.rssm_modules])
                ema_params = chain.from_iterable([m.parameters() for m in self._ema_rssm_modules])
                for param, ema_param in zip(rssm_params, ema_params):
                    ema_param[:] = self.ema_coeff * ema_param + (1 - self.ema_coeff) * param

        return losses, pred, targets

    def _latent_overshooting(self, pred_tf, targets, n_lo):
        losses_lo = {}
        for l in range(self.levels):
            offset = n_lo[l]
            # prepare start states for latent overshooting (we us our own hand-made masks)
            with torch.no_grad():
                start_state, _ = filter_mem_state_seq_to_batch(pred_tf[l], targets[l]['terminal'], i_end=-offset)
                start_state_detached = rssm_detach_state(*start_state)
                # prepare action, posterior and mask windows that contain for every start state the next n_lo time steps
                actions = torch.stack(pred_tf[l]['a']).detach()  # make tensor (time x batch x d_a)
                mask = compute_mask(targets[l]['terminal'])  # compute mask from groundtruth sequences
                z_post_params = torch.stack(pred_tf[l]['z_post'])  # don't take z_post from start_state_detached

                # select for every start state the next n_lo actions, mask items and posteriors
                # since the rssm states are always recorded after an action was applied, correct actions and terminal
                # flags for a start state at time step t start from t+1
                action_windows, mask_windows, z_post_windows = [], [], []
                for t in range(1, actions.shape[0] - offset + 1):
                    action_windows.append(actions[t: t + offset])
                    mask_windows.append(mask[t: t + offset])
                    z_post_windows.append(z_post_params[t: t + offset])

                # concat all windows along the batch dimension for parallel loss calculation
                actions = torch.concat(action_windows, dim=1)
                mask = torch.concat(mask_windows, dim=1)
                z_post_params = torch.concat(z_post_windows, dim=1)

            # do the model rollout
            trajectory = {'a': actions, 'o': None, 'r': None, 'terminal': None}
            pred_lo_lvl, _, _ = self.forward_static(trajectory, start_state=start_state_detached, level=l,
                                                    n_warmup=0, sample_state=True, reconstruct=False)
            z_prior_params = torch.stack(pred_lo_lvl['z_prior'])

            if GlobalLogger.can_log('mask_latent_overshooting', self._current_train_step):
                with TempFigure(figsize=(5, 5)) as fig:
                    plt.matshow(mask.detach().cpu().numpy().squeeze(), fignum=fig, aspect='auto')
                    plt.colorbar()
                    GlobalLogger.logger.log_plot(fig_to_img(fig),
                                                 Scope.TRAIN() / f'model/l{l}_latent_overshooting_mask',
                                                 time_step=self._current_train_step)

            if self.kl_balance is not None:
                z_prior = self.rssm_modules[l].z_dist(z_prior_params)
                z_prior_detached = self.rssm_modules[l].z_dist(z_prior_params.detach())
                z_post = self.rssm_modules[l].z_dist(z_post_params)
                z_post_detached = self.rssm_modules[l].z_dist(z_post_params.detach())
                kl_0 = torch.mean(torch.distributions.kl_divergence(z_post_detached, z_prior) * (1 - mask).squeeze(-1))
                kl_1 = torch.mean(torch.distributions.kl_divergence(z_post, z_prior_detached) * (1 - mask).squeeze(-1))
                kl = (self.kl_balance * kl_0 + (1 - self.kl_balance) * kl_1)
            else:
                kl = torch.mean(torch.distributions.kl_divergence(z_post_detached, z_prior) * (1 - mask).squeeze(-1))
            losses_lo[f'kl_latent_overshooting_{l}'] = self.kl_betas[l] * kl / offset
        return losses_lo

    def _eval_step(self,
                   training_data: Dict[str, torch.Tensor],
                   **kwargs):
        model_steps = kwargs.pop('model_steps', None)
        if model_steps is None:
            raise RuntimeError('Please specify how many steps the model should run using the model_steps kwarg')

        warmup_steps = kwargs.pop('force_warmup', self.warmup_steps)
        warmup_steps = self.maybe_sample_warmup_steps(training_data, model_steps, warmup_steps)
        ground_truth_trajs = {k: v for k, v in training_data.items() if k in ('o', 'a', 'r', 'terminal', 'mask')}
        pred, pred_ema, targets = self.forward_all_levels(ground_truth_trajectory=ground_truth_trajs,
                                                          warmup_steps=warmup_steps,
                                                          model_steps=model_steps,
                                                          **kwargs)
        # average losses and calculate masks
        losses = {}
        for level in range(self.levels):
            if GlobalLogger.can_log('mask_model', self._current_train_step):
                fig = plt.figure(figsize=(5, 5))
                plt.matshow(targets[level]['mask'].detach().cpu().numpy().squeeze(), fignum=fig, aspect='auto')
                plt.colorbar()
                GlobalLogger.logger.log_plot(fig_to_img(fig), Scope.TRAIN() / f'model/l{level}_loss_mask',
                                             time_step=self._current_train_step)
                plt.close(fig)
                del fig

            loss_level = self.rssm_loss(pred[level], pred_ema[level], targets[level], targets[level]['mask'],
                                        self.kl_betas[level], self.kl_reg_betas[level], level)

            #pessimistic_loss = self.pessimistic_loss(pred, targets, level)
            #loss_level['total_pessimistic_loss'] = pessimistic_loss

            if level == 0:
                s_embeddings = torch.stack(pred[level]['s_embedding']).detach()
                goal_loss = self.goal_autoencoder.eval_step(x=s_embeddings, mask=targets[level]['mask'])
                goal_loss = {f'{k}_goal_autoenc': v for k, v in goal_loss.items()}
                loss_level.update(goal_loss)

            if level < self.levels - 1:
                # action autoencoder with static upfiltered trajectory data
                loss_act_autoenc = self.action_autoencoder_loss(targets[level]['a'], targets[level]['mask'], level + 1)
                loss_level.update(loss_act_autoenc)

            loss_level = {f'{k}_{level}': v for k, v in loss_level.items()}
            losses.update(loss_level)

        losses['total'] = torch.stack([v for k, v in losses.items() if k.startswith('total')]).mean()

        if len(self.latent_overshooting) > 0:
            loss_lo = self._latent_overshooting(pred_tf=pred, targets=targets, n_lo=self.latent_overshooting)
            losses.update(loss_lo)
            for v in loss_lo.values():
                losses['total'] += v

        return losses, pred, targets

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

    def action_autoencoder_loss(self,
                                a_orig: torch.Tensor | List[torch.Tensor],
                                mask: torch.Tensor,
                                target_lvl: int,
                                a_enc_targets: None | torch.Tensor | List[torch.Tensor] = None):
        a_orig = stack_if_list(a_orig).detach()
        losses = self.upwards_filters[target_lvl]['a'].eval_step(a_orig, mask=mask)
        losses = {f'{k}_act_autoencoder': v for k, v in losses.items()}
        return losses

        # OLD CODE BELOW, WE NOW HAVE A SEPARATE UPWARDS FILTER OBJECT FOR THIS

        # actually don't need to re-compute this but we just do it anyways
        """
        mask_next_level = MaxUpwardsFilter(self.strides[target_lvl])(mask).detach()
        valid = 1 - mask
        valid_next_level = 1 - mask_next_level

        # a_enc from filter_up was detached so we need to re-encode the original actions
        a_orig = stack_if_list(a_orig).detach()
        a_reshaped = UpwardsFilter(self.strides[target_lvl])(a_orig)
        a_reshaped = torch.permute(a_reshaped, (0, 2, 1, 3))
        a_enc_dist, a_enc = self.act_enc(a_reshaped, sample=True)
        a_enc_dist = self.act_enc.dist(a_enc_dist)
        _, a_rec = self.act_dec(a_enc)
        a_rec = torch.permute(a_rec, (0, 2, 1, 3))
        a_rec = a_rec.reshape(a_rec.shape[0] * a_rec.shape[1], a_rec.shape[2], a_rec.shape[3])
        a_rec = a_rec[:len(a_orig)]  # a_orig might have been padded before filtering so a_rec could be too long

        a_rec_loss = self._mse(a_rec, a_orig, valid)

        # https://www.tensorflow.org/agents/api_docs/python/tf_agents/distributions/utils/SquashToSpecNormal says that
        # kl divergence between two distributions is invariant if both are transformed with the same invertible function
        reg_dist = torchd.Normal(loc=torch.zeros_like(a_enc), scale=torch.ones_like(a_enc))
        kl = torchd.kl_divergence(a_enc_dist.base_dist.base_dist, reg_dist)
        # clip kl divergence at 1 nat
        kl = torch.maximum(1.0, kl)
        # we removed the Independent() wrapper from a_enc_dist and have to manualls sum over event dimension
        kl = kl.sum(dim=-1, keepdim=True)
        a_kl_loss = 0.1 * masked_mean(kl, mask_next_level)

        if a_enc_targets is not None:
            a_enc_targets = stack_if_list(a_enc_targets).detach()
            a_enc_loss = self._mse(a_enc, a_enc_targets, valid_next_level)
        else:
            a_enc_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        total = a_rec_loss + a_kl_loss + a_enc_loss

        return {'total_a_autoenc': total, 'a_rec': a_rec_loss, 'a_kl': a_kl_loss, 'a_enc': a_enc_loss}
        """

    def rssm_loss(self,
                  pred: Dict[str, List[torch.Tensor]],
                  pred_ema: Dict[str, List[torch.Tensor]],
                  targets: Dict[str, torch.Tensor],
                  mask: torch.Tensor,
                  kl_beta: float,
                  kl_reg_beta: float = 0.0,
                  level: int = 0):
        rssm_cell = self.rssm_modules[level]
        valid = 1 - mask  # use mask to multiply irrelevant steps with zero

        assert (valid >= 0.0).all(), 'Negative valid values detected!'

        o_dist = rssm_cell.o_decoder.dist(torch.stack(pred['o_dist']))
        r_dist = rssm_cell.r_decoder.dist(torch.stack(pred['r_dist']))
        term_dist = rssm_cell.term_decoder.dist(torch.stack(pred['terminal_dist']))
        #rec_o = self._neg_log_prob(o_dist, targets['o'], valid)
        rec_o = self._mse(o_dist.base_dist.loc, targets['o'], valid)  # rllib and official dreamer code use MSE for obs loss
        rec_r = self._neg_log_prob(r_dist, targets['r'], valid)
        rec_term = self._neg_log_prob(term_dist, targets['terminal'], valid)

        states_stacked = rssm_stack_states(pred['h'], pred['z'], pred['z_prior'], pred['z_post'], pred['rnn_state'],
                                           pred['s_embedding'])
        states_stacked = rssm_add_labels(states_stacked)
        z_prior = self.rssm_modules[level].z_dist(states_stacked['z_prior'])
        z_post = self.rssm_modules[level].z_dist(states_stacked['z_post'])
        if self.kl_balance is not None:
            kl_0 = self._kl_div(z_post, z_prior, valid, detach_ps=True, free_nats=1.0)
            kl_1 = self._kl_div(z_post, z_prior, valid, detach_qs=True, free_nats=1.0)
            kl_z = self.kl_balance * kl_0 + (1 - self.kl_balance) * kl_1
        else:
            kl_z = self._kl_div(z_post, z_prior, valid, free_nats=1.0)
        kl_reg_z = self.kl_reg(z_post, valid)

        # minimize difference of reconstructions given slightly perturbed z
        # factor = states_stacked['z'].std(dim=[0, 1]) * 0.01
        # z_pert = states_stacked['z'] + factor.reshape(1, 1, -1) * torch.rand_like(states_stacked['z'])
        # h = states_stacked['h']
        # s = torch.concat([h, z_pert], dim=-1)
        # s = rssm_cell.s_embedding(s)
        ## don't change obs decoder parameters, as the similarity loss should affect only latent z
        # with FreezeParameters([rssm_cell.o_decoder]):
        #    decoded_pert = rssm_cell.decode(s, sample=False, reconstruct_observation=True)
        # similarity_loss = self._mse(decoded_pert['o'], torch.stack(pred['o']), valid)
        # similarity_loss = similarity_loss + self._mse(decoded_pert['r'], torch.stack(pred['r']), valid)
        # similarity_loss = similarity_loss + self._mse(decoded_pert['terminal'], torch.stack(pred['terminal']), valid)

        contrastive_z = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        # contrastive_z = 0.05 * self._contrastive_loss(pred['z'], pred['terminal'], valid[:1])

        with torch.no_grad():
            mae_o = self._mae(pred['o'], targets['o'], valid)
            mae_r = self._mae(pred['r'], targets['r'], valid)  # / 2.0
            mae_term = self._mae(pred['terminal'], targets['terminal'], valid)

        total = rec_o + rec_r + rec_term + kl_z * kl_beta + kl_reg_z * kl_reg_beta + contrastive_z  # + similarity_loss
        loss = {'total': total, 'o': rec_o, 'r': rec_r, 'term': rec_term, 'kl_z': kl_z, 'kl_reg_z': kl_reg_z,
                'monitoring_o': mae_o, 'monitoring_r': mae_r, 'monitoring_term': mae_term,
                'contrastive_z': contrastive_z}  # , 'similarity_loss': similarity_loss}

        # if level > 0:
        #    nom = torch.sum(-o_dist.entropy() * valid)
        #    denom = torch.sum(valid).to(torch.float32)
        #    denom = torch.where(denom == 0, 1.0, denom)
        #    goal_dist_entropy = 0.1 * (nom / denom)
        #    loss['monitoring_goal_dist_entropy'] = goal_dist_entropy
        #    loss['total'] += goal_dist_entropy

        if self.ema_regularization:
            raise NotImplementedError('this has been deactivated')
            # cons_o = self._kl_div(pred['o_dist'], pred_ema['o_dist'], valid, detach_qs=True)
            # cons_r = self._kl_div(pred['r_dist'], pred_ema['r_dist'], valid, detach_qs=True)
            # cons_term = self._kl_div(pred['terminal_dist'], pred_ema['terminal_dist'], valid, detach_qs=True)
            # cons_z_prior = self._kl_div(pred['z_prior'], pred_ema['z_prior'], valid, detach_qs=True)
            # cons_z_post = self._kl_div(pred['z_post'], pred_ema['z_post'], valid, detach_qs=True)
            # loss['ema_reg'] = self.ema_regularization * (cons_o + cons_r + cons_term + cons_z_prior + cons_z_post)
            # loss['total'] += loss['ema_reg']

        if self.temporal_activation_regularization > 0:
            loss['temporal_act_reg'] = self._mse(pred['s_embedding'][:-1], torch.stack(pred['s_embedding'][1:]),
                                                 valid[:-1])
            loss['temporal_act_reg'] *= self.temporal_activation_regularization
            loss['total'] += loss['temporal_act_reg']
            # loss['temporal_act_reg'] = self._mse(pred['h'][:-1], torch.stack(pred['h'][1:]), valid[:-1])
            # loss['temporal_act_reg'] *= self.temporal_activation_regularization
            # loss['total'] += loss['temporal_act_reg']
            # loss['temporal_act_reg'] = self._mse(pred['z'][:-1], torch.stack(pred['z'][1:]), valid[:-1])
            # loss['temporal_act_reg'] *= self.temporal_activation_regularization
            # loss['total'] += loss['temporal_act_reg']

        # if level > 0:
        #    raise RuntimeError('_avg_z_distance is shared by all levels whereas there should be one per level!')
        #    with torch.no_grad():
        #        weights = torch.mean((torch.stack(pred['z'][:-1]) - torch.stack(pred['z'][1:])) ** 2, dim=-1, keepdim=True)
        #        self._avg_z_distance.copy_(0.95 * self._avg_z_distance + 0.05 * weights.mean())
        #        # for z with or above average distance, contrastive goal loss weight is 0
        #        weights = torch.maximum(1 - weights / self._avg_z_distance,
        #                                torch.tensor(0.0, dtype=torch.float32, device=self.device))
        #        weighted_valid = valid[:1] * weights
        #    #weighted_valid = valid[:1]
        #    loss['monitoring_average_z_distance'] = self._avg_z_distance
        #    loss['contrastive_goal_loss'] = self._contrastive_loss(pred['o'], pred['terminal'], weighted_valid)
        #    loss['total'] += loss['contrastive_goal_loss']

        return loss

    def _contrastive_loss(self,
                          x: List[torch.Tensor],
                          terminal: List[torch.Tensor],
                          valid: torch.Tensor):
        # Don't detach terminal time steps as we want them to be different from their predecessors as well
        x = torch.stack(x)
        x_src = x[:-1]
        x_dst = x[1:]
        diff = (x_src - x_dst) ** 2
        cont_loss = 2.0 * torch.mean(torch.maximum(torch.tensor(0.0, device=x.device), (1.0 - diff)) * valid)
        return cont_loss

    def _neg_log_prob(self,
                      distribution: torch.distributions.Distribution,
                      x_target: torch.Tensor,
                      valid: torch.Tensor):
        if isinstance(distribution, torch.distributions.RelaxedOneHotCategorical):
            # smooth out targets a bit to avoid inf/nan log probs with RelaxedOneHotCategorical
            x_target = torch.abs(x_target - 1e-5)
            x_target /= x_target.sum(dim=-1, keepdim=True)

        d_data_dim = np.prod(x_target.shape[2:])
        # all dists use Independent wrapper class to make event shape span the entire data dimension
        neg_log_prob = -distribution.log_prob(x_target) / d_data_dim
        # x = masked_mean(neg_log_prob, 1 - valid.squeeze(-1))
        x = torch.mean(neg_log_prob * valid.squeeze(-1))

        return x

    # should not be compiled since it causes constant re-compilation for some reason
    def _kl_div(self,
                ps: torch.distributions.Distribution,
                qs: torch.distributions.Distribution,
                valid: torch.Tensor,
                detach_ps: bool = False,
                detach_qs: bool = False,
                free_nats: float = 0.0):
        if detach_ps:
            ps = detach_dist(ps)
        if detach_qs:
            qs = detach_dist(qs)

        kl = torch.distributions.kl.kl_divergence(ps, qs)
        if free_nats > 0:
            kl = torch.maximum(kl, torch.tensor(1.0).to(kl))
        # account for masked items in mean is not so important for minimizing the loss, so we can use normal mean
        # x = masked_mean(kl, 1 - valid.squeeze(-1))
        x = torch.mean(kl * valid.squeeze(-1))

        return x

    @staticmethod
    # @torch.compile(disable=disable_torch_compile)
    def kl_reg(ps: torch.distributions.Distribution,
               valid: torch.Tensor):
        # two sources of invalidity:
        # 1) ps can contain None elements, they're filled with distributions that have the same parameters as the
        #    regularizing ones to produce zero kl divergence, so they effectively don't count
        # 2) valid tensor can be 0 somewhere, this is filtered out after kl divergence computation
        if isinstance(ps, torchd.Independent):
            dummy = ps.base_dist
        else:
            dummy = ps

        if isinstance(dummy, torchd.Normal):  # assume at least first distribution is not None
            reg_dist = torch.distributions.Normal(loc=torch.zeros_like(ps.base_dist.loc),
                                                  scale=torch.ones_like(ps.base_dist.scale))
            reg_dist = torch.distributions.Independent(reg_dist, 1)
        # elif isinstance(ps[0], torch.distributions.ContinuousBernoulli):
        #    reg_dist = torch.distributions.ContinuousBernoulli(probs=torch.full_like(ps.probs, 0.5))
        elif isinstance(dummy, torchd.OneHotCategorical):
            reg_dist = torch.distributions.OneHotCategorical(logits=torch.ones_like(ps.logits))
        else:
            raise ValueError(f'No regularization distribution for distribution {ps} found')

        kl_reg = torch.mean(torch.distributions.kl.kl_divergence(ps, reg_dist) * valid.squeeze(-1))

        # kl_reg = [kl_divergence(p, reg_dist) * m for p, m in zip(ps, mask) if p is not None]
        # kl_reg = torch.stack(kl_reg, dim=0).mean()
        return kl_reg

    def _mae(self,
             y_hats: List[torch.Tensor],
             ys: torch.Tensor,
             valid: torch.Tensor):
        y_hats = stack_if_list(y_hats, 0)
        diff = torch.abs(ys - y_hats)
        x = masked_mean(diff, 1 - valid)
        return x

    def _mse(self,
             y_hats: List[torch.Tensor],
             ys: torch.Tensor,
             valid: torch.Tensor):
        y_hats = stack_if_list(y_hats, 0)
        diff = torch.abs(ys - y_hats) ** 2
        x = masked_mean(diff, 1 - valid)
        return x
