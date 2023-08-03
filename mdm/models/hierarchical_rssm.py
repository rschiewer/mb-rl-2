from __future__ import annotations

import copy
import time
from itertools import chain
import random
import sys
from collections import OrderedDict
from typing import Dict

import torch
import torch.distributions as torchd
from torch.distributions import kl_divergence
from torch.nn import ModuleList, ModuleDict
import matplotlib.pyplot as plt
from torch.profiler import record_function

from mdm.models.building_blocks import *
from mdm.models.building_blocks import RSSMCell
from mdm.models.dynamics_model import DynamicsModel
from mdm.policies.actor_critic_agent import ActorCriticAgent
from mdm.utils.torch_tools import *
from mdm.utils.utils import rssm_states_seq_to_batch, fig_to_img, update_memory, TempFigure
from mdm.logging.logger import GlobalLogger, Scope


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
        rssm_modules = [torch.jit.script(m) for m in rssm_modules]
        self.rssm_modules = ModuleList(rssm_modules)
        self.links = (*links, lvl_k_link)
        self.upwards_filters = ModuleList(upwards_filters)
        self.warmup_steps = tuple(warmup_steps)
        self.kl_betas = tuple(kl_betas)
        self.kl_reg_betas = tuple(kl_reg_betas)
        self.r_max_agents = tuple(r_max_agents)  # no module list to shield the agents from any pytorch functions
        self.goal_seeking_agents = tuple(goal_seeking_agents)
        self.latent_overshooting = latent_overshooting if latent_overshooting else []
        self.ema_regularization = ema_regularization
        self.ema_coeff = ema_coeff
        self.ema_update_interval = ema_update_interval
        self.temporal_activation_regularization = temporal_activation_regularization
        self.kl_balance = kl_balance
        self.dbg_timestep = 0
        # self._avg_z_distance = torch.nn.parameter.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=False)
        self.avg_chunk_dist_early = ModuleList([RunningMeanStd(shape=(mod.d_z,)) for mod in self.rssm_modules[:-1]])
        self.avg_chunk_dist_mid = ModuleList([RunningMeanStd(shape=(mod.d_z,)) for mod in self.rssm_modules[:-1]])
        self.avg_chunk_dist_late = ModuleList([RunningMeanStd(shape=(mod.d_z,)) for mod in self.rssm_modules[:-1]])

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

    def forward_dynamic(self,
                        start_state: Tuple[
                            torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor],
                        start_state_below: Tuple[
                            torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor],
                        level: int,
                        n_steps: int,
                        n_warmup: int = 0,
                        memory: Dict[str, torch.Tensor] | None = None,
                        memory_other: Dict[str, torch.Tensor] | None = None,
                        memory_targets: Dict[str, torch.Tensor] | None = None,
                        memory_states_below: Dict[str, torch.Tensor] | None = None,
                        sample_state: bool = True,
                        sample_output: bool = True,
                        reconstruct: bool = True,
                        use_ema_modules: bool = False,
                        start_time_step: torch.Tensor | None = None):
        with record_function('forward_dynamic'):
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
            assert agent.observation_key == 'z', 'only latent agent supported'
            lvl_below = level - 1
            lower_level_steps = self.strides[level]  # + 1  # give goal seeking agent some slack to achieve goals

            state = start_state
            state_below = start_state_below
            goal_rewards, distance_rewards, reachability_rewards, terminals = [], [], [], []
            for t in range(n_steps):
                _, a_t, = agent(state[0], sample=True, disable_exploration=True)
                a_t = a_t.detach()  # prevent gradients from flowing through agent back into world model
                s, next_state = mdl(a=a_t, last_state=state, use_posterior=False, sample_state=sample_state)
                _, o_smpl = mdl.o_decoder(s, sample_output)
                # get simulated ground truth for this step from level below
                # if a chunk starts in an invalid trajectory part (beyond terminal state), this is filtered out later
                simulation = self.goal_seeking_agents[lvl_below][0].act_in_sim(env_state=state_below, sim_env=self,
                                                                               n_steps=lower_level_steps,
                                                                               goal=o_smpl,
                                                                               sample_actions=True, sample_model=True,
                                                                               disable_exploration=True,
                                                                               reconstruct=True)
                simulated_ground_truth = self.filter_up(o=simulation['model']['z'], r=simulation['model']['r'],
                                                        terminal=simulation['model']['terminal'], level=level,
                                                        n_steps=lower_level_steps, respect_terminal_flag=True,
                                                        window_size=lower_level_steps)
                simulated_ground_truth = {k: v.squeeze(0) for k, v in simulated_ground_truth.items()}  # remove time dim

                reachability_reward = 0.1 * simulation['agent']['r'][-1].detach()
                distance_reward = torch.mean((state_below[0] - simulation['model']['z'][-1]) ** 2, dim=-1,
                                             keepdim=True).detach()
                # augment reward with how reachable the goal was for lower level
                # simulated_ground_truth['r'] += reachability_reward
                # augment reward with how different the final state of the agent is from the start state
                # simulated_ground_truth['r'] += distance_reward

                goal_rewards.append(simulation['agent']['r'][-1])
                reachability_rewards.append(reachability_reward)
                distance_rewards.append(distance_reward)
                terminals.append(simulated_ground_truth['terminal'])

                # TODO: sample_output changed to fixed false to reduce variance of training targets
                if t < n_warmup:  # if still in warmup, re-do last step with simulated ground truth and use posterior
                    o_enc = mdl.o_encoder(simulated_ground_truth['o'])
                    s, next_state = mdl(a=a_t, o_enc=o_enc, last_state=state,
                                        use_posterior=True, sample_state=sample_state)
                    # pred_other, next_state_other = mdl_other(a=a_t, o_enc=o_enc,
                    #                                         last_state=state, use_posterior=True,
                    #                                         sample_state=sample_state,
                    #                                         sample_output=False, reconstruct=reconstruct)
                # else:  # do ema model prediction in any case for model regularization
                #    pred_other, next_state_other = mdl_other(a=a_t, last_state=state, use_posterior=False,
                #                                             sample_state=sample_state,
                #                                             sample_output=False, reconstruct=True)

                pred = mdl.decode(s, sample=sample_output, reconstruct_observation=reconstruct)
                update_memory(memory, **pred, **RSSMCell.add_labels(next_state), a=a_t)
                # update_memory(memory_other, **pred_other, **next_state_other)
                update_memory(memory_other, **pred, **RSSMCell.add_labels(next_state), a=a_t)
                update_memory(memory_targets, **simulated_ground_truth)
                update_memory(memory_states_below, **RSSMCell.add_labels(simulation['model_state']))

                # important: update model states
                state = next_state
                state_below = simulation['model_state']

            if GlobalLogger.can_log('simulated_ground_truth_goal_distance', self._current_train_step):
                sim_ground_truth_r = torch.stack(memory_targets['r'][1:]).mean(dim=1).detach().cpu().numpy()
                goal_rewards = torch.stack(goal_rewards).mean(dim=1).detach().cpu().numpy().squeeze()
                # reachability_rewards = torch.stack(reachability_rewards).mean(dim=1).detach().cpu().numpy().squeeze()
                distance_rewards = torch.stack(distance_rewards).mean(dim=1).detach().cpu().numpy().squeeze()
                terminals = torch.stack(terminals).mean(dim=1).detach().cpu().numpy().squeeze()
                fig = plt.figure(dpi=60)
                plt.plot(sim_ground_truth_r, label='sim ground truth r', marker='o')
                plt.plot(goal_rewards, label='agent goal reward', marker='o')
                # plt.plot(reachability_rewards, label='reachability reward', marker='o')
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

            return memory, memory_other, memory_targets, state, memory_states_below

    def forward_static(self,
                       trajectory: Dict[str, torch.Tensor],
                       start_state: Dict[str, torch.Tensor],
                       level: int = 0,
                       n_steps: int = -1,
                       n_warmup: int = -1,
                       memory: Dict[str, torch.Tensor] | None = None,
                       memory_other: Dict[str, torch.Tensor] | None = None,
                       sample_state: bool = True,
                       sample_output: bool = True,
                       reconstruct: bool = True,
                       use_ema_modules: bool = False):
        with record_function('forward_static'):
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

            s_mem, state_mem = mdl.scan(a, o, start_state, n_warmup, sample_state)

            # TODO: rework decoders to return tensors instead of distributions so mdl.decode() can be called once
            #       before this loop to decode all time steps at once
            for a_t, s_t, state_t in zip(a, s_mem, state_mem):
                pred = mdl.decode(s_t, sample=sample_output, reconstruct_observation=reconstruct)
                update_memory(memory, **pred, **RSSMCell.add_labels(state_t), a=a_t)
                update_memory(memory_other, **pred, **RSSMCell.add_labels(state_t), a=a_t)

            state = state_mem[-1]

            """
            state = start_state
            for t in range(n_steps):
                if t < n_warmup:
                    o_t = o[t]
                    use_posterior = True
                else:
                    o_t = None
                    use_posterior = False
                a_t = a[t]

                s, next_state = mdl(a=a_t, o_enc=o_t, last_state=state, use_posterior=use_posterior,
                                       sample_state=sample_state)
                # pred_other, next_state_other = mdl_other(a=a_t, o_enc=o_t, last_state=state, use_posterior=use_posterior,
                #                                         sample_state=sample_state, sample_output=sample_output,
                #                                         reconstruct=reconstruct)

                pred = mdl.decode(s, sample=sample_output, reconstruct_observation=reconstruct)
                update_memory(memory, **pred, **RSSMCell.add_labels(next_state), a=a_t)
                # update_memory(memory_other, **pred_other, **next_state_other)
                update_memory(memory_other, **pred, **RSSMCell.add_labels(next_state), a=a_t)

                state = next_state
            """

            return memory, memory_other, state

    def filter_up(self,
                  o: List[torch.Tensor] | None = None,
                  r: List[torch.Tensor] | None = None,
                  terminal: List[torch.Tensor] | None = None,
                  level: int = 0,
                  n_steps: int = -1,
                  respect_terminal_flag: bool = True,
                  time_step: List[torch.Tensor] | None = None,
                  window_size: int | None = None,
                  **kwargs):
        if n_steps == -1:
            n_steps = len(o)
        flt = self.upwards_filters[level]

        if respect_terminal_flag:
            assert terminal is not None
            mask = compute_mask(stack_if_list(terminal[:n_steps]), mode='deterministic', threshold=0.8)
        else:
            mask = None

        simulated_ground_truth = {}
        if o is not None:
            simulated_ground_truth['o'] = flt['o'](stack_if_list(o[:n_steps]), mask=mask,
                                                   window_size=window_size).detach()
        if r is not None:
            simulated_ground_truth['r'] = flt['r'](stack_if_list(r[:n_steps]), mask=mask,
                                                   window_size=window_size).detach()
        if terminal is not None:
            simulated_ground_truth['terminal'] = flt['terminal'](stack_if_list(terminal[:n_steps]), mask=mask,
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
        with record_function('ground_level'):
            assert level > 0, 'Level 0 grounding is done automatically in forward_static() method'

            memory_targets = {} if memory_targets is None else memory_targets
            memory_states_below = {} if memory_states_below is None else memory_states_below
            n_steps = self.strides[level]
            assert n_steps <= len(trajectory_below['z']), f'Not enough below level time steps to ground level {level}'
            simulated_ground_truth = self.filter_up(o=trajectory_below['z'], r=trajectory_below['r'],
                                                    terminal=trajectory_below['terminal'], level=level,
                                                    respect_terminal_flag=False,
                                                    n_steps=n_steps)
            d_batch = trajectory_below['o'][0].shape[0]
            device = trajectory_below['o'][0].device
            simulated_ground_truth['a'] = self.rssm_modules[level].zero_a(d_batch, device).unsqueeze(
                0)  # one dummy action

            for k, v in simulated_ground_truth.items():
                assert v.shape[0] == 1, f'k is too long: {v.shape[0]}'

            mem, mem_other, state = self.forward_static(simulated_ground_truth, start_state=start_state, level=level,
                                                        n_steps=1, n_warmup=1, memory=memory, memory_other=memory_other,
                                                        sample_state=sample_state, sample_output=False,
                                                        reconstruct=reconstruct, use_ema_modules=use_ema_modules)

            simulated_ground_truth = {k: v.squeeze(0) for k, v in simulated_ground_truth.items()}  # remove time dim

            update_memory(memory_targets, **simulated_ground_truth)

            # reconstruct state of model below at time step that corresponds to one step by current level
            state_below = {k: trajectory_below[k][n_steps - 1] for k in RSSMCell.state_keys()}
            update_memory(memory_states_below, **state_below)
            state_below = RSSMCell.remove_labels(state_below)

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
            simulation = self.forward_dynamic(start_state=model_state[l], start_state_below=state_below, level=l,
                                              n_steps=n_steps, n_warmup=n_warmup, memory=memory[l],
                                              memory_other=memory_ema[l], memory_targets=targets[l],
                                              memory_states_below=memory_states_below[l], sample_state=sample_state,
                                              sample_output=sample_output, reconstruct=reconstruct, )
            memory[l], memory_ema[l], targets[l], model_state[l], memory_states_below[l] = simulation
        return memory, memory_ema, targets, memory_states_below

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
        losses, pred, targets, states_below = self.eval_step(training_data,
                                                             force_warmup=[-1 for _ in self.rssm_modules],
                                                             **kwargs)

        for k, v in losses.items():
            if torch.isnan(v).any() or torch.isinf(v).any():
                raise RuntimeError(f'Invalid loss detected: {k}, {v}')

        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        optimizer.step()

        # if self._current_train_step % self.ema_update_interval == 0:
        #    self._update_ema_modules()

        return losses, pred, targets, states_below

    # @torch.compile(disable=disable_torch_compile)
    def _update_ema_modules(self):
        with torch.no_grad():
            rssm_params = chain.from_iterable([m.parameters() for m in self.rssm_modules])
            ema_params = chain.from_iterable([m.parameters() for m in self._ema_rssm_modules])
            for param, ema_param in zip(rssm_params, ema_params):
                ema_param[:] = self.ema_coeff * ema_param + (1 - self.ema_coeff) * param

    # @torch.compile(disable=disable_torch_compile)
    def _latent_overshooting(self, pred_tf, targets, n_lo):
        with record_function('latent_overshooting'):
            losses_lo = {}
            for l in range(self.levels):
                offset = n_lo[l]
                # prepare start states for latent overshooting (we us our own hand-made masks)
                start_state, _ = rssm_states_seq_to_batch(pred_tf[l], targets[l]['terminal'], i_end=-offset)
                start_state_detached = RSSMCell.detach_state(*start_state)

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
                    kl_0 = torch.mean(torch.distributions.kl_divergence(z_post_detached, z_prior) * (1 - mask))
                    kl_1 = torch.mean(torch.distributions.kl_divergence(z_post, z_prior_detached) * (1 - mask))
                    kl = (self.kl_balance * kl_0 + (1 - self.kl_balance) * kl_1)
                else:
                    kl = torch.mean(torch.distributions.kl_divergence(z_post_detached, z_prior) * (1 - mask))
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
        ground_truth_trajs = {k: v for k, v in training_data.items() if k in ('o', 'a', 'r', 'terminal')}
        pred, pred_ema, targets, states_below = self.forward_all_levels(ground_truth_trajectory=ground_truth_trajs,
                                                                        warmup_steps=warmup_steps,
                                                                        model_steps=model_steps,
                                                                        **kwargs)
        # average losses and calculate masks
        losses = {}
        for i_lvl in range(self.levels):
            mask_lvl = compute_mask(targets[i_lvl]['terminal'])

            if GlobalLogger.can_log('mask_model', self._current_train_step):
                fig = plt.figure(figsize=(5, 5))
                plt.matshow(mask_lvl.detach().cpu().numpy().squeeze(), fignum=fig, aspect='auto')
                plt.colorbar()
                GlobalLogger.logger.log_plot(fig_to_img(fig), Scope.TRAIN() / f'model/l{i_lvl}_loss_mask',
                                             time_step=self._current_train_step)
                plt.close(fig)
                del fig

            loss_level = self.calc_loss(pred[i_lvl], pred_ema[i_lvl], targets[i_lvl], mask_lvl, self.kl_betas[i_lvl],
                                        self.kl_reg_betas[i_lvl], i_lvl)

            loss_level = {k + f'_{i_lvl}': v for k, v in loss_level.items()}
            losses.update(loss_level)

        losses['total'] = torch.stack([v for k, v in losses.items() if k.startswith('total')]).mean()

        if len(self.latent_overshooting) > 0:
            loss_lo = self._latent_overshooting(pred_tf=pred, targets=targets, n_lo=self.latent_overshooting)
            losses.update(loss_lo)
            for v in loss_lo.values():
                losses['total'] += v

        return losses, pred, targets, states_below

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
                  kl_reg_beta: float = 0.0,
                  level: int = 0):
        with record_function('calc_loss'):
            valid = 1 - mask  # use mask to multiply irrelevant steps with zero

            assert (valid >= 0.0).all(), 'Negative valid values detected!'

            rec_o = self._neg_log_prob(pred['o_dist'], targets['o'], valid)
            rec_r = self._neg_log_prob(pred['r_dist'], targets['r'], valid)
            rec_term = self._neg_log_prob(pred['terminal_dist'], targets['terminal'], valid)

            states_stacked = RSSMCell.stack_states(**pred)
            states_stacked = RSSMCell.add_labels(states_stacked)
            z_prior = self.rssm_modules[level].z_dist(states_stacked['z_prior'])
            z_post = self.rssm_modules[level].z_dist(states_stacked['z_post'])
            if self.kl_balance is not None:
                # kl_0 = self._kl_div(pred['z_post'], pred['z_prior'], valid, detach_ps=True)
                # kl_1 = self._kl_div(pred['z_post'], pred['z_prior'], valid, detach_qs=True)
                kl_0 = self._kl_div_2(z_post, z_prior, valid, detach_ps=True)
                kl_1 = self._kl_div_2(z_post, z_prior, valid, detach_qs=True)
                kl_z = self.kl_balance * kl_0 + (1 - self.kl_balance) * kl_1
            else:
                # kl_z = self._kl_div(pred['z_post'], pred['z_prior'], valid)
                kl_z = self._kl_div_2(z_post, z_prior, valid)
            # kl_reg_z = self._kl_reg_2(pred['z_post'], valid)
            kl_reg_z = self._kl_reg_2(z_post, valid)
            contrastive_z = torch.tensor(0.0, dtype=torch.float32,
                                         device=self.device)  # 0.05 * self._contrastive_loss(pred['z'], pred['terminal'], valid[:1])

            mae_o = self._mae(pred['o'], targets['o'], valid)
            mae_r = self._mae(pred['r'], targets['r'], valid)
            mae_term = self._mae(pred['terminal'], targets['terminal'], valid)

            total = rec_o + rec_r + rec_term + kl_z * kl_beta + kl_reg_z * kl_reg_beta + contrastive_z
            loss = {'total': total, 'o': rec_o, 'r': rec_r, 'term': rec_term, 'kl_z': kl_z, 'kl_reg_z': kl_reg_z,
                    'monitoring_o': mae_o, 'monitoring_r': mae_r, 'monitoring_term': mae_term,
                    'contrastive_z': contrastive_z}

            if self.ema_regularization:
                cons_o = self._kl_div(pred['o_dist'], pred_ema['o_dist'], valid, detach_qs=True)
                cons_r = self._kl_div(pred['r_dist'], pred_ema['r_dist'], valid, detach_qs=True)
                cons_term = self._kl_div(pred['terminal_dist'], pred_ema['terminal_dist'], valid, detach_qs=True)
                cons_z_prior = self._kl_div(pred['z_prior'], pred_ema['z_prior'], valid, detach_qs=True)
                cons_z_post = self._kl_div(pred['z_post'], pred_ema['z_post'], valid, detach_qs=True)
                loss['ema_reg'] = self.ema_regularization * (cons_o + cons_r + cons_term + cons_z_prior + cons_z_post)
                loss['total'] += loss['ema_reg']

            if self.temporal_activation_regularization > 0:
                # loss['temporal_act_reg'] = self._mse(pred['h'][:-1], torch.stack(pred['h'][1:]), valid[:-1])
                # loss['temporal_act_reg'] *= self.temporal_activation_regularization
                # loss['total'] += loss['temporal_act_reg']
                loss['temporal_act_reg'] = self._mse(pred['z'][:-1], torch.stack(pred['z'][1:]), valid[:-1])
                loss['temporal_act_reg'] *= self.temporal_activation_regularization
                loss['total'] += loss['temporal_act_reg']

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

    @staticmethod
    # @torch.compile(disable=disable_torch_compile)
    def _contrastive_loss(x: List[torch.Tensor],
                          terminal: List[torch.Tensor],
                          valid: torch.Tensor):
        # Don't detach terminal time steps as we want them to be different from their predecessors as well
        x = torch.stack(x)
        x_src = x[:-1]
        x_dst = x[1:]
        diff = (x_src - x_dst) ** 2
        cont_loss = 2.0 * torch.mean(torch.maximum(torch.tensor(0.0, device=x.device), (1.0 - diff)) * valid)
        return cont_loss

    @staticmethod
    # @torch.compile(disable=disable_torch_compile)
    def _neg_log_prob(distributions: List[torch.distributions.Distribution],
                      x_target: torch.Tensor,
                      valid: torch.Tensor):
        if isinstance(distributions[0], torch.distributions.RelaxedOneHotCategorical):
            # smooth out targets a bit to avoid inf/nan log probs with RelaxedOneHotCategorical
            x_target = torch.abs(x_target - 1e-5)
            x_target /= x_target.sum(dim=-1, keepdim=True)
        valid = unsqueeze_right(valid, x_target)
        # neg_log_prob = [-d.log_prob(x) * m for d, x, m in zip(distributions, x_target, mask)]
        # neg_log_prob = torch.stack(neg_log_prob, dim=0).mean()

        d_tmp = stack_dists(distributions)
        neg_log_prob = -torch.mean(d_tmp.log_prob(x_target) * valid)

        return neg_log_prob

    @staticmethod
    # should not be compiled since it causes constant re-compilation for some reason
    def _kl_div(ps: List[torch.distributions.Distribution],
                qs: List[torch.distributions.Distribution],
                valid: torch.Tensor,
                detach_ps: bool = False,
                detach_qs: bool = False):
        ps_valid, qs_valid = [], []
        for p, q in zip(ps, qs):
            # if p or q hold None entries, just use the other on for that time step, prevents gradients in those cases
            ps_valid.append(p if p is not None else q)
            qs_valid.append(q if q is not None else p)

        ps = stack_dists(ps_valid)
        qs = stack_dists(qs_valid)

        if detach_ps:
            ps = detach_dist(ps)
        if detach_qs:
            qs = detach_dist(qs)

        kl = torch.sum(torch.distributions.kl.kl_divergence(ps, qs) * valid, dim=-1)
        kl = kl.mean()

        return kl

    @staticmethod
    # should not be compiled since it causes constant re-compilation for some reason
    def _kl_div_2(ps: torch.distributions.Distribution,
                  qs: torch.distributions.Distribution,
                  valid: torch.Tensor,
                  detach_ps: bool = False,
                  detach_qs: bool = False):
        if detach_ps:
            ps = detach_dist(ps)
        if detach_qs:
            qs = detach_dist(qs)

        kl = torch.sum(torch.distributions.kl.kl_divergence(ps, qs) * valid, dim=-1)
        kl = kl.mean()

        return kl

    @staticmethod
    # @torch.compile(disable=disable_torch_compile)
    def _kl_reg_2(ps: torch.distributions.Distribution,
                  valid: torch.Tensor):
        # two sources of invalidity:
        # 1) ps can contain None elements, they're filled with distributions that have the same parameters as the
        #    regularizing ones to produce zero kl divergence, so they effectively don't count
        # 2) valid tensor can be 0 somewhere, this is filtered out after kl divergence computation

        if isinstance(ps, torchd.Normal):  # assume at least first distribution is not None
            reg_dist = torch.distributions.Normal(loc=torch.zeros_like(ps.loc), scale=torch.ones_like(ps.scale))

        # elif isinstance(ps[0], torch.distributions.ContinuousBernoulli):
        #    reg_dist = torch.distributions.ContinuousBernoulli(probs=torch.full_like(ps.probs, 0.5))
        elif isinstance(ps, torchd.OneHotCategorical):
            reg_dist = torch.distributions.OneHotCategorical(logits=torch.ones_like(ps.logits))
        else:
            raise ValueError(f'No regularization distribution for distribution {ps} found')

        kl_reg = torch.sum(torch.distributions.kl.kl_divergence(ps, reg_dist) * valid, dim=-1)
        kl_reg = kl_reg.mean()

        # kl_reg = [kl_divergence(p, reg_dist) * m for p, m in zip(ps, mask) if p is not None]
        # kl_reg = torch.stack(kl_reg, dim=0).mean()
        return kl_reg

    @staticmethod
    def _kl_reg(ps: List[torch.distributions.Distribution],
                valid: torch.Tensor):
        # two sources of invalidity:
        # 1) ps can contain None elements, they're filled with distributions that have the same parameters as the
        #    regularizing ones to produce zero kl divergence, so they effectively don't count
        # 2) valid tensor can be 0 somewhere, this is filtered out after kl divergence computation

        if isinstance(ps[0], torchd.Normal):  # assume at least first distribution is not None
            ps_valid = [p if p is not None else torchd.Normal(loc=torch.zeros_like(ps[0].loc),
                                                              scale=torch.ones_like(ps[0].scale)) for p in ps]
            ps = stack_dists(ps_valid)
            reg_dist = torch.distributions.Normal(loc=torch.zeros_like(ps.loc), scale=torch.ones_like(ps.scale))

        # elif isinstance(ps[0], torch.distributions.ContinuousBernoulli):
        #    reg_dist = torch.distributions.ContinuousBernoulli(probs=torch.full_like(ps.probs, 0.5))
        elif isinstance(ps[0], torchd.OneHotCategorical):
            ps_valid = [p if p is not None else torchd.OneHotCategorical(torch.zeros_like(ps[0].probs)) for p in ps]
            ps = stack_dists(ps_valid)
            reg_dist = torch.distributions.OneHotCategorical(logits=torch.ones_like(ps.logits))
        else:
            raise ValueError(f'No regularization distribution for distribution {ps} found')

        kl_reg = torch.sum(torch.distributions.kl.kl_divergence(ps, reg_dist) * valid, dim=-1)
        kl_reg = kl_reg.mean()

        # kl_reg = [kl_divergence(p, reg_dist) * m for p, m in zip(ps, mask) if p is not None]
        # kl_reg = torch.stack(kl_reg, dim=0).mean()
        return kl_reg

    @staticmethod
    def _mae(y_hats: List[torch.Tensor], ys: torch.Tensor, valid: torch.Tensor):
        y_hats = torch.stack(y_hats, dim=0)
        valid = valid.reshape(*valid.shape + (1,) * (y_hats.ndim - valid.ndim))  # append size 1 dim for broadcasting
        return torch.mean(torch.abs(ys - y_hats) * valid)

    @staticmethod
    def _mse(y_hats: List[torch.Tensor],
             ys: torch.Tensor,
             valid: torch.Tensor):
        valid = unsqueeze_right(valid, ys)
        y_hats = torch.stack(y_hats)
        return torch.mean(((y_hats - ys) ** 2) * valid)


class CoreLoopWrapper(torch.nn.Module):

    def __init__(self,
                 rssm_module: torch.jit.ScriptModule):
        super().__init__()
        self.rssm_module = rssm_module

    @torch.compile
    def forward(self,
                a: torch.Tensor,
                n_steps: int,
                n_warmup: int,
                o: Optional[torch.Tensor],
                sample_state: bool,
                state: Optional[RSSMStateType]):
        mdl = self.rssm_module
        s_mem: List[torch.Tensor] = []
        state_mem: List[RSSMStateType] = []
        if o is None:
            o = mdl.zero_o(a.shape[1], a.device)
        for t in range(n_steps):
            if t < n_warmup:
                use_posterior = True
            else:
                # o_t will be ignored in this case
                use_posterior = False

            s, next_state = mdl(a=a[t], o_enc=o[t], last_state=state, use_posterior=use_posterior,
                                sample_state=sample_state)
            # pred_other, next_state_other = mdl_other(a=a_t, o_enc=o_t, last_state=state, use_posterior=use_posterior,
            #                                         sample_state=sample_state, sample_output=sample_output,
            #                                         reconstruct=reconstruct)

            s_mem.append(s)
            state_mem.append(next_state)

            state = next_state

        return s_mem, state_mem
