from __future__ import annotations

import copy
from itertools import chain, combinations
import random
import sys

import torch.distributions as torchd
import torch.nn
from torch.nn import ModuleList, ModuleDict

from mdm.models.building_blocks import *
from mdm.models.dynamics_model import DynamicsModel
from mdm.models.rssm_cell import RSSMCell, rssm_stack_states, rssm_detach_state, rssm_state_keys, rssm_add_labels, \
    RSSMStateType
from mdm.policies.actor_critic_agent import ActorCriticAgent
from mdm.utils.torch_tools import *
from mdm.utils.utils import (filter_mem_state_seq_to_batch, fig_to_img, append_memory, extend_memory, TempFigure,
                             list_of_tuples_to_tuple_of_lists, list_of_dicts_to_dict_of_lists)
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
                 pessimism_coeff: float = 0.0,
                 reachability_penalty: bool = False,
                 kl_balance: int | bool = False):
        super(HierarchicalRSSM, self).__init__()

        assert len(links) == len(rssm_modules) - 1
        assert len(upwards_filters) == len(rssm_modules) - 1
        for filters in upwards_filters:
            window_sizes = set([f.window_size for f in filters.values()])
            assert len(window_sizes) == 1, f'All filters must have the same window size'

        lvl_k_link = 'o'  # only here to make the loop in eval_step() method work
        lvl_0_filters = {k: IdentityUpwardsFilter() for k in self._filter_names}
        for i_level, level in enumerate(upwards_filters):
            assert level.keys() <= set(self._filter_names), (f'Allowed filter names: {self._filter_names}, found'
                                                             f'filter names: {level.keys()}')
            d_a_below = rssm_modules[i_level].d_a
            d_a_above = rssm_modules[i_level + 1].d_a
            d_s_embedding = rssm_modules[i_level].d_s_embedding
            window_size = level['o'].window_size
            level['a'] = AutoencodingUpwardsFilter(s_x_orig=(window_size, d_a_below),
                                                   d_x_enc=d_a_above,
                                                   window_size=window_size, encoder_lws=[200, 100, 100],
                                                   encoder_type='squashed_normal', decoder_lws=[100, 100, 200],
                                                   decoder_type='squashed_normal', activation='relu', layer_norm=True,
                                                   epsilon=0.1, beta=0.1)
            # level['a'] = EMAClustering(window_size=window_size, s_x_orig=d_a_below, n_centroids=d_a_above, alpha=0.01,
            #                           dead_zone_mode='off', dead_zone_size=1.0)
            # level['a'] = RandomProjectionUpwardsFilter(window_size=window_size, s_x_orig=(window_size, d_a_below),
            #                                           d_x_filtered=d_a_above)
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
        self.pessimism_coeff = pessimism_coeff
        self.reachability_penalty = reachability_penalty
        self.dbg_timestep = 0
        self.avg_chunk_dist_early = ModuleList([RunningMeanStd(shape=(mod.d_z,)) for mod in self.rssm_modules[:-1]])
        self.avg_chunk_dist_mid = ModuleList([RunningMeanStd(shape=(mod.d_z,)) for mod in self.rssm_modules[:-1]])
        self.avg_chunk_dist_late = ModuleList([RunningMeanStd(shape=(mod.d_z,)) for mod in self.rssm_modules[:-1]])

        self.goal_autoencoder = VAE(d_x=rssm_modules[0].d_s_embedding, d_z=50, latent_dist='normal',
                                    encoder_lws=[200, 200, 100], decoder_lws=[100, 200, 200], activation='relu',
                                    n_latent_categories=0, layer_norm=True, beta=0.01)
        d_z = self.rssm_modules[0].d_z_smpl
        d_goal_embedding = 50
        self.goal_embedder = torch.nn.Sequential(lwa(lws=[d_z, 200, 200, 200, d_goal_embedding], activation='relu',
                                                     layer_norm=True, name='goal_embedding'))

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

    @torch.jit.export
    def forward_static(self,
                       trajectory: Dict[str, torch.Tensor],
                       start_state: Dict[str, torch.Tensor],
                       level: int,
                       n_steps: int,
                       n_warmup: int,
                       sample_state: bool,
                       sample_output: bool,
                       reconstruct: bool,
                       use_ema_modules: bool = False,
                       memory: Optional[Dict[str, torch.Tensor]] = None):
        mdl = self._ema_rssm_modules[level] if use_ema_modules else self.rssm_modules[level]
        mdl_other = self.rssm_modules[level] if use_ema_modules else self._ema_rssm_modules[level]
        memory = {} if memory is None else memory
        o, a = trajectory['o'], trajectory['a']

        if n_steps < 0:
            n_steps = a.shape[0]
        else:
            assert n_steps <= a.shape[0], f'Not enough actions available ({a.shape[0]}) to go {n_steps} steps'
            a = a[:n_steps]
            o = o[:n_steps]

        if n_warmup < 0:
            n_warmup = n_steps

        if n_warmup > 0:
            o = mdl.o_encoder(o)

        if start_state is None:
            start_state = mdl.init_state(a.shape[1], a.device)

        # perform simulation using a and o and start_state (last two depending on availability)
        s_mem_t0_to_T = mdl.scan(a, o, start_state, posterior_steps=n_warmup, sample_state=sample_state)
        s_final = s_mem_t0_to_T[-1]  # store away final state to return from this method
        # add start state and remove last state to build start states for 1-step prediction with EMA model
        s_mem_tm1_to_Tm1 = [start_state] + s_mem_t0_to_T[:-1]

        # make one big state tuple containing all time steps per element
        s_mem_t0_to_T = list_of_tuples_to_tuple_of_lists(s_mem_t0_to_T)
        s_mem_tm1_to_Tm1 = list_of_tuples_to_tuple_of_lists(s_mem_tm1_to_Tm1)

        # decode predictions from states
        s_embed_t0_to_T = torch.stack(s_mem_t0_to_T[-1])
        pred_t0_to_T = mdl.decode(s_embed_t0_to_T, sample=sample_output, reconstruct_observation=reconstruct)
        # adhere to convention and make first dimension a list
        pred_t0_to_T = {k: dim_to_list(v, 0) for k, v in pred_t0_to_T.items()}
        # add rssm state components to memory
        pred_t0_to_T.update(rssm_add_labels(s_mem_t0_to_T))
        # add action to memory
        pred_t0_to_T['a'] = dim_to_list(a, 0)
        extend_memory(memory, pred_t0_to_T)

        # 1-step prediction with EMA model
        # stack each element in state memory
        s_mem_tm1_to_Tm1 = [torch.stack(x) for x in s_mem_tm1_to_Tm1]
        # fold time dim into batch dim to form start states and actions
        s_TxB = [x.reshape(x.shape[0] * x.shape[1], *x.shape[2:]) for x in s_mem_tm1_to_Tm1]
        a_TxB = a.reshape(a.shape[0] * a.shape[1], *a.shape[2:])
        # do one prediction step
        s_mem_TxB = mdl_other(a_TxB, last_state=s_TxB, use_posterior=False, sample_state=sample_state)
        # get prior distribution parameters
        z_prior_EMA_TxB = s_mem_TxB[2]
        # reshape prior distribution parameters from (TxB, ...) back to (T, B, ...)
        z_prior_EMA_t0_to_T = z_prior_EMA_TxB.reshape(n_steps, -1, *z_prior_EMA_TxB.shape[1:])
        # get z prior distribution parameters from online model rollout above
        z_prior_t0_to_T = torch.stack(s_mem_t0_to_T[2])

        # compute disagreement as MSE between EMA and online model distribution parameters
        disagreement = torch.nn.functional.mse_loss(z_prior_EMA_t0_to_T, z_prior_t0_to_T, reduction='none')
        # flatten out distribution parameter dimensions and average MSE over them
        disagreement = disagreement.reshape(*disagreement.shape[:2], -1)
        disagreement = disagreement.mean(dim=-1, keepdim=True)
        # add disagreement to memory
        memory['model_disagreement'] = dim_to_list(disagreement, 0)

        return memory, s_final

    @torch.no_grad()
    def filter_up(self,
                  o: List[torch.Tensor] | torch.Tensor | None = None,
                  a: List[torch.Tensor] | torch.Tensor | None = None,
                  r: List[torch.Tensor] | torch.Tensor | None = None,
                  terminal: List[torch.Tensor] | torch.Tensor | None = None,
                  mask: List[torch.Tensor] | torch.Tensor | None = None,
                  rnn_states: List[torch.Tensor] | torch.Tensor | None = None,
                  z: List[torch.Tensor] | torch.Tensor | None = None,
                  s_embedding: List[torch.Tensor] | torch.Tensor | None = None,
                  level: int = None,
                  n_steps: int = -1,
                  respect_mask: bool = True,
                  window_size: int | None = None,
                  sample_action_autoencoder: bool = False,
                  **kwargs):
        assert level is not None

        if n_steps == -1:
            if o:
                n_steps = len(o)
            elif a:
                n_steps = len(a)
            elif r:
                n_steps = len(r)
            elif terminal:
                n_steps = len(terminal)
            else:
                n_steps = len(mask)

        flt = self.upwards_filters[level]

        if respect_mask:
            assert mask is not None
            assert torch.allclose(torch.round(mask), mask), 'Mask seems to contain values other than 1.0 and 0.0'

        simulated_ground_truth = {}
        if o is not None:
            simulated_ground_truth['o'] = flt['o'](stack_if_list(o[:n_steps]), mask=mask,
                                                   window_size=window_size).detach()
        if a is not None:
            simulated_ground_truth['a'] = flt['a'](stack_if_list(a[:n_steps]), mask=mask,
                                                   window_size=window_size, sample=sample_action_autoencoder).detach()
        if r is not None:
            simulated_ground_truth['r'] = flt['r'](stack_if_list(r[:n_steps]), mask=mask,
                                                   window_size=window_size).detach()
            if level > 0 and self.reachability_penalty:
                #obs_diff = torch.mean((simulated_ground_truth['o'][:-1] - simulated_ground_truth['o'][1:]) ** 2,
                #                      dim=-1, keepdim=True)
                #simulated_ground_truth['r'][1:] += obs_diff

                # filter out states and state embeddings from end of each chunk
                state_flt = PickOneUpwardsFilter(window_size=self.strides[level], offset=-1)
                rnn_states = state_flt(stack_if_list(rnn_states[:n_steps]), mask=mask, window_size=window_size).detach()
                z = state_flt(stack_if_list(z[:n_steps]), mask=mask, window_size=window_size).detach()
                s_embedding = state_flt(stack_if_list(s_embedding[:n_steps]), mask=mask,
                                        window_size=window_size).detach()
                # compute reachability with the resulting states
                # max reachability is 1, which means the starting state and the goal are directly adjacent
                # min reachabilitiy is 0, which means the agent needed all steps or even more
                reach_penalty = self.reachability(rnn_states, z, s_embedding, level)
                # avoid that rewards becones zero at full penalty, this could accidentally drown out negative rewards
                reach_penalty = torch.clamp(reach_penalty, 0.0, 0.5)
                simulated_ground_truth['r'] = torch.where(simulated_ground_truth['r'] > 0,
                                                          simulated_ground_truth['r'] * (1 - reach_penalty),
                                                          simulated_ground_truth['r'] * (1 + reach_penalty))

                if GlobalLogger.can_log('reachability_penalty', self._current_train_step):
                    msg = {'reachability_penalty': reach_penalty.mean().unsqueeze(0).detach().cpu().numpy()}
                    GlobalLogger.logger.log(msg,
                                            Scope.TRAIN() / f'model/{level}/reachability_penalty',
                                            time_step=self._current_train_step)

        if terminal is not None:
            simulated_ground_truth['terminal'] = flt['terminal'](stack_if_list(terminal[:n_steps]), mask=mask,
                                                                 window_size=window_size).detach()
        if mask is not None:
            simulated_ground_truth['mask'] = flt['mask'](stack_if_list(mask[:n_steps]), mask=None,
                                                         window_size=window_size).detach()

        return simulated_ground_truth

    def reachability(self,
                     rnn_states: torch.Tensor,
                     z: torch.Tensor,
                     s_embedding: torch.Tensor,
                     level: int):
        # Let GSA start in final state of one chunk and navigate to final state of next chunk to see how easy the chunk
        # is traversable for the GSA.
        # The first elements in rnn_states and z come from the end of the first chunk. Thus, we technically omit the
        # first chunk when computing the penalty.

        # create aliases for various subtrajectories
        states_t0_to_Tm1 = rnn_states[:-1]
        z_t0_to_Tm1 = z[:-1]
        s_embed_t0_to_Tm1 = s_embedding[:-1]
        s_embed_t1_to_T = s_embedding[1:]
        # create aliases for time and batch dimension sizes
        T, B = s_embedding.shape[:2]
        Tm1 = T - 1

        # fold time into batch dimension
        states_t0_to_Tm1_rs = states_t0_to_Tm1.reshape(Tm1 * B, *states_t0_to_Tm1.shape[2:])
        z_t0_to_Tm1_rs = z_t0_to_Tm1.reshape(Tm1 * B, *z_t0_to_Tm1.shape[2:])
        s_embed_t0_to_Tm1_rs = s_embed_t0_to_Tm1.reshape(Tm1 * B, *s_embed_t0_to_Tm1.shape[2:])
        s_embed_t1_to_T_rs = s_embed_t1_to_T.reshape(Tm1 * B, *s_embed_t1_to_T.shape[2:])

        # we only need the rnn_state, z and s_embed fields in RSSM state to start simulation, rest can be zeros
        start_state_t0_to_Tm1 = self.rssm_modules[level - 1].init_state(Tm1 * B, z.device)
        start_state_t0_to_Tm1 = (start_state_t0_to_Tm1[0],
                                 z_t0_to_Tm1_rs,
                                 start_state_t0_to_Tm1[2],
                                 start_state_t0_to_Tm1[3],
                                 states_t0_to_Tm1_rs,
                                 s_embed_t0_to_Tm1_rs)

        # do simulation
        gsa = self.goal_seeking_agents[level - 1][0]
        with FreezeParameters([gsa]):
            simulation = gsa.act_in_sim(env_start_state=start_state_t0_to_Tm1, sim_env=self,
                                        n_steps=self.strides[level], goal=s_embed_t1_to_T_rs)

        # penalize chunks that have goals reachable in fewer steps than chunk_size
        goal_terminals = torch.stack(simulation['agent']['terminal'])
        goal_terminals = goal_terminals[1:]  # remove first time step which is start state
        # each steps after a terminal transition is masked, i.e. the mask tells us how many steps the gsa needed
        # to get to the goal
        is_beyond_terminal = compute_mask(goal_terminals)
        # the needed steps are the max steps available (i.e. the chunk length) minus the masked steps
        steps_needed = self.strides[level] - is_beyond_terminal.sum(dim=0)
        # normalize the penalty
        steps_needed_penalty = steps_needed / self.strides[level]
        # unfold batch and time dimensions, remember that we omitted first time step
        steps_needed_penalty_rs = steps_needed_penalty.reshape(Tm1, B, 1)
        # add zero penalty for first chunk to match shapes
        penalty_mock_first_step = torch.zeros_like(steps_needed_penalty_rs[0:1])
        steps_needed_penalty_rs = torch.concat([penalty_mock_first_step, steps_needed_penalty_rs])

        return steps_needed_penalty_rs

    def forward_all_levels(self,
                           ground_truth_trajectory: Dict[str, torch.Tensor],
                           warmup_steps: List[int],
                           model_steps: List[int],
                           model_state: List[Dict[str, torch.Tensor]] | None = None,
                           sample_state: bool = True,
                           sample_output: bool = True,
                           reconstruct: bool = True,
                           dynamic: bool = True):
        d_batch = ground_truth_trajectory['a'].shape[1]
        memory = [{} for _ in range(self.levels)]
        if model_state is None:
            model_state = [rssm.init_state(d_batch, self.device) for rssm in self.rssm_modules]

        targets = [ground_truth_trajectory] + [{} for _ in range(self.levels - 1)]

        # lvl 0 is special and gets static ground truth trajectories
        memory[0], model_state[0] = self.forward_static(ground_truth_trajectory,
                                                        start_state=model_state[0], level=0,
                                                        n_steps=model_steps[0], n_warmup=warmup_steps[0],
                                                        sample_state=sample_state,
                                                        sample_output=sample_output,
                                                        reconstruct=reconstruct)
        for l in range(1, self.levels):
            filtered_trajectory = self.filter_up(o=memory[l - 1]['s_embedding'], a=targets[l - 1]['a'],
                                                 r=targets[l - 1]['r'], terminal=targets[l - 1]['terminal'],
                                                 mask=targets[l - 1]['mask'], rnn_states=memory[l - 1]['rnn_state'],
                                                 z=memory[l - 1]['z'], s_embedding=memory[l - 1]['s_embedding'],
                                                 level=l, respect_mask=True, sample_action_autoencoder=True)
            memory[l], model_state[l] = self.forward_static(filtered_trajectory,
                                                            start_state=model_state[l], level=l,
                                                            n_steps=-1, n_warmup=warmup_steps[l],
                                                            sample_state=sample_state,
                                                            sample_output=sample_output,
                                                            reconstruct=reconstruct)
            targets[l] = filtered_trajectory

        return memory, targets, model_state

    def pessimistic_loss(self, memory, targets, level):
        rollout_lengths = [10, 5]
        start_state, start_state_mask = filter_mem_state_seq_to_batch(memory[level], targets[level]['mask'])
        start_state = rssm_detach_state(*start_state)
        rma, _ = self.r_max_agents[level]
        world = self.rssm_modules[level]
        mem = {}

        # do rollout from ground truth states, freeze agent parameters as we want to only update the model
        with FreezeParameters([rma]):
            current_state = start_state
            for t in range(rollout_lengths[level]):
                agent_o = rma.o_from_state(current_state)
                _, a = rma(agent_o, sample=True)
                current_state = world(a=a, last_state=current_state, use_posterior=False)
                pred = world.decode(current_state[-1], sample=True, reconstruct_observation=True)
                append_memory(mem, **rssm_add_labels(current_state), **pred, a=a)

            # compute eqn (4) from https://arxiv.org/abs/2204.12581
            # T(s', r | s, s) factorizes to p(z_t | z_t-1, h_t-1, a_t-1) * p(r_t | z_t) in this model and h_t and
            # s_embedding_t are deterministic.
            mask = compute_mask(mem['terminal'], first_step_mask=start_state_mask)
            z = torch.stack(mem['z'])
            z_dist_params = torch.stack(mem['z_prior'])
            z_dist = self.rssm_modules[level].z_dist(z_dist_params)
            r = torch.stack(mem['r'])
            r_dist_params = torch.stack(mem['r_dist'])
            r_dist = self.rssm_modules[level].r_decoder.dist(r_dist_params)
            o = torch.stack(mem['s_embedding'])
            # calc state values
            v = rma.critic_net(o)

            # v = torch.minimum(rma.critic_net(o), rma.ema_critic_net(o))
            # v = rma.return_running_average.normalize(v, rma.return_running_average.mean, rma.return_running_average.var)

            z_log_prob = z_dist.log_prob(z.detach()).unsqueeze(-1)
            r_log_prob = r_dist.log_prob(r.detach()).unsqueeze(-1)
            pessimistic_loss = (r.detach() + 0.99 * v) * z_log_prob * r_log_prob
            #pessimistic_loss = (r[:-1] + 0.99 * v[1:])
            # pessimistic_loss = calc_lambda_returns(r[1:], terminal[1:], v[:-1], v[-1], 0.99, 0.95)
            pessimistic_loss = self.pessimism_coeff * masked_mean(pessimistic_loss, mask)
            # pessimistic_loss = 0.001 * masked_mean(v, mask)
        return pessimistic_loss

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
                start_state, _ = filter_mem_state_seq_to_batch(pred_tf[l], targets[l]['mask'], i_end=-offset)
                start_state_detached = rssm_detach_state(*start_state)
                # prepare action, posterior and mask windows that contain for every start state the next n_lo time steps
                actions = torch.stack(pred_tf[l]['a']).detach()  # make tensor (time x batch x d_a)
                mask = targets[l]['mask']  # compute mask from groundtruth sequences
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
            pred_lo_lvl, _ = self.forward_static(trajectory, start_state=start_state_detached, level=l, n_steps=-1,
                                                 n_warmup=0, sample_state=True, reconstruct=False,
                                                 sample_output=False)
            z_prior_params = torch.stack(pred_lo_lvl['z_prior'])

            if GlobalLogger.can_log('mask_latent_overshooting', self._current_train_step):
                with TempFigure(figsize=(5, 5)) as fig:
                    plt.matshow(mask.detach().cpu().numpy().squeeze(), fignum=fig, aspect='auto')
                    plt.colorbar()
                    GlobalLogger.logger.log_plot(fig_to_img(fig),
                                                 Scope.TRAIN() / f'model/l{l}_latent_overshooting_mask',
                                                 time_step=self._current_train_step)

            z_prior = self.rssm_modules[l].z_dist(z_prior_params)
            z_post_detached = self.rssm_modules[l].z_dist(z_post_params.detach())
            kl = masked_mean(torch.distributions.kl_divergence(z_post_detached, z_prior).unsqueeze(-1), mask)

            losses_lo[f'kl_latent_overshooting_{l}'] = self.kl_betas[l] * kl
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
        pred, targets, _ = self.forward_all_levels(ground_truth_trajectory=ground_truth_trajs,
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

            loss_level = self.rssm_loss(pred[level], targets[level], targets[level]['mask'],
                                        self.kl_betas[level], self.kl_reg_betas[level], level)

            pessimistic_loss = self.pessimistic_loss(pred, targets, level)
            loss_level['total_pessimistic_loss'] = pessimistic_loss

            # markov_loss = self.markovianity_loss(targets[level], level=level, delta_max=5)#len(targets[level]['o']))
            # loss_level.update(markov_loss)

            # markov_loss = self.markov_goal_embedding(targets[level], level=level, delta_max=5)
            # loss_level.update(markov_loss)

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

            # pass-through model disagreement loss
            loss_level['monitoring_model_disagreement'] = torch.stack(pred[level]['model_disagreement']).mean()

            losses.update(loss_level)

        losses['total'] = torch.stack([v for k, v in losses.items() if k.startswith('total')]).mean()

        if len(self.latent_overshooting) > 0:
            loss_lo = self._latent_overshooting(pred_tf=pred, targets=targets, n_lo=self.latent_overshooting)
            losses.update(loss_lo)
            for v in loss_lo.values():
                losses['total'] += v

        return losses, pred, targets

    def markovianity_loss(self,
                          targets: Dict[str, torch.Tensor],
                          level: int,
                          delta_max: int,
                          delta_min: int = 2):
        assert delta_max > delta_min, f'delta_max expected to be larger than delta_min'
        assert delta_max <= len(targets['o']), f'delta_max can\'t be larger than available time steps'

        d_time, d_batch = targets['a'].shape[:2]
        model = self.rssm_modules[level]

        # Get time windows from training data. Iterate over all time steps and cut out multiple subtrajectories of
        # varying length that end in that time step. Note that the windows heavily overlap, but that's ok.
        subtrajectories = []
        for delta in range(delta_min, delta_max):
            subtraj_current_delta = []
            for t in reversed(range(delta, d_time)):
                subtraj = {k: v[t - delta:t] for k, v in targets.items()}
                subtraj_current_delta.append(subtraj)
            # transform list of time window dicts to one large dict and concatenate at batch dimension
            subtraj_current_delta = list_of_dicts_to_dict_of_lists(subtraj_current_delta)
            subtraj_current_delta = {k: torch.concat(v, dim=1) for k, v in subtraj_current_delta.items()}
            subtrajectories.append(subtraj_current_delta)
        # Each element in subtrajectories is a batch of length <= delta_max that contains one subtrajectory for each
        # batch item and time step except the first few time steps, as here the possible length for the subtrajectories
        # is capped.

        # produce rollouts and store final destination state for comparison
        predictions = []
        for batch in subtrajectories:
            o = model.o_encoder(batch['o'])
            a = batch['a']
            states = model.scan(a=a, o_enc=o, start_state=None, posterior_steps=a.shape[0], sample_state=True)
            # keep only last time step's s_embedding
            final_state = states[-1]
            s_embed = final_state[-1]
            predictions.append(s_embed)
        # for batches with smaller delta, more time steps were added to the batch as we could take them from closer to
        # the trajectory start. Get rid of those for now for simplicity.
        # cutoff = len(predictions[-1])  # choose batch with largest time window
        # predictions = [pred[:cutoff] for pred in predictions]

        # Each element in predictions is the final s_embedding of a subtrajectory rollout of some length. The rollout
        # spans all possible time steps of the original trajectories in targets, starting with the last one to the
        # earliest time step possible. Thue, by construction the batch index and end state stays the same over all the
        # various rollouts in predictions except that shorter rollouts can be performed for earlier time
        # steps of the trajectories. We can compute the similarity loss in a straightforward manner. Use expensive n^2
        # comparison for now.
        sim_loss = list(combinations(predictions, 2))  # for some reason, we need to explicitly make a list here
        # In case we compare two rollouts where one subtrajectory was shorter, it covers more states from the
        # beginning of the input trajectories. Here, truncate the states we have no comparison partner for.
        sim_loss = [(a[:min(len(a), len(b))], b[:min(len(a), len(b))]) for a, b in sim_loss]
        # compute MSE for each pair of rollouts
        sim_loss = [torch.mean((pair[0] - pair[1]) ** 2) for pair in sim_loss]
        sim_loss = torch.stack(sim_loss).mean()

        # contrastive term that encourages the model to increase the difference of adjacent states, irrespective
        # of the amount of time steps that came before them
        contr_loss = [torch.mean((pred[:-1] - pred[1:]) ** 2) for pred in predictions]
        contr_loss = -torch.stack(contr_loss).mean()

        # clamp contrastive loss term at [-1.0, inf] to prevent it from destabilizing learning
        total_markov_loss = sim_loss + torch.maximum(contr_loss, torch.tensor(-1.0).to(contr_loss))

        return {'total_markov_loss': total_markov_loss, 'similarity_markov_loss': sim_loss,
                'contrastive_markov_loss': contr_loss}

    def markov_goal_embedding(self,
                              targets: Dict[str, torch.Tensor],
                              level: int,
                              delta_max: int,
                              delta_min: int = 2):
        assert delta_max > delta_min, f'delta_max expected to be larger than delta_min'
        assert delta_max <= len(targets['o']), f'delta_max can\'t be larger than available time steps'

        d_time, d_batch = targets['a'].shape[:2]
        model = self.rssm_modules[level]

        # Get time windows from training data. Iterate over all time steps and cut out multiple subtrajectories of
        # varying length that end in that time step. Note that the windows heavily overlap, but that's ok.
        subtrajectories = []
        for delta in range(delta_min, delta_max):
            subtraj_current_delta = []
            for t in reversed(range(delta, d_time)):
                subtraj = {k: v[t - delta:t] for k, v in targets.items()}
                subtraj_current_delta.append(subtraj)
            # transform list of time window dicts to one large dict and concatenate at batch dimension
            subtraj_current_delta = list_of_dicts_to_dict_of_lists(subtraj_current_delta)
            subtraj_current_delta = {k: torch.concat(v, dim=1) for k, v in subtraj_current_delta.items()}
            subtrajectories.append(subtraj_current_delta)
        # Each element in subtrajectories is a batch of length <= delta_max that contains one subtrajectory for each
        # batch item and time step except the first few time steps, as here the possible length for the subtrajectories
        # is capped.

        # produce rollouts and store final destination state for comparison
        predictions = []
        for batch in subtrajectories:
            o = model.o_encoder(batch['o'])
            a = batch['a']
            states = model.scan(a=a, o_enc=o, start_state=None, posterior_steps=a.shape[0], sample_state=True)
            # keep only last time step's s_repr
            final_state = states[-1]
            s_repr = final_state[-1]
            # make sure we don't modify weights of the RSSM but only the goal embedder
            s_repr = s_repr.detach()
            # use the goal embedder to produce a goal embedding for the last time step's state
            s_embedding = self.goal_embedder(s_repr)
            predictions.append(s_embedding)
        # for batches with smaller delta, more time steps were added to the batch as we could take them from closer to
        # the trajectory start. Get rid of those for now for simplicity.
        # cutoff = len(predictions[-1])  # choose batch with largest time window
        # predictions = [pred[:cutoff] for pred in predictions]

        # Each element in predictions is the final s_embedding of a subtrajectory rollout of some length. The rollout
        # spans all possible time steps of the original trajectories in targets, starting with the last one to the
        # earliest time step possible. Thue, by construction the batch index and end state stays the same over all the
        # various rollouts in predictions except that shorter rollouts can be performed for earlier time
        # steps of the trajectories. We can compute the similarity loss in a straightforward manner. Use expensive n^2
        # comparison for now.
        sim_loss = list(combinations(predictions, 2))  # for some reason, we need to explicitly make a list here
        # In case we compare two rollouts where one subtrajectory was shorter, it covers more states from the
        # beginning of the input trajectories. Here, truncate the states we have no comparison partner for.
        sim_loss = [(a[:min(len(a), len(b))], b[:min(len(a), len(b))]) for a, b in sim_loss]
        # compute MSE for each pair of rollouts
        sim_loss = [torch.mean((pair[0] - pair[1]) ** 2) for pair in sim_loss]
        sim_loss = torch.stack(sim_loss).mean()

        # contrastive term that encourages the model to increase the difference of adjacent states, irrespective
        # of the amount of time steps that came before them
        contr_loss = [torch.mean((pred[:-1] - pred[1:]) ** 2) for pred in predictions]
        contr_loss = -torch.stack(contr_loss).mean()

        # clamp contrastive loss term at [-1.0, inf] to prevent it from destabilizing learning
        total_markov_loss = sim_loss + torch.maximum(contr_loss, torch.tensor(-1.0).to(contr_loss))

        return {'total_markov_loss': total_markov_loss, 'similarity_markov_loss': sim_loss,
                'contrastive_markov_loss': contr_loss}

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

    def rssm_loss(self,
                  pred: Dict[str, List[torch.Tensor]],
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
        rec_o = self._neg_log_prob(o_dist, targets['o'], valid)
        # rec_o = self._mse(o_dist.base_dist.loc, targets['o'],
        #                  valid)  # rllib and official dreamer code use MSE for obs loss
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
        kl_reg_z = torch.tensor(0.0).to(rec_r)  # self.kl_reg(z_post, valid)

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

        with torch.no_grad():
            # generate some statistics about the z distribution
            if self.rssm_modules[level].latent_dist == 'normal':
                all_z = torch.flatten(torch.stack(pred['z']), start_dim=0, end_dim=-2)
                z_mean = all_z.mean()
                z_std = all_z.std()
                z_inter_dim_std = all_z.mean(dim=0).std()
                z_dist_stats = {'z_mean': z_mean, 'z_std': z_std, 'inter_dim_z_std': z_inter_dim_std}
            elif self.rssm_modules[level].latent_dist == 'categorical':
                all_z = torch.flatten(torch.stack(pred['z']), start_dim=0, end_dim=-2)
                all_z = torch.argmax(all_z, dim=-1).to(dtype=torch.float32)
                z_mean = all_z.mean()
                z_std = all_z.std()
                # z_most_often = torch.bincount(all_z)
                z_dist_stats = {'z_mean': z_mean, 'z_std': z_std}
            else:
                z_dist_stats = {}

            z_dist_stats = {f'monitoring_{k}': v for k, v in z_dist_stats.items()}
            loss.update(z_dist_stats)

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
        if (isinstance(distribution, torch.distributions.Independent)
                and isinstance(distribution.base_dist, torch.distributions.RelaxedOneHotCategorical)):
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
            reg_dist = torch.distributions.Normal(loc=torch.zeros_like(dummy.loc),
                                                  scale=torch.ones_like(dummy.scale))
            reg_dist = torch.distributions.Independent(reg_dist, 1)
        # elif isinstance(ps[0], torch.distributions.ContinuousBernoulli):
        #    reg_dist = torch.distributions.ContinuousBernoulli(probs=torch.full_like(ps.probs, 0.5))
        elif isinstance(dummy, torchd.OneHotCategorical):
            reg_dist = torch.distributions.OneHotCategorical(logits=torch.ones_like(dummy.logits))
        elif isinstance(dummy, torchd.Bernoulli):
            reg_dist = torchd.Bernoulli(logits=torch.ones_like(dummy.logits))
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
