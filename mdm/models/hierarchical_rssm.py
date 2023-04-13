from __future__ import annotations

import copy
import random
from collections import OrderedDict
from typing import List, Dict, Tuple, Any

import torch
from torch.nn import ModuleList, ModuleDict
from torch.distributions import kl_divergence

from mdm.models.building_blocks import *
from mdm.models.dynamics_model import DynamicsModel
from mdm.policies.actor_critic_agent import ActorCriticAgent
from mdm.utils.torch_tools import FuzzyDeviceMixin
from mdm.utils.torch_tools import get_dist_params, detach_dist, update_ema_modules
from mdm.utils.utils import expand_shape_right


class RSSMCell(torch.nn.Module):

    def __init__(self,
                 d_z: int,
                 d_h: int,
                 d_a: int,
                 o_encoder: 'InputEncoder',
                 o_decoder: 'OutputDecoder',
                 r_decoder: 'OutputDecoder',
                 term_decoder: 'OutputDecoder',
                 d_context: int = 0,
                 n_hidden_layers: int = 1,
                 hidden_dropout: float = 0.1,
                 epsilon: float = 0.01,
                 z_prior_lws: Sequence[int] = (32, 32),
                 z_post_lws: Sequence[int] = (32, 32),
                 layer_norm: bool = False,
                 activation: str = 'relu',
                 rnn_type: str = 'lstm',
                 latent_dist: str = 'normal'):
        super().__init__()

        assert o_decoder.d_x_encoded == d_z + d_h
        assert r_decoder.d_x_encoded == d_z + d_h
        assert term_decoder.d_x_encoded == d_z + d_h

        self.d_z = d_z
        self.d_h = d_h
        self.d_a = d_a
        self.d_context = d_context
        self.o_encoder = o_encoder
        self.o_decoder = o_decoder
        self.r_decoder = r_decoder
        self.term_decoder = term_decoder
        self.n_hidden_layers = n_hidden_layers
        self.epsilon = epsilon
        self.layer_norm = layer_norm
        self.activation = activation
        self.rnn_type = rnn_type
        self.latent_dist = latent_dist

        self.n_latent_categories = 16
        if latent_dist == 'normal':
            d_z_post_in = d_h + d_z * 2
            d_z_final = d_z * 2
            d_z_smpl = d_z
        elif latent_dist == 'bernoulli':
            d_z_post_in = d_h + d_z
            d_z_final = d_z
            d_z_smpl = d_z
        elif latent_dist == 'categorical':
            d_z_post_in = d_h + d_z * self.n_latent_categories
            d_z_final = d_z * self.n_latent_categories
            d_z_smpl = d_z * self.n_latent_categories
        else:
            raise ValueError(f'Unknown latent distribution type: {latent_dist}')
        self.d_z_smpl = d_z_smpl
        self.d_x_posterior = self.d_o_encoded + 2  # observation + reward + terminal

        z_prior_lws = (d_h, *z_prior_lws, d_z_final)
        z_post_lws = (d_z_post_in + self.d_x_posterior, *z_post_lws, d_z_final)

        if rnn_type == 'lstm':
            rnn_constr = torch.nn.LSTM
        elif rnn_type == 'gru':
            rnn_constr = torch.nn.GRU
        else:
            raise ValueError(f'Unsupported rnn type: {rnn_type}')

        d_det_core = d_z_smpl + d_a
        self._rnn = rnn_constr(d_det_core, hidden_size=d_h, num_layers=n_hidden_layers, batch_first=False,
                               dropout=hidden_dropout)
        self._z_prior = torch.nn.Sequential(lwa(z_prior_lws, activation, layer_norm=layer_norm, name='z_prior'))
        self._z_post = torch.nn.Sequential(lwa(z_post_lws, activation, layer_norm=layer_norm, name='z_post'))

    @property
    def o_shape(self):
        return self.o_encoder.s_x_orig

    @property
    def d_o_encoded(self):
        return self.o_encoder.d_x_encoded

    def init_state(self,
                   d_batch: int,
                   device: torch.device):
        z = self.zero_z(d_batch, device)
        rnn_state = self.zero_rnn_state(d_batch, device)
        return {'z': z, 'z_prior': None, 'z_post': None, 'rnn_state': rnn_state}

    def zero_s(self,
               d_batch: int,
               device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, self.d_state, device=device)

    def zero_z(self,
               d_batch: int,
               device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, self.d_z_smpl, device=device)

    def zero_o(self,
               d_batch: int,
               device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, *self.o_shape, device=device)

    def zero_a(self,
               d_batch: int,
               device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, self.d_a, device=device)

    def zero_r(self,
               d_batch: int,
               device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, 1, device=device)

    def zero_term(self,
                  d_batch: int,
                  device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, 1, device=device)

    def zero_rnn_state(self,
                       d_batch: int,
                       device: torch.device) -> RnnStateType:
        if self.rnn_type == 'lstm':
            return (torch.zeros(self.n_hidden_layers, d_batch, self.d_h, device=device),
                    torch.zeros(self.n_hidden_layers, d_batch, self.d_h, device=device))
        else:
            return torch.zeros(self.n_hidden_layers, d_batch, self.d_h, device=device)

    def zero_context(self,
                     d_batch: int,
                     device: torch.device) -> torch.Tensor:
        return torch.zeros(d_batch, self.d_context, device=device)

    def imagine(self,
                a: torch.Tensor,
                last_state: Dict[str, torch.Tensor],
                context: Optional[torch.Tensor] = None,
                sample: bool = True):
        if context is None:
            context = self.zero_context(a.shape[0], a.device)

        inp = torch.concat([last_state['z'], a, context], dim=-1)
        inp = inp.unsqueeze(0)  # add time dim
        h, next_rnn_state = self._rnn(inp, last_state['rnn_state'])
        h = h.squeeze(0)  # remove time dim
        z_prior, z_smpl = self.build_z_prior(h, sample)

        return h, {'z': z_smpl, 'z_prior': z_prior, 'z_post': None, 'rnn_state': next_rnn_state}

    def observe(self,
                a: torch.Tensor,
                o_current: torch.Tensor,
                r_current: torch.Tensor,
                term_current: torch.Tensor,
                last_state: Dict[str, torch.Tensor],
                context: Optional[torch.Tensor] = None,
                sample: bool = True):
        h, next_state = self.imagine(a, last_state, context, sample)
        x_current_groundtruth = torch.concat([self.o_encoder(o_current), r_current, term_current], dim=-1)
        z_post, z_smpl = self.build_z_post(h, next_state['z_prior'], x_current_groundtruth, sample)

        next_state['z'] = z_smpl  # overwrite with posterior sample
        next_state['z_post'] = z_post
        return h, next_state

    def forward(self,
                a: torch.Tensor,
                o_current: torch.Tensor | None = None,
                r_current: torch.Tensor | None = None,
                term_current: torch.Tensor | None = None,
                last_state: Dict[str, torch.Tensor] | None = None,
                context: Optional[torch.Tensor] = None,
                use_posterior: bool = True,
                reconstruct: bool = True,
                sample_state: bool = True,
                sample_output: bool = True):
        if o_current is None and last_state is None:
            raise ValueError('Need at least (o_current, r_current, term_current) or last_state')
        if o_current is None and use_posterior:
            raise ValueError('Can\'t use posterior if no ground truth data is provided')

        # compute next world state
        if use_posterior:
            h, next_state = self.observe(a, o_current, r_current, term_current, last_state, context, sample_state)
        else:
            h, next_state = self.imagine(a, last_state, context, sample_state)

        # predict outputs
        s = torch.concat([h, next_state['z']], dim=-1)
        r_dist, r_smpl = self.r_decoder(s, sample_output)
        term_dist, term_smpl = self.term_decoder(s, sample_output)
        if reconstruct:
            o_dist, o_smpl = self.o_decoder(s, sample_output)
        else:
            o_dist, o_smpl = None, None

        reconstruction = {'o': o_smpl, 'o_dist': o_dist, 'a': a, 'r_dist': r_dist, 'r': r_smpl,
                          'terminal_dist': term_dist, 'terminal': term_smpl, 's': s, 'h': h}

        return reconstruction, next_state

    def build_z_prior(self,
                      h: torch.Tensor,
                      sample: bool = True) -> [torch.distributions.Distribution, torch.Tensor]:
        z_prior_params = self._z_prior(h)
        if self.latent_dist == 'normal':
            mu, logvar = torch.tensor_split(z_prior_params, 2, dim=-1)
            sigma = torch.exp(0.5 * logvar) + self.epsilon
            z_prior = torch.distributions.Normal(loc=mu, scale=sigma)
            z_smpl = z_prior.rsample() if sample else mu
        elif self.latent_dist == 'bernoulli':
            z_prior = torch.distributions.ContinuousBernoulli(logits=z_prior_params)
            z_smpl = z_prior.rsample() if sample else z_prior.probs
        else:  # categorical
            z_prior_params = z_prior_params.reshape((z_prior_params.shape[0], self.d_z, self.n_latent_categories))
            z_prior = torch.distributions.OneHotCategorical(logits=z_prior_params)
            probs = torch.nn.functional.softmax(z_prior.probs, dim=-1)
            if sample:
                z_smpl = z_prior.sample() + probs - probs.detach()
            else:
                z_smpl = probs
        return z_prior, z_smpl

    def build_z_post(self,
                     h: torch.Tensor,
                     z_prior: torch.distributions.Distribution,
                     x_posterior: torch.Tensor,
                     sample: bool = True) -> [torch.distributions.Distribution, torch.Tensor]:
        z_prior_params = torch.concat(get_dist_params(z_prior), dim=-1)
        z_post_inp = torch.concat([h, z_prior_params, x_posterior], dim=-1)
        z_post_params = self._z_post(z_post_inp)
        if self.latent_dist == 'normal':
            mu, logvar = torch.tensor_split(z_post_params, 2, dim=-1)
            sigma = torch.exp(0.5 * logvar) + self.epsilon
            z_post = torch.distributions.Normal(loc=mu, scale=sigma)
            z_smpl = z_post.rsample() if sample else mu
        elif self.latent_dist == 'bernoulli':
            z_post = torch.distributions.ContinuousBernoulli(logits=z_post_params)
            z_smpl = z_post.rsample() if sample else z_prior.probs
        else:  # categorical
            z_post_params = z_post_params.reshape((z_post_params.shape[0], self.d_z, self.n_latent_categories))
            z_post = torch.distributions.OneHotCategorical(logits=z_post_params)
            probs = torch.nn.functional.softmax(z_post.probs, dim=-1)
            if sample:
                z_smpl = z_post.sample() + probs - probs.detach()
            else:
                z_smpl = probs
        return z_post, z_smpl


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
                 ema_coeff: float = 0.0):
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
            mask_and_action_filters = {'mask': MinUpwardsFilter(window_size), 'a': AvgUpwardsFilter(window_size)}
            level.update(mask_and_action_filters)
        upwards_filters = [ModuleDict(lvl_0_filters)] + [ModuleDict(x) for x in upwards_filters]

        # generate EMA shadow copies of the RSSMs
        self._ema_rssm_modules = ModuleList([copy.deepcopy(m) for m in rssm_modules])
        for i_mod, ema_mod in enumerate(self._ema_rssm_modules):  # disable gradient computation in ema shadow copies
            for i_param, param in enumerate(ema_mod.parameters()):
                param.detach_()

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

    @property
    def levels(self) -> int:
        return len(self.rssm_modules)

    @property
    def strides(self) -> List[int]:
        return [filters['o'].window_size for filters in self.upwards_filters]

    def forward_old(self, o, a, r, terminal, n_warmup: int = -1, level: int = 0,
                    memory: Optional[dict] = None, start_state: Optional[dict] = None, sample_state: bool = True,
                    sample_output: bool = True, reconstruct: bool = True, use_ema_modules: bool = False):
        assert o.shape[0] == r.shape[0] == terminal.shape[0]

        device = self.device
        mdl = self._ema_rssm_modules[level] if use_ema_modules else self.rssm_modules[level]
        n_pred_steps, d_batch = a.shape[:2]
        n_groundtruth_steps = o.shape[0]
        mem = {} if memory is None else memory
        state = mdl.init_state(d_batch, device) if start_state is None else start_state
        n_warmup = n_warmup if n_warmup >= 0 else n_pred_steps

        for t, a_t in enumerate(a):
            if t < n_groundtruth_steps and t < n_warmup:
                o_t, r_t, term_t = o[t], r[t], terminal[t]
                use_posterior = True
            else:
                o_t, r_t, term_t = None, None, None
                use_posterior = False

            pred, state = mdl(a=a_t, o_current=o_t, r_current=r_t, term_current=term_t, last_state=state,
                              use_posterior=use_posterior, sample_state=sample_state, sample_output=sample_output,
                              reconstruct=reconstruct)

            for k, v in {**pred, **state}.items():
                data = mem.get(k, [])
                data.append(v)
                mem[k] = data

            # store a as well for the record
            actions = mem.get('a', [])
            actions.append(a_t)
            mem['a'] = actions

        return mem, state

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
            pred, state = mdl(a=a_t, o_current=o_t, r_current=r_t, term_current=term_t, last_state=state,
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

    def simulate(self,
                 n_steps: int,
                 before_history: Dict[str, torch.Tensor],
                 start_state: Dict[str, torch.Tensor],
                 agent_goal: torch.Tensor | None = None,
                 level: int = 0,
                 agent_memory: Dict[str, torch.Tensor] | None = None,
                 sample_state: bool = True,
                 sample_output: bool = True,
                 reconstruct: bool = True,
                 use_ema_modules: bool = False):
        if n_steps == 0:
            return before_history, start_state, {}

        mdl = self._ema_rssm_modules[level] if use_ema_modules else self.rssm_modules[level]
        mdl_other = self.rssm_modules[level] if use_ema_modules else self._ema_rssm_modules[level]
        agent_memory = {} if agent_memory is None else agent_memory
        state = start_state

        # determine whether we need goal-augmented observations or not
        if agent_goal is None:
            agent = self.r_max_agents[level][0]

            def process_o(_step):
                return _step[agent.observation_key]

            def process_r(_step):
                return _step['r']
        else:
            agent = self.goal_seeking_agents[level][0]

            def process_o(_step):
                # TODO: should agent_goal be detached?
                return torch.concat([_step[agent.observation_key], agent_goal], dim=-1)

            def process_r(_step):
                # TODO: should agent_goal be detached?
                return agent.goal_similarity(_step[agent.observation_key], agent_goal)

        def agent_decide(_step):
            agent_o = process_o(_step)
            a_dist, a, v = agent(agent_o)
            ema_a_dist, ema_a, ema_v = agent(agent_o, use_ema_modules=True)
            return {'a_dist': a_dist, 'a': a, 'v': v, 'ema_a_dist': ema_a_dist, 'ema_a': ema_a, 'ema_v': ema_v}

        # start simulation, first agent observation is the last one from before_history
        current = {agent.observation_key: before_history[agent.observation_key][-1].detach()}
        for t in range(n_steps):
            pred_agent = agent_decide(current)
            current, next_state = mdl(a=pred_agent['a'], last_state=state, use_posterior=False,
                                      sample_state=sample_state, sample_output=sample_output, reconstruct=reconstruct)
            current.update(next_state)
            # this is a hack, other model gets for every step last_state of normal model, so it can't deviate a lot
            current_other, next_state_other = mdl_other(a=pred_agent['a'], last_state=state, use_posterior=False,
                                                        sample_state=sample_state, sample_output=sample_output,
                                                        reconstruct=reconstruct)
            current_other.update(next_state_other)

            # add missing quantities to agent memory
            novelty = kl_divergence(detach_dist(current['z_prior']), current_other['z_prior']).mean(dim=-1)
            pred_agent['model_novelty'] = novelty
            pred_agent['r'] = process_r(current)
            pred_agent['terminal'] = current['terminal']

            for k, v in current.items():
                data = before_history.get(k, [])
                data.append(v)
                before_history[k] = data

            for k, v in pred_agent.items():
                data = agent_memory.get(k, [])
                data.append(v)
                agent_memory[k] = data

            state = next_state

        return before_history, state, agent_memory

    def forward2(self, o, a, r, terminal, n_warmup: int = -1, n_agent_steps: int = 0, level: int = 0,
                 memory: Optional[dict] = None, start_state: Optional[dict] = None, sample_state: bool = True,
                 sample_output: bool = True, reconstruct: bool = True, use_ema_modules: bool = False,
                 agent_goal: torch.Tensor = None):

        # TODO: This version of forward should delegate to observe(), imagine() and simulate() and just orchestrate them
        pass

    def forward(self, o, a, r, terminal, n_warmup: int = -1, n_agent_steps: int = 0, level: int = 0,
                memory: Optional[dict] = None, start_state: Optional[dict] = None, sample_state: bool = True,
                sample_output: bool = True, reconstruct: bool = True, use_ema_modules: bool = False,
                agent_goal: torch.Tensor = None):
        """
        Implements core functionality of this class. Can continue earlier calls if a start state is provided.
        The length of the rollout is controlled by the amount of action time steps and :n_agent_steps:.
        The final rollout length is the amount of provided actions plus the requested :n_agent_steps: on top. After
        all predefined actions are used up, the internal agent is used for :n_agent_steps:. If ground truth
        data is provided for a time step, and this time step is within the warmup period, the ground truth data is used.
        :param o: observations to warm up world model
        :param a: predefined actions for the rollout
        :param r: rewards to warm up world model
        :param terminal: terminal flags to warm up model
        :param n_warmup: amount of steps where the warm up data is actually used (can be less steps than data is there)
        :param n_agent_steps: amount of steps where actions are chosen by the internal agent, those are executed after
        all predefined actions are used up
        :param level: world model hierarchy index
        :param memory: optional memory to use for storing generated rollout data, a blank one is created otherwise
        :param start_state: optional start state for the RSSMCell to continue a previous rollout
        :param sample_state: toggles sampling of z in RSSMCell
        :param sample_output: toggles sampling of o, r, terminal predictions in RSSMCell
        :param reconstruct: toggles generation of o in RSSMCell
        :param use_ema_modules: toggles whether the live (trained by gradient descent) or EMA RSSMCell should be used
        :param agent_goal: optional goal to determine whether the actions for :n_agent_steps: come from the reward
        maximizing or the goal maximizing agent, in case of the latter this is the goal the agent should achieve
        during this rollout
        :return: a memory with all the world model related generated data
        """

        assert o.shape[0] == r.shape[0] == terminal.shape[0]

        device = self.device
        mdl = self._ema_rssm_modules[level] if use_ema_modules else self.rssm_modules[level]
        n_predefined_actions, d_batch = a.shape[:2]
        n_groundtruth_steps = o.shape[0]
        mem = {} if memory is None else memory
        state = mdl.init_state(d_batch, device) if start_state is None else start_state
        n_warmup = n_warmup if n_warmup >= 0 else n_predefined_actions
        total_steps = n_predefined_actions + n_agent_steps

        assert n_warmup <= n_predefined_actions

        # configure correct agent for action making
        if n_agent_steps > 0:
            if agent_goal is None:
                def get_a(mem):
                    agent_o = mem[self.r_max_agents[level][0].observation_key][-1]
                    a_dist, a, v = self.r_max_agents[level][0](agent_o)
                    return a
            else:
                def get_a(mem):
                    agent_o = mem[self.goal_seeking_agents[level][0].observation_key][-1]
                    a_dist, a, v = self.goal_seeking_agents[level][0](torch.concat([agent_o, agent_goal], dim=-1))
                    return a

        # perform simulation
        for t in range(total_steps):
            if t < n_groundtruth_steps and t < n_warmup:
                o_t, r_t, term_t = o[t], r[t], terminal[t]
                use_posterior = True
            else:
                o_t, r_t, term_t = None, None, None
                use_posterior = False

            a_t = a[t] if t < n_predefined_actions else get_a(mem)

            pred, state = mdl(a=a_t, o_current=o_t, r_current=r_t, term_current=term_t, last_state=state,
                              use_posterior=use_posterior, sample_state=sample_state, sample_output=sample_output,
                              reconstruct=reconstruct)

            for k, v in {**pred, **state}.items():
                data = mem.get(k, [])
                data.append(v)
                mem[k] = data

        return mem, state

    def forward_all_hierarchies_old(self,
                                    training_data: Dict[str, torch.Tensor],
                                    warmup_steps: List[int]):
        pred = []
        pred_ema = []
        targets = []
        inp_lvl = {k: v for k, v in training_data.items() if k in ('o', 'a', 'r', 'terminal')}  # ground truth data lvl0
        for i_lvl, (filters, link, n_warmup) in enumerate(zip(self.upwards_filters, self.links, warmup_steps)):
            # prep current lvl input
            filtered_inp_level = {k: filters[k](inp_lvl[k]) for k in inp_lvl}
            # if f'a_{i_lvl}' in training_data:
            #    filtered_inp_level['a'] = training_data[f'a_{i_lvl}']

            n_warmup = random.randint(1, filtered_inp_level['o'].shape[0]) if n_warmup == 'rand' else n_warmup
            # do prediction
            mem, _ = self(**filtered_inp_level, n_warmup=n_warmup, level=i_lvl, sample_state=True,
                          sample_output=True, reconstruct=True)
            if self.ema_regularization:
                mem_ema, _ = self(**filtered_inp_level, n_warmup=n_warmup, level=i_lvl, sample_state=True,
                                  sample_output=True, reconstruct=True, use_ema_modules=True)
            else:
                mem_ema = None
            # prep next lvl input
            # TODO: just a test, remove again later
            # inp_lvl = {'o': torch.stack(mem[link]), 'a': filtered_inp_level['a'], 'r': torch.stack(mem['r']),
            #           'terminal': torch.stack(mem['terminal'])}
            inp_lvl = {'o': torch.stack(mem[link]).detach(), 'a': filtered_inp_level['a'], 'r': filtered_inp_level['r'],
                       'terminal': filtered_inp_level['terminal']}
            # inp_lvl = {'o': filtered_inp_level['o'], 'a': filtered_inp_level['a'], 'r': filtered_inp_level['r'],
            #           'terminal': filtered_inp_level['terminal']}

            pred.append(mem)
            pred_ema.append(mem_ema)
            targets.append(filtered_inp_level)
        return pred, pred_ema, targets

    def forward_all_hierarchies(self,
                                training_data: Dict[str, torch.Tensor],
                                warmup_steps: Sequence[int],
                                agent_steps: Sequence[int],
                                start_state_lvl_0: Dict[str, torch.Tensor] | None = None):
        pred, pred_ema, targets, r_max_agents, goal_seeking_agents, states = [], [], [], [], [], []
        next_input = {k: v for k, v in training_data.items() if k in ('o', 'a', 'r', 'terminal')}  # lvl 0 input
        for i_lvl, (filters, n_wu, n_as) in enumerate(zip(self.upwards_filters, warmup_steps, agent_steps)):
            # preparations
            if i_lvl == 0:
                if n_wu == 'rand':
                    n_wu = random.randint(1, next_input['o'].shape[0])
                elif n_wu == -1:
                    n_wu = next_input['a'].shape[0]
                start_state = start_state_lvl_0
            else:  # for all abstract levels, take only init o, r, term and zero action per definition
                next_input = {k: filters[k](v) for k, v in next_input.items()}
                next_input['a'] = torch.zeros_like(next_input['a'])
                n_wu = 1
                start_state = None

            # do prediction
            wu_inp = {k: v[:n_wu] for k, v in next_input.items()}
            mem, state = self.observe(**wu_inp, level=i_lvl, start_state=start_state)
            mem, state = self.imagine(next_input['a'][n_wu:], start_state=state, level=i_lvl, memory=mem)
            mem, state, r_max_agent_mem = self.simulate(n_steps=n_as, before_history=mem, start_state=state,
                                                        level=i_lvl)
            if self.ema_regularization:
                mem_ema, init_state_ema = self.observe(**wu_inp, level=i_lvl, use_ema_modules=True)
                mem_ema, _ = self.imagine(next_input['a'][n_wu:], start_state=init_state_ema, level=i_lvl,
                                          memory=mem_ema, use_ema_modules=True)
                mem_ema, _, _ = self.simulate(n_steps=n_as, before_history=mem_ema, start_state=init_state_ema,
                                              level=i_lvl, use_ema_modules=True)
            else:
                mem_ema = None

            # store targets for this level
            if i_lvl == 0:  # targets are groundtruth data from real env
                targets.append(next_input)
            else:  # targets have to be computed with goal seeking agent on lower level model
                simulated_ground_truth, goal_seeking_agent_mem = self.compute_targets(i_lvl, mem, pred[i_lvl - 1])
                simulated_ground_truth = {'o': torch.stack(simulated_ground_truth[self.links[i_lvl - 1]]),
                                          'r': torch.stack(simulated_ground_truth['r']),
                                          'terminal': torch.stack(simulated_ground_truth['terminal'])}
                simulated_ground_truth = {k: filters[k](v) for k, v in simulated_ground_truth.items()}
                simulated_ground_truth = {k: v.detach() for k, v in simulated_ground_truth.items()}
                targets.append(simulated_ground_truth)
                goal_seeking_agents.append(goal_seeking_agent_mem)

            # choose inputs for next level
            next_input = {'o': torch.stack(mem[self.links[i_lvl]]), 'a': torch.stack(mem['a']),
                          'r': torch.stack(mem['r']), 'terminal': torch.stack(mem['terminal'])}
            next_input = {k: v.detach() for k, v in next_input.items()}  # prevent gradient flow from higher to lower

            pred.append(mem)
            pred_ema.append(mem_ema)
            r_max_agents.append(r_max_agent_mem)
            states.append(state)
        return pred, pred_ema, targets, r_max_agents, goal_seeking_agents, states

    def compute_targets(self,
                        lvl_current: int,
                        r_max_mem_current: Dict[str, torch.Tensor],
                        r_max_mem_below: Dict[str, torch.Tensor]):
        lvl_below = lvl_current - 1
        chunk_size = self.strides[lvl_current]
        init_mem = {k: torch.stack(r_max_mem_below[k][:chunk_size]) for k in ('o', 'a', 'r', 'terminal')}  # first chunk
        mem_below, state_below = self.observe(**init_mem, level=lvl_below)
        agent_mem = {}
        for intermediate_goal in r_max_mem_current['o'][1:]:
            mem_below, state_below, agent_mem = self.simulate(n_steps=chunk_size, before_history=mem_below,
                                                              start_state=state_below, agent_goal=intermediate_goal,
                                                              level=lvl_below, agent_memory=agent_mem)
        return mem_below, agent_mem

    def _train_step(self,
                    training_data: Dict[str, torch.Tensor],
                    optimizer: torch.optim.Optimizer,
                    **kwargs) -> Dict[str, torch.Tensor]:
        optimizer.zero_grad(set_to_none=True)
        losses_tf = self.eval_step(training_data, force_warmup=[-1 for _ in self.rssm_modules])
        losses_one = self.eval_step(training_data, force_warmup=[1 for _ in self.rssm_modules])
        losses_wu = self.eval_step(training_data)

        losses, r_max_agent_losses, goal_seeking_agent_losses = {}, {}, {}
        if False:  # disable agent training inside model for now
            # if kwargs.get('train_agents', False):  # either train agents...
            if random.random() < 0.5:  # either r_max agents
                for i_lvl, agent_losses in enumerate(losses_tf['r_max_agents']):  # TODO: should I use losses_tf here?
                    agent, act_opt, crit_opt = self.r_max_agents[i_lvl]
                    agent.update_step(agent_losses, act_opt, crit_opt)
                    loss_level = {k + f'_{i_lvl}': v for k, v in agent_losses.items()}
                    r_max_agent_losses.update(loss_level)
            else:  # or goal_seeking agents
                for i_lvl, agent_losses in enumerate(
                        losses_tf['goal_seeking_agents']):  # TODO: should I use losses_tf here?
                    agent, act_opt, crit_opt = self.goal_seeking_agents[i_lvl]
                    agent.update_step(agent_losses, act_opt, crit_opt)
                    loss_level = {k + f'_{i_lvl}': v for k, v in agent_losses.items()}
                    goal_seeking_agent_losses.update(loss_level)
        else:  # ... or train model
            for k in losses_tf['model']:
                losses[k] = (losses_tf['model'][k] + losses_wu['model'][k] + losses_one['model'][k]) / 3
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
            # update ema modules
            rssm_params = [OrderedDict(m.named_parameters()) for m in self.rssm_modules]
            ema_params = [OrderedDict(m.named_parameters()) for m in self._ema_rssm_modules]
            update_ema_modules(rssm_params, ema_params, self.ema_coeff)
            r_max_agent_losses, goal_seeking_agent_losses = {}, {}

        return {'model': losses, 'r_max_agents': r_max_agent_losses, 'goal_seeking_agents': goal_seeking_agent_losses}

    def _eval_step(self,
                   training_data: Dict[str, torch.Tensor],
                   **kwargs):
        warmup_steps = kwargs.get('force_warmup', self.warmup_steps)
        agent_steps = [0, 10, 5]  # TODO: this is arbitrary and only for testing
        pred, pred_ema, targets, _, _, _ = self.forward_all_hierarchies(training_data, warmup_steps, agent_steps)

        # model losses
        losses = {}
        mask_lvl = training_data['mask']
        for i_lvl in range(self.levels):
            if i_lvl == 0:  # TODO: hack, make this nicer
                mask_lvl = self.upwards_filters[i_lvl]['mask'](mask_lvl)
            else:
                mask_lvl = torch.zeros_like(torch.stack(pred[i_lvl]['r']))
            loss_level = self.calc_loss(pred[i_lvl], pred_ema[i_lvl], targets[i_lvl], mask_lvl, self.kl_betas[i_lvl],
                                        self.kl_reg_betas[i_lvl])
            loss_level = {k + f'_{i_lvl}': v for k, v in loss_level.items()}
            losses.update(loss_level)
        losses['total'] = torch.stack([v for k, v in losses.items() if k.startswith('total')]).mean()

        # agent losses
        r_max_agent_losses = {}
        # for i_lvl in range(1, self.levels):
        #    r_max_loss_level = self.r_max_agents[i_lvl][0].eval_step(**r_max_agent_mem[i_lvl])
        #    r_max_agent_losses.append(r_max_loss_level)
        goal_seeking_agent_losses = {}
        # for i_lvl in range(self.levels - 1):
        #    goal_seeking_loss_level = self.goal_seeking_agents[i_lvl][0].eval_step(**goal_seeking_agent_mem[i_lvl])
        #    goal_seeking_agent_losses.append(goal_seeking_loss_level)

        return {'model': losses, 'r_max_agents': r_max_agent_losses, 'goal_seeking_agents': goal_seeking_agent_losses}

    def calc_loss(self,
                  pred: Dict[str, List[Union[torch.Tensor, torch.distributions.Distribution]]],
                  pred_ema: Dict[str, List[Union[torch.Tensor, torch.distributions.Distribution]]],
                  targets: Dict[str, torch.Tensor],
                  mask: torch.Tensor,
                  kl_beta: float,
                  kl_reg_beta: float = 0.0):
        mask = 1 - mask  # use mask to multiply irrelevant steps with zero
        rec_o = self._neg_log_prob(pred['o_dist'], targets['o'], mask)
        rec_r = self._neg_log_prob(pred['r_dist'], targets['r'], mask)
        rec_term = self._neg_log_prob(pred['terminal_dist'], targets['terminal'], mask)
        kl_z = self._kl_div(pred['z_post'], pred['z_prior'], mask)
        kl_reg_z = self._kl_reg(pred['z_post'], mask)
        contrastive_z = self.ema_regularization * self._contrastive_loss(pred['z'], mask)

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
    def _contrastive_loss(x: List[torch.Tensor],
                          mask: torch.Tensor):
        x = torch.stack(x)
        d_time, d_batch = x.shape[:2]
        # t_offset = random.randint(1, d_time - 1)
        # b_offset = random.randint(1, d_batch - 1)
        t_offset = 1
        b_offset = 0
        mask = expand_shape_right(mask, x)
        diff = x - x.roll(shifts=[t_offset, b_offset], dims=[0, 1])
        cont_loss = torch.mean(torch.maximum(torch.tensor(0.0, device=x.device), 1.0 - (diff ** 2) * mask))
        return cont_loss

    @staticmethod
    def _neg_log_prob(distributions: List[torch.distributions.Distribution],
                      x_target: torch.Tensor,
                      mask: torch.Tensor):
        if isinstance(distributions[0], torch.distributions.RelaxedOneHotCategorical):
            # smooth out targets a bit to avoid inf/nan log probs with RelaxedOneHotCategorical
            x_target = torch.abs(x_target - 1e-5)
            x_target /= x_target.sum(dim=-1, keepdim=True)
        mask = expand_shape_right(mask, x_target)
        neg_log_prob = [-d.log_prob(x) * m for d, x, m in zip(distributions, x_target, mask)]
        neg_log_prob = torch.stack(neg_log_prob, dim=0).mean()
        return neg_log_prob

    @staticmethod
    def _kl_div(ps: List[torch.distributions.Distribution],
                qs: List[torch.distributions.Distribution],
                mask: torch.Tensor,
                detach_ps: bool = False,
                detach_qs: bool = False):
        if detach_ps:
            ps = [detach_dist(p) if p is not None else None for p in ps]
        if detach_qs:
            qs = [detach_dist(q) if q is not None else None for q in qs]
        kl = [torch.distributions.kl_divergence(p, q) * m for p, q, m in zip(ps, qs, mask) if None not in (p, q)]
        kl = torch.stack(kl, dim=0).mean()
        # if len(kl) > 1:
        #    kl = kl[1:].mean()  # ignore first prior since it's totally uninformed
        # else:
        #    kl = torch.tensor(0, dtype=torch.float32)
        return kl

    @staticmethod
    def _kl_reg(ps: List[torch.distributions.Distribution],
                mask: torch.Tensor):
        if isinstance(ps[0], torch.distributions.Normal):
            reg_dist = torch.distributions.Normal(loc=torch.zeros_like(ps[0].loc), scale=torch.ones_like(ps[0].scale))
        elif isinstance(ps[0], torch.distributions.ContinuousBernoulli):
            reg_dist = torch.distributions.ContinuousBernoulli(probs=torch.full_like(ps[0].logits, 0.5))
        elif isinstance(ps[0], torch.distributions.OneHotCategorical):
            reg_dist = torch.distributions.OneHotCategorical(logits=torch.ones_like(ps[0].logits))
        else:
            raise ValueError(f'No regularization distribution for distribution {ps[0]} found')

        kl_reg = [torch.distributions.kl_divergence(p, reg_dist) * m for p, m in zip(ps, mask) if p is not None]
        kl_reg = torch.stack(kl_reg, dim=0).mean()
        return kl_reg

    @staticmethod
    def _mae(y_hats: List[torch.Tensor], ys: torch.Tensor, mask: torch.Tensor):
        y_hats = torch.stack(y_hats, dim=0)
        mask = mask.reshape(*mask.shape + (1,) * (y_hats.ndim - mask.ndim))  # append size 1 dimensions for broadcasting
        return torch.mean(torch.abs(ys - y_hats) * mask)
