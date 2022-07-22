from typing import Dict, Optional
import random

import torch

from mdm.models.building_blocks import AbstractActionModel, RSSM, RnnStateType
from mdm.utils.torch_tools import *
from mdm.models.dynamics_model import DynamicsModel


class MultiscaleDynamicsModelMK2(DynamicsModel, FuzzyDeviceMixin):

    def __init__(self,
                 primitive_model: RSSM,
                 abstract_model: RSSM,
                 abstract_action_model: AbstractActionModel,
                 abstract_step_size: int,
                 beta_kl_primitive: float = 0.01,
                 beta_reg_primitive: float = 0.001,
                 beta_kl_abstract: float = 0.01,
                 beta_reg_abstract: float = 0.001,
                 beta_abstract_model: float = 1.0,
                 beta_abstract_action: float = 1.0,
                 n_warmup_prim: Union[int, Sequence[int]] = 1,
                 n_warmup_abstr: Union[int, Sequence[int]] = 1,
                 detach_posteriors: bool = False):
        super().__init__()
        self.primitive_model = primitive_model
        self.abstract_model = abstract_model
        self.abstract_action_model = abstract_action_model
        self.abstract_step_size = abstract_step_size
        self.d_state = primitive_model.d_state
        self.d_action = primitive_model.d_action
        self.d_reward = primitive_model.d_reward
        self.d_observation = primitive_model.d_observation
        self.d_abstract_state = abstract_model.d_state
        self.d_abstract_action = abstract_model.d_action
        self.d_abstract_reward = abstract_model.d_reward
        self.beta_kl_prim = beta_kl_primitive
        self.beta_reg_prim = beta_reg_primitive
        self.beta_kl_abstr = beta_kl_abstract
        self.beta_reg_abstr = beta_reg_abstract
        self.beta_abstract_model = beta_abstract_model
        self.beta_abstract_action = beta_abstract_action
        self.n_warmup_prim = n_warmup_prim
        self.n_warmup_abstr = n_warmup_abstr
        self.detach_posteriors = detach_posteriors
        self.context_projector = torch.nn.Sequential(
            torch.nn.Linear(primitive_model.d_hidden, 2),
        )

        if isinstance(n_warmup_prim, int):
            assert self.abstract_step_size >= n_warmup_prim >= 1
        else:
            assert n_warmup_prim[0] < n_warmup_prim[1]
            assert 1 <= n_warmup_prim[0] and n_warmup_prim[1] <= self.abstract_step_size

        if isinstance(n_warmup_abstr, int):
            assert n_warmup_abstr >= 1
        else:
            assert n_warmup_abstr[0] < n_warmup_abstr[1]
            assert 1 <= n_warmup_abstr[0]

    def _warmup_prim(self) -> int:
        if isinstance(self.n_warmup_prim, int):
            return random.randint(1, self.n_warmup_prim)
        else:
            return random.randint(self.n_warmup_prim[0], self.n_warmup_prim[1])

    def _warmup_abstr(self) -> int:
        if isinstance(self.n_warmup_abstr, int):
            return random.randint(1, self.n_warmup_abstr)
        else:
            return random.randint(self.n_warmup_abstr[0], self.n_warmup_abstr[1])

    def forward(self,
                o: torch.Tensor,
                a: torch.Tensor,
                r: torch.Tensor,
                term: torch.Tensor,
                abstr_r: torch.Tensor,
                abstr_term: torch.Tensor):
        assert o.shape[1] == r.shape[1] == term.shape[1] == a.shape[1]
        assert torch.all(r[:, 0] == 0)
        assert torch.all(term[:, 0] == 0)
        assert torch.all(a[:, 0] == 0)

        mem = self.gen_mem()
        d_batch, n_steps_prim = a.shape[:2]
        a_binned = bin_every_k_steps(a, self.abstract_step_size, self.device, padding_val=0)

        prim_current = self.primitive_model.gen_init_values(d_batch, self.device)
        abstr_current = self.abstract_model.gen_init_values(d_batch, self.device)

        n_warmup_prim = self._warmup_prim()
        n_warmup_abstr = self._warmup_abstr()
        for i_chunk in range(a_binned.shape[1]):
            i_start = i_chunk * self.abstract_step_size
            i_end = min((i_chunk + 1) * self.abstract_step_size, n_steps_prim)

            # primitive model rollout
            ctx_high_level = self.primitive_model.zero_ctx_high_level(d_batch, self.device)
            mem, prim_current = self.rollout_primitive(a=a[:, i_start: i_end], o=o[:, i_start: i_end],
                                                       r=r[:, i_start: i_end], term=term[:, i_start: i_end],
                                                       s=prim_current['s'], rnn_state=prim_current['rnn_state'],
                                                       ctx_high_level=ctx_high_level,
                                                       n_posterior_steps=n_warmup_prim,
                                                       mem=mem, sample=True)

            # input to abstr act mdl needs to be always of same length, so take a_binned instead of a[i_start: i_end]
            abstr_a = self.abstract_action_model(a_binned[:, i_chunk])
            mem['abstr_a'].append(abstr_a)

            # abstract model rollout
            ctx_low_level = self.fuse_state(prim_current['s'], prim_current['rnn_state'])#.detach()
            mem['ctx_low_level'].append(ctx_low_level)

            mem, abstr_current = self.rollout_abstract(a=add_time_dim(abstr_a), r=add_time_dim(abstr_r[:, i_chunk]),
                                                       term=add_time_dim(abstr_term[:, i_chunk]),
                                                       primitive_history=add_time_dim(ctx_low_level),
                                                       s=abstr_current['s'],
                                                       rnn_state=abstr_current['rnn_state'],
                                                       n_posterior_steps=1 if n_warmup_abstr > 0 else 0,
                                                       mem=mem, sample=True)

            # exchange primitive model's internal state with prediction from abstract model
            #prim_s, prim_rnn_state = self.unfuse_state(abstr_current['o'])
            #prim_current['s'] = prim_s
            #prim_current['rnn_state'] = prim_rnn_state

            # warmup should only happen at first sequence chunk
            n_warmup_prim = 0
            n_warmup_abstr = max(n_warmup_abstr - 1, 0)

        mem = self.pack_mem(mem)

        return mem

    def _invoke_primitive_model(self,
                                prim_current: Dict,
                                x_post: Optional[torch.Tensor],
                                mem: Optional[Dict],
                                ctx_high_level: torch.Tensor,
                                use_posterior: bool = True,
                                sample: bool = True):
        d_batch = prim_current['a'].shape[0]
        device = prim_current['a'].device

        # do prediction
        pred = self.primitive_model(s=prim_current['s'], a=prim_current['a'], x_current_groundtruth=x_post,
                                    ctx_high_level=ctx_high_level, rnn_state=prim_current['rnn_state'],
                                    use_posterior=use_posterior, sample=sample)

        # sometimes use s_prior to provide next step's s
        # d_batch = pred['s'].shape[0]
        # pred['s'] = torch.where(torch.rand(d_batch, 1, device=self.device) < 0.2, pred['s_prior'].rsample(), pred['s'])

        # update primitive state
        prim_current['s'] = pred['s']
        prim_current['rnn_state'] = pred['rnn_state']
        prim_current['o'] = pred['o']
        prim_current['r'] = pred['r']
        prim_current['term'] = pred['term']

        # store things
        mem['prim_s'].append(pred['s'])
        mem['prim_s_prior'].append(pred['s_prior'])
        if use_posterior:
            mem['prim_s_post'].append(pred['s_post'])
        else:  # hack for loss calculation
            mem['prim_s_post'].append(pred['s_prior'])
        mem['prim_rnn_state'].append(pred['rnn_state'])
        mem['prim_o'].append(pred['o'])
        mem['prim_r'].append(pred['r'])
        mem['prim_term'].append(pred['term'])

    def _invoke_abstract_model(self,
                               abstr_current: Dict,
                               x_post: torch.Tensor,
                               mem: Dict,
                               use_posterior: bool = True,
                               sample: bool = True):
        d_batch = abstr_current['a'].shape[0]
        device = abstr_current['a'].device

        ctx_high_level = self.abstract_model.zero_ctx_high_level(d_batch, device)

        # do prediction
        pred = self.abstract_model(s=abstr_current['s'], a=abstr_current['a'],
                                   x_current_groundtruth=x_post, ctx_high_level=ctx_high_level,
                                   rnn_state=abstr_current['rnn_state'], use_posterior=use_posterior, sample=sample)

        # update abstract state
        abstr_current['s'] = pred['s']
        abstr_current['rnn_state'] = pred['rnn_state']
        abstr_current['o'] = pred['o']
        abstr_current['r'] = pred['r']
        abstr_current['term'] = pred['term']

        # store things
        mem['abstr_s'].append(pred['s'])
        mem['abstr_s_prior'].append(pred['s_prior'])
        if use_posterior:
            mem['abstr_s_post'].append(pred['s_post'])
        else:  # hack for loss calculation
            mem['abstr_s_post'].append(pred['s_prior'])
        mem['abstr_rnn_state'].append(pred['rnn_state'])
        mem['abstr_o'].append(pred['o'])
        mem['abstr_o_dist'].append(pred['o_dist'])
        mem['abstr_r'].append(pred['r'])
        mem['abstr_term'].append(pred['term'])

    @staticmethod
    def gen_mem():
        mem = {'prim_o': [], 'prim_r': [], 'prim_term': [], 'prim_s_prior': [], 'prim_s_post': [], 'prim_s': [],
               'prim_rnn_state': [], 'abstr_o': [], 'abstr_o_dist': [], 'abstr_a': [], 'abstr_r': [], 'abstr_term': [],
               'abstr_s_prior': [], 'abstr_s_post': [], 'abstr_s': [], 'abstr_rnn_state': [], 'ctx_low_level': []}
        return mem

    @staticmethod
    def pack_mem(mem):
        for k, v in mem.items():
            if isinstance(v, List) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                mem[k] = torch.stack(mem[k], dim=1)
        return mem

    def train_step(self,
                   o_ground_truth: torch.Tensor,
                   a_ground_truth: torch.Tensor,
                   r_ground_truth: torch.Tensor,
                   term_ground_truth: torch.Tensor,
                   optimizer: torch.optim.Optimizer) -> Dict[str, torch.Tensor]:
        optimizer.zero_grad(set_to_none=True)
        losses = self.eval_step(o_ground_truth, a_ground_truth, r_ground_truth, term_ground_truth)
        losses['total'].backward()
        optimizer.step()

        return losses

    def eval_step(self,
                  o_ground_truth: torch.Tensor,
                  a_ground_truth: torch.Tensor,
                  r_ground_truth: torch.Tensor,
                  term_ground_truth: torch.Tensor) -> Dict[str, torch.Tensor]:
        device = self._device

        # sum makes sense for rewards, use padding=0 to not affect sum for last element
        # start at time step 1 because the first time step is just the initial observation to start prediction
        abstr_r_ground_truth = bin_every_k_steps(r_ground_truth, self.abstract_step_size, device=device,
                                                 padding_val=0).sum(dim=2)
        # terminal flag can only be 0 or 1, so mean value with automatic padding should be used
        # start at time step 1 because the first time step is just the initial observation to start prediction
        abstr_term_ground_truth = bin_every_k_steps(term_ground_truth, self.abstract_step_size,
                                                    device=device).max(dim=2).values

        pred = self(o_ground_truth, a_ground_truth, r_ground_truth, term_ground_truth, abstr_r_ground_truth,
                    abstr_term_ground_truth)

        # primitive model loss
        prim_rec_o = torch.nn.functional.mse_loss(pred['prim_o'], o_ground_truth)
        prim_rec_r = torch.nn.functional.mse_loss(pred['prim_r'], r_ground_truth)
        prim_rec_term = torch.nn.functional.binary_cross_entropy(pred['prim_term'], term_ground_truth)
        prim_kl_s = self.beta_kl_prim * kl_loss_normal(pred['prim_s_prior'], pred['prim_s_post'],
                                                       detach_posterior=self.detach_posteriors)
        prim_kl_s_reg = self.beta_reg_prim * kl_regularizer_normal(pred['prim_s_post'])

        # abstract model loss
        #abstr_o_processed = self.context_projector(pred['abstr_o'])
        #prim_ctx_processed = self.context_projector(pred['ctx_low_level'])
        #abstr_o_rec = torch.nn.functional.mse_loss(abstr_o_processed, prim_ctx_processed)

        #abstr_o_rec = torch.nn.functional.mse_loss(pred['abstr_o'], pred['ctx_low_level'])
        abstr_o_rec = kl_loss_normal(pred['abstr_o_dist'], pred['prim_s_post'], detach_posterior=True)
        abstr_rec_r = torch.nn.functional.mse_loss(pred['abstr_r'], abstr_r_ground_truth)
        abstr_rec_term = torch.nn.functional.binary_cross_entropy(pred['abstr_term'], abstr_term_ground_truth)
        abstr_kl_s = self.beta_kl_abstr * kl_loss_normal(pred['abstr_s_prior'], pred['abstr_s_post'],
                                                         detach_posterior=self.detach_posteriors)
        abstr_kl_s_reg = self.beta_reg_abstr * kl_regularizer_normal(pred['abstr_s_post'])

        # abstract action model loss
        scale = pred['abstr_a'].reshape(-1, self.abstract_model.d_action).std(dim=0, unbiased=False)
        abstr_a_loss = torch.distributions.Normal(loc=0.0, scale=scale + 0.001).entropy()
        abstr_a_loss = -torch.sum(torch.abs(abstr_a_loss))
        abstr_a_loss *= self.beta_abstract_action

        if pred['abstr_o'].shape[1] > 1:
            abstr_factor = self.beta_abstract_model
        else:
            abstr_factor = 0  # disable abstract model loss in case we only use the primitive level

        total = prim_rec_o + prim_rec_r + prim_rec_term + prim_kl_s + prim_kl_s_reg
        total += abstr_factor * (abstr_o_rec + abstr_rec_r + abstr_rec_term + abstr_kl_s + abstr_kl_s_reg
                                 + abstr_a_loss)

        return {'total': total, 'prim_o': prim_rec_o, 'prim_r': prim_rec_r, 'prim_term': prim_rec_term,
                'prim_kl_s': prim_kl_s, 'prim_kl_s_reg': prim_kl_s_reg, 'abstr_o': abstr_o_rec,
                'abstr_r': abstr_rec_r, 'abstr_term': abstr_rec_term, 'abstr_kl_s': abstr_kl_s,
                'abstr_kl_s_reg': abstr_kl_s_reg, 'abstr_a_var': abstr_a_loss}

    def input_compatible(self,
                         o_ground_truth: torch.Tensor,
                         a_ground_truth: torch.Tensor,
                         r_ground_truth: torch.Tensor) -> Tuple[bool, str]:
        # laziness ahead
        return True, ''

    def flatten_rnn_state(self,
                          h: RnnStateType) -> torch.Tensor:
        if self.primitive_model.rnn_type == 'lstm' and self.abstract_model.rnn_type == self.primitive_model.rnn_type:
            h = torch.concat(h, dim=0)  # concat h and c tensors of LSTM along the layer dimension, this is arbitrary
        h = torch.transpose(h, 0, 1)  # bring batch dimension to front
        h = torch.flatten(h, start_dim=1)  # fold h/c/layer dimension into d_hidden
        return h

    def reconstruct_rnn_state(self,
                              filtered_h: torch.Tensor) -> RnnStateType:
        d_batch = filtered_h.shape[0]
        if self.primitive_model.rnn_type == 'lstm':
            unfiltered = filtered_h.reshape(d_batch, 2 * self.primitive_model.n_hidden_layers,
                                            self.primitive_model.d_hidden)
            unfiltered = unfiltered.transpose(0, 1)
            unfiltered = torch.tensor_split(unfiltered, 2, dim=0)
            unfiltered = unfiltered[0].contiguous(), unfiltered[1].contiguous()
        else:
            unfiltered = filtered_h.reshape(d_batch, self.primitive_model.n_hidden_layers,
                                            self.primitive_model.d_hidden)
            unfiltered = unfiltered.transpose(0, 1)
            unfiltered = unfiltered.contiguous()
        return unfiltered

    def fuse_state(self,
                   s: torch.Tensor,
                   rnn_state: RnnStateType) -> torch.Tensor:
        #h_filtered = self.flatten_rnn_state(rnn_state)
        #fused = torch.concat([s, h_filtered], dim=-1)
        #fused = rnn_state[0][-1]
        fused = s
        return fused

    def unfuse_state(self,
                     fused_state: torch.Tensor) -> Tuple[torch.Tensor, RnnStateType]:
        if fused_state.ndim == 1:
            s = fused_state[:self.primitive_model.d_state]
            h_filtered = fused_state[self.primitive_model.d_state:].unsqueeze(0)
            h = self.reconstruct_rnn_state(h_filtered)
            h = h[0].squeeze(1), h[1].squeeze(1)
        elif fused_state.ndim == 2:
            s = fused_state[:, :self.primitive_model.d_state]
            h_filtered = fused_state[:, self.primitive_model.d_state:]
            h = self.reconstruct_rnn_state(h_filtered)
        else:
            raise ValueError('Expected tensor with maximum one batch and one data dimension')
        return s, h

    def rollout_primitive(self,
                          a: torch.Tensor,
                          o: Optional[torch.Tensor] = None,
                          r: Optional[torch.Tensor] = None,
                          term: Optional[torch.Tensor] = None,
                          s: Optional[torch.Tensor] = None,
                          rnn_state: Optional[RnnStateType] = None,
                          ctx_high_level: Optional[torch.Tensor] = None,
                          mem: Optional[Dict] = None,
                          sample: bool = True,
                          n_posterior_steps: int = -1):
        d_batch, n_steps = a.shape[:2]
        prim_current = self.primitive_model.gen_init_values(d_batch, self.device)

        if s is not None:
            prim_current['s'] = s
        if rnn_state is not None:
            prim_current['rnn_state'] = rnn_state

        if ctx_high_level is None:
            ctx_high_level = self.primitive_model.zero_ctx_high_level(d_batch, self.device)

        if mem is None:
            mem = self.gen_mem()

        if None in (o, r, term):
            n_groundtruth_available = 0
        else:
            assert o is not None and r is not None and term is not None, 'Need o, r, and term groundtruth'
            n_groundtruth_available = o.shape[1]

        # default argument means we use as much ground truth data as possible with the posterior
        if n_posterior_steps == -1:
            n_posterior_steps = n_groundtruth_available
        elif n_posterior_steps > n_groundtruth_available:
            raise ValueError(f'Can\'t perform more posterior steps as groundtruth data is available.')

        # prediction
        for t in range(n_steps):
            if t < n_groundtruth_available:
                x_posterior = torch.concat([o[:, t], r[:, t], term[:, t]], dim=-1)
            else:
                x_posterior = torch.zeros(d_batch, self.primitive_model.d_observation + self.primitive_model.d_reward
                                          + 1, device=self.device)

            use_posterior = t < n_posterior_steps
            #use_posterior = torch.rand(()) < 0.2

            prim_current['a'] = a[:, t]
            self._invoke_primitive_model(prim_current, x_posterior, mem, ctx_high_level, use_posterior=use_posterior,
                                         sample=sample)

        return mem, prim_current

    def rollout_abstract(self,
                         a: torch.Tensor,
                         primitive_history: Optional[torch.Tensor] = None,
                         r: Optional[torch.Tensor] = None,
                         term: Optional[torch.Tensor] = None,
                         s: Optional[torch.Tensor] = None,
                         rnn_state: Optional[RnnStateType] = None,
                         mem: Optional[Dict] = None,
                         sample: bool = True,
                         n_posterior_steps: int = -1):
        d_batch, n_steps = a.shape[:2]
        abstr_current = self.abstract_model.gen_init_values(d_batch, self.device)

        if s is not None:
            abstr_current['s'] = s
        if rnn_state is not None:
            abstr_current['rnn_state'] = rnn_state

        if mem is None:
            mem = self.gen_mem()

        # prediction
        if primitive_history is None:
            n_groundtruth_available = 0
        else:
            n_groundtruth_available = primitive_history.shape[1]

        # default argument means we use as much ground truth data as possible with the posterior
        if n_posterior_steps == -1:
            n_posterior_steps = n_groundtruth_available
        elif n_posterior_steps > n_groundtruth_available:
            raise ValueError(f'Can\'t perform more posterior steps as groundtruth data is available.')

        for t in range(n_steps):
            if t < n_groundtruth_available:
                x_posterior = torch.concat([primitive_history[:, t], r[:, t], term[:, t]], dim=-1)
            else:
                x_posterior = torch.zeros(d_batch, self.abstract_model.d_observation + self.abstract_model.d_reward
                                          + 1, device=self.device)

            use_posterior = t < n_posterior_steps

            abstr_current['a'] = a[:, t]
            self._invoke_abstract_model(abstr_current, x_posterior, mem, use_posterior=use_posterior, sample=sample)

        return mem, abstr_current
