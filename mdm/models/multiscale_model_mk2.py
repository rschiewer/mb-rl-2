from typing import Dict, Optional

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
        self.detach_posteriors = detach_posteriors

    def forward(self,
                o: torch.Tensor,
                a: torch.Tensor,
                r: torch.Tensor,
                term: torch.Tensor,
                abstr_r: torch.Tensor,
                abstr_term: torch.Tensor,
                n_warmup_prim: int = 1,
                n_warmup_abstr: int = 0):
        assert o.shape[1] == r.shape[1] == term.shape[1] == a.shape[1] + 1
        assert torch.all(r[:, 0] == 0)
        assert torch.all(term[:, 0] == 0)
        assert self.abstract_step_size >= n_warmup_prim >= 1

        mem = self.gen_mem()
        d_batch, n_steps_prim = a.shape[:2]
        a_binned = bin_every_k_steps(a, self.abstract_step_size, self.device, padding_val=0)

        prim_current = self.primitive_model.gen_init_values(d_batch, self.device)
        abstr_current = self.abstract_model.gen_init_values(d_batch, self.device)

        init_o = o[:, :n_warmup_prim]
        init_r = r[:, :n_warmup_prim]
        init_term = term[:, :n_warmup_prim]
        init_o_abstr = add_time_dim(self.abstract_model.zero_o(d_batch, self.device))
        init_r_abstr = add_time_dim(self.abstract_model.zero_r(d_batch, self.device))
        init_term_abstr = add_time_dim(self.abstract_model.zero_term(d_batch, self.device))

        # exclude first step trajectory data because it's never a prediction target
        o_target = o[:, 1:]
        r_target = r[:, 1:]
        term_target = term[:, 1:]

        for i_chunk in range(a_binned.shape[1]):
            use_posterior_prim = True#torch.rand((1,)) > 0.5
            use_posterior_abstr = True#torch.rand((1,)) > 0.5

            i_start = i_chunk * self.abstract_step_size
            i_end = min((i_chunk + 1) * self.abstract_step_size, n_steps_prim)


            # primitive model rollout
            ctx_high_level = self.primitive_model.zero_ctx_high_level(d_batch, self.device)
            mem, prim_current = self.rollout_primitive(a=a[:, i_start: i_end], init_o=init_o, init_r=init_r,
                                                       init_term=init_term, init_s=prim_current['s'],
                                                       init_rnn_state=prim_current['rnn_state'],
                                                       ctx_high_level=ctx_high_level,
                                                       o_target=o_target[:, i_start: i_end],
                                                       r_target=r_target[:, i_start: i_end],
                                                       term_target=term_target[:, i_start: i_end],
                                                       mem=mem, use_posterior=use_posterior_prim, sample=True)
            # update primitive init data for next chunk
            init_o = add_time_dim(prim_current['o'])
            init_r = add_time_dim(prim_current['r'])
            init_term = add_time_dim(prim_current['term'])
            #init_o = None
            #init_r = None
            #init_term = None

            # input to abstr act mdl needs to be always of same length, so take a_binned instead of a[i_start: i_end]
            abstr_a = self.abstract_action_model(a_binned[:, i_chunk])
            mem['abstr_a'].append(abstr_a)
            ctx_low_level = self.fuse_state(prim_current['s'], prim_current['rnn_state'])
            mem, abstr_current = self.rollout_abstract(a=add_time_dim(abstr_a), init_r=init_r_abstr,
                                                       init_term=init_term_abstr,
                                                       init_o=init_o_abstr,
                                                       init_s=abstr_current['s'],
                                                       init_rnn_state=abstr_current['rnn_state'],
                                                       o_target=add_time_dim(ctx_low_level),
                                                       r_target=add_time_dim(abstr_r[:, i_chunk]),
                                                       term_target=add_time_dim(abstr_term[:, i_chunk]),
                                                       mem=mem, use_posterior=use_posterior_abstr, sample=True)
            mem['ctx_low_level'].append(ctx_low_level)

            # update abstract init data for next chunk
            init_o_abstr = add_time_dim(abstr_current['o'])
            init_r_abstr = add_time_dim(abstr_current['r'])
            init_term_abstr = add_time_dim(abstr_current['term'])

            # exchange primitive model's internal state with prediction from abstract model
            #prim_s, prim_rnn_state = self.unfuse_state(abstr_current['o'])
            #prim_current['s'] = prim_s
            #prim_current['rnn_state'] = prim_rnn_state

        mem = self.pack_mem(mem)
        #mem['prim_rnn_state'] = prim_current['rnn_state']
        #mem['abstr_rnn_state'] = abstr_current['rnn_state']

        return mem

    def forward_old(self,
                o: torch.Tensor,
                a: torch.Tensor,
                r: torch.Tensor,
                term: torch.Tensor,
                n_warmup: int = 1):
        assert o.shape[1] == r.shape[1] == term.shape[1] == a.shape[1] + 1
        assert torch.all(r[:, 0] == 0)
        assert torch.all(term[:, 0] == 0)
        assert n_warmup >= 1

        device = self._device
        d_batch, n_steps = a.shape[:2]
        actions_binned = bin_every_k_steps(a, self.abstract_step_size, self.device, padding_val=0)

        mem = self.gen_mem()
        prim_current = self.primitive_model.gen_init_values(d_batch, self.device)
        abstr_current = self.abstract_model.gen_init_values(d_batch, self.device)

        for t in range(n_steps):
            if t % self.abstract_step_size == 0 and t > 0:
                i_a = t // self.abstract_step_size - 1
                abstr_current['a'] = self.abstract_action_model(actions_binned[:, i_a])
                x_posterior = torch.concat([prim_current['s'], self.flatten_rnn_state(prim_current['rnn_state'])], dim=-1)
                self._invoke_abstract_model(abstr_current, x_posterior, mem)

                # cut information flow for primitive model
                prim_current['s'] = self.primitive_model.zero_s(d_batch, device)
                prim_current['rnn_state'] = self.primitive_model.zero_h(d_batch, device)
                prim_current['o'] = self.primitive_model.zero_o(d_batch, device)
                prim_current['r'] = self.primitive_model.zero_r(d_batch, device)
                prim_current['term'] = self.primitive_model.zero_term(d_batch, device)

            prim_current['a'] = a[:, t]
            if t < n_warmup:
                prim_current['o'] = o[:, t]
                prim_current['r'] = r[:, t]
                prim_current['term'] = term[:, t]

            x_posterior = torch.concat([o[:, t+1], r[:, t+1], term[:, t+1]], dim=-1)
            ctx_high_level = torch.concat([abstr_current['s'], self.flatten_rnn_state(abstr_current['rnn_state'])], dim=-1)
            self._invoke_primitive_model(prim_current, x_posterior, mem, ctx_high_level)

        # do a final prediction on abstract level
        abstr_current['a'] = self.abstract_action_model(actions_binned[:, -1])
        x_posterior = torch.concat([prim_current['s'], self.flatten_rnn_state(prim_current['rnn_state'])], dim=-1)
        self._invoke_abstract_model(abstr_current, x_posterior, mem)

        mem = self.pack_mem(mem)
        mem['prim_rnn_state'] = prim_current['rnn_state']
        mem['abstr_rnn_state'] = abstr_current['rnn_state']

        return mem

    # this method is more of a note on how to work with arbitrary model hierarchies
    def _invoke_model(self,
                      mdl: RSSM,
                      mdl_current: Dict,
                      ctx_low_level: torch.Tensor,
                      ctx_high_level: torch.Tensor):
        if ctx_low_level is None:
            use_posterior = True
        else:
            use_posterior = False

        pred = mdl(s=mdl_current['s'], a=mdl_current['a'], ctx_low_level=ctx_low_level, ctx_high_level=ctx_high_level,
                   h=mdl_current['h'], use_posterior=use_posterior)
        mdl_current['s'] = pred['s']
        mdl_current['h'] = pred['h']
        return pred

    def _invoke_primitive_model(self,
                                prim_current: Dict,
                                x_post: torch.Tensor,
                                mem: Optional[Dict],
                                ctx_high_level: torch.Tensor,
                                use_posterior: bool = True,
                                sample: bool = True):
        x_hat = torch.concat([prim_current['o'], prim_current['r'], prim_current['term']], dim=-1)

        # do prediction
        pred = self.primitive_model(s=prim_current['s'], a=prim_current['a'], x_hat=x_hat, x_post_groundtruth=x_post,
                                    ctx_high_level=ctx_high_level, rnn_state=prim_current['rnn_state'],
                                    use_posterior=use_posterior, sample=sample)
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
        x_hat = torch.concat([abstr_current['o'], abstr_current['r'], abstr_current['term']], dim=-1)

        # do prediction
        pred = self.abstract_model(s=abstr_current['s'], a=abstr_current['a'], x_hat=x_hat,
                                   x_post_groundtruth=x_post, ctx_high_level=ctx_high_level,
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
        mem['abstr_r'].append(pred['r'])
        mem['abstr_term'].append(pred['term'])

    @staticmethod
    def gen_mem():
        mem = { 'prim_o': [], 'prim_r': [], 'prim_term': [], 'prim_s_prior': [], 'prim_s_post': [], 'prim_s': [],
                'prim_rnn_state': [], 'abstr_o': [], 'abstr_a': [], 'abstr_r': [], 'abstr_term': [],
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
                   optimizer: torch.optim.Optimizer,
                   n_warmup: int = 1) -> Dict[str, torch.Tensor]:
        optimizer.zero_grad(set_to_none=True)
        losses = self.eval_step(o_ground_truth, a_ground_truth, r_ground_truth, term_ground_truth, n_warmup)
        losses['total'].backward()
        optimizer.step()

        return losses

    def eval_step(self,
                  o_ground_truth: torch.Tensor,
                  a_ground_truth: torch.Tensor,
                  r_ground_truth: torch.Tensor,
                  term_ground_truth: torch.Tensor,
                  n_warmup: int = 1) -> Dict[str, torch.Tensor]:
        device = self._device

        # sum makes sense for rewards, use padding=0 to not affect sum for last element
        # start at time step 1 because the first time step is just the initial observation to start prediction
        abstr_r_ground_truth = bin_every_k_steps(r_ground_truth[:, 1:], self.abstract_step_size, device=device,
                                           padding_val=0).sum(dim=2)
        # terminal flag can only be 0 or 1, so mean value with automatic padding should be used
        # start at time step 1 because the first time step is just the initial observation to start prediction
        abstr_term_ground_truth = bin_every_k_steps(term_ground_truth[:, 1:], self.abstract_step_size,
                                              device=device).max(dim=2).values

        pred = self(o_ground_truth, a_ground_truth, r_ground_truth, term_ground_truth, abstr_r_ground_truth,
                    abstr_term_ground_truth, n_warmup)

        # primitive model loss
        prim_rec_o = torch.nn.functional.mse_loss(pred['prim_o'], o_ground_truth[:, 1:])
        prim_rec_r = torch.nn.functional.mse_loss(pred['prim_r'], r_ground_truth[:, 1:])
        prim_rec_term = torch.nn.functional.binary_cross_entropy(pred['prim_term'], term_ground_truth[:, 1:])
        prim_kl_s = self.beta_kl_prim * kl_loss_normal(pred['prim_s_prior'], pred['prim_s_post'],
                                                       detach_posterior=self.detach_posteriors)
        prim_kl_s_reg = self.beta_reg_prim * kl_regularizer_normal(pred['prim_s_post'])

        # abstract model loss
        abstr_o_rec = torch.nn.functional.mse_loss(pred['abstr_o'], pred['ctx_low_level'])
        abstr_rec_r = torch.nn.functional.mse_loss(pred['abstr_r'], abstr_r_ground_truth)
        abstr_rec_term = torch.nn.functional.binary_cross_entropy(pred['abstr_term'], abstr_term_ground_truth)
        abstr_kl_s = self.beta_kl_abstr * kl_loss_normal(pred['abstr_s_prior'], pred['abstr_s_post'],
                                                         detach_posterior=self.detach_posteriors)
        abstr_kl_s_reg = self.beta_reg_abstr * kl_regularizer_normal(pred['abstr_s_post'])

        # abstract action model loss
        scale = pred['abstr_a'].reshape(-1, self.abstract_model.d_action).std(dim=0, unbiased=False)
        abstr_a_loss = torch.distributions.Normal(loc=0.0, scale=scale + 0.001).entropy()
        abstr_a_loss = -0.1 * torch.sum(torch.abs(abstr_a_loss))

        if pred['abstr_o'].shape[1] > 1:
            abstr_factor = 1.0
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
        h_filtered = self.flatten_rnn_state(rnn_state)
        fused = torch.concat([s, h_filtered], dim=-1)
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
                          init_o: Optional[torch.Tensor] = None,
                          init_r: Optional[torch.Tensor] = None,
                          init_term: Optional[torch.Tensor] = None,
                          init_s: Optional[torch.Tensor] = None,
                          init_rnn_state: Optional[RnnStateType] = None,
                          ctx_high_level: Optional[torch.Tensor] = None,
                          o_target: Optional[torch.Tensor] = None,
                          r_target: Optional[torch.Tensor] = None,
                          term_target: Optional[torch.Tensor] = None,
                          mem: Optional[Dict] = None,
                          use_posterior: bool = True,
                          sample: bool = True):
        d_batch, n_steps = a.shape[:2]
        prim_current = self.primitive_model.gen_init_values(d_batch, self.device)

        # default arguments
        if init_o is None:
            init_o = add_time_dim(self.primitive_model.zero_o(d_batch, self.device))
        if init_r is None:
            init_r = add_time_dim(self.primitive_model.zero_r(d_batch, self.device))
        if init_term is None:
            init_term = add_time_dim(self.primitive_model.zero_term(d_batch, self.device))

        if init_s is None:
            prim_current['s'] = self.primitive_model.zero_s(d_batch, self.device)
        else:
            prim_current['s'] = init_s
        if init_rnn_state is not None:
            prim_current['rnn_state'] = init_rnn_state

        if ctx_high_level is None:
            ctx_high_level = self.primitive_model.zero_ctx_high_level(d_batch, self.device)

        if mem is None:
            mem = self.gen_mem()

        # prediction
        n_warmup = init_o.shape[1]
        for t in range(n_steps):
            if use_posterior:
                x_posterior = torch.concat([o_target[:, t], r_target[:, t], term_target[:, t]], dim=-1)
            else:
                x_posterior = self.primitive_model.zero_x_post(d_batch, self.device)

            if t < n_warmup:
                prim_current['o'] = init_o[:, t]
                prim_current['r'] = init_r[:, t]
                prim_current['term'] = init_term[:, t]

            prim_current['a'] = a[:, t]
            self._invoke_primitive_model(prim_current, x_posterior, mem, ctx_high_level, use_posterior=use_posterior,
                                         sample=sample)

        return mem, prim_current

    def rollout_abstract(self,
                         a: torch.Tensor,
                         init_o: Optional[torch.Tensor] = None,
                         init_r: Optional[torch.Tensor] = None,
                         init_term: Optional[torch.Tensor] = None,
                         init_s: Optional[torch.Tensor] = None,
                         init_rnn_state: Optional[RnnStateType] = None,
                         o_target: Optional[torch.Tensor] = None,
                         r_target: Optional[torch.Tensor] = None,
                         term_target: Optional[torch.Tensor] = None,
                         mem: Optional[Dict] = None,
                         use_posterior: bool = True,
                         sample: bool = True):
        d_batch, n_steps = a.shape[:2]
        abstr_current = self.abstract_model.gen_init_values(d_batch, self.device)

        # default arguments
        if init_o is None:
            init_o = add_time_dim(self.abstract_model.zero_o(d_batch, self.device))
        if init_r is None:
            init_r = add_time_dim(self.abstract_model.zero_r(d_batch, self.device))
        if init_term is None:
            init_term = add_time_dim(self.abstract_model.zero_term(d_batch, self.device))

        if init_s is not None:
            abstr_current['s'] = init_s
        if init_rnn_state is not None:
            abstr_current['rnn_state'] = init_rnn_state

        if mem is None:
            mem = self.gen_mem()

        # prediction
        n_warmup = init_r.shape[1]
        for t in range(n_steps):
            if use_posterior:
                x_posterior = torch.concat([o_target[:, t], r_target[:, t], term_target[:, t]], dim=-1)
            else:
                x_posterior = self.abstract_model.zero_x_post(d_batch, self.device)

            if t < n_warmup:
                abstr_current['o'] = init_o[:, t]
                abstr_current['r'] = init_r[:, t]
                abstr_current['term'] = init_term[:, t]

            abstr_current['a'] = a[:, t]
            self._invoke_abstract_model(abstr_current, x_posterior, mem, use_posterior=use_posterior, sample=sample)

        return mem, abstr_current
