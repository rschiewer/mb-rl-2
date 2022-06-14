from typing import Dict, Optional

from mdm.models.building_blocks import AbstractActionModel, RSSM
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
                 beta_reg_abstract: float = 0.001):
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

    def forward(self,
                o: torch.Tensor,
                a: torch.Tensor,
                r: torch.Tensor,
                term: torch.Tensor,
                n_warmup_prim: int = 1,
                n_warmup_abstr: int = 0):
        assert o.shape[1] == r.shape[1] == term.shape[1] == a.shape[1] + 1
        assert torch.all(r[:, 0] == 0)
        assert torch.all(term[:, 0] == 0)
        assert n_warmup_prim >= 1

        mem = self._gen_mem()
        d_batch, n_steps_prim = a.shape[:2]

        a_binned = bin_every_k_steps(a, self.abstract_step_size, self.device, padding_val=0)
        r_binned = bin_every_k_steps(r[:, 1:], self.abstract_step_size, self.device, padding_val=0)
        term_binned = bin_every_k_steps(term[:, 1:], self.abstract_step_size, self.device, padding_val=0)

        prim_current = self.primitive_model.gen_init_values(d_batch, self.device)
        abstr_current = self.abstract_model.gen_init_values(d_batch, self.device)

        prim_current['o'] = o[:, 0]
        prim_current['r'] = r[:, 0]
        prim_current['term'] = term[:, 0]
        # exclude initial trajectory data because it's in prim_current now
        o = o[:, 1:]
        r = r[:, 1:]
        term = term[:, 1:]

        for i_chunk in range(a_binned.shape[1]):
            i_start = i_chunk * self.abstract_step_size
            i_end = min((i_chunk + 1) * self.abstract_step_size, n_steps_prim)
            ctx_high_level = self.fuse_state(abstr_current['s'], abstr_current['rnn_state'])
            mem, prim_current = self.rollout_primitive(init_values=prim_current, a=a[:, i_start: i_end],
                                                       o=o[:, i_start: i_end], r=r[:, i_start: i_end],
                                                       term=term[:, i_start: i_end], ctx_high_level=ctx_high_level,
                                                       mem=mem, n_warmup=n_warmup_prim, use_posterior=True, sample=True)

            a_abstr = add_time_dim(self.abstract_action_model(a_binned[:, i_chunk]))
            x_posterior = add_time_dim(self.fuse_state(prim_current['s'], prim_current['rnn_state']))
            mem, abstr_current = self.rollout_abstract(init_abstr=abstr_current, a_abstr=a_abstr,
                                                       r_abstr=r_binned[:, i_chunk], term_abstr=term_binned[:, i_chunk],
                                                       ctx_low_level=x_posterior, mem=mem, n_warmup=n_warmup_abstr,
                                                       use_posterior=True, sample=True)

            n_warmup_prim = max(n_warmup_prim - self.abstract_step_size, 0)
            n_warmup_abstr = max(n_warmup_abstr - 1, 0)

        mem = self._pack_mem(mem)
        mem['prim_rnn_state'] = prim_current['rnn_state']
        mem['abstr_rnn_state'] = abstr_current['rnn_state']

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

        mem = self._gen_mem()
        prim_current = self.primitive_model.gen_init_values(d_batch, self.device)
        abstr_current = self.abstract_model.gen_init_values(d_batch, self.device)

        for t in range(n_steps):
            if t % self.abstract_step_size == 0 and t > 0:
                i_a = t // self.abstract_step_size - 1
                abstr_current['a'] = self.abstract_action_model(actions_binned[:, i_a])
                x_posterior = torch.concat([prim_current['s'], self.filter_rnn_h(prim_current['rnn_state'])], dim=-1)
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
            ctx_high_level = torch.concat([abstr_current['s'], self.filter_rnn_h(abstr_current['rnn_state'])], dim=-1)
            self._invoke_primitive_model(prim_current, x_posterior, mem, ctx_high_level)

        # do a final prediction on abstract level
        abstr_current['a'] = self.abstract_action_model(actions_binned[:, -1])
        x_posterior = torch.concat([prim_current['s'], self.filter_rnn_h(prim_current['rnn_state'])], dim=-1)
        self._invoke_abstract_model(abstr_current, x_posterior, mem)

        mem = self._pack_mem(mem)
        mem['prim_rnn_state'] = prim_current['rnn_state']
        mem['abstr_rnn_state'] = abstr_current['rnn_state']

        return mem

    # this method is more of a note on how to work with arbitrary model hierarchies
    def _invoke_model(self,
                      mdl: RSSM,
                      mdl_current: dict,
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
                                prim_current: dict,
                                x_post: torch.Tensor,
                                mem: Optional[dict],
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
        mem['prim_s_post'].append(pred['s_post'])
        mem['prim_o'].append(pred['o'])
        mem['prim_r'].append(pred['r'])
        mem['prim_term'].append(pred['term'])

    def _invoke_abstract_model(self,
                               abstr_current: dict,
                               x_post: torch.Tensor,
                               mem: dict,
                               use_posterior: bool = True,
                               sample: bool = True):
        d_batch = abstr_current['a'].shape[0]
        device = abstr_current['a'].device

        #if use_posterior:
        #    # TODO: check if this is the correct time step's s and h
        #    x_post_gt = torch.concat([prim_current['s'], self.filter_rnn_hc(prim_current['rnn_state'])], dim=-1)
        #else:
        #    x_post_gt = self.primitive_model.zero_x_post_groundtruth(d_batch, device)
        ctx_high_level = self.abstract_model.zero_ctx_high_level(d_batch, device)
        x_hat = torch.concat([abstr_current['r'], abstr_current['term']], dim=-1)

        # do prediction
        pred = self.abstract_model(s=abstr_current['s'], a=abstr_current['a'], x_hat=x_hat,
                                   x_post_groundtruth=x_post, ctx_high_level=ctx_high_level,
                                   rnn_state=abstr_current['rnn_state'], use_posterior=use_posterior, sample=sample)
        # update abstract state
        abstr_current['s'] = pred['s']
        abstr_current['rnn_state'] = pred['rnn_state']
        abstr_current['r'] = pred['r']
        abstr_current['term'] = pred['term']

        # store things
        mem['abstr_s'].append(pred['s'])
        mem['abstr_s_prior'].append(pred['s_prior'])
        mem['abstr_s_post'].append(pred['s_post'])
        mem['abstr_r'].append(pred['r'])
        mem['abstr_term'].append(pred['term'])

    @staticmethod
    def _gen_mem():
        mem = { 'prim_o': [], 'prim_r': [], 'prim_term': [], 'prim_s_prior': [], 'prim_s_post': [], 'prim_s': [],
                'abstr_s_prior': [], 'abstr_s_post': [], 'abstr_s': [], 'abstr_r': [], 'abstr_term': []}
        return mem


    @staticmethod
    def _pack_mem(mem):
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
        abstr_r_target = bin_every_k_steps(r_ground_truth[:, 1:], self.abstract_step_size, device=device,
                                           padding_val=0).sum(dim=2)
        # terminal flag can only be 0 or 1, so mean value with automatic padding should be used
        # start at time step 1 because the first time step is just the initial observation to start prediction
        abstr_term_target = bin_every_k_steps(term_ground_truth[:, 1:], self.abstract_step_size,
                                              device=device).max(dim=2).values

        pred = self(o_ground_truth, a_ground_truth, r_ground_truth, term_ground_truth, n_warmup)

        # primitive model loss
        prim_rec_o = torch.nn.functional.mse_loss(pred['prim_o'], o_ground_truth[:, 1:])
        prim_rec_r = torch.nn.functional.mse_loss(pred['prim_r'], r_ground_truth[:, 1:])
        prim_rec_term = torch.nn.functional.binary_cross_entropy(pred['prim_term'], term_ground_truth[:, 1:])
        prim_kl_s = self.beta_kl_prim * kl_loss_normal(pred['prim_s_prior'], pred['prim_s_post'], detach_posterior=True)
        prim_kl_s_reg = self.beta_reg_prim * kl_regularizer_normal(pred['prim_s_post'])

        # abstract model loss
        abstr_rec_r = torch.nn.functional.mse_loss(pred['abstr_r'], abstr_r_target)
        abstr_rec_term = torch.nn.functional.binary_cross_entropy(pred['abstr_term'], abstr_term_target)
        abstr_kl_s = self.beta_kl_abstr * kl_loss_normal(pred['abstr_s_prior'], pred['abstr_s_post'], detach_posterior=True)
        abstr_kl_s_reg = self.beta_reg_abstr * kl_regularizer_normal(pred['abstr_s_post'])

        total = prim_rec_o + prim_rec_r + prim_rec_term + prim_kl_s + prim_kl_s_reg
        total += abstr_rec_r + abstr_rec_term + abstr_kl_s + abstr_kl_s_reg

        return {'total': total, 'prim_o': prim_rec_o, 'prim_r': prim_rec_r, 'prim_term': prim_rec_term,
                'prim_kl_s': prim_kl_s, 'prim_kl_s_reg': prim_kl_s_reg, 'abstr_r': abstr_rec_r,
                'abstr_term': abstr_rec_term, 'abstr_kl_s': abstr_kl_s, 'abstr_kl_s_reg': abstr_kl_s_reg}

    def input_compatible(self,
                         o_ground_truth: torch.Tensor,
                         a_ground_truth: torch.Tensor,
                         r_ground_truth: torch.Tensor) -> Tuple[bool, str]:
        # laziness ahead
        return True, ''

    def filter_rnn_h(self, h: Tuple[torch.Tensor, torch.Tensor]):
        #if self.primitive_model.rnn_type == 'lstm' and self.abstract_model.rnn_type == self.primitive_model.rnn_type:
        #    h = torch.concat(h, dim=0)  # concat h and c tensors of LSTM along the layer dimension, this is arbitrary
        #h = torch.transpose(h, 0, 1)  # bring batch dimension to front
        #h = torch.flatten(h, start_dim=1)  # fold h/c/layer dimension into d_hidden
        #return h
        if self.primitive_model.rnn_type == 'lstm':
            h = h[0]
        return h[-1]

    def fuse_state(self, s: torch.Tensor, h: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
        h_filtered = self.filter_rnn_h(h)
        fused = torch.concat([s, h_filtered], dim=-1)
        return fused

    def rollout_primitive(self,
                          init_values: dict,
                          a: torch.Tensor,
                          o: torch.Tensor,
                          r: torch.Tensor,
                          term: torch.Tensor,
                          ctx_high_level: torch.Tensor,
                          mem: dict,
                          n_warmup: int,
                          use_posterior: bool = True,
                          sample: bool = True):
        d_batch, n_steps = a.shape[:2]

        for t in range(n_steps):
            if use_posterior:
                x_posterior = torch.concat([o[:, t], r[:, t], term[:, t]], dim=-1)
            else:
                x_posterior = self.primitive_model.zero_x_post(d_batch, self._device)

            init_values['a'] = a[:, t]  # init_values['a'] is currently unused, all actions should be in a
            self._invoke_primitive_model(init_values, x_posterior, mem, ctx_high_level, use_posterior=use_posterior,
                                         sample=sample)
            if t < n_warmup:
                init_values['o'] = o[:, t]
                init_values['r'] = r[:, t]
                init_values['term'] = term[:, t]

        return mem, init_values

    def rollout_abstract(self,
                         init_abstr: dict,
                         a_abstr: torch.Tensor,
                         r_abstr: torch.Tensor,
                         term_abstr: torch.Tensor,
                         ctx_low_level: torch.Tensor,
                         mem: dict,
                         n_warmup: int,
                         use_posterior: bool = True,
                         sample: bool = True):
        #if use_posterior:
        #    assert len(ctx_low_level) == a_abstr.shape[1] + 1

        d_batch, n_steps = a_abstr.shape[:2]

        if ctx_low_level is None:
            ctx_low_level = [None] * n_steps

        for t in range(n_steps):
            if use_posterior:
                x_posterior = ctx_low_level[:, t]
            else:
                x_posterior = self.abstract_model.zero_x_post(d_batch, self._device)

            init_abstr['a'] = a_abstr[:, t]
            self._invoke_abstract_model(init_abstr, x_posterior, mem, use_posterior=use_posterior, sample=sample)

            if t < n_warmup:
                init_abstr['r'] = r_abstr[:, t]
                init_abstr['term'] = term_abstr[:, t]

        return mem, init_abstr
