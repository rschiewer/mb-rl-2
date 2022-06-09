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
                start_observations: torch.Tensor,
                start_rewards: torch.Tensor,
                start_terminals: torch.Tensor,
                actions: torch.Tensor):
        device = actions.device
        d_batch, n_steps = actions.shape[:2]
        n_s_start = start_observations.shape[1]
        actions_binned = bin_every_k_steps(actions, self.abstract_step_size, self.device, padding_val=0)

        mem = self._gen_mem()
        prim_current = self.primitive_model.gen_init_values(d_batch, self.device)
        abstr_current = self.abstract_model.gen_init_values(d_batch, self.device)

        for t in range(n_steps):
            if t % self.abstract_step_size == 0 and t > 0:
                abstr_current['a'] = self.abstract_action_model(actions_binned[:, t // self.abstract_step_size - 1])
                self._invoke_abstract_model(prim_current, abstr_current, mem)

                # cut information flow for primitive model
                prim_current['s'] = self.primitive_model.zero_s(d_batch, device)
                prim_current['h'] = self.primitive_model.zero_h(d_batch, device)
                prim_use_posterior = False
            else:
                prim_use_posterior = True

            prim_current['a'] = actions[:, t]
            if t < n_s_start:
                prim_current['o'] = start_observations[:, t]
                prim_current['r'] = start_rewards[:, t]
                prim_current['term'] = start_terminals[:, t]

            self._invoke_primitive_model(prim_current, abstr_current, mem, use_posterior=prim_use_posterior)

        # do a final prediction on abstract level
        abstr_current['a'] = self.abstract_action_model(actions_binned[:, -1])
        self._invoke_abstract_model(prim_current, abstr_current, mem)

        mem = self._pack_mem(mem)
        mem['prim_h'] = prim_current['h']
        mem['abstr_h'] = abstr_current['h']

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
                                abstr_current: Optional[dict],
                                mem: dict,
                                use_posterior: bool = True,
                                sample: bool = True):
        d_batch = abstr_current['a'].shape[0]
        device = abstr_current['a'].device

        # checks
        if use_posterior:
            ctx_low_level = torch.concat([prim_current['o'], prim_current['r'], prim_current['term']], dim=-1)
        else:
            ctx_low_level = self.primitive_model.zero_ctx_low_level(d_batch, device)

        if abstr_current is None:
            ctx_high_level = self.primitive_model.zero_ctx_high_level(d_batch, device)
        else:
            ctx_high_level = torch.concat([abstr_current['s'], self.filter_rnn_state(abstr_current['h'])], dim=-1)

        # do prediction
        pred = self.primitive_model(s=prim_current['s'], a=prim_current['a'], ctx_low_level=ctx_low_level,
                                    ctx_high_level=ctx_high_level, h=prim_current['h'], use_posterior=use_posterior,
                                    sample=sample)
        # update primitive state
        prim_current['s'] = pred['s']
        prim_current['h'] = pred['h']

        # store things
        mem['prim_s'].append(pred['s'])
        mem['prim_s_prior'].append(pred['s_prior'])
        if use_posterior:  # hack to make loss calculation work like usual but also always provide a distribution for s
            mem['prim_s_post'].append(pred['s_post'])
        else:
            mem['prim_s_post'].append(pred['s_prior'])
        mem['prim_o'].append(pred['o'])
        mem['prim_r'].append(pred['r'])
        mem['prim_term'].append(pred['term'])

    def _invoke_abstract_model(self,
                               prim_current: Optional[dict],
                               abstr_current: dict,
                               mem: dict,
                               use_posterior: bool = True,
                               sample: bool = True):
        d_batch = abstr_current['a'].shape[0]
        device = abstr_current['a'].device

        if use_posterior:
            ctx_low_level = torch.concat([prim_current['s'], self.filter_rnn_state(prim_current['h'])], dim=-1)
        else:
            ctx_low_level = self.primitive_model.zero_ctx_low_level(d_batch, device)
        ctx_high_level = self.abstract_model.zero_ctx_high_level(d_batch, device)

        # do prediction
        pred = self.abstract_model(s=abstr_current['s'], a=abstr_current['a'], ctx_low_level=ctx_low_level,
                                   ctx_high_level=ctx_high_level, h=abstr_current['h'],
                                   use_posterior=use_posterior, sample=sample)
        # update abstract state
        abstr_current['s'] = pred['s']
        abstr_current['h'] = pred['h']

        # store things
        mem['abstr_s'].append(pred['s'])
        mem['abstr_s_prior'].append(pred['s_prior'])
        mem['abstr_s_post'].append(pred['s_post'])  # TODO: posterior might not be populated here
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
        abstr_r_target = bin_every_k_steps(r_ground_truth, self.abstract_step_size, device=device,
                                           padding_val=0).sum(dim=2)
        # terminal flag can only be 0 or 1, so mean value with automatic padding should be used
        abstr_term_target = bin_every_k_steps(term_ground_truth, self.abstract_step_size,
                                              device=device).mean(dim=2)

        start_observations = o_ground_truth[:, :n_warmup]
        start_rewards = r_ground_truth[:, :n_warmup]
        start_terminals = term_ground_truth[:, :n_warmup]
        pred = self(start_observations, start_rewards, start_terminals, a_ground_truth)

        # primitive model loss
        prim_rec_o = torch.nn.functional.mse_loss(pred['prim_o'], o_ground_truth)
        prim_rec_r = torch.nn.functional.mse_loss(pred['prim_r'], r_ground_truth)
        prim_rec_term = torch.nn.functional.binary_cross_entropy(pred['prim_term'], term_ground_truth)
        prim_kl_s = self.beta_kl_prim * kl_loss_normal(pred['prim_s_prior'], pred['prim_s_post'])
        prim_kl_s_reg = self.beta_reg_prim * kl_regularizer_normal(pred['prim_s_post'])

        # abstract model loss
        abstr_rec_r = torch.nn.functional.mse_loss(pred['abstr_r'], abstr_r_target)
        abstr_rec_term = torch.nn.functional.binary_cross_entropy(pred['abstr_term'], abstr_term_target)
        abstr_kl_s = self.beta_kl_abstr * kl_loss_normal(pred['abstr_s_prior'], pred['abstr_s_post'])
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

    def filter_rnn_state(self, h: Tuple[torch.Tensor, torch.Tensor]):
        #if self.primitive_model.rnn_type == 'lstm' and self.abstract_model.rnn_type == self.primitive_model.rnn_type:
        #    h = torch.concat(h, dim=0)  # concat h and c tensors of LSTM along the layer dimension, this is arbitrary
        #h = torch.transpose(h, 0, 1)  # bring batch dimension to front
        #h = torch.flatten(h, start_dim=1)  # fold h/c/layer dimension into d_hidden
        #return h
        if self.primitive_model.rnn_type == 'lstm':
            h = h[0]
        return h[-1]

    def rollout_primitive(self,
                          start_observations: Optional[torch.Tensor],
                          start_rewards: Optional[torch.Tensor],
                          start_terminals: Optional[torch.Tensor],
                          actions: torch.Tensor,
                          abstr_s: Optional[torch.Tensor] = None,
                          abstr_h: Optional[torch.Tensor] = None,
                          sample: bool = True):
        d_batch, n_steps = actions.shape[:2]
        device = self._device
        prim_current = self.primitive_model.gen_init_values(d_batch, device)

        if None in (start_observations, start_rewards, start_terminals):
            n_warmup = 0
            prim_current = {k: None for k in prim_current}
        else:
            n_warmup = start_observations.shape[1]

        #TODO: here

        mem = self._gen_mem()

        for t in range(n_steps):
            prim_current['a'] = actions[:, t]
            if t < n_warmup:
                prim_current['o'] = start_observations[:, t]
                prim_current['r'] = start_rewards[:, t]
                prim_current['term'] = start_terminals[:, t]
            self._invoke_primitive_model(prim_current, prim_current['o'], ctx_high_level, mem, sample=sample)

        mem = self._pack_mem(mem)
        mem['prim_h'] = prim_current['h']

        return mem

    def rollout_abstract(self,
                         abstr_start_states: torch.Tensor,
                         abstr_actions: torch.Tensor,
                         ctx_low_level: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                         sample: bool = True):
        d_batch, n_steps = abstr_actions.shape[:2]
        n_warmup = abstr_start_states.shape[1]
        device = self._device

        mem = self._gen_mem()
        abstr_current = self.abstract_model.gen_init_values(d_batch, device)

        if ctx_low_level is None:
            ctx_low_level = [None] * n_steps

        for t in range(n_steps):
            abstr_current['a'] = abstr_actions[:, t]
            if t < n_warmup:
                abstr_current['s'] = abstr_start_states[:, t]
            self._invoke_abstract_model(ctx_low_level[t], abstr_current, mem, sample=sample)

        mem = self._pack_mem(mem)
        mem['abstr_h'] = abstr_current['h']

        return mem
