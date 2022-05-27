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
                actions: torch.Tensor):
        d_batch, n_steps = actions.shape[:2]
        n_s_start = start_observations.shape[1]

        #if n_steps % self.abstract_step_size != 0:
        #    raise ValueError(f'Provided trajectories\' length must be divisible by abstract step size but is not, '
        #                     f'length is {n_steps} and abstract step size is {self.abstract_step_size}')

        actions_binned = bin_every_k_steps(actions, self.abstract_step_size, self.device, padding_val=0)

        mem = { 'o': [], 'r': [], 'term': [], 'prim_s_prior': [], 'prim_s_post': [], 'abstr_s_prior': [],
                'abstr_s_post': [], 'abstr_r': [], 'abstr_term': []}
        prim_current = self.primitive_model.gen_init_values(d_batch, self.device)
        abstr_current = self.abstract_model.gen_init_values(d_batch, self.device)

        for t in range(n_steps):
            if t % self.abstract_step_size == 0 and t > 0:
                abstr_current['a'] = self.abstract_action_model(actions_binned[:, t // self.abstract_step_size - 1])
                self._invoke_abstract_model(abstr_current, prim_current['h'], mem)

            prim_current['a'] = actions[:, t]
            if t < n_s_start:
                prim_current['o'] = start_observations[:, t]

            self._invoke_primitive_model(prim_current, abstr_current['s'], mem)

        # do a final prediction on abstract level
        abstr_current['a'] = self.abstract_action_model(actions_binned[:, -1])
        self._invoke_abstract_model(abstr_current, prim_current['h'], mem)

        mem = self._pack_mem(mem)
        mem['prim_h'] = prim_current['h']
        mem['abstr_h'] = abstr_current['h']

        return mem

    def _invoke_primitive_model(self,
                                prim_current: dict,
                                abstr_s: Union[torch.Tensor, None],  # this is high level context
                                mem: dict):
        # checks
        if abstr_s is None:
            d_batch = prim_current['o'].shape[0]
            abstr_s = torch.zeros(d_batch, self.d_abstract_state, device=self.device)

        # do prediction
        pred = self.primitive_model(s=prim_current['s'], a=prim_current['a'], ctx_low_level=prim_current['o'],
                                    ctx_high_level=abstr_s, h=prim_current['h'])
        # update primitive state
        prim_current['s'] = pred['s']
        prim_current['h'] = pred['h']

        # store things
        mem['prim_s_prior'].append(pred['s_prior'])
        mem['prim_s_post'].append(pred['s_post'])
        mem['o'].append(pred['o'])
        mem['r'].append(pred['r'])
        mem['term'].append(pred['term'])

    def _invoke_abstract_model(self,
                               abstr_current: dict,
                               h_primitive: Union[Tuple[torch.Tensor, torch.Tensor], None],  # this is low level context
                               mem: dict):
        # checks
        if h_primitive is None:
            use_posterior = False
        else:
            use_posterior = True
            h_primitive = self._filter_h(h_primitive)

        # do prediction
        pred = self.abstract_model(s=abstr_current['s'], a=abstr_current['a'], ctx_low_level=h_primitive,
                                   h=abstr_current['h'], use_posterior=use_posterior)
        # update abstract state
        abstr_current['s'] = pred['s']
        abstr_current['h'] = pred['h']

        # store things
        mem['abstr_s_prior'].append(pred['s_prior'])
        mem['abstr_s_post'].append(pred['s_post'])
        mem['abstr_r'].append(pred['r'])
        mem['abstr_term'].append(pred['term'])

    def _pack_mem(self, mem):
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
        abstr_term_target = bin_every_k_steps(term_ground_truth, self.abstract_step_size, device=device).mean(dim=2)

        start_observations = o_ground_truth[:, :n_warmup]
        pred = self(start_observations, a_ground_truth)

        # primitive model loss
        prim_rec_o = torch.nn.functional.mse_loss(pred['o'], o_ground_truth)
        prim_rec_r = torch.nn.functional.mse_loss(pred['r'], r_ground_truth)
        prim_rec_term = torch.nn.functional.binary_cross_entropy(pred['term'], term_ground_truth)
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

    def _filter_h(self, h: Tuple[torch.Tensor, torch.Tensor]):
        h = torch.concat(h, dim=0)  # concat h and c tensors of LSTM along the layer dimension, this is arbitrary
        h = torch.transpose(h, 0, 1)  # bring batch dimension to front
        h = torch.flatten(h, start_dim=1)  # fold h/c/layer dimension into d_hidden
        return h

    def rollout_primitive(self,
                          start_observations: torch.Tensor,
                          actions: torch.Tensor,
                          ctx_high_level: Optional[torch.Tensor] = None):
        d_batch, n_steps = actions.shape[:2]
        n_s_start = start_observations.shape[1]
        device = self._device

        mem = { 'o': [], 'r': [], 'term': [], 'prim_s_prior': [], 'prim_s_post': [], 'abstr_s_prior': [],
                'abstr_s_post': [], 'abstr_r': [], 'abstr_term': []}
        prim_current = self.primitive_model.gen_init_values(d_batch, device)

        for t in range(n_steps):
            prim_current['a'] = actions[:, t]
            if t < n_s_start:
                prim_current['o'] = start_observations[:, t]
            self._invoke_primitive_model(prim_current, ctx_high_level, mem)

        mem = self._pack_mem(mem)
        mem['prim_h'] = prim_current['h']
        mem['abstr_h'] = None

        return mem

    def rollout_abstract(self,
                         abstr_start_states: torch.Tensor,
                         abstr_actions: torch.Tensor,
                         ctx_low_level: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None):
        d_batch, n_steps = abstr_actions.shape[:2]
        device = self._device

        mem = { 'o': [], 'r': [], 'term': [], 'prim_s_prior': [], 'prim_s_post': [], 'abstr_s_prior': [],
                'abstr_s_post': [], 'abstr_r': [], 'abstr_term': []}
        abstr_current = self.abstract_model.gen_init_values(d_batch, device)
        abstr_current['s'] = abstr_start_states

        for t in range(n_steps):
            abstr_current['a'] = abstr_actions[:, t]
            self._invoke_abstract_model(abstr_current, ctx_low_level[t], mem)

        mem = self._pack_mem(mem)
        mem['prim_h'] = None
        mem['abstr_h'] = abstr_current['h']

        return mem
