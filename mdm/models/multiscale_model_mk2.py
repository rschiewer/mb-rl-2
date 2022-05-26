from typing import Dict

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
        device = self._device

        #if n_steps % self.abstract_step_size != 0:
        #    raise ValueError(f'Provided trajectories\' length must be divisible by abstract step size but is not, '
        #                     f'length is {n_steps} and abstract step size is {self.abstract_step_size}')

        actions_binned = bin_every_k_steps(actions, self.abstract_step_size, device, padding_val=0)

        mem = { 'o': [], 'r': [], 'term': [], 'prim_s_prior': [], 'prim_s_post': [], 'abstr_s_prior': [],
                'abstr_s_post': [], 'abstr_r': [], 'abstr_term': []}
        prim_current = self.primitive_model.gen_init_values(d_batch, device)
        abstr_current = self.abstract_model.gen_init_values(d_batch, device)

        for t in range(n_steps):
            if t % self.abstract_step_size == 0 and t > 0:
                abstr_current['a'] = self.abstract_action_model(actions_binned[:, t // self.abstract_step_size - 1])
                self._invoke_abstract_model(prim_current, abstr_current, mem)

            if t < n_s_start:
                prim_current['o'] = start_observations[:, t]

            pred = self.primitive_model(s=prim_current['s'], a=actions[:, t], ctx_low_level=prim_current['o'],
                                        ctx_high_level=abstr_current['s'], cell_state=prim_current['h'])
            # update primitive state
            prim_current['s'] = pred['s']
            prim_current['h'] = pred['h']

            # store things
            mem['prim_s_prior'].append(pred['s_prior'])
            mem['prim_s_post'].append(pred['s_post'])
            mem['o'].append(pred['o'])
            mem['r'].append(pred['r'])
            mem['term'].append(pred['term'])

        # do a final prediction on abstract level
        abstr_current['a'] = self.abstract_action_model(actions_binned[:, -1])
        self._invoke_abstract_model(prim_current, abstr_current, mem)

        mem = {k: torch.stack(v, dim=1) if isinstance(v[0], torch.Tensor) else v for k, v in mem.items()}

        return mem

    def _invoke_abstract_model(self,
                               prim_current: dict,
                               abstr_current: dict,
                               mem: dict):
        # prepare input for next abstract state prediction
        h_flat = MultiscaleDynamicsModelMK2._filter_h(prim_current['h'])

        # do prediction
        pred = self.abstract_model(s=abstr_current['s'], a=abstr_current['a'], ctx_low_level=h_flat,
                                   cell_state=abstr_current['h'])
        # update abstract state
        abstr_current['s'] = pred['s']
        abstr_current['h'] = pred['h']

        # store things
        mem['abstr_s_prior'].append(pred['s_prior'])
        mem['abstr_s_post'].append(pred['s_post'])
        mem['abstr_r'].append(pred['r'])
        mem['abstr_term'].append(pred['term'])

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

        abstr_r_target = bin_every_k_steps(r_ground_truth, self.abstract_step_size, device=device).sum(dim=2)
        abstr_term_target = bin_every_k_steps(term_ground_truth, self.abstract_step_size, device=device).sum(dim=2)

        start_observations = o_ground_truth[:, :n_warmup]
        pred = self(start_observations, a_ground_truth)

        # primitive model loss
        prim_rec_o = reconstruction_loss(pred['o'], o_ground_truth)
        prim_rec_r = reconstruction_loss(pred['r'], r_ground_truth)
        prim_rec_term = reconstruction_loss(pred['term'], term_ground_truth)
        prim_kl_s = self.beta_kl_prim * kl_loss_normal(pred['prim_s_prior'], pred['prim_s_post'])
        prim_kl_s_reg = self.beta_reg_prim * kl_regularizer_normal(pred['prim_s_post'])

        # abstract model loss
        abstr_rec_r = reconstruction_loss(pred['abstr_r'], abstr_r_target)
        abstr_rec_term = reconstruction_loss(pred['abstr_term'], abstr_term_target)
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
        return True, ''

    @staticmethod
    def _filter_h(h: Tuple[torch.Tensor, torch.Tensor]):
        h = torch.concat(h, dim=0)  # concat h and c tensors of LSTM along the layer dimension, this is arbitrary
        h = torch.transpose(h, 0, 1)  # bring batch dimension to front
        h = torch.flatten(h, start_dim=1)  # fold h/c/layer dimension into d_hidden
        return h
