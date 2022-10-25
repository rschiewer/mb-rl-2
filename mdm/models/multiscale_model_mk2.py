from typing import Dict, Optional
import random
from math import cos

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
                 abstract_pred_target: str,
                 beta_kl_primitive: float = 0.01,
                 beta_reg_primitive: float = 0.001,
                 beta_kl_abstract: float = 0.01,
                 beta_reg_abstract: float = 0.001,
                 beta_abstract_model: float = 1.0,
                 beta_abstract_action: float = 1.0,
                 n_warmup_prim: Union[int, Sequence[int]] = 1,
                 n_warmup_abstr: Union[int, Sequence[int]] = 1,
                 detach_posteriors: bool = False):
        if not abstract_pred_target.startswith('prim_'):
            raise ValueError(f'Abstract model\'s prediciton target should be from primitive model and start ',
                             f'with "prim_", but is {abstract_pred_target} instead.')
        if abstract_pred_target not in self.gen_mem().keys():
            raise ValueError(f'Unknown abstract model prediction target: {abstract_pred_target}')

        super().__init__()
        self.primitive_model = primitive_model
        self.abstract_model = abstract_model
        self.abstract_action_model = abstract_action_model
        self.abstract_step_size = abstract_step_size
        self.abstr_pred_target = abstract_pred_target
        self.d_state = primitive_model.d_z
        self.d_action = primitive_model.d_action
        self.d_reward = primitive_model.d_reward
        self.d_observation = primitive_model.d_observation
        self.d_abstract_state = abstract_model.d_z
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
        self._current_train_step = None

    def prepare_for_training(self):
        self._current_train_step = 0

    def forward(self,
                o: torch.Tensor,
                a: torch.Tensor,
                r: torch.Tensor,
                term: torch.Tensor,
                abstr_r: torch.Tensor,
                abstr_term: torch.Tensor,
                n_warmup_prim: int,
                n_warmup_abstr: int):
        assert o.shape[1] == r.shape[1] == term.shape[1] == a.shape[1]

        mem = self.gen_mem()
        d_batch, n_steps_prim = a.shape[:2]
        a_binned = bin_every_k_steps(a, self.abstract_step_size, padding_val=0)

        prim_current = self.primitive_model.gen_init_values(d_batch, self.device)
        abstr_current = self.abstract_model.gen_init_values(d_batch, self.device)

        warmup_steps_prim_left = n_warmup_prim
        warmup_steps_abstr_left = n_warmup_abstr
        for i_chunk in range(a_binned.shape[1]):
            i_start = i_chunk * self.abstract_step_size
            i_end = min((i_chunk + 1) * self.abstract_step_size, n_steps_prim)

            # primitive model rollout
            #n_warmup_prim = min(warmup_steps_prim_left, self.abstract_step_size)
            mem, prim_current = self.rollout_primitive(a=a[:, i_start: i_end], o=o[:, i_start: i_end],
                                                       r=r[:, i_start: i_end], term=term[:, i_start: i_end],
                                                       z=prim_current['z'], rnn_state=prim_current['rnn_state'],
                                                       n_posterior_steps=warmup_steps_prim_left,
                                                       mem=mem, sample=True)

            # input to abstr act mdl needs to be always of same length, so take a_binned instead of a[i_start: i_end]
            abstr_a = self.abstract_action_model(a_binned[:, i_chunk])

            # abstract model rollout
            # ctx_low_level = self.fuse_state(prim_current['z'], prim_current['rnn_state']).detach()
            # mem['ctx_low_level'].append(ctx_low_level)
            prim_data = mem[self.abstr_pred_target][-1].detach()
            mem['abstr_o_target'].append(prim_data)

            #n_warmup_abstr = 1 if warmup_steps_abstr_left > 0 else 0
            mem, abstr_current = self.rollout_abstract(a=add_time_dim(abstr_a), r=add_time_dim(abstr_r[:, i_chunk]),
                                                       term=add_time_dim(abstr_term[:, i_chunk]),
                                                       prim_data=add_time_dim(prim_data),
                                                       z=abstr_current['z'],
                                                       rnn_state=abstr_current['rnn_state'],
                                                       n_posterior_steps=warmup_steps_abstr_left,
                                                       mem=mem, sample=True)

            # exchange primitive model's internal state with prediction from abstract model
            # prim_data, prim_rnn_state = self.unfuse_state(abstr_current['o'])
            # prim_current['z'] = prim_data
            # prim_current['rnn_state'] = prim_rnn_state
            # prim_current['z'] = abstr_current['o']

            # warmup should only happen at first sequence chunk
            if warmup_steps_prim_left > 0:
                warmup_steps_prim_left -= max(self.abstract_step_size, 0)
            if warmup_steps_abstr_left > 0:
                warmup_steps_abstr_left -= 1
            #warmup_steps_prim_left = max(warmup_steps_prim_left - self.abstract_step_size, 0)
            #warmup_steps_abstr_left = max(warmup_steps_abstr_left - 1, 0)

        mem = self.pack_mem(mem)

        return mem

    def _invoke_primitive_model(self,
                                prim_current: Dict,
                                x_post: Optional[torch.Tensor],
                                mem: Optional[Dict],
                                use_posterior: bool = True,
                                sample: bool = True):
        d_batch = prim_current['a'].shape[0]
        ctx_high_level = self.primitive_model.zero_ctx_high_level(d_batch, self.device)

        # do prediction
        pred = self.primitive_model(z=prim_current['z'], a=prim_current['a'], x_current_groundtruth=x_post,
                                    ctx_high_level=ctx_high_level, rnn_state=prim_current['rnn_state'],
                                    use_posterior=use_posterior, sample=sample)

        # sometimes use s_prior to provide next step's s
        # d_batch = pred['z'].shape[0]
        # pred['z'] = torch.where(torch.rand(d_batch, 1, device=self.device) < 0.2, pred['s_prior'].rsample(), pred['z'])

        # update primitive state
        prim_current['z'] = pred['z']
        prim_current['rnn_state'] = pred['rnn_state']

        # store things
        mem['prim_z'].append(pred['z'])
        mem['prim_z_prior'].append(pred['z_prior'])
        if use_posterior:
            mem['prim_z_post'].append(pred['z_post'])
        else:
            mem['prim_z_post'].append(pred['z_prior'])  # hack to make loss calculation easier
        mem['prim_rnn_state'].append(pred['rnn_state'])
        mem['prim_h'].append(pred['h'])
        mem['prim_s'].append(pred['s'])
        mem['prim_a'].append(prim_current['a'])
        mem['prim_o'].append(pred['o'])
        mem['prim_o_dist'].append(pred['o_dist'])
        mem['prim_r'].append(pred['r'])
        mem['prim_r_dist'].append(pred['r_dist'])
        mem['prim_term'].append(pred['term'])

    def _invoke_abstract_model(self,
                               abstr_current: Dict,
                               x_post: torch.Tensor,
                               mem: Dict,
                               use_posterior: bool = True,
                               sample: bool = True):
        d_batch = abstr_current['a'].shape[0]
        ctx_high_level = self.abstract_model.zero_ctx_high_level(d_batch, self.device)

        # do prediction
        pred = self.abstract_model(z=abstr_current['z'], a=abstr_current['a'],
                                   x_current_groundtruth=x_post, ctx_high_level=ctx_high_level,
                                   rnn_state=abstr_current['rnn_state'], use_posterior=use_posterior, sample=sample)

        # update abstract state
        abstr_current['z'] = pred['z']
        abstr_current['rnn_state'] = pred['rnn_state']

        # store things
        mem['abstr_z'].append(pred['z'])
        mem['abstr_z_prior'].append(pred['z_prior'])
        if use_posterior:
            mem['abstr_z_post'].append(pred['z_post'])
        else:
            mem['abstr_z_post'].append(pred['z_prior'])  # hack to make loss calculation easier
        mem['abstr_rnn_state'].append(pred['rnn_state'])
        mem['abstr_h'].append(pred['h'])
        mem['abstr_s'].append(pred['s'])
        mem['abstr_a'].append(abstr_current['a'])
        mem['abstr_o'].append(pred['o'])
        mem['abstr_o_dist'].append(pred['o_dist'])
        mem['abstr_r'].append(pred['r'])
        mem['abstr_r_dist'].append(pred['r_dist'])
        mem['abstr_term'].append(pred['term'])

    @staticmethod
    def gen_mem():
        mem = {'prim_a': [], 'prim_o': [], 'prim_o_dist': [], 'prim_r': [], 'prim_r_dist': [], 'prim_term': [],
               'prim_z_prior': [], 'prim_h': [],
               'prim_z_post': [], 'prim_z': [], 'prim_s': [], 'prim_rnn_state': [], 'abstr_o': [], 'abstr_o_dist': [],
               'abstr_a': [], 'abstr_h': [],
               'abstr_a_dist': [], 'abstr_r': [], 'abstr_r_dist': [], 'abstr_term': [], 'abstr_z_prior': [],
               'abstr_z_post': [], 'abstr_z': [], 'abstr_s': [], 'abstr_rnn_state': [],
               'abstr_o_target': []}
        return mem

    @staticmethod
    def pack_mem(mem):
        for k, v in mem.items():
            if isinstance(v, List) and len(v) > 0:
                if k.endswith('_rnn_state') and isinstance(v[0], (tuple, torch.Tensor)):
                    mem[k] = torch.stack([pack_rnn_state(state) for state in v], dim=1)
                elif isinstance(v[0], torch.Tensor):
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
        torch.nn.utils.clip_grad_norm(self.parameters(), 1.0)
        self._current_train_step += 1
        return losses

    def calc_abstr_r_ground_truth(self, r_ground_truth: torch.Tensor):
        # sum makes sense for rewards, use padding=0 to not affect sum for last element
        return bin_every_k_steps(r_ground_truth, self.abstract_step_size, padding_val=None).sum(dim=2)

    def calc_abstr_term_ground_truth(self, term_ground_truth: torch.Tensor):
        # terminal flag can only be 0 or 1, so mean value with automatic padding should be used
        return bin_every_k_steps(term_ground_truth, self.abstract_step_size, padding_val=None).max(dim=2).values

    def eval_step(self,
                  o_ground_truth: torch.Tensor,
                  a_ground_truth: torch.Tensor,
                  r_ground_truth: torch.Tensor,
                  term_ground_truth: torch.Tensor) -> Dict[str, torch.Tensor]:
        prim_steps = a_ground_truth.shape[1]

        abstr_r_ground_truth = self.calc_abstr_r_ground_truth(r_ground_truth)
        abstr_term_ground_truth = self.calc_abstr_term_ground_truth(term_ground_truth)

        pred_mixed = self(o_ground_truth, a_ground_truth, r_ground_truth, term_ground_truth, abstr_r_ground_truth,
                          abstr_term_ground_truth, self.n_warmup_prim, self.n_warmup_abstr)
        #pred_post = self(o_ground_truth, a_ground_truth, r_ground_truth, term_ground_truth, abstr_r_ground_truth,
        #                 abstr_term_ground_truth, prim_steps, abstr_steps)

        beta = 1  # min((self._current_train_step / self.n_warmup_schedule), 1)

        # primitive model loss
        loss_mixed = self.calc_loss(pred_mixed, o_ground_truth, term_ground_truth, abstr_r_ground_truth,
                                    abstr_term_ground_truth, r_ground_truth, beta)
        #loss_post = self._calc_loss(pred_post, o_ground_truth, term_ground_truth, abstr_r_ground_truth,
        #                            abstr_term_ground_truth, r_ground_truth, beta)

        #prim_s_prior = build_gaussian(pred_mixed['prim_z_prior'])
        #prim_s_post = build_gaussian(pred_post['prim_z_post'])
        #prim_consistency = self.beta_kl_prim * (torch.distributions.kl_divergence(prim_s_post, prim_s_prior).mean())

        #abstr_s_prior = build_gaussian(pred_mixed['abstr_z_prior'])
        #abstr_s_post = build_gaussian(pred_post['abstr_z_post'])
        #abstr_consistency = self.beta_kl_abstr * (torch.distributions.kl_divergence(abstr_s_post, abstr_s_prior).mean())

        #loss_mixed['total'] += loss_post['total'] + prim_consistency + abstr_consistency

        return loss_mixed
        #return loss_post

    def calc_loss(self, pred, o_ground_truth, term_ground_truth, abstr_r_ground_truth, abstr_term_ground_truth,
                  r_ground_truth, beta):
        d_batch, d_time = o_ground_truth.shape[:2]
        #prim_rec_o = torch.nn.functional.mse_loss(pred['prim_o'], o_ground_truth)
        #prim_rec_r = torch.nn.functional.mse_loss(pred['prim_r'], r_ground_truth)
        prim_rec_term = torch.nn.functional.binary_cross_entropy(pred['prim_term'], term_ground_truth)
        prim_rec_o = - build_gaussian(pred['prim_o_dist']).log_prob(o_ground_truth).mean()
        #prim_rec_o /= (d_batch * d_time)
        prim_rec_r = - build_gaussian(pred['prim_r_dist']).log_prob(r_ground_truth).mean()
        #prim_rec_r /= (d_batch * d_time)

        #prim_rec_term = - build_bernoulli(pred['prim_term']).log_prob(term_ground_truth).mean()
        #prim_rec_term = torch.nn.functional.binary_cross_entropy(pred['prim_term'], term_ground_truth)
        prim_s_prior = build_gaussian(pred['prim_z_prior'])
        prim_s_post = build_gaussian(pred['prim_z_post'])
        unit_gaussian_prim = torch.distributions.Normal(loc=torch.zeros_like(prim_s_post.loc),
                                                        scale=torch.ones_like(prim_s_post.scale))
        prim_kl_s = beta * self.beta_kl_prim * (torch.distributions.kl_divergence(prim_s_post, prim_s_prior).mean())
        # + torch.distributions.kl_divergence(prim_s_prior, prim_s_post).mean())
        prim_kl_s_reg = beta * self.beta_reg_prim * torch.distributions.kl_divergence(prim_s_post,
                                                                                      unit_gaussian_prim).mean()
        # ctx_low_level_dist = build_gaussian(pred_mixed['ctx_low_level'].detach())
        # abstr_o_dist = build_gaussian(pred_mixed['abstr_o_dist'])
        # abstr_rec_o = self.beta_abstract_model * torch.distributions.kl_divergence(ctx_low_level_dist, abstr_o_dist).mean()
        # abstr_rec_o = torch.nn.functional.mse_loss(pred_mixed['abstr_o'], pred_mixed['ctx_low_level'])

        # TODO: Think about this, right now if the final chunk takes less than abstract_step_size steps,
        # the abstract model is trained towards predicting prim_s after less than abstract_step_size steps.
        # This predicts the correct prim_s, but is inconsistent with the rest of the training procedure.

        # abstr_o_target = bin_every_k_steps(pred_mixed['prim_s'], self.abstract_step_size, padding_val=0).detach()
        # abstr_o_target = abstr_o_target[:, :, -1]
        # diff = torch.sum(abstr_o_target - pred_mixed['abstr_o_target'], dim=(1, 2))
        #abstr_rec_o = torch.nn.functional.mse_loss(pred['abstr_o'], pred['abstr_o_target'])
        #abstr_rec_r = torch.nn.functional.mse_loss(pred['abstr_r'], abstr_r_ground_truth)
        abstr_rec_term = torch.nn.functional.binary_cross_entropy(pred['abstr_term'], abstr_term_ground_truth)
        abstr_rec_o = - build_gaussian(pred['abstr_o_dist']).log_prob(pred['abstr_o_target']).mean()
        #abstr_rec_o /= (d_batch * d_time)
        abstr_rec_r = - build_gaussian(pred['abstr_r_dist']).log_prob(abstr_r_ground_truth).mean()
        #abstr_rec_r /= (d_batch * d_time)
        #abstr_rec_term = torch.nn.functional.binary_cross_entropy(pred['abstr_term'], abstr_term_ground_truth)

        abstr_s_prior = build_gaussian(pred['abstr_z_prior'])
        abstr_s_post = build_gaussian(pred['abstr_z_post'])
        unit_gaussian_abstr = torch.distributions.Normal(loc=torch.zeros_like(abstr_s_post.loc),
                                                         scale=torch.ones_like(abstr_s_post.scale))
        abstr_kl_s = beta * self.beta_kl_abstr * (torch.distributions.kl_divergence(abstr_s_post, abstr_s_prior).mean())
        # + torch.distributions.kl_divergence(abstr_s_prior, abstr_s_post).mean())
        abstr_kl_s_reg = beta * self.beta_reg_abstr * torch.distributions.kl_divergence(abstr_s_post,
                                                                                        unit_gaussian_abstr).mean()
        # abstract action model loss
        scale = pred['abstr_a'].reshape(-1, self.abstract_model.d_action).std(dim=0, unbiased=False)
        abstr_a_loss = torch.distributions.Normal(loc=0.0, scale=scale + 0.001).entropy()
        abstr_a_loss = -torch.sum(torch.abs(abstr_a_loss))
        abstr_a_loss *= self.beta_abstract_action

        if pred['abstr_o'].shape[1] > 1:
            abstr_factor = 1
        else:
            abstr_factor = 0  # disable abstract model loss in case we only use the primitive level

        total = prim_rec_o + prim_rec_r + prim_rec_term + prim_kl_s + prim_kl_s_reg
        total += abstr_factor * (abstr_rec_o + abstr_rec_r + abstr_rec_term + abstr_kl_s + abstr_kl_s_reg
                                 + abstr_a_loss)
        prim_o_mae = torch.mean(torch.abs(pred['prim_o'] - o_ground_truth))
        prim_r_mae = torch.mean(torch.abs(pred['prim_r'] - r_ground_truth))
        prim_term_mae = torch.mean(torch.abs(pred['prim_term'] - term_ground_truth))
        prim_kl_unscaled = torch.distributions.kl_divergence(prim_s_post, prim_s_prior).mean()
        prim_kl_reg_unscaled = torch.distributions.kl_divergence(prim_s_post, unit_gaussian_prim).mean()
        abstr_o_mae = torch.mean(torch.abs(pred['abstr_o'] - pred['abstr_o_target']))
        abstr_r_mae = torch.mean(torch.abs(pred['abstr_r'] - abstr_r_ground_truth))
        abstr_term_mae = torch.mean(torch.abs(pred['abstr_term'] - abstr_term_ground_truth))
        abstr_kl_unscaled = torch.distributions.kl.kl_divergence(abstr_s_post, abstr_s_prior).mean()
        abstr_kl_reg_unscaled = torch.distributions.kl_divergence(abstr_s_post, unit_gaussian_abstr).mean()

        return {'total': total, 'prim_o': prim_rec_o, 'prim_r': prim_rec_r, 'prim_term': prim_rec_term,
                'prim_kl_s': prim_kl_s, 'prim_kl_s_reg': prim_kl_s_reg, 'abstr_o': abstr_rec_o,
                'abstr_r': abstr_rec_r, 'abstr_term': abstr_rec_term, 'abstr_kl_s': abstr_kl_s,
                'abstr_kl_s_reg': abstr_kl_s_reg, 'abstr_a_var': abstr_a_loss, 'monitoring_prim_o': prim_o_mae,
                'monitoring_prim_r': prim_r_mae, 'monitoring_prim_term': prim_term_mae,
                'monitoring_prim_kl': prim_kl_unscaled, 'monitoring_prim_kl_reg': prim_kl_reg_unscaled,
                'monitoring_abstr_o': abstr_o_mae, 'monitoring_abstr_r': abstr_r_mae,
                'monitoring_abstr_term': abstr_term_mae, 'monitoring_abstr_kl': abstr_kl_unscaled,
                'monitoring_abstr_kl_reg': abstr_kl_reg_unscaled}

    def input_compatible(self,
                         o_ground_truth: torch.Tensor,
                         a_ground_truth: torch.Tensor,
                         r_ground_truth: torch.Tensor) -> Tuple[bool, str]:
        # laziness ahead
        return True, ''

    def filter_rnn_state(self,
                         h: RnnStateType):
        if isinstance(h, tuple):
            return h[0][-1]
        else:
            return h[-1]

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
            reconstructed = filtered_h.reshape(d_batch, 2 * self.primitive_model.n_hidden_layers,
                                               self.primitive_model.d_h)
            reconstructed = reconstructed.transpose(0, 1)
            reconstructed = torch.tensor_split(reconstructed, 2, dim=0)
            reconstructed = reconstructed[0].contiguous(), reconstructed[1].contiguous()
        else:
            reconstructed = filtered_h.reshape(d_batch, self.primitive_model.n_hidden_layers,
                                               self.primitive_model.d_h)
            reconstructed = reconstructed.transpose(0, 1)
            reconstructed = reconstructed.contiguous()
        return reconstructed

    def rollout_primitive(self,
                          a: torch.Tensor,
                          o: Optional[torch.Tensor] = None,
                          r: Optional[torch.Tensor] = None,
                          term: Optional[torch.Tensor] = None,
                          z: Optional[torch.Tensor] = None,
                          rnn_state: Optional[RnnStateType] = None,
                          mem: Optional[Dict] = None,
                          sample: bool = True,
                          n_posterior_steps: int = -1):
        d_batch, n_steps = a.shape[:2]
        prim_current = self.primitive_model.gen_init_values(d_batch, self.device)

        if z is not None:
            prim_current['z'] = z
        if rnn_state is not None:
            prim_current['rnn_state'] = rnn_state

        if mem is None:
            mem = self.gen_mem()

        if None in (o, r, term):
            n_groundtruth_available = 0
        else:
            assert o is not None and r is not None and term is not None, 'Need o, r, and term groundtruth'
            assert o.shape[1] == r.shape[1] == term.shape[1], 'All groundtruth data has to have the same length'
            n_groundtruth_available = o.shape[1]

        # default argument means we use as much ground truth data as possible with the posterior
        if n_posterior_steps == -1:
            n_posterior_steps = n_groundtruth_available
        #elif n_posterior_steps > n_groundtruth_available:
        #    raise ValueError(f'Can\'t perform more posterior steps as groundtruth data is available.')

        # prediction
        for t in range(n_steps):
            if t < n_groundtruth_available:
                x_posterior = torch.concat([o[:, t], r[:, t], term[:, t]], dim=-1)
            else:
                x_posterior = torch.zeros(d_batch, self.primitive_model.d_observation + self.primitive_model.d_reward
                                          + 1, device=self.device)

            use_posterior = t < n_posterior_steps
            #if not use_posterior and t < n_groundtruth_available and self.training:
            #    use_posterior = True
            #    use_posterior = torch.rand(()) < 0.5

            prim_current['a'] = a[:, t]
            self._invoke_primitive_model(prim_current, x_posterior, mem, use_posterior=use_posterior,
                                         sample=sample)

        return mem, prim_current

    def rollout_abstract(self,
                         a: torch.Tensor,
                         prim_data: Optional[torch.Tensor] = None,
                         r: Optional[torch.Tensor] = None,
                         term: Optional[torch.Tensor] = None,
                         z: Optional[torch.Tensor] = None,
                         rnn_state: Optional[RnnStateType] = None,
                         mem: Optional[Dict] = None,
                         sample: bool = True,
                         n_posterior_steps: int = -1):
        d_batch, n_steps = a.shape[:2]
        abstr_current = self.abstract_model.gen_init_values(d_batch, self.device)

        if z is not None:
            abstr_current['z'] = z
        if rnn_state is not None:
            abstr_current['rnn_state'] = rnn_state

        if mem is None:
            mem = self.gen_mem()

        # prediction
        if prim_data is None:
            n_groundtruth_available = 0
        else:
            n_groundtruth_available = prim_data.shape[1]

        # default argument means we use as much ground truth data as possible with the posterior
        if n_posterior_steps == -1:
            n_posterior_steps = n_groundtruth_available
        #elif n_posterior_steps > n_groundtruth_available:
        #    raise ValueError(f'Can\'t perform more posterior steps as groundtruth data is available.')

        #if a.min() < -1 or a.max() > 1:
        #    raise ValueError('Abstract actions should not contain values outside of the interval [-1, 1]')

        for t in range(n_steps):
            if t < n_groundtruth_available:
                x_posterior = torch.concat([prim_data[:, t], r[:, t], term[:, t]], dim=-1)
            else:
                x_posterior = torch.zeros(d_batch, self.abstract_model.d_observation + self.abstract_model.d_reward
                                          + 1, device=self.device)

            use_posterior = t < n_posterior_steps
            #if not use_posterior and t < n_groundtruth_available and self.training:
            #    use_posterior = True
            #    use_posterior = torch.rand(()) < 0.5

            abstr_current['a'] = a[:, t]
            self._invoke_abstract_model(abstr_current, x_posterior, mem, use_posterior=use_posterior, sample=sample)

        return mem, abstr_current
