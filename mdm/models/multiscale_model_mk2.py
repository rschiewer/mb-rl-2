from typing import Dict, Optional
import random
from math import cos

import torch

from mdm.models.building_blocks import AbstractActionModel, RSSM, RnnStateType, InputEncoder, OutputDecoder
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
                 beta_sched_stay: int = 1000,
                 beta_sched_rise: int = 1000,
                 n_warmup_prim: Union[int, Sequence[int]] = 1,
                 n_warmup_abstr: Union[int, Sequence[int]] = 1,
                 latent_overshooting: bool = False,
                 overshooting_stride: int = 1,
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
        self.beta_kl_prim = beta_kl_primitive
        self.beta_reg_prim = beta_reg_primitive
        self.beta_kl_abstr = beta_kl_abstract
        self.beta_reg_abstr = beta_reg_abstract
        self.beta_abstract_model = beta_abstract_model
        self.beta_abstract_action = beta_abstract_action
        self.beta_stay = beta_sched_stay
        self.beta_rise = beta_sched_rise
        self.n_warmup_prim = n_warmup_prim
        self.n_warmup_abstr = n_warmup_abstr
        self.latent_overshooting = latent_overshooting
        self.overshooting_stride = overshooting_stride
        self.detach_posteriors = detach_posteriors
        self._current_train_step = None

    def prepare_for_training(self):
        self._current_train_step = 0
        for m in self.modules():
            if isinstance(m, ManagedStatefulTrainingModule):
                m.prepare_for_training()

    def forward(self,
                o: torch.Tensor,
                a: torch.Tensor,
                r: torch.Tensor,
                term: torch.Tensor,
                abstr_r: torch.Tensor,
                abstr_term: torch.Tensor,
                n_warmup_prim: int,
                n_warmup_abstr: int,
                prim_z_start: torch.Tensor = None,
                prim_rnn_state_start: torch.Tensor = None,
                abstr_z_start: torch.Tensor = None,
                abstr_rnn_state_start: torch.Tensor = None,
                reconstruct: bool = True,
                sample: bool = True):
        assert o.shape[:1] == r.shape[:1] == term.shape[:1] == a.shape[:1]

        mem = self.gen_mem()
        n_steps_prim, d_batch = a.shape[:2]
        a_binned = bin_every_k_steps(a, self.abstract_step_size, padding_val=0)
        n_steps_abstr = a_binned.shape[0]

        prim_current = self.primitive_model.gen_init_values(d_batch, self.device)
        if prim_z_start is not None:
            prim_current['z'] = prim_z_start
        if prim_rnn_state_start is not None:
            prim_current['rnn_state'] = prim_rnn_state_start
        abstr_current = self.abstract_model.gen_init_values(d_batch, self.device)
        if abstr_z_start is not None:
            abstr_current['z'] = abstr_z_start
        if abstr_rnn_state_start is not None:
            abstr_current['rnn_state'] = abstr_rnn_state_start

        warmup_steps_prim_left = n_warmup_prim
        warmup_steps_abstr_left = n_warmup_abstr
        for i_chunk in range(n_steps_abstr):
            i_start = i_chunk * self.abstract_step_size
            i_end = min((i_chunk + 1) * self.abstract_step_size, n_steps_prim)

            # primitive model rollout
            # n_warmup_prim = min(warmup_steps_prim_left, self.abstract_step_size)
            mem, prim_current = self.rollout_primitive(a=a[i_start: i_end], o=o[i_start: i_end],
                                                       r=r[i_start: i_end], term=term[i_start: i_end],
                                                       z=prim_current['z'], rnn_state=prim_current['rnn_state'],
                                                       n_posterior_steps=warmup_steps_prim_left,
                                                       mem=mem, reconstruct=reconstruct, sample=sample)

            # input to abstr act mdl needs to be always of same length, so take a_binned instead of a[i_start: i_end]
            abstr_a = self.abstract_action_model(a_binned[i_chunk].swapaxes(0, 1))

            # abstract model rollout
            # ctx_low_level = self.fuse_state(prim_current['z'], prim_current['rnn_state']).detach()
            # mem['ctx_low_level'].append(ctx_low_level)
            prim_data = mem[self.abstr_pred_target][-1].detach()
            #prim_data = o[i_end - 1]
            mem['abstr_o_target'].append(prim_data)
            abstr_r_groundtruth = torch.stack(mem['prim_r'][i_start:i_end], dim=0)
            abstr_r_groundtruth = self.calc_abstr_r_ground_truth(abstr_r_groundtruth)
            abstr_term_groundtruth = torch.stack(mem['prim_term'][i_start:i_end], dim=0)
            abstr_term_groundtruth = self.calc_abstr_term_ground_truth(abstr_term_groundtruth)
            #abstr_r_groundtruth = abstr_r[i_chunk].unsqueeze(0)
            #abstr_term_groundtruth = abstr_term[i_chunk].unsqueeze(0)

            mem, abstr_current = self.rollout_abstract(a=add_time_dim(abstr_a), r=abstr_r_groundtruth,
                                                       term=abstr_term_groundtruth,
                                                       prim_data=add_time_dim(prim_data),
                                                       z=abstr_current['z'],
                                                       rnn_state=abstr_current['rnn_state'],
                                                       n_posterior_steps=warmup_steps_abstr_left,
                                                       mem=mem, reconstruct=reconstruct, sample=sample)
            # abstr_r_groundtruth = self.calc_abstr_r_ground_truth(abstr_r_groundtruth)
            # abstr_term_groundtruth = self.calc_abstr_term_ground_truth(abstr_term_groundtruth)
            # mem, abstr_current = self.rollout_abstract(a=add_time_dim(abstr_a), r=add_time_dim(abstr_r[:, i_chunk]),
            #                                           term=add_time_dim(abstr_term[:, i_chunk]),
            #                                           prim_data=add_time_dim(prim_data),
            #                                           z=abstr_current['z'],
            #                                           rnn_state=abstr_current['rnn_state'],
            #                                           n_posterior_steps=warmup_steps_abstr_left,
            #                                           mem=mem, sample=True)

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
            # warmup_steps_prim_left = max(warmup_steps_prim_left - self.abstract_step_size, 0)
            # warmup_steps_abstr_left = max(warmup_steps_abstr_left - 1, 0)

        # mem = self.pack_mem(mem)

        return mem

    def _invoke_primitive_model(self,
                                prim_current: Dict,
                                o: Optional[torch.Tensor],
                                r: Optional[torch.Tensor],
                                term: Optional[torch.Tensor],
                                mem: Dict,
                                use_posterior: bool = True,
                                sample: bool = True,
                                reconstruct: bool = True):
        d_batch = prim_current['a'].shape[0]
        ctx_high_level = self.primitive_model.zero_ctx_high_level(d_batch, self.device)

        # do prediction
        pred = self.primitive_model(z=prim_current['z'], a=prim_current['a'], o_current=o, r_current=r,
                                    term_current=term, ctx_high_level=ctx_high_level,
                                    rnn_state=prim_current['rnn_state'], use_posterior=use_posterior,
                                    reconstruct=reconstruct, sample=sample)

        # sometimes use s_prior to provide next step's s
        # d_batch = pred['z'].shape[0]
        # pred['z'] = torch.where(torch.rand(d_batch, 1, device=self.device) < 0.2, pred['s_prior'].rsample(), pred['z'])

        # update primitive state
        prim_current['z'] = pred['z']
        prim_current['rnn_state'] = pred['rnn_state']

        # store things
        mem['prim_z'].append(pred['z'])
        mem['prim_z_prior'].append(pred['z_prior'])
        mem['prim_z_post'].append(pred['z_post'])
        # if use_posterior:
        #    mem['prim_z_post'].append(pred['z_post'])
        # else:
        #    mem['prim_z_post'].append(None)  # hack to make loss calculation easier
        mem['prim_rnn_state'].append(pred['rnn_state'])
        mem['prim_h'].append(pred['h'])
        mem['prim_s'].append(pred['s'])
        mem['prim_a'].append(prim_current['a'])
        mem['prim_o'].append(pred['o'])
        mem['prim_o_dist'].append(pred['o_dist'])
        mem['prim_r'].append(pred['r'])
        mem['prim_r_dist'].append(pred['r_dist'])
        mem['prim_term'].append(pred['term'])
        mem['prim_term_dist'].append(pred['term_dist'])

    def _invoke_abstract_model(self,
                               abstr_current: Dict,
                               o_abstr: Optional[torch.Tensor],
                               r_abstr: Optional[torch.Tensor],
                               term_abstr: Optional[torch.Tensor],
                               mem: Dict,
                               use_posterior: bool = True,
                               sample: bool = True,
                               reconstruct: bool = True):
        d_batch = abstr_current['a'].shape[0]
        ctx_high_level = self.abstract_model.zero_ctx_high_level(d_batch, self.device)

        # do prediction
        pred = self.abstract_model(z=abstr_current['z'], a=abstr_current['a'],
                                   o_current=o_abstr, r_current=r_abstr, term_current=term_abstr,
                                   ctx_high_level=ctx_high_level, rnn_state=abstr_current['rnn_state'],
                                   use_posterior=use_posterior, reconstruct=reconstruct, sample=sample)

        # update abstract state
        abstr_current['z'] = pred['z']
        abstr_current['rnn_state'] = pred['rnn_state']

        # store things
        mem['abstr_z'].append(pred['z'])
        mem['abstr_z_prior'].append(pred['z_prior'])
        mem['abstr_z_post'].append(pred['z_post'])
        # if use_posterior:
        #    mem['abstr_z_post'].append(pred['z_post'])
        # else:
        #    mem['abstr_z_post'].append(pred['z_prior'])  # hack to make loss calculation easier
        mem['abstr_rnn_state'].append(pred['rnn_state'])
        mem['abstr_h'].append(pred['h'])
        mem['abstr_s'].append(pred['s'])
        mem['abstr_a'].append(abstr_current['a'])
        mem['abstr_o'].append(pred['o'])
        mem['abstr_o_dist'].append(pred['o_dist'])
        mem['abstr_r'].append(pred['r'])
        mem['abstr_r_dist'].append(pred['r_dist'])
        mem['abstr_term'].append(pred['term'])
        mem['abstr_term_dist'].append(pred['term_dist'])

    @staticmethod
    def gen_mem():
        mem = {'prim_a': [], 'prim_o': [], 'prim_o_dist': [], 'prim_r': [], 'prim_r_dist': [], 'prim_term': [],
               'prim_term_dist': [], 'prim_z_prior': [], 'prim_h': [], 'prim_z_post': [], 'prim_z': [], 'prim_s': [],
               'prim_rnn_state': [], 'abstr_o': [], 'abstr_o_dist': [], 'abstr_a': [], 'abstr_h': [],
               'abstr_a_dist': [], 'abstr_r': [], 'abstr_r_dist': [], 'abstr_term': [],
               'abstr_term_dist': [], 'abstr_z_prior': [], 'abstr_z_post': [], 'abstr_z': [], 'abstr_s': [],
               'abstr_rnn_state': [], 'abstr_o_target': []}
        return mem

    @staticmethod
    def pack_mem(mem):
        for k, v in mem.items():
            if isinstance(v, List) and len(v) > 0:
                if k.endswith('_rnn_state') and isinstance(v[0], (tuple, torch.Tensor)):
                    mem[k] = torch.stack([pack_rnn_state(state) for state in v], dim=0)
                elif isinstance(v[0], torch.Tensor):
                    mem[k] = torch.stack(mem[k], dim=0)
                # elif isinstance(v[0], torch.distributions.Normal):
                #    mem[k] =  [torch.concat([d.loc, d.scale]) for d in v]
        return mem

    def train_step(self,
                   o_ground_truth: torch.Tensor,
                   a_ground_truth: torch.Tensor,
                   r_ground_truth: torch.Tensor,
                   term_ground_truth: torch.Tensor,
                   mask: torch.Tensor,
                   optimizer: torch.optim.Optimizer) -> Dict[str, torch.Tensor]:
        optimizer.zero_grad(set_to_none=True)
        losses = self.eval_step(o_ground_truth, a_ground_truth, r_ground_truth, term_ground_truth, mask)
        losses['total'].backward()
        #torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()

        self._current_train_step += 1
        for m in self.modules():
            if isinstance(m, ManagedStatefulTrainingModule):
                m.increase_train_step()

        return losses

    def calc_abstr_a(self, a: torch.Tensor):
        a_binned = bin_every_k_steps(a, self.abstract_step_size, padding_val=0)

        a_abstr = [self.abstract_action_model(a_bin.swapaxes(0, 1)) for a_bin in a_binned]
        a_abstr = torch.stack(a_abstr)
        return a_abstr

    def calc_abstr_r_ground_truth(self, r_ground_truth: torch.Tensor):
        # sum makes sense for rewards, use padding=0 to not affect sum for last element
        return bin_every_k_steps(r_ground_truth, self.abstract_step_size, padding_val=0).sum(dim=1)

    def calc_abstr_term_ground_truth(self, term_ground_truth: torch.Tensor):
        # terminal flag can only be 0 or 1, so mean value with automatic padding should be used
        return bin_every_k_steps(term_ground_truth, self.abstract_step_size, padding_val=None).max(dim=1).values

    def _calc_beta_schedule(self,
                            t: int,
                            max_beta: float):
        """
        Calculates truncated sawtooth beta value like this:
            ___    ___    ___
          /   |  /   |  /   |
        /     |/     |/     | ...
        """
        beta = min(t % (self.beta_rise + self.beta_stay) * max_beta / self.beta_rise, max_beta)
        return beta

    def eval_step(self,
                  o_ground_truth: torch.Tensor,
                  a_ground_truth: torch.Tensor,
                  r_ground_truth: torch.Tensor,
                  term_ground_truth: torch.Tensor,
                  mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        abstr_steps = ceil(a_ground_truth.shape[0] / self.abstract_step_size)

        abstr_r_ground_truth = self.calc_abstr_r_ground_truth(r_ground_truth)
        abstr_term_ground_truth = self.calc_abstr_term_ground_truth(term_ground_truth)
        beta = self._calc_beta_schedule(self._current_train_step, self.beta_kl_prim)

        pred_post = self(o_ground_truth, a_ground_truth, r_ground_truth, term_ground_truth, abstr_r_ground_truth,
                         abstr_term_ground_truth, n_warmup_prim=-1, n_warmup_abstr=-1, sample=True)
        loss = self.calc_loss(pred_post, o_ground_truth, r_ground_truth, term_ground_truth, abstr_r_ground_truth,
                              abstr_term_ground_truth, mask, beta)
        loss['beta'] = torch.tensor(beta)

        if self.latent_overshooting:
            for i_chunk in range(1, abstr_steps, self.abstract_step_size * self.overshooting_stride):
                t = i_chunk * self.abstract_step_size
                prim_z = pred_post['prim_z'][t]
                prim_rnn_state = pred_post['prim_rnn_state'][t]
                abstr_z = pred_post['abstr_z'][i_chunk]
                abstr_rnn_state = pred_post['abstr_rnn_state'][i_chunk]
                pred_latent = self(o_ground_truth[t:], a_ground_truth[t:], r_ground_truth[t:],
                                   term_ground_truth[t:], abstr_r_ground_truth[i_chunk:],
                                   abstr_term_ground_truth[i_chunk:], prim_z_start=prim_z,
                                   prim_rnn_state_start=prim_rnn_state,
                                   abstr_z_start=abstr_z, abstr_rnn_state_start=abstr_rnn_state,
                                   n_warmup_prim=self.abstract_step_size,
                                   n_warmup_abstr=1, sample=True)
                mask_prim = 1 - mask[t:]
                mask_abstr = bin_every_k_steps(mask_prim, self.abstract_step_size).max(dim=1).values
                if self.detach_posteriors:
                    prim_kl_target = [detach_dist(d) for d in pred_post['prim_z_post'][t:]]
                    abstr_kl_target = [detach_dist(d) for d in pred_post['abstr_z_post'][i_chunk:]]
                else:
                    prim_kl_target = pred_post['prim_z_post'][t:]
                    abstr_kl_target = pred_post['abstr_z_post'][i_chunk:]
                kl_prim = self._kl_div(prim_kl_target, pred_latent['prim_z_prior'], mask_prim)
                kl_abstr = self._kl_div(abstr_kl_target, pred_latent['abstr_z_prior'], mask_abstr)
                loss['prim_kl_s'] += beta * self.beta_kl_prim * kl_prim
                loss['abstr_kl_s'] += beta * self.beta_kl_abstr * kl_abstr
                loss['monitoring_prim_kl'] += kl_prim
                loss['monitoring_abstr_kl'] += kl_abstr

            # each of the loss terms below was calculated once for pred_post call and then abstr_steps - 1 times in the loop
            #normalizing_factor = max(abstr_steps, 1)
            #loss['prim_kl_s'] /= normalizing_factor
            #loss['abstr_kl_s'] /= normalizing_factor
            #loss['monitoring_prim_kl'] /= normalizing_factor
            #loss['monitoring_abstr_kl'] /= normalizing_factor

        return loss

    @staticmethod
    def _neg_log_prob(distributions: List[torch.distributions.Distribution],
                      x_target: torch.Tensor,
                      mask: torch.Tensor):
        if isinstance(distributions[0], torch.distributions.RelaxedOneHotCategorical):  # mse as max log prob fails
            # smooth out targets a little bit to avoid inf/nan log probs with RelaxedOneHotCategorical
            x_target = torch.abs(x_target - 1e-5)
            x_target /= x_target.sum(dim=-1, keepdim=True)
            mask = mask.reshape(*mask.shape + (1,) * (x_target.ndim - mask.ndim))  # append size 1 dim for broadcasting
            #neg_log_prob = [(d.rsample() - x) ** 2 * m for d, x, m in zip(distributions, x_target, mask)]
        #else:
        neg_log_prob = [-d.log_prob(x) * m for d, x, m in zip(distributions, x_target, mask)]
        neg_log_prob = torch.stack(neg_log_prob, dim=0).mean()
        return neg_log_prob

    @staticmethod
    def _kl_div(ps: List[torch.distributions.Distribution],
                qs: List[torch.distributions.Distribution],
                mask: torch.Tensor,
                detach_posterior: bool = False):
        if detach_posterior:
            ps = [detach_dist(p) for p in ps]
        kl_div = [torch.distributions.kl_divergence(p, q) * m for p, q, m in zip(ps, qs, mask) if None not in (p, q)]
        kl_div = torch.stack(kl_div, dim=0)
        if len(kl_div) > 1:
            kl_div = kl_div[1:].mean()  # ignore first prior since it's totally uninformed
        else:
            kl_div = torch.tensor(0, dtype=torch.float32)
        #if torch.isnan(kl_div):
        #    print('!!!')
        return kl_div

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
    def _mse(y_hats: List[torch.Tensor], ys: torch.Tensor, mask: torch.Tensor):
        y_hats = torch.stack(y_hats, dim=0)
        mask = mask.reshape(*mask.shape + (1,) * (y_hats.ndim - mask.ndim))  # append size 1 dimensions for broadcasting
        return torch.mean((ys - y_hats) ** 2 * mask)

    @staticmethod
    def _mae(y_hats: List[torch.Tensor], ys: torch.Tensor, mask: torch.Tensor):
        y_hats = torch.stack(y_hats, dim=0)
        mask = mask.reshape(*mask.shape + (1,) * (y_hats.ndim - mask.ndim))  # append size 1 dimensions for broadcasting
        return torch.mean(torch.abs(ys - y_hats) * mask)

    def calc_loss(self, pred, o_ground_truth, r_ground_truth, term_ground_truth, abstr_r_ground_truth,
                  abstr_term_ground_truth, mask, beta):
        assert mask.ndim == 3
        assert mask.shape[-1] == 1

        mask = 1 - mask
        mask_abstr = bin_every_k_steps(mask, self.abstract_step_size).max(dim=1).values

        prim_rec_o = self._neg_log_prob(pred['prim_o_dist'], o_ground_truth, mask)
        prim_rec_r = self._neg_log_prob(pred['prim_r_dist'], r_ground_truth, mask)
        prim_rec_term = self._neg_log_prob(pred['prim_term_dist'], term_ground_truth, mask)
        prim_kl_z_unscaled = self._kl_div(pred['prim_z_post'], pred['prim_z_prior'], mask)
        prim_kl_z = beta * self.beta_kl_prim * prim_kl_z_unscaled
        prim_kl_z_reg_unscaled = self._kl_reg(pred['prim_z_post'], mask)
        prim_kl_z_reg = beta * self.beta_reg_prim * prim_kl_z_reg_unscaled

        # TODO: Think about this, right now if the final chunk takes less than abstract_step_size steps,
        # the abstract model is trained towards predicting prim_s after less than abstract_step_size steps.
        # This predicts the correct prim_s, but is inconsistent with the rest of the training procedure.

        abstr_o_target = torch.stack(pred['abstr_o_target'], dim=0)
        abstr_rec_o = self._neg_log_prob(pred['abstr_o_dist'], abstr_o_target, mask_abstr)
        abstr_rec_r = self._neg_log_prob(pred['abstr_r_dist'], abstr_r_ground_truth, mask_abstr)
        abstr_rec_term = self._neg_log_prob(pred['abstr_term_dist'], abstr_term_ground_truth, mask_abstr)
        abstr_kl_z_unscaled = self._kl_div(pred['abstr_z_post'], pred['abstr_z_prior'], mask_abstr)
        abstr_kl_z = beta * self.beta_kl_abstr * abstr_kl_z_unscaled
        abstr_kl_z_reg_unscaled = self._kl_reg(pred['abstr_z_post'], mask_abstr)
        abstr_kl_z_reg = beta * self.beta_reg_abstr * abstr_kl_z_reg_unscaled

        # disable abstract model loss in case we only use the primitive level
        abstr_factor = 1 if len(pred['abstr_o']) > 1 else 0
        abstr_factor *= self.beta_abstract_model

        total = prim_rec_o + prim_rec_r + prim_rec_term + prim_kl_z + prim_kl_z_reg
        total += abstr_factor * (abstr_rec_o + abstr_rec_r + abstr_rec_term + abstr_kl_z + abstr_kl_z_reg)
        prim_o_mae = self._mae(pred['prim_o'], o_ground_truth, mask)
        prim_r_mae = self._mae(pred['prim_r'], r_ground_truth, mask)
        prim_term_mae = self._mae(pred['prim_term'], term_ground_truth, mask)
        abstr_o_mae = self._mae(pred['abstr_o'], abstr_o_target, mask_abstr)
        abstr_r_mae = self._mae(pred['abstr_r'], abstr_r_ground_truth, mask_abstr)
        abstr_term_mae = self._mae(pred['abstr_term'], abstr_term_ground_truth, mask_abstr)

        return {'total': total, 'prim_o': prim_rec_o, 'prim_r': prim_rec_r, 'prim_term': prim_rec_term,
                'prim_kl_s': prim_kl_z, 'prim_kl_s_reg': prim_kl_z_reg, 'abstr_o': abstr_rec_o,
                'abstr_r': abstr_rec_r, 'abstr_term': abstr_rec_term, 'abstr_kl_s': abstr_kl_z,
                'abstr_kl_s_reg': abstr_kl_z_reg, 'monitoring_prim_o': prim_o_mae,
                'monitoring_prim_r': prim_r_mae, 'monitoring_prim_term': prim_term_mae,
                'monitoring_prim_kl': prim_kl_z_unscaled, 'monitoring_prim_kl_reg': prim_kl_z_reg_unscaled,
                'monitoring_abstr_o': abstr_o_mae, 'monitoring_abstr_r': abstr_r_mae,
                'monitoring_abstr_term': abstr_term_mae, 'monitoring_abstr_kl': abstr_kl_z_unscaled,
                'monitoring_abstr_kl_reg': abstr_kl_z_reg_unscaled}

    def filter_rnn_state(self,
                         h: RnnStateType):
        if isinstance(h, tuple):
            return h[0][-1]
        else:
            return h[-1]

    def rollout_primitive(self,
                          a: torch.Tensor,
                          o: Optional[torch.Tensor] = None,
                          r: Optional[torch.Tensor] = None,
                          term: Optional[torch.Tensor] = None,
                          z: Optional[torch.Tensor] = None,
                          rnn_state: Optional[RnnStateType] = None,
                          mem: Optional[Dict] = None,
                          sample: bool = True,
                          reconstruct: bool = True,
                          n_posterior_steps: int = -1):
        n_steps, d_batch = a.shape[:2]
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
            n_groundtruth_available = o.shape[0]

        # default argument means we use as much ground truth data as possible with the posterior
        if n_posterior_steps == -1:
            n_posterior_steps = n_groundtruth_available

        # prediction
        for t in range(n_steps):
            if t < n_groundtruth_available:
                o_post, r_post, term_post = o[t], r[t], term[t]
            else:
                o_post = self.primitive_model.zero_o(d_batch, self.device)
                r_post = self.primitive_model.zero_r(d_batch, self.device)
                term_post = self.primitive_model.zero_term(d_batch, self.device)

            use_posterior = t < n_posterior_steps
            # if use_posterior and self.training:
            #    use_posterior = torch.rand(()) < 0.2
            # if not use_posterior and t < n_groundtruth_available and self.training:
            #    use_posterior = True
            #    use_posterior = torch.rand(()) < 0.5

            prim_current['a'] = a[t]
            self._invoke_primitive_model(prim_current, o_post, r_post, term_post, mem, use_posterior=use_posterior,
                                         reconstruct=reconstruct, sample=sample)

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
                         reconstruct: bool = True,
                         n_posterior_steps: int = -1):
        n_steps, d_batch = a.shape[:2]
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
            n_groundtruth_available = prim_data.shape[0]

        # default argument means we use as much ground truth data as possible with the posterior
        if n_posterior_steps == -1:
            n_posterior_steps = n_groundtruth_available

        for t in range(n_steps):
            if t < n_groundtruth_available:
                o_post = prim_data[t]
                r_post = r[t]
                term_post = term[t]
            else:
                o_post = self.abstract_model.zero_o(d_batch, self.device)
                r_post = self.abstract_model.zero_r(d_batch, self.device)
                term_post = self.abstract_model.zero_term(d_batch, self.device)

            use_posterior = t < n_posterior_steps
            # if use_posterior and self.training:
            #    use_posterior = torch.rand(()) < 0.2
            # if not use_posterior and t < n_groundtruth_available and self.training:
            #    use_posterior = True
            #    use_posterior = torch.rand(()) < 0.5

            abstr_current['a'] = a[t]
            self._invoke_abstract_model(abstr_current, o_post, r_post, term_post, mem, use_posterior=use_posterior,
                                        reconstruct=reconstruct, sample=sample)

        return mem, abstr_current
