import random
from typing import Sequence, List, Dict

import torch
from torch.nn import ModuleList, ModuleDict

from mdm.utils.torch_tools import layers_with_activation, RnnStateType, FuzzyDeviceMixin
from mdm.models.building_blocks import *
from mdm.models.dynamics_model import DynamicsModel
from mdm.utils.torch_tools import get_dist_params, detach_dist
from mdm.utils.utils import expand_shape_right


class StandardRSSM(torch.nn.Module):

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
                o_current: torch.Tensor,
                r_current: torch.Tensor,
                term_current: torch.Tensor,
                last_state: Dict[str, torch.Tensor],
                context: Optional[torch.Tensor] = None,
                use_posterior: bool = True,
                reconstruct: bool = True,
                sample_state: bool = True,
                sample_output: bool = True):
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

        reconstruction = {'o': o_smpl, 'o_dist': o_dist, 'r_dist': r_dist, 'r': r_smpl, 'term_dist': term_dist,
                          'term': term_smpl, 's': s, 'h': h}

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

    def __init__(self,
                 rssm_modules: Sequence[StandardRSSM],
                 links: Sequence[str],  # links associate output from one lvl below with inputs on this lvl
                 upwards_filters: Sequence[Dict[str, UpwardsFilter]],
                 warmup_steps: Sequence[Union[int, str]],
                 kl_betas: Sequence[float]):
        super(HierarchicalRSSM, self).__init__()

        assert len(links) == len(rssm_modules) - 1
        assert len(upwards_filters) == len(rssm_modules) - 1
        for filters in upwards_filters:
            window_sizes = set([f.window_size for f in filters.values()])
            assert len(window_sizes) == 1

        lvl_k_link = 'o'  # only here to make the loop in eval_step() method work
        lvl_0_filters = {k: IdentityUpwardsFilter() for k in ('o', 'a', 'r', 'term', 'mask')}
        for level in upwards_filters:
            mask_filter = {'mask': MinUpwardsFilter(level['o'].window_size)}
            level.update(mask_filter)
        upwards_filters = [ModuleDict(lvl_0_filters)] + [ModuleDict(x) for x in upwards_filters]

        self.rssm_modules = ModuleList(list(rssm_modules))
        self.links = [*links, lvl_k_link]
        self.upwards_filters = ModuleList(upwards_filters)
        self.warmup_steps = warmup_steps
        self.kl_betas = tuple(kl_betas)

    @property
    def strides(self):
        return [filters['o'].window_size for filters in self.upwards_filters[1:]]

    def forward(self, o, a, r, term, n_warmup: int = -1, level: int = 0, memory: Optional[dict] = None,
                start_state: Optional[dict] = None, sample_state: bool = True, sample_output: bool = True,
                reconstruct: bool = True):
        assert o.shape[0] == r.shape[0] == term.shape[0]

        device = self.device
        n_pred_steps, d_batch = a.shape[:2]
        n_groundtruth_steps = o.shape[0]
        mdl = self.rssm_modules[level]
        mem = {} if memory is None else memory
        state = mdl.init_state(d_batch, device) if start_state is None else start_state
        n_warmup = n_warmup if n_warmup > 0 else n_pred_steps

        for t, a_t in enumerate(a):
            if t < n_groundtruth_steps and t < n_warmup:
                o_t, r_t, term_t = o[t], r[t], term[t]
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

        return mem, state

    def _train_step(self,
                    o_ground_truth: torch.Tensor,
                    a_ground_truth: torch.Tensor,
                    r_ground_truth: torch.Tensor,
                    term_ground_truth: torch.Tensor,
                    mask: torch.Tensor,
                    optimizer: torch.optim.Optimizer,
                    **kwargs) -> Dict[str, torch.Tensor]:
        optimizer.zero_grad(set_to_none=True)
        losses = self.eval_step(o_ground_truth, a_ground_truth, r_ground_truth, term_ground_truth, mask)
        losses['total'].backward()
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()

        return losses

    def _eval_step(self,
                   o_ground_truth: torch.Tensor,
                   a_ground_truth: torch.Tensor,
                   r_ground_truth: torch.Tensor,
                   term_ground_truth: torch.Tensor,
                   mask: torch.Tensor,
                   **kwargs) -> Dict[str, torch.Tensor]:
        # execute all levels
        pred = []
        targets = []
        inp_lvl = {'o': o_ground_truth, 'a': a_ground_truth, 'r': r_ground_truth, 'term': term_ground_truth}
        for i_level, (filters, link, n_warmup) in enumerate(zip(self.upwards_filters, self.links, self.warmup_steps)):
            # prep current lvl input
            filtered_inp_level = {k: filters[k](inp_lvl[k]) for k in inp_lvl}
            if n_warmup == 'rand':
                n_warmup = random.randint(1, filtered_inp_level['o'].shape[0])
            # do prediction
            mem, _ = self(**filtered_inp_level, n_warmup=n_warmup, level=i_level, sample_state=True,
                          sample_output=True, reconstruct=True)
            # prep next lvl input
            inp_lvl = {'o': torch.stack(mem[link]), 'a': filtered_inp_level['a'], 'r': torch.stack(mem['r']),
                       'term': torch.stack(mem['term'])}

            pred.append(mem)
            targets.append(filtered_inp_level)

        # compute losses
        losses = {}
        mask_lvl = mask
        for i_level, (pred_lvl, tarted_lvl, filters) in enumerate(zip(pred, targets, self.upwards_filters)):
            mask_lvl = filters['mask'](mask_lvl)
            loss_level = self.calc_loss(pred_lvl, tarted_lvl, mask_lvl, self.kl_betas[i_level])
            loss_level = {k + f'_{i_level}': v for k, v in loss_level.items()}
            losses.update(loss_level)
        losses['total'] = torch.stack([v for k, v in losses.items() if k.startswith('total')]).mean()

        return losses

    def calc_loss(self,
                  predictions: Dict[str, List[Union[torch.Tensor, torch.distributions.Distribution]]],
                  targets: Dict[str, torch.Tensor],
                  mask: torch.Tensor,
                  kl_beta: float,
                  kl_reg_beta: float = 0.0):
        mask = 1 - mask  # use mask to multiply irrelevant steps with zero
        rec_o = self._neg_log_prob(predictions['o_dist'], targets['o'], mask)
        rec_r = self._neg_log_prob(predictions['r_dist'], targets['r'], mask)
        rec_term = self._neg_log_prob(predictions['term_dist'], targets['term'], mask)
        kl_z = self._kl_div(predictions['z_post'], predictions['z_prior'], mask)
        kl_reg_z = self._kl_reg(predictions['z_post'], mask)

        mae_o = self._mae(predictions['o'], targets['o'], mask)
        mae_r = self._mae(predictions['r'], targets['r'], mask)
        mae_term = self._mae(predictions['term'], targets['term'], mask)

        total = rec_o + rec_r + rec_term + kl_z * kl_beta + kl_reg_z * kl_reg_beta
        return {'total': total, 'o': rec_o, 'r': rec_r, 'term': rec_term, 'kl_z': kl_z, 'kl_reg_z': kl_reg_z,
                'monitoring_o': mae_o, 'monitoring_r': mae_r, 'monitoring_term': mae_term}

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
                detach_posterior: bool = False):
        if detach_posterior:
            ps = [detach_dist(p) for p in ps]
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
