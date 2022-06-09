from typing import Tuple, Optional, Sequence
from enum import Enum

import torch
import haste_pytorch as haste

from mdm.utils.torch_tools import FuzzyDeviceMixin, layers_with_activation, add_time_dim, remove_time_dim


class AbstractActionModel(torch.nn.Module):

    def __init__(self,
                 d_action: int,
                 abstract_step_size: int,
                 d_abstract_action: int,
                 lws: tuple = (64, 64),
                 activation: str = 'relu'):
        super(AbstractActionModel, self).__init__()

        lws = (d_action * abstract_step_size, *lws, d_abstract_action)
        self.flatten_layer = torch.nn.Flatten(start_dim=1)
        self.det_mdl = torch.nn.Sequential(*layers_with_activation(lws, activation))

    def forward(self,
                actions: torch.Tensor) -> torch.Tensor:
        # actions.shape = (d_batch, n_abstract_steps, d_action)
        macro_action = self.flatten_layer(actions)
        macro_action = self.det_mdl(macro_action)
        macro_action = torch.tanh(macro_action)
        # macro_action = torch.softmax(macro_action, dim=-1)
        # macro_action = F.gumbel_softmax(macro_action, hard=True)
        return macro_action


class RSSM(torch.nn.Module):

    def __init__(self,
                 d_state: int,
                 d_action: int,
                 d_observation: int,
                 d_reward: int,
                 d_high_level_ctx: int,
                 d_low_level_ctx: int,
                 d_hidden: int,
                 n_hidden_layers: int = 1,
                 hidden_dropout: float = 0.1,
                 epsilon: float = 0.01,
                 s_prior_lws: Sequence[int] = (32, 32),
                 s_post_lws: Sequence[int] = (32, 32),
                 o_lws: Sequence[int] = (32, 32),
                 r_lws: Sequence[int] = (32, 32),
                 term_lws: Sequence[int] = (32, 32),
                 activation: str = 'relu',
                 layer_norm: bool = False,
                 rnn_type: str = 'lstm'):
        super().__init__()

        self.d_state = d_state
        self.d_action = d_action
        self.d_observation = d_observation
        self.d_reward = d_reward
        self.d_high_level_ctx = d_high_level_ctx
        self.d_low_level_ctx = d_low_level_ctx
        self.d_hidden = d_hidden
        self.n_hidden_layers = n_hidden_layers
        self.epsilon = epsilon
        self.activation = activation
        self.layer_norm = layer_norm
        self.rnn_type = rnn_type

        s_prior_lws = (d_hidden, *s_prior_lws, d_state * 2)
        s_post_lws = (d_hidden, *s_post_lws, d_state * 2)
        o_lws = (d_hidden + d_state, *o_lws, d_observation * 2)
        r_lws = (d_hidden + d_state, *r_lws, d_reward * 2)
        term_lws = (d_hidden + d_state, *term_lws, 1)

        if rnn_type == 'lstm':
            rnn_constr = torch.nn.LSTM
        elif rnn_type == 'gru':
            rnn_constr = torch.nn.GRU
        else:
            raise ValueError(f'Unsupported rnn type: {rnn_type}')
        self.det_core = rnn_constr(d_state + d_action + d_low_level_ctx + d_high_level_ctx, hidden_size=d_hidden,
                                   num_layers=n_hidden_layers, batch_first=True, dropout=hidden_dropout)
        #self.det_core = haste.LayerNormLSTM(d_state + d_action + d_high_level_ctx, hidden_size=d_hidden,
        #                                    zoneout=0.05, dropout=hidden_dropout, batch_first=True)
        if layer_norm:
            self.det_core_norm = torch.nn.LayerNorm(d_hidden)
        else:
            self.det_core_norm = None

        self.s_prior = torch.nn.Sequential(*layers_with_activation(s_prior_lws, activation, layer_norm=layer_norm))
        self.s_post = torch.nn.Sequential(*layers_with_activation(s_post_lws, activation, layer_norm=layer_norm))
        self.r_dist = torch.nn.Sequential(*layers_with_activation(r_lws, activation, layer_norm=layer_norm))
        self.term_dist = torch.nn.Sequential(*layers_with_activation(term_lws, activation, layer_norm=layer_norm))

        if d_high_level_ctx == 0:
            self._det_core_fwd = self._det_core_without_high_level_ctx
        else:
            self._det_core_fwd = self._det_core_with_high_level_ctx

        if d_observation == 0:
            self._observation = self._zero_observation
            self.o_dist = None
        else:
            self._observation = self._nonzero_observation
            self.o_dist = torch.nn.Sequential(*layers_with_activation(o_lws, activation))

    @property
    def top_node(self):
        return self.d_high_level_ctx == 0

    @property
    def inner_node(self):
        return self.d_high_level_ctx > 0 and self.d_observation == 0

    @property
    def bottom_node(self):
        return self. self.d_observation > 0

    def _det_core_without_high_level_ctx(self,
                                         s: torch.Tensor,
                                         a: torch.Tensor,
                                         low_level_ctx: torch.Tensor,
                                         high_level_ctx: torch.Tensor,
                                         h: Tuple[torch.Tensor, torch.Tensor]):
        s, a, low_level_ctx = add_time_dim(s, a, low_level_ctx)
        x_det, new_h = self.det_core(torch.concat([s, a, low_level_ctx], dim=-1), h)
        x_det = remove_time_dim(x_det)
        #if self.layer_norm:
        #    new_h[0] = self.det_core_norm(new_h[0])
        return x_det, new_h

    def _det_core_with_high_level_ctx(self,
                                      s: torch.Tensor,
                                      a: torch.Tensor,
                                      low_level_ctx: torch.Tensor,
                                      high_level_ctx: torch.Tensor,
                                      h: Tuple[torch.Tensor, torch.Tensor]):
        s, a, low_level_ctx, high_level_ctx = add_time_dim(s, a, low_level_ctx, high_level_ctx)
        x_det, new_h = self.det_core(torch.concat([s, a, low_level_ctx, high_level_ctx], dim=-1), h)
        x_det = remove_time_dim(x_det)
        #if self.layer_norm:
        #    new_h[0] = self.det_core_norm(new_h[0])
        return x_det, new_h

    def gen_init_values(self,
                        d_batch: int,
                        device: torch.device):
        s = self.zero_s(d_batch, device)
        o = self.zero_o(d_batch, device)
        a = self.zero_a(d_batch, device)
        h = self.zero_h(d_batch, device)
        return {'s': s, 'o': o, 'a': a, 'h': h}

    def zero_s(self,
               d_batch: int,
               device: torch.device):
        return torch.zeros(d_batch, self.d_state, device=device)

    def zero_o(self,
               d_batch: int,
               device: torch.device):
        if self.d_observation > 0:
            o = torch.zeros(d_batch, self.d_observation, device=device)
        else:
            o = torch.tensor(0, device=device)
        return o

    def zero_a(self,
               d_batch: int,
               device: torch.device):
        return torch.zeros(d_batch, self.d_action, device=device)

    def zero_h(self,
               d_batch: int,
               device: torch.device):
        h = torch.zeros(self.n_hidden_layers, d_batch, self.d_hidden, device=device)
        if self.rnn_type == 'lstm':
            return h, h
        else:
            return h

    def zero_ctx_low_level(self,
                           d_batch: int,
                           device: torch.device):
        return torch.zeros(d_batch, self.d_low_level_ctx, device=device)

    def forward(self,
                s: torch.Tensor,
                a: torch.Tensor,
                #o: torch.Tensor,  # those should be in ctx_low_level
                #r: torch.Tensor,
                #term: torch.Tensor,
                ctx_low_level: Optional[torch.Tensor] = None,
                ctx_high_level: Optional[torch.Tensor] = None,
                h: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_posterior: bool = True,
                sample: bool = True):
        zero_ctx_low_level = self.zero_ctx_low_level(s.shape[0], s.device)
        x_det, new_h = self._det_core_fwd(s, a, zero_ctx_low_level, ctx_high_level, h)

        s_prior = self._prior(x_det)
        if use_posterior:
            x_det, new_h = self._det_core_fwd(s, a, ctx_low_level, ctx_high_level, h)
            s_post = self._posterior(x_det)
            s_dist = s_post
        else:
            s_post = None
            s_dist = s_prior

        if sample:
            s_smpl = s_dist.rsample()
        else:
            s_smpl = s_dist.loc

        o_dist, o_smpl = self._observation(x_det, s_smpl, sample)
        r_dist, r_smpl = self._reward(x_det, s_smpl, sample)
        term_dist, term_smpl = self._terminal(x_det, s_smpl, sample)

        return {'s': s_smpl, 's_prior': s_prior, 's_post': s_post, 'o': o_smpl, 'r': r_smpl, 'term': term_smpl,
                'h': new_h}

    def _prior(self,
               x_det: torch.Tensor) -> torch.distributions.Normal:
        s_prior_params = self.s_prior(x_det)
        mu, sigma = torch.tensor_split(s_prior_params, 2, dim=-1)
        sigma = torch.abs(sigma) + self.epsilon
        s_prior = torch.distributions.Normal(loc=mu, scale=sigma)
        return s_prior

    def _posterior(self,
                   x_det: torch.Tensor) -> torch.distributions.Normal:
        s_post_params = self.s_post(x_det)
        mu, sigma = torch.tensor_split(s_post_params, 2, dim=-1)
        sigma = torch.abs(sigma) + self.epsilon
        s_post = torch.distributions.Normal(loc=mu, scale=sigma)
        return s_post

    def _zero_observation(self,
                          x_det: torch.Tensor,
                          s_smpl: torch.Tensor,
                          sample: bool = True):
        return None, torch.tensor(0.0)

    def _nonzero_observation(self,
                             x_det: torch.Tensor,
                             s_smpl: torch.Tensor,
                             sample: bool = True):
        x_in = torch.concat([x_det, s_smpl], dim=-1)
        o_params = self.o_dist(x_in)
        mu, sigma = torch.tensor_split(o_params, 2, dim=-1)
        sigma = torch.abs(sigma) + self.epsilon
        o_dist = torch.distributions.Normal(loc=mu, scale=sigma)
        o_smpl = o_dist.rsample() if sample else mu
        return o_dist, o_smpl

    def _reward(self,
                x_det: torch.Tensor,
                s_smpl: torch.Tensor,
                sample: bool = True) -> Tuple[torch.distributions.Normal, torch.Tensor]:
        x_in = torch.concat([x_det, s_smpl], dim=-1)
        r_params = self.r_dist(x_in)
        mu, sigma = torch.tensor_split(r_params, 2, dim=-1)
        sigma = torch.abs(sigma) + self.epsilon
        r_dist = torch.distributions.Normal(loc=mu, scale=sigma)
        r_smpl = r_dist.rsample() if sample else mu
        return r_dist, r_smpl

    def _terminal(self,
                  x_det: torch.Tensor,
                  s_smpl: torch.Tensor) -> torch.Tensor:
        x_in = torch.concat([x_det, s_smpl], dim=-1)
        term_params = self.term_dist(x_in)
        #term_params = torch.sigmoid(term_params)
        #term_params = torch.clamp(term_params, 0.01, 0.99)
        #term_dist = torch.distributions.ContinuousBernoulli(logits=term_params)
        #term_smpl = term_dist.rsample() if sample else term_params
        #return term_dist, term_smpl
        return torch.sigmoid(term_params)
