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
                 d_ctx_high_level: int,
                 d_x_posterior: int,
                 d_hidden: int,
                 n_hidden_layers: int = 1,
                 hidden_dropout: float = 0.1,
                 epsilon: float = 0.01,
                 s_prior_lws: Sequence[int] = (32, 32),
                 s_post_lws: Sequence[int] = (32, 32),
                 o_lws: Sequence[int] = (32, 32),
                 r_lws: Sequence[int] = (32, 32),
                 term_lws: Sequence[int] = (32, 32),
                 layer_norm: bool = False,
                 activation: str = 'relu',
                 rnn_type: str = 'lstm'):
        super().__init__()

        self.d_state = d_state
        self.d_action = d_action
        self.d_observation = d_observation
        self.d_reward = d_reward
        self.d_high_level_ctx = d_ctx_high_level
        self.d_low_level_ctx = d_x_posterior
        self.d_hidden = d_hidden
        self.n_hidden_layers = n_hidden_layers
        self.epsilon = epsilon
        self.activation = activation
        self.layer_norm = layer_norm
        self.rnn_type = rnn_type

        s_prior_lws = (d_hidden, *s_prior_lws, d_state * 2)
        s_post_lws = (d_hidden + d_x_posterior, *s_post_lws, d_state * 2)
        o_lws = (d_hidden + d_state, *o_lws, d_observation * 2)
        r_lws = (d_hidden + d_state, *r_lws, d_reward * 2)
        term_lws = (d_hidden + d_state, *term_lws, 1)

        if rnn_type == 'lstm':
            rnn_constr = torch.nn.LSTM
        elif rnn_type == 'gru':
            rnn_constr = torch.nn.GRU
        else:
            raise ValueError(f'Unsupported rnn type: {rnn_type}')
        self._rnn = rnn_constr(d_state + d_action + d_observation + d_reward + 1 + d_ctx_high_level,
                               hidden_size=d_hidden, num_layers=n_hidden_layers, batch_first=True,
                               dropout=hidden_dropout)
        #self._det_core = haste.LayerNormLSTM(d_state + d_action + d_high_level_ctx, hidden_size=d_hidden,
        #                                    zoneout=0.05, dropout=hidden_dropout, batch_first=True)
        if layer_norm:
            self.det_core_norm = torch.nn.LayerNorm(d_hidden)
        else:
            self.det_core_norm = None

        self._s_prior = torch.nn.Sequential(*layers_with_activation(s_prior_lws, activation, layer_norm=layer_norm))
        self._s_post = torch.nn.Sequential(*layers_with_activation(s_post_lws, activation, layer_norm=layer_norm))
        self._o_dist = torch.nn.Sequential(*layers_with_activation(o_lws, activation, layer_norm=layer_norm))
        self._r_dist = torch.nn.Sequential(*layers_with_activation(r_lws, activation, layer_norm=layer_norm))
        self._term_dist = torch.nn.Sequential(*layers_with_activation(term_lws, activation, layer_norm=layer_norm))

    @property
    def top_node(self):
        return self.d_high_level_ctx == 0

    @property
    def inner_node(self):
        return self.d_high_level_ctx > 0 and self.d_observation == 0

    @property
    def bottom_node(self):
        return self. self.d_observation > 0

    def _det_core(self,
                  s: torch.Tensor,
                  a: torch.Tensor,
                  x_hat: torch.Tensor,
                  ctx_high_level: torch.Tensor,
                  rnn_state: Tuple[torch.Tensor, torch.Tensor]):
        inp = torch.concat([s, a, x_hat, ctx_high_level], dim=-1)
        inp = add_time_dim(inp)
        x_det, new_h = self._rnn(inp, rnn_state)
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
        r = self.zero_r(d_batch, device)
        term = self.zero_term(d_batch, device)
        rnn_state = self.zero_h(d_batch, device)
        return {'s': s, 'o': o, 'a': a, 'r': r, 'term': term, 'rnn_state': rnn_state}

    def zero_s(self,
               d_batch: int,
               device: torch.device):
        return torch.zeros(d_batch, self.d_state, device=device)

    def zero_o(self,
               d_batch: int,
               device: torch.device):
        return torch.zeros(d_batch, self.d_observation, device=device)

    def zero_a(self,
               d_batch: int,
               device: torch.device):
        return torch.zeros(d_batch, self.d_action, device=device)

    def zero_r(self,
               d_batch: int,
               device: torch.device):
        return torch.zeros(d_batch, self.d_reward, device=device)

    def zero_term(self,
                  d_batch: int,
                  device: torch.device):
        return torch.zeros(d_batch, 1, device=device)

    def zero_h(self,
               d_batch: int,
               device: torch.device):
        h = torch.zeros(self.n_hidden_layers, d_batch, self.d_hidden, device=device)
        if self.rnn_type == 'lstm':
            return h, h
        else:
            return h

    def zero_x_post_groundtruth(self,
                                d_batch: int,
                                device: torch.device):
        return torch.zeros(d_batch, self.d_low_level_ctx, device=device)

    def zero_ctx_high_level(self,
                            d_batch: int,
                            device: torch.device):
        return torch.zeros(d_batch, self.d_high_level_ctx, device=device)

    def forward(self,
                s: torch.Tensor,
                a: torch.Tensor,
                x_hat: torch.Tensor,  # o, r, term from last prediction step (since is only implicitly contained in s)
                x_post_groundtruth: torch.Tensor,  # ground truth o, r, term for this prediction step for the posterior
                ctx_high_level: torch.Tensor,  # abstr_s, abstr_h in primitive model
                rnn_state: Tuple[torch.Tensor, torch.Tensor],  # memory from previous step
                use_posterior: bool = True,
                sample: bool = True):
        h, next_rnn_state = self._det_core(s, a, x_hat, ctx_high_level, rnn_state)
        s_prior = self.build_s_prior(h)

        if use_posterior:
            s_post = self.build_s_post(h, x_post_groundtruth)
            s_dist = s_post
        else:
            s_post = None
            s_dist = s_prior

        if sample:
            s_smpl = s_dist.rsample()
        else:
            s_smpl = s_dist.loc

        o_dist, o_smpl = self._build_o_dist(h, s_smpl, sample)
        r_dist, r_smpl = self._build_r_dist(h, s_smpl, sample)
        term_smpl = self._build_terminal_dist(h, s_smpl)

        return {'s': s_smpl, 's_prior': s_prior, 's_post': s_post, 'o': o_smpl, 'r': r_smpl, 'term': term_smpl,
                'rnn_state': next_rnn_state}

    def build_s_prior(self,
                      h: torch.Tensor) -> torch.distributions.Normal:
        s_prior_params = self._s_prior(h)
        mu, sigma = torch.tensor_split(s_prior_params, 2, dim=-1)
        sigma = torch.abs(sigma) + self.epsilon
        s_prior = torch.distributions.Normal(loc=mu, scale=sigma)
        return s_prior

    def build_s_post(self,
                     h: torch.Tensor,
                     x_posterior: torch.Tensor) -> torch.distributions.Normal:
        inp = torch.concat([h, x_posterior], dim=-1)
        s_post_params = self._s_post(inp)
        mu, sigma = torch.tensor_split(s_post_params, 2, dim=-1)
        sigma = torch.abs(sigma) + self.epsilon
        s_post = torch.distributions.Normal(loc=mu, scale=sigma)
        return s_post

    def _build_o_dist(self,
                      h: torch.Tensor,
                      s_smpl: torch.Tensor,
                      sample: bool = True):
        inp = torch.concat([h, s_smpl], dim=-1)
        o_params = self._o_dist(inp)
        mu, sigma = torch.tensor_split(o_params, 2, dim=-1)
        sigma = torch.abs(sigma) + self.epsilon
        o_dist = torch.distributions.Normal(loc=mu, scale=sigma)
        o_smpl = o_dist.rsample() if sample else mu
        return o_dist, o_smpl

    def _build_r_dist(self,
                      h: torch.Tensor,
                      s_smpl: torch.Tensor,
                      sample: bool = True) -> Tuple[torch.distributions.Normal, torch.Tensor]:
        x_in = torch.concat([h, s_smpl], dim=-1)
        r_params = self._r_dist(x_in)
        mu, sigma = torch.tensor_split(r_params, 2, dim=-1)
        sigma = torch.abs(sigma) + self.epsilon
        r_dist = torch.distributions.Normal(loc=mu, scale=sigma)
        r_smpl = r_dist.rsample() if sample else mu
        return r_dist, r_smpl

    def _build_terminal_dist(self,
                             h: torch.Tensor,
                             s_smpl: torch.Tensor) -> torch.Tensor:
        x_in = torch.concat([h, s_smpl], dim=-1)
        term_params = self._term_dist(x_in)
        #term_params = torch.sigmoid(term_params)
        #term_params = torch.clamp(term_params, 0.01, 0.99)
        #term_dist = torch.distributions.ContinuousBernoulli(logits=term_params)
        #term_smpl = term_dist.rsample() if sample else term_params
        #return term_dist, term_smpl
        return torch.sigmoid(term_params)
