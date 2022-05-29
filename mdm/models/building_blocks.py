from typing import Tuple, Optional, Sequence
from enum import Enum

import torch

from mdm.utils.torch_tools import DeviceMixin, layers_with_activation, add_time_dim, remove_time_dim


class AbstractActionModel(torch.nn.Module, DeviceMixin):

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
                 activation: str = 'relu'):
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

        s_prior_lws = (d_hidden, *s_prior_lws, d_state * 2)
        s_post_lws = (d_hidden + d_low_level_ctx, *s_post_lws, d_state * 2)
        o_lws = (d_hidden + d_state, *o_lws, d_observation * 2)
        r_lws = (d_hidden + d_state, *r_lws, d_reward * 2)
        term_lws = (d_hidden + d_state, *term_lws, 1)

        self.det_core = torch.nn.LSTM(d_state + d_action + d_high_level_ctx, hidden_size=d_hidden,
                                      num_layers=n_hidden_layers, batch_first=True, dropout=hidden_dropout)
        self.s_prior = torch.nn.Sequential(*layers_with_activation(s_prior_lws, activation))
        self.s_post = torch.nn.Sequential(*layers_with_activation(s_post_lws, activation))
        self.o_dist = torch.nn.Sequential(*layers_with_activation(o_lws, activation))
        self.r_dist = torch.nn.Sequential(*layers_with_activation(r_lws, activation))
        self.term_dist = torch.nn.Sequential(*layers_with_activation(term_lws, activation))

        if d_high_level_ctx == 0:
            self._det_core_fwd = self._det_core_without_ctx
        else:
            self._det_core_fwd = self._det_core_with_ctx

        if d_observation == 0:
            self._observation = self._zero_observation
        else:
            self._observation = self._nonzero_observation

    @property
    def top_node(self):
        return self.d_high_level_ctx == 0

    @property
    def inner_node(self):
        return self.d_high_level_ctx > 0 and self.d_observation == 0

    @property
    def bottom_node(self):
        return self. self.d_observation > 0

    def _det_core_without_ctx(self,
                              s: torch.Tensor,
                              a: torch.Tensor,
                              high_level_ctx: torch.Tensor,
                              h: Tuple[torch.Tensor, torch.Tensor]):
        s, a = add_time_dim(a, s)
        x_det, new_h = self.det_core(torch.concat([s, a], dim=-1), h)
        x_det = remove_time_dim(x_det)
        return x_det, new_h

    def _det_core_with_ctx(self,
                           s: torch.Tensor,
                           a: torch.Tensor,
                           high_level_ctx: torch.Tensor,
                           h: Tuple[torch.Tensor, torch.Tensor]):
        s, a, high_level_ctx = add_time_dim(a, s, high_level_ctx)
        x_det, new_h = self.det_core(torch.concat([s, a, high_level_ctx], dim=-1), h)
        x_det = remove_time_dim(x_det)
        return x_det, new_h

    def _zero_observation(self,
                          x_det: torch.Tensor,
                          s_smpl: torch.Tensor):
        return None, torch.tensor(0.0)

    def _nonzero_observation(self,
                             x_det: torch.Tensor,
                             s_smpl: torch.Tensor):
        x_in = torch.concat([x_det, s_smpl], dim=-1)
        o_params = self.o_dist(x_in)
        mu, sigma = torch.tensor_split(o_params, 2, dim=-1)
        sigma = torch.abs(sigma) + self.epsilon
        o_dist = torch.distributions.Normal(loc=mu, scale=sigma)
        o_smpl = o_dist.rsample()
        return o_dist, o_smpl

    def gen_init_values(self, d_batch: int,  device: torch.device):
        s = torch.zeros(d_batch, self.d_state, device=device)
        o = torch.zeros(d_batch, self.d_observation, device=device)
        a = torch.zeros(d_batch, self.d_action, device=device)
        h = torch.zeros(self.n_hidden_layers, d_batch, self.d_hidden, device=device)
        return {'s': s, 'o': o, 'a': a, 'h': (h, h)}

    def forward(self,
                s: torch.Tensor,
                a: torch.Tensor,
                #o: torch.Tensor,  # those should be in ctx_low_level
                #r: torch.Tensor,
                #term: torch.Tensor,
                ctx_low_level: Optional[torch.Tensor] = None,
                ctx_high_level: Optional[torch.Tensor] = None,
                h: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_posterior: bool = True):
        x_det, new_h = self._det_core_fwd(s, a, ctx_high_level, h)

        s_prior = self._prior(x_det)
        if use_posterior:
            s_post = self._posterior(x_det, ctx_low_level)
            s_smpl = s_post.rsample()
        else:
            s_post = None
            s_smpl = s_prior.rsample()

        o_dist, o_smpl = self._observation(x_det, s_smpl)
        r_dist, r_smpl = self._reward(x_det, s_smpl)
        term_dist, term_smpl = self._terminal(x_det, s_smpl)

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
                   x_det: torch.Tensor,
                   ctx_low_level: torch.Tensor) -> torch.distributions.Normal:
        x_in = torch.concat([x_det, ctx_low_level], dim=-1)
        s_post_params = self.s_post(x_in)
        mu, sigma = torch.tensor_split(s_post_params, 2, dim=-1)
        sigma = torch.abs(sigma) + self.epsilon
        s_post = torch.distributions.Normal(loc=mu, scale=sigma)
        return s_post


    def _reward(self,
                x_det: torch.Tensor,
                s_smpl: torch.Tensor) -> Tuple[torch.distributions.Normal, torch.Tensor]:
        x_in = torch.concat([x_det, s_smpl], dim=-1)
        r_params = self.r_dist(x_in)
        mu, sigma = torch.tensor_split(r_params, 2, dim=-1)
        sigma = torch.abs(sigma) + self.epsilon
        r_dist = torch.distributions.Normal(loc=mu, scale=sigma)
        r_smpl = r_dist.rsample()
        return r_dist, r_smpl

    def _terminal(self,
                  x_det: torch.Tensor,
                  s_smpl: torch.Tensor) -> Tuple[torch.distributions.ContinuousBernoulli, torch.Tensor]:
        x_in = torch.concat([x_det, s_smpl], dim=-1)
        term_params = self.term_dist(x_in)
        term_params = torch.sigmoid(term_params)
        term_dist = torch.distributions.ContinuousBernoulli(probs=term_params)
        term_smpl = term_dist.rsample()
        return term_dist, term_smpl
