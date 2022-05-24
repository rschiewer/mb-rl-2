from typing import Tuple, Optional

import torch

from mdm.utils.torch_tools import DeviceMixin, FeedforwardBlock, layers_with_activation, add_time_dim, remove_time_dim


class AbstractActionModel(torch.nn.Module, DeviceMixin):

    def __init__(self,
                 d_action: int,
                 n_abstract_steps: int,
                 d_abstract_action: int,
                 lws: tuple = (64, 64),
                 activation: str = 'relu'):
        super(AbstractActionModel, self).__init__()

        lws = (d_action * n_abstract_steps, *lws, d_abstract_action)
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
                 d_high_level_context: int,
                 d_low_level_context: int,
                 d_hidden: int,
                 n_hidden_layers: int = 1,
                 hidden_dropout: float = 0.1,
                 epsilon: float = 0.01,
                 s_prior_lws: Tuple[int] = (32, 32),
                 s_post_lws: Tuple[int] = (32, 32),
                 o_lws: Tuple[int] = (32, 32),
                 r_lws: Tuple[int] = (32, 32),
                 term_lws: Tuple[int] = (32, 32),
                 activation: str = 'relu'):
        super().__init__()

        self.d_state = d_state
        self.d_action = d_action
        self.d_observation = d_observation
        self.d_high_level_context = d_high_level_context
        self.d_low_level_context = d_low_level_context
        self.d_hidden = d_hidden
        self.n_hidden_layers = n_hidden_layers
        self.epsilon = epsilon

        s_prior_lws = (d_hidden, *s_prior_lws, d_state * 2)
        s_post_lws = (d_hidden, *s_post_lws, d_state * 2)
        o_lws = (d_hidden, *o_lws, d_observation * 2)
        r_lws = (d_hidden, *r_lws, 2)
        term_lws = (d_hidden, *term_lws, 1)

        self.det_core = torch.nn.LSTM(d_state + d_action + d_high_level_context, hidden_size=d_hidden,
                                      num_layers=n_hidden_layers, batch_first=True, dropout=hidden_dropout)
        self.s_prior = torch.nn.Sequential(*layers_with_activation(s_prior_lws, activation))
        self.s_post = torch.nn.Sequential(*layers_with_activation(s_post_lws, activation))
        self.o_dist = torch.nn.Sequential(*layers_with_activation(o_lws, activation))
        self.r_dist = torch.nn.Sequential(*layers_with_activation(r_lws, activation))
        self.term_dist = torch.nn.Sequential(*layers_with_activation(term_lws, activation))

        if d_high_level_context == 0:
            def det_core_forward_fn(s_, a_, high_level_ctx_, cell_state_):
                s_, a_ = add_time_dim(a_, s_)
                x_det_, new_cell_state_ = self.det_core(torch.concat([s_, a_], dim=-1), cell_state_)
                x_det_ = remove_time_dim(x_det_)
                return x_det_, new_cell_state_
        else:
            def det_core_forward_fn(s_, a_, high_level_ctx_, cell_state_):
                s_, a_, high_level_ctx_ = add_time_dim(a_, s_, high_level_ctx_)
                x_det_, new_cell_state_ = self.det_core(torch.concat([s_, a_, high_level_ctx_], dim=-1), cell_state_)
                x_det_ = remove_time_dim(x_det_)
                return x_det_, new_cell_state_

        self._det_core_fwd = det_core_forward_fn

    def forward(self,
                s: torch.Tensor,
                a: torch.Tensor,
                #o: torch.Tensor,  # those should be in ctx_low_level
                #r: torch.Tensor,
                #term: torch.Tensor,
                ctx_low_level: torch.Tensor,
                ctx_high_level: torch.Tensor = None,
                cell_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        x_det, new_cell_state = self._det_core_fwd(s, a, ctx_high_level, cell_state)

        s_prior = self._prior(x_det)
        s_post = self._posterior(x_det, ctx_low_level)
        s_post_smpl = s_post.rsample()

        o_dist = self._observation(x_det, s_post_smpl)
        o_smpl = o_dist.rsample()
        r_dist = self._reward(x_det, s_post_smpl)
        r_smpl = r_dist.rsample()
        term_dist = self._terminal(x_det, s_post_smpl)
        term_smpl = term_dist.rsample()

        return ({'s': s_post_smpl, 's_prior': s_prior, 's_post': s_post, 'o': o_smpl, 'r': r_smpl, 'term': term_smpl},
                new_cell_state)

    def predict_with_prior(self,
                           s: torch.Tensor,
                           a: torch.Tensor,
                           ctx_high_level: torch.Tensor,
                           cell_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        x_det, new_cell_state = self._det_core_fwd(s, a, ctx_high_level, cell_state)

        s_prior = self._prior(x_det)
        s_prior_smpl = s_prior.rsample()

        o_dist = self._observation(x_det, s_prior_smpl)
        o_smpl = o_dist.rsample()
        r_dist = self._reward(x_det, s_prior_smpl)
        r_smpl = r_dist.rsample()
        term_dist = self._terminal(x_det, s_prior_smpl)
        term_smpl = term_dist.rsample()

        return {'s': s_prior_smpl, 's_prior': s_prior, 'o': o_smpl, 'r': r_smpl, 'term': term_smpl}, new_cell_state

    def _prior(self,
               x_det: torch.Tensor) -> torch.distributions.Normal:
        s_prior_params = self.s_prior(x_det)
        mu, sigma = torch.tensor_split(s_prior_params, 2, dim=-1)
        sigma = torch.abs(sigma) + self.epsilon
        s_prior = torch.nn.distributions.Normal(loc=mu, scale=sigma)
        return s_prior

    def _posterior(self,
                   x_det: torch.Tensor,
                   ctx_low_level: torch.Tensor) -> torch.distributions.Normal:
        x_in = torch.concat([x_det, ctx_low_level], dim=-1)
        s_post_params = self.s_post(x_in)
        mu, sigma = torch.tensor_split(s_post_params, 2, dim=-1)
        sigma = torch.abs(sigma) + self.epsilon
        s_post = torch.nn.distributions.Normal(loc=mu, scale=sigma)
        return s_post

    def _observation(self,
                     x_det: torch.Tensor,
                     s_smpl: torch.Tensor) -> torch.distributions.Normal:
        x_in = torch.concat([x_det, s_smpl], dim=-1)
        o_params = self.o_dist(x_in)
        mu, sigma = torch.tensor_split(o_params, 2, dim=-1)
        sigma = torch.abs(sigma) + self.epsilon
        o_dist = torch.nn.distributions.Normal(loc=mu, scale=sigma)
        return o_dist

    def _reward(self,
                x_det: torch.Tensor,
                s_smpl: torch.Tensor) -> torch.distributions.Normal:
        x_in = torch.concat([x_det, s_smpl], dim=-1)
        r_params = self.r_dist(x_in)
        mu, sigma = torch.tensor_split(r_params, 2, dim=-1)
        sigma = torch.abs(sigma) + self.epsilon
        r_dist = torch.nn.distributions.Normal(loc=mu, scale=sigma)
        return r_dist

    def _terminal(self,
                  x_det: torch.Tensor,
                  s_smpl: torch.Tensor) -> torch.distributions.ContinuousBernoulli:
        x_in = torch.concat([x_det, s_smpl], dim=-1)
        term_params = self.term_dist(x_in)
        term_params = torch.sigmoid(term_params)
        term_dist = torch.nn.distributions.ContinuousBernoulli(probs=term_params)
        return term_dist
