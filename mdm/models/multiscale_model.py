from typing import Tuple, Dict
from collections import deque
from math import ceil

import torch
import torch.nn.functional as F

from mdm.utils.torch_tools import *
from mdm.models.dynamics_model import DynamicsModel


class MacroActionModel(torch.nn.Module, DeviceMixin):

    def __init__(self,
                 d_action: int,
                 n_abstract_steps: int,
                 d_macro_action: int):
        super(MacroActionModel, self).__init__()

        self.flatten_layer = torch.nn.Flatten(start_dim=1)
        self.det_mdl = FeedforwardBlock(d_action * n_abstract_steps, lws=(64, d_macro_action))

    def forward(self, actions: torch.Tensor):
        # actions.shape = (d_batch, n_abstract_steps, d_action)
        macro_action = self.flatten_layer(actions)
        macro_action = self.det_mdl(macro_action)
        macro_action = F.gumbel_softmax(macro_action, hard=True)
        return macro_action


class SingleStepModel(torch.nn.Module, DeviceMixin):

    def __init__(self,
                 det_mdl: RecurrentBlock,
                 sampling_mdl_s: GaussianBlock,
                 sampling_mdl_r: GaussianBlock,
                 sampling_mdl_term: ContinuousBernoulliBlock):
        super(SingleStepModel, self).__init__()

        self.det_mdl = det_mdl
        self.sampling_mdl_s = sampling_mdl_s
        self.sampling_mdl_r = sampling_mdl_r
        self.sampling_mdl_term = sampling_mdl_term

    def forward(self,
                s: torch.Tensor,
                a: torch.Tensor,
                macro_s: torch.Tensor,
                macro_a: torch.Tensor,
                macro_r: torch.Tensor,
                macro_s_next: torch.Tensor,
                h: torch.Tensor):
        s, a, macro_s, macro_a, macro_r, macro_s_next = add_time_dim(s, a, macro_s, macro_a, macro_r, macro_s_next,
                                                                     batch_first=self.det_mdl.batch_first)

        x_det, h = self.det_mdl(s, a, macro_s, macro_a, macro_r, macro_s_next, h=h)
        x_det = remove_time_dim(x_det, batch_first=self.det_mdl.batch_first)
        s_next_dist = self.sampling_mdl_s(x_det)
        r_dist = self.sampling_mdl_r(x_det)
        term_dist = self.sampling_mdl_term(x_det)

        return {'s_next_dist': s_next_dist,
                'r_dist': r_dist,
                'term_dist': term_dist}, h


class AbstractModel(torch.nn.Module, DeviceMixin):

    def __init__(self,
                 det_mdl: FeedforwardBlock,
                 sampling_mdl_s_prior: GaussianBlock,
                 sampling_mdl_s_posterior: GaussianBlock,
                 sampling_mdl_r_prior: GaussianBlock,
                 sampling_mdl_r_posterior: GaussianBlock,
                 sampling_mdl_term_prior: ContinuousBernoulliBlock,
                 sampling_mdl_term_posterior: ContinuousBernoulliBlock):
        super(AbstractModel, self).__init__()

        self.det_mdl = det_mdl  # same det_mdl is used for prior and posterior
        self.sampling_mdl_s_prior = sampling_mdl_s_prior
        self.sampling_mdl_s_posterior = sampling_mdl_s_posterior
        self.sampling_mdl_r_prior = sampling_mdl_r_prior
        self.sampling_mdl_r_posterior = sampling_mdl_r_posterior
        self.sampling_mdl_term_prior = sampling_mdl_term_prior
        self.sampling_mdl_term_posterior = sampling_mdl_term_posterior

    def forward(self,
                macro_s: torch.Tensor,
                macro_a: torch.Tensor,
                single_step_model_history: torch.Tensor = None):
        if single_step_model_history is None:
            d_batch = macro_s.shape[0]
            d_memory = self.det_mdl.d_inputs[-1]
            single_step_model_history = torch.zeros(d_batch, d_memory, device=self.device)

            macro_det = self.det_mdl(macro_s, macro_a, single_step_model_history)
            macro_s_next_dist = self.sampling_mdl_s_prior(macro_det)
            macro_r_dist = self.sampling_mdl_r_prior(macro_det)
            macro_term_dist = self.sampling_mdl_term_prior(macro_det)
        else:
            macro_det = self.det_mdl(macro_s, macro_a, single_step_model_history)
            macro_s_next_dist = self.sampling_mdl_s_posterior(macro_det)
            macro_r_dist = self.sampling_mdl_r_posterior(macro_det)
            macro_term_dist = self.sampling_mdl_term_posterior(macro_det)

        return {'macro_s_next_dist': macro_s_next_dist,
                'macro_r_dist': macro_r_dist,
                'macro_term_dist': macro_term_dist}


class MultiscaleDynamicsModel(DynamicsModel):

    def __init__(self,
                 single_step_model: SingleStepModel,
                 abstract_model: AbstractModel,
                 macro_action_model: torch.nn.Module,
                 abstract_step_size: int,
                 d_state: int,
                 d_action: int,
                 d_reward: int,
                 d_macro_state: int,
                 d_macro_action: int,
                 d_macro_reward: int):
        super(MultiscaleDynamicsModel, self).__init__()

        if not single_step_model.det_mdl.batch_first and macro_action_model.batch_first:
            raise ValueError('This model requires all its components to obey the batch_first convention')

        self.single_step_model = single_step_model
        self.abstract_model = abstract_model
        self.macro_action_model = macro_action_model
        self.macro_step_size = abstract_step_size
        self.d_state = d_state
        self.d_action = d_action
        self.d_reward = d_reward
        self.d_macro_state = d_macro_state
        self.d_macro_action = d_macro_action
        self.d_macro_reward = d_macro_reward

    @staticmethod
    def reconstruction_loss(x_pred: torch.Tensor, x_true: torch.Tensor):
        l = torch.mean((x_pred - x_true) ** 2)
        return l

    @staticmethod
    def maximum_likelihood_loss(x_dist: List[torch.distributions.Distribution], x_true: torch.Tensor):
        l = torch.zeros(x_dist[0].batch_shape, device=x_true.device)
        for d, x in zip(x_dist, x_true):
            l -= d.log_prob(x)
        return torch.mean(l)

    @staticmethod
    def kl_loss_normal(priors: List[torch.distributions.Normal],
                       posteriors: List[torch.distributions.Normal],
                       detach_posterior: bool = True):
        l = torch.zeros_like(priors[0].loc)
        for prior, posterior in zip(priors, posteriors):
            if detach_posterior:
                posterior = torch.distributions.Normal(loc=posterior.loc.detach(), scale=posterior.scale.detach())
            l += torch.distributions.kl.kl_divergence(posterior, prior)
        return torch.mean(l)

    @staticmethod
    def kl_loss_bernolli(priors: List[torch.distributions.ContinuousBernoulli],
                         posteriors: List[torch.distributions.ContinuousBernoulli],
                         detach_posterior: bool = True):
        if priors[0].probs is None:
            l = torch.zeros_like(priors[0].logits)
        else:
            l = torch.zeros_like(priors[0].probs)

        for prior, posterior in zip(priors, posteriors):
            if detach_posterior:
                posterior = torch.distributions.ContinuousBernoulli(logits=posterior.logits)  # todo: make if-else for logits vs probs
            l += torch.distributions.kl.kl_divergence(posterior, prior)
        return torch.mean(l)

    @staticmethod
    def kl_regularizer_normal(priors: List[torch.distributions.Normal]):
        uniform_gauss = torch.distributions.Normal(loc=torch.zeros_like(priors[0].loc, requires_grad=False),
                                                   scale=torch.ones_like(priors[0].scale, requires_grad=False))
        l = torch.zeros_like(priors[0].loc)
        for prior in priors:
            l += torch.distributions.kl.kl_divergence(prior, uniform_gauss)
        return torch.mean(l)

    @staticmethod
    def kl_regularizer_bernoulli(priors: List[torch.distributions.ContinuousBernoulli]):
        uniform_gauss = torch.distributions.ContinuousBernoulli(logits=priors[0].logits)
        l = torch.zeros_like(priors[0].logits)
        for prior in priors:
            l += torch.distributions.kl.kl_divergence(prior, uniform_gauss)
        return torch.mean(l)

    @staticmethod
    def macro_reward_loss(macro_r_pred,
                          macro_r_true):
        l = torch.mean((macro_r_pred - macro_r_true) ** 2)
        return l

    def average_kstep_value(self,
                            rewards: torch.Tensor,
                            k: int):
        d_batch, d_time, d_data = rewards.shape
        n_macro_steps = ceil(d_time / k)
        d_padding = n_macro_steps * k - d_time

        padding = torch.zeros(d_batch, d_padding, d_data, device=self.device)
        rewards = torch.concat([rewards, padding], dim=1)

        avg = []
        for i in range(n_macro_steps):
            i_start = i * k
            i_stop = (i + 1) * k
            avg.append(torch.mean(rewards[:, i_start:i_stop], dim=1, keepdim=True))

        return torch.concat(avg, dim=1)

    def forward(self,
                start_states: torch.Tensor,
                actions: torch.Tensor,
                context: torch.Tensor = None):
        d_batch, n_steps = actions.shape[:2]
        n_start_states = start_states.shape[1]
        device = self.device

        s = torch.zeros(d_batch, self.d_state, device=device)
        h = self.single_step_model.det_mdl.gen_h_placeholder(d_batch)
        macro_s = torch.zeros(d_batch, self.d_macro_state, device=device)
        #macro_a = self.macro_action_model(self._next_single_step_actions(actions, 0))
        macro_a = torch.zeros(d_batch, self.d_macro_action, device=device)
        macro_r = torch.zeros(d_batch, self.d_macro_reward, device=device)
        macro_s_next = torch.zeros_like(macro_s, device=device)

        s_mem, s_dist_mem = [], []
        r_mem, r_dist_mem = [], []
        term_mem, term_dist_mem = [], []
        macro_s_prior_mem, macro_s_post_mem = [], []
        macro_r_mem = []
        macro_r_prior_mem, macro_r_post_mem = [], []
        macro_term_mem = []
        macro_term_prior_mem, macro_term_post_mem = [], []

        for t in range(n_steps):
            if t % self.macro_step_size == 0 and t > 0:  # invoke abstract model every k time steps
                # TODO: think about this model more closely
                macro_a = self.macro_action_model(self._next_single_step_actions(actions, t))
                h_flat = self.filter_single_step_model_history(h)
                macro_s = macro_s_next  # update current macro state to previously predicted one

                # invoke models
                pred_macro_prior = self.abstract_model(macro_s, macro_a)
                pred_macro_post = self.abstract_model(macro_s, macro_a, h_flat)

                # sample from more informed posterior distributions to get inputs for single step model
                macro_s_next = pred_macro_post['macro_s_next_dist'].rsample()
                macro_r = pred_macro_post['macro_r_dist'].rsample()
                macro_term = pred_macro_post['macro_term_dist'].rsample()

                # store for loss calculation
                macro_s_prior_mem.append(pred_macro_prior['macro_s_next_dist'])
                macro_s_post_mem.append(pred_macro_post['macro_s_next_dist'])

                macro_r_mem.append(macro_r)
                macro_r_prior_mem.append(pred_macro_prior['macro_r_dist'])
                macro_r_post_mem.append(pred_macro_post['macro_r_dist'])

                macro_term_mem.append(macro_term)
                macro_term_prior_mem.append(pred_macro_prior['macro_term_dist'])
                macro_term_post_mem.append(pred_macro_post['macro_term_dist'])

                # prevent memory leakage beyond macro steps
                h = self.single_step_model.det_mdl.gen_h_placeholder(d_batch)
                s = torch.zeros_like(s, device=device)

            if t < n_start_states:  # if still in warmup period, use teacher forcing for states
                s = start_states[:, t]
            a = actions[:, t]

            #h = (torch.zeros_like(h[0], device=device), torch.zeros_like(h[1], device=device))
            pred_ss, h = self.single_step_model(s, a, macro_s, macro_a, macro_r, macro_s_next, h)
            s_next = pred_ss['s_next_dist'].rsample()
            r = pred_ss['r_dist'].rsample()
            term = pred_ss['term_dist'].rsample()

            # store single step states and rewards for loss calculation
            s_mem.append(s_next)
            r_mem.append(r)
            term_mem.append(term)
            s_dist_mem.append(pred_ss['s_next_dist'])
            r_dist_mem.append(pred_ss['r_dist'])
            term_dist_mem.append(pred_ss['term_dist'])

            # set next state to upcoming time step's current state
            s = s_next

        return {'s': s_mem,
                's_dist': s_dist_mem,
                'r': r_mem,
                'r_dist': r_dist_mem,
                'term': term_mem,
                'term_dist': term_dist_mem,
                'macro_s_prior': macro_s_prior_mem,
                'macro_s_post': macro_s_post_mem,
                'macro_r': macro_r_mem,
                'macro_r_prior': macro_r_prior_mem,
                'macro_r_post': macro_r_post_mem,
                'macro_term': macro_term_mem,
                'macro_term_prior': macro_term_prior_mem,
                'macro_term_post': macro_term_post_mem}

    def _next_single_step_actions(self, actions: torch.Tensor, t: int):
        n_steps = actions.shape[1]
        if t + self.macro_step_size > n_steps:
            diff = t + self.macro_step_size - n_steps
            next_actions = actions[:, t:]
            next_actions = torch.concat([next_actions, torch.zeros(actions.shape[0], diff, actions.shape[2],
                                                                   device=self.device)], dim=1)
        else:
            next_actions = actions[:, t: t + self.macro_step_size]
        return next_actions

    def train_step(self,
                   s_ground_truth: torch.Tensor,
                   a_ground_truth: torch.Tensor,
                   r_ground_truth: torch.Tensor,
                   term_ground_truth: torch.Tensor,
                   optimizer: torch.optim.Optimizer,
                   n_warmup: int = 1):
        optimizer.zero_grad(set_to_none=True)

        losses = self.eval_step(s_ground_truth, a_ground_truth, r_ground_truth, term_ground_truth, n_warmup)
        losses['total'].backward()
        #torch.nn.utils.clip_grad_value_(self.parameters(), 1.0)
        optimizer.step()

        return losses

    def eval_step(self,
                  s_ground_truth: torch.Tensor,
                  a_ground_truth: torch.Tensor,
                  r_ground_truth: torch.Tensor,
                  term_ground_truth: torch.Tensor,
                  n_warmup: int = 1) -> Dict:
        rec_loss = MultiscaleDynamicsModel.reconstruction_loss
        rec_loss_ml = MultiscaleDynamicsModel.maximum_likelihood_loss
        kl_loss_norm = MultiscaleDynamicsModel.kl_loss_normal
        kl_loss_bern = MultiscaleDynamicsModel.kl_loss_bernolli
        kl_reg_norm = MultiscaleDynamicsModel.kl_regularizer_normal
        kl_reg_bern = MultiscaleDynamicsModel.kl_regularizer_bernoulli
        macro_r_loss = MultiscaleDynamicsModel.reconstruction_loss
        macro_term_loss = MultiscaleDynamicsModel.reconstruction_loss

        # target macro reward can be pre-computed from the single step rewards
        macro_r_target = self.average_kstep_value(r_ground_truth, self.macro_step_size)
        macro_r_target = macro_r_target[:, 1:]  # exclude first chunk since macro model is inactive there

        # target macro terminal transition probability can be pre-computed as well
        macro_term_target = self.average_kstep_value(term_ground_truth, self.macro_step_size)
        macro_term_target = macro_term_target[:, 1:]  # exclude first chunk since macro model is inactive there
        
        warmup_states = s_ground_truth[:, :n_warmup, :]
        pred = self(warmup_states, a_ground_truth)

        # https://stats.stackexchange.com/questions/332179/how-to-weight-kld-loss-vs-reconstruction-loss-in-variational-auto-encoder
        # and beta VAE paper for further explanation
        #M = self.abstract_model.sampling_mdl_s_prior.lws[-1] / 2
        #N = self.abstract_model.sampling_mdl_s_prior.lws[0]
        #beta = 10
        #beta_norm = M / N * beta

        rec_s = rec_loss(torch.stack(pred['s'], dim=1), s_ground_truth)
        rec_r = rec_loss(torch.stack(pred['r'], dim=1), r_ground_truth)
        rec_term = rec_loss(torch.stack(pred['term'], dim=1), term_ground_truth)
        #rec_s = rec_loss_ml(pred['s_dist'], torch.transpose(s_ground_truth, 0, 1))
        #rec_r = rec_loss_ml(pred['r_dist'], torch.transpose(r_ground_truth, 0, 1))

        if len(pred['macro_s_prior']) > 0:
            kl_s = 0.001 * kl_loss_norm(pred['macro_s_prior'], pred['macro_s_post'], detach_posterior=True)
            kl_r = 0.001 * kl_loss_norm(pred['macro_r_prior'], pred['macro_r_post'], detach_posterior=True)
            kl_term = 0.001 * kl_loss_bern(pred['macro_term_prior'], pred['macro_term_post'], detach_posterior=True)
            reg_s = 0.0001 * kl_reg_norm(pred['macro_s_post'])
            reg_r = 0.0001 * kl_reg_norm(pred['macro_r_post'])
            reg_term = 0.0001 * kl_reg_bern(pred['macro_term_post'])
            mr = macro_r_loss(torch.stack(pred['macro_r'], dim=1), macro_r_target)
            mt = macro_term_loss(torch.stack(pred['macro_term'], dim=1), macro_term_target)
        else:
            kl_s, kl_r, kl_term = torch.tensor(0), torch.tensor(0), torch.tensor(0)
            reg_s, reg_r, reg_term = torch.tensor(0), torch.tensor(0), torch.tensor(0)
            mr, mt = torch.tensor(0), torch.tensor(0)

        kl = kl_s + kl_r #+ kl_term
        reg = reg_s + reg_r #+ reg_term
        loss = rec_s + rec_r + rec_term + kl + reg + mr + mt

        return {'total': loss, 'rec_s': rec_s, 'rec_r': rec_r, 'rec_term': rec_term, 'kl_s': kl_s, 'kl_r': kl_r, 'kl_reg': reg,
                'macro_r': mr, 'macro_term': mt}

    def input_compatible(self,
                         s_ground_truth: torch.Tensor,
                         a_ground_truth: torch.Tensor,
                         r_ground_truth: torch.Tensor) -> Tuple[bool, str]:
        compatible = True
        tens_name = None
        shape = None

        # test lengths of shapes first
        if len(s_ground_truth.shape) != 3:
            compatible = False
            tens_name = 'state'
            shape = s_ground_truth.shape
        elif len(a_ground_truth.shape) != 3:
            compatible = False
            tens_name = 'action'
            shape = a_ground_truth.shape
        elif len(r_ground_truth.shape) != 3:
            compatible = False
            tens_name = 'reward'
            shape = r_ground_truth.shape

        if not compatible:
            msg = f'{tens_name} tensor should have 3 dimensions (d_batch, d_time, d_data) even if d_data is 1 but has '
            msg += f'shape {tuple(shape)}'
            return compatible, msg

        # test data shapes
        if s_ground_truth.shape[2] != self.d_state:
            compatible = False
            tens_name = 'state'
            shape = (s_ground_truth.shape[2], self.d_state)
        elif a_ground_truth.shape[2] != self.d_action:
            compatible = False
            tens_name = 'action'
            shape = (a_ground_truth.shape[2], self.d_action)
        elif r_ground_truth.shape[2] != self.d_reward:
            compatible = False
            tens_name = 'reward'
            shape = (r_ground_truth.shape[2], self.d_reward)

        if not compatible:
            msg = f'{tens_name} tensor\'s data dimension does not match the expected size (found {shape[0]}, '
            msg += f'expected {shape[1]})'
            return compatible, msg

        return compatible, tens_name

    def rollout_single_step(self,
                            start_states: torch.Tensor,
                            actions: torch.Tensor,
                            macro_s: torch.Tensor = None,
                            macro_a: torch.Tensor = None,
                            macro_r: torch.Tensor = None,
                            macro_s_next: torch.Tensor = None,
                            h: Tuple[torch.Tensor, torch.Tensor] = None):
        d_batch, n_steps = actions.shape[:2]
        n_start_states = start_states.shape[1]
        device = self.device

        # prepare necessary placeholders
        s = torch.zeros(d_batch, self.d_state, device=device)
        h = self.single_step_model.det_mdl.gen_h_placeholder(d_batch) if h is None else h
        macro_s = torch.zeros(d_batch, self.d_macro_state, device=device) if macro_s is None else macro_s
        macro_a = torch.zeros(d_batch, self.d_macro_action, device=device) if macro_a is None else macro_a
        macro_r = torch.zeros(d_batch, self.d_macro_reward, device=device) if macro_r is None else macro_r
        macro_s_next = torch.zeros_like(macro_s, device=device) if macro_s_next is None else macro_s_next

        s_mem, s_dist_mem = [], []
        r_mem, r_dist_mem = [], []
        term_mem, term_dist_mem = [], []
        for t in range(n_steps):
            if t < n_start_states:  # if still in warmup period, use teacher forcing for states
                s = start_states[:, t]
            a = actions[:, t]

            pred_ss, h = self.single_step_model(s, a, macro_s, macro_a, macro_r, macro_s_next, h)
            s_next = pred_ss['s_next_dist'].sample()
            r = pred_ss['r_dist'].sample()
            term = pred_ss['term_dist'].sample()

            s_mem.append(s_next)
            r_mem.append(r)
            term_mem.append(term)
            s_dist_mem.append(pred_ss['s_next_dist'])
            r_dist_mem.append(pred_ss['r_dist'])
            term_dist_mem.append(pred_ss['term_dist'])

            s = s_next

        r_mem = torch.stack(r_mem, dim=1)
        s_mem = torch.stack(s_mem, dim=1)
        term_mem = torch.stack(term_mem, dim=1)

        return {'s': s_mem,
                's_dist': s_dist_mem,
                'r': r_mem,
                'r_dist': r_dist_mem,
                'term': term_mem,
                'term_dist': term_dist_mem,
                'h': h}

    def rollout_abstract(self,
                         macro_start_states: torch.Tensor,
                         macro_actions: torch.Tensor):
        d_batch, n_steps = macro_actions.shape[:2]
        n_start_states = macro_start_states.shape[1]
        device = self.device

        macro_s = torch.zeros(d_batch, self.d_macro_state, device=device)

        macro_s_prior_mem, macro_s_prior_dist_mem = [], []
        macro_r_prior_mem, macro_r_prior_dist_mem = [], []
        for t in range(n_steps):
            if t < n_start_states:
                macro_s = macro_start_states[:, t]
            macro_a = macro_actions[:, t]

            #_, macro_s_next_prior, macro_r_next_prior = self.abstract_model(macro_s, macro_a)
            #macro_s_next = macro_s_next_prior.sample()
            #macro_r_next = macro_r_next_prior.sample()
            pred_macro_prior = self.abstract_model(macro_s, macro_a)
            macro_s_next = pred_macro_prior['macro_s_next_dist'].sample()
            macro_r = pred_macro_prior['macro_r_dist'].sample()

            macro_s_prior_mem.append(macro_s_next)
            macro_r_prior_mem.append(macro_r)
            macro_s_prior_dist_mem.append(pred_macro_prior['macro_s_next_dist'])
            macro_r_prior_dist_mem.append(pred_macro_prior['macro_r_dist'])

            macro_s = macro_s_next

        macro_s_prior_mem = torch.stack(macro_s_prior_mem, dim=1)
        macro_r_prior_mem = torch.stack(macro_r_prior_mem, dim=1)

        return macro_s_prior_mem, macro_s_prior_dist_mem, macro_r_prior_mem, macro_r_prior_dist_mem

    def macro_next_posterior(self,
                             macro_s: torch.Tensor,
                             macro_a: torch.Tensor,
                             h: Tuple[torch.Tensor, torch.Tensor]):
        h_flat = self.filter_single_step_model_history(h)
        #_, macro_s_next_post_dist, macro_r_next_post_dist = self.abstract_model(macro_s, macro_a, h_flat)
        #macro_s_next = macro_s_next_post_dist.sample()
        #macro_r_next_post = macro_r_next_post_dist.sample()
        pred = self.abstract_model(macro_s, macro_a, h_flat)
        macro_s_next = pred['macro_s_next_dist'].sample()
        macro_s_next_dist = pred['macro_s_next_dist']
        macro_r_dist = pred['macro_r_dist']
        macro_r = pred['macro_r_dist'].sample()

        return macro_s_next, macro_s_next_dist, macro_r, macro_r_dist

    def filter_single_step_model_history(self, h: Tuple[torch.Tensor, torch.Tensor]):
        h = torch.concat(h, dim=0)  # concat h and c tensors of LSTM along the layer dimension, this is arbitrary
        h = torch.transpose(h, 0, 1)  # bring batch dimension to front
        h = torch.flatten(h, start_dim=1)  # fold h/c/layer dimension into d_hidden
        return h


def build_single_step_model(d_macro_state: int,
                            d_macro_action: int,
                            d_macro_reward: int,
                            d_state: int,
                            d_action: int,
                            d_reward: int,
                            d_hidden: int,
                            n_rec_layers: int,
                            s_lws: Iterable[int],
                            r_lws: Iterable[int],
                            term_lws: Iterable[int],
                            batch_first: bool = True) -> SingleStepModel:
    # the deterministic model receives s, a, marco_s, macro_a, macro_s_next
    det_mdl = RecurrentBlock(d_state, d_action, d_macro_state, d_macro_action, d_macro_reward, d_macro_state,
                             d_hidden=d_hidden, n_layers=n_rec_layers, batch_first=batch_first, dropout=0.1)
    # the sampling model receives the output of the deterministic model and no additional input
    s_lws = (*s_lws, d_state)
    r_lws = (*r_lws, d_reward)
    term_lws = (*term_lws, 1)
    sampling_mdl_s = GaussianBlock(d_hidden, lws=s_lws)
    sampling_mdl_r = GaussianBlock(d_hidden, lws=r_lws)
    sampling_mdl_term = ContinuousBernoulliBlock(d_hidden, lws=term_lws)
    single_step_mdl = SingleStepModel(det_mdl, sampling_mdl_s, sampling_mdl_r, sampling_mdl_term)

    return single_step_mdl


def build_abstract_model(d_macro_state: int,
                         d_macro_action: int,
                         d_macro_reward: int,
                         d_memory: int,
                         n_memory_layers: int,
                         ff_lws: Iterable[int],
                         s_prior_lws: Iterable[int],
                         s_post_lws: Iterable[int],
                         r_prior_lws: Iterable[int],
                         r_post_lws: Iterable[int],
                         term_lws: Iterable[int]) -> AbstractModel:
    # final hidden state info from single step model contains h and c of all LSTM layers
    d_hidden_final = d_memory * 2 * n_memory_layers
    # assume markovian dynamics at this level of abstraction, so no recurrency
    ff_lws = tuple(ff_lws)
    det_mdl = FeedforwardBlock(d_macro_state, d_macro_action, d_hidden_final, lws=ff_lws)
    # the sampling model receives the output of the deterministic model and no additional input
    s_prior_lws = (*s_prior_lws, d_macro_state)
    s_post_lws = (*s_post_lws, d_macro_state)
    r_prior_lws = (*r_prior_lws, d_macro_reward)
    r_post_lws = (*r_post_lws, d_macro_reward)
    s_prior = GaussianBlock(ff_lws[-1], lws=s_prior_lws)
    s_posterior = GaussianBlock(ff_lws[-1], lws=s_post_lws)
    r_prior = GaussianBlock(ff_lws[-1], lws=r_prior_lws)
    r_posterior = GaussianBlock(ff_lws[-1], lws=r_post_lws)
    term_prior = ContinuousBernoulliBlock(ff_lws[-1], lws=term_lws)
    term_posterior = ContinuousBernoulliBlock(ff_lws[-1], lws=term_lws)
    abstract_model = AbstractModel(det_mdl=det_mdl, sampling_mdl_s_prior=s_prior, sampling_mdl_s_posterior=s_posterior,
                                   sampling_mdl_r_prior=r_prior, sampling_mdl_r_posterior=r_posterior,
                                   sampling_mdl_term_prior=term_prior, sampling_mdl_term_posterior=term_posterior)

    return abstract_model
