import unittest

import torch

from models.multiscale_model import *


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.d_state = 10
        self.d_action = 4
        self.d_reward = 1
        self.d_hidden = 32
        self.n_rec_layers = 2
        self.d_macro_state = 20
        self.d_macro_action = 8
        self.d_macro_reward = 1
        self.n_abstract_steps = 3

        single_step_mdl = build_single_step_model(self.d_macro_state, self.d_macro_action, self.d_macro_reward,
                                                  self.d_state, self.d_action, self.d_reward, self.d_hidden,
                                                  self.n_rec_layers)
        abstract_mdl = build_abstract_model(self.d_macro_state, self.d_macro_action, self.d_macro_reward, self.d_hidden,
                                            self.n_rec_layers)
        macro_action_mdl = MacroActionModel(self.d_action, self.n_abstract_steps, self.d_macro_action)

        multiscale_mdl = MultiscaleDynamicsModel(single_step_mdl, abstract_mdl, macro_action_mdl, self.n_abstract_steps,
                                                 self.d_state, self.d_action, self.d_reward, self.d_macro_state,
                                                 self.d_macro_action, self.d_macro_reward)
        self.mdl = multiscale_mdl

    def test_shapes(self):
        d_batch = 32
        d_time = 20
        n_start_states = 3

        start_states = torch.ones(d_batch, n_start_states, self.d_state)
        actions = torch.ones(d_batch, d_time, self.d_action)

        predictions = self.mdl(start_states, actions)
        s_mem, s_dist_mem, r_mem, r_dist_mem = predictions[:4]
        macro_s_prior_mem, macro_s_posterior_mem = predictions[4:6]
        macro_r_prior_mem, macro_r_posterior_mem = predictions[6:]

        for x in [s_mem, s_dist_mem, r_mem, r_dist_mem]:
            self.assertEqual(len(x), d_time)

        for x in [macro_s_prior_mem, macro_s_posterior_mem, macro_r_prior_mem, macro_r_posterior_mem]:
            self.assertEqual(len(x), d_time // self.n_abstract_steps + 1)

        for s in s_mem:
            self.assertEqual(s.shape, (d_batch, self.d_state))
        for r in r_mem:
            self.assertEqual(r.shape, (d_batch, self.d_reward))
        for dist in macro_s_prior_mem + macro_s_posterior_mem:
            smpl = dist.sample()
            self.assertEqual(smpl.shape, (d_batch, self.d_macro_state))
        for dist in macro_r_prior_mem + macro_r_posterior_mem:
            smpl = dist.sample()
            self.assertEqual(smpl.shape, (d_batch, self.d_macro_reward))

    def test_training(self):
        d_batch = 32
        d_time = 20
        n_start_states = 3
        lr = 0.0001
        momentum = 0.9

        def reconstr_loss(s_pred, r_pred, s_true, r_true):
            l = torch.mean((s_pred - s_true) ** 2) + torch.mean((r_pred - r_true) ** 2)
            return l

        def kl_loss(s_priors, s_posteriors, r_priors, r_posteriors):
            l = 0
            for s_prior, s_posterior, r_prior, r_posterior in zip(s_priors, s_posteriors, r_priors, r_posteriors):
                l += torch.distributions.kl.kl_divergence(s_prior, s_posterior)
                l += torch.distributions.kl.kl_divergence(r_prior, r_posterior)
            return torch.mean(l)

        s_ground_truth = torch.ones(d_batch, d_time, self.d_state)
        r_ground_truth = torch.ones(d_batch, d_time, self.d_state)
        start_states = s_ground_truth[:, :n_start_states, :]
        actions = torch.ones(d_batch, d_time, self.d_action)
        optimizer = torch.optim.SGD(self.mdl.parameters(), lr=lr, momentum=momentum)

        # one train step
        optimizer.zero_grad(True)
        predictions = self.mdl(start_states, actions)
        s_mem, s_dist_mem, r_mem, r_dist_mem = predictions[:4]
        macro_s_prior_mem, macro_s_posterior_mem = predictions[4:6]
        macro_r_prior_mem, macro_r_posterior_mem = predictions[6:]

        rec = reconstr_loss(torch.stack(s_mem, dim=1), torch.stack(r_mem, dim=1), s_ground_truth, r_ground_truth)
        kl = kl_loss(macro_s_prior_mem, macro_s_posterior_mem, macro_r_prior_mem, macro_r_posterior_mem)
        loss = rec + kl
        print(rec)
        print(kl)
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    unittest.main()
