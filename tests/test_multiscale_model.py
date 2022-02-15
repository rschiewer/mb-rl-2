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
        macro_r_mem = predictions[6]
        macro_r_prior_mem, macro_r_posterior_mem = predictions[7:]

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

    def test_train_step(self):
        d_batch = 32
        d_time = 20
        lr = 0.0001
        momentum = 0.9
        n_warmup = 3

        s_ground_truth = torch.ones(d_batch, d_time, self.d_state)
        a_ground_truth = torch.ones(d_batch, d_time, self.d_action)
        r_ground_truth = torch.ones(d_batch, d_time, self.d_reward)
        optimizer = torch.optim.SGD(self.mdl.parameters(), lr=lr, momentum=momentum)

        loss, rec_loss, kl_loss, macro_r_loss = self.mdl.train_step(s_ground_truth, a_ground_truth, r_ground_truth,
                                                                    n_warmup, optimizer)

if __name__ == '__main__':
    unittest.main()
