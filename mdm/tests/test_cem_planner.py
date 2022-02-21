import unittest

import torch

from mdm.planning.cem_planner import CrossentropyPlanner, DistributionType


class CategoricalCEPlannerTest(unittest.TestCase):

    def setUp(self) -> None:
        self.dist_type = DistributionType.CATEGORICAL
        self.ce_planner = CrossentropyPlanner(self.dist_type)
        self.n_start_states = 3
        self.d_state = 5
        self.n_actions = 10

    def test_plan_pattern_1(self):
        d_batch = 128
        n_opt_steps = 10
        n_plan_steps = 20
        winning_perc = 0.15
        discount = 1
        act_noise = 0.01
        r_good, r_bad = 1.0, 0.0

        start_states = torch.zeros(d_batch, self.n_start_states, self.d_state)
        ground_truth_best_a = torch.randint(0, self.n_actions, (n_plan_steps,))

        def rollout_fn(start_states: torch.Tensor, actions: torch.Tensor):
            d_batch, d_time = actions.shape[:2]
            rewards = torch.zeros(d_batch, d_time)
            for t in range(d_time):
                rewards[:, t] = torch.where(actions[:, t] == ground_truth_best_a[t], r_good, r_bad)
            return rewards

        best_a = self.ce_planner.plan(rollout_fn=rollout_fn, start_states=start_states, d_dist=self.n_actions,
                                      n_plan_steps=n_plan_steps, n_evolution_steps=n_opt_steps,
                                      winning_perc=winning_perc, discount=discount, act_noise=act_noise)

        n_fails = len(torch.nonzero(ground_truth_best_a - best_a).squeeze())
        self.assertLess(n_fails, 3)
        #print(ground_truth_best_a - best_a)

if __name__ == '__main__':
    unittest.main()
