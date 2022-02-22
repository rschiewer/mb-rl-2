import unittest

import torch

from mdm.planning.cem_planner import CrossentropyPlanner, DistributionType


class CrossentropyMethodPlanner(unittest.TestCase):

    def setUp(self) -> None:
        self.n_start_states = 3
        self.d_state = 5

    def test_plan_categorical_1(self):
        d_batch = 128
        n_action = 10
        n_opt_steps = 10
        n_plan_steps = 20
        winning_perc = 0.25
        discount = 1
        act_noise = 0.01
        r_good, r_bad = 1.0, 0.0
        dist_type = DistributionType.CATEGORICAL

        ce_planner = CrossentropyPlanner(dist_type)
        start_states = torch.zeros(d_batch, self.n_start_states, self.d_state)
        ground_truth_best_a = torch.randint(0, n_action, (n_plan_steps,))

        def rollout_fn(start_states: torch.Tensor, actions: torch.Tensor):
            d_batch, d_time = actions.shape[:2]
            rewards = torch.zeros(d_batch, d_time)
            for t in range(d_time):
                rewards[:, t] = torch.where(actions[:, t] == ground_truth_best_a[t], r_good, r_bad)
            return rewards

        winner_a, dist, i_winners = ce_planner.plan(rollout_fn=rollout_fn, start_states=start_states,
                                                    d_dist=n_action, n_plan_steps=n_plan_steps,
                                                    n_evolution_steps=n_opt_steps, winning_perc=winning_perc,
                                                    discount=discount, act_noise=act_noise)

        champion_a = winner_a[0]
        champion_dist_mode = torch.argmax(dist.probs[i_winners[0]], dim=-1)
        all_dist_mode = torch.argmax(dist.probs[i_winners].mean(dim=(0)), dim=-1)

        print(len(torch.nonzero(ground_truth_best_a - champion_a)))
        print(len(torch.nonzero(ground_truth_best_a - champion_dist_mode)))
        print(len(torch.nonzero(ground_truth_best_a - all_dist_mode)))

    def test_plan_normal_1(self):
        d_batch = 128
        n_action = 10
        n_opt_steps = 20
        n_plan_steps = 20
        winning_perc = 0.25
        discount = 1
        act_noise = 0.1
        dist_type = DistributionType.NORMAL

        ce_planner = CrossentropyPlanner(dist_type)
        start_states = torch.zeros(d_batch, self.n_start_states, self.d_state)
        ground_truth_best_a = torch.rand(n_plan_steps, n_action)

        def rollout_fn(start_states: torch.Tensor, actions: torch.Tensor):
            rewards = - torch.mean(torch.abs(actions - ground_truth_best_a.unsqueeze(0)), dim=2)
            return rewards

        winner_a, dist, i_winners = ce_planner.plan(rollout_fn=rollout_fn, start_states=start_states,
                                                    d_dist=n_action, n_plan_steps=n_plan_steps,
                                                    n_evolution_steps=n_opt_steps, winning_perc=winning_perc,
                                                    discount=discount, act_noise=act_noise)

        champion_a = winner_a[0]
        champion_dist_mode = dist.mean[i_winners[0]]
        all_dist_mode = dist.mean[i_winners].mean(dim=0)

        print(torch.mean(torch.abs(ground_truth_best_a - champion_a)))
        print(torch.mean(torch.abs(ground_truth_best_a - champion_dist_mode)))
        print(torch.mean(torch.abs(ground_truth_best_a - all_dist_mode)))


if __name__ == '__main__':
    unittest.main()
