import time
import unittest

import torch

from mdm.planning.cem_planner import CrossentropyPlanner, DistributionType
from mdm.gridworld.gridworld import Gridworld
from mdm.utils.utils import here


class CrossentropyMethodPlanner(unittest.TestCase):

    def setUp(self) -> None:
        self.n_start_states = 3
        self.d_state = 5

    def test_plan_categorical_1(self):
        d_batch = 64
        n_action = 10
        n_opt_steps = 15
        n_plan_steps = 20
        winning_perc = 0.35
        discount = 1
        act_noise = 0.01
        r_good, r_bad = torch.tensor(1.0), torch.tensor(0.0)
        dist_type = DistributionType.CATEGORICAL

        ce_planner = CrossentropyPlanner(dist_type)
        ground_truth_best_a = torch.randint(0, n_action, (n_plan_steps,))

        def rollout_fn(actions: torch.Tensor):
            d_batch, d_time = actions.shape[:2]
            rewards = torch.zeros(d_batch, d_time)
            for t in range(d_time):
                rewards[:, t] = torch.where(actions[:, t] == ground_truth_best_a[t], r_good, r_bad)
            return rewards, None, {}

        actions, dist, i_winners, R_winners, rollout_data = ce_planner.plan(rollout_fn=rollout_fn, d_dist=n_action,
                                                                            n_rollouts=d_batch,
                                                                            n_plan_steps=n_plan_steps,
                                                                            n_evolution_steps=n_opt_steps,
                                                                            winning_perc=winning_perc,
                                                                            discount=discount, act_noise=act_noise)

        champion_a = actions[i_winners[0]]
        champion_dist_mode = torch.argmax(dist.probs[i_winners[0]], dim=-1)
        all_dist_mode = torch.argmax(dist.probs[i_winners].mean(dim=(0)), dim=-1)

        print(len(torch.nonzero(ground_truth_best_a - champion_a)))
        print(len(torch.nonzero(ground_truth_best_a - champion_dist_mode)))
        print(len(torch.nonzero(ground_truth_best_a - all_dist_mode)))

    def test_plan_normal_1(self):
        d_batch = 256
        n_action = 10
        n_opt_steps = 20
        n_plan_steps = 20
        winning_perc = 0.1
        discount = 0.99
        act_noise = 0.1
        dist_type = DistributionType.NORMAL

        ce_planner = CrossentropyPlanner(dist_type)
        ground_truth_best_a = 10 * torch.rand(n_plan_steps, n_action) - 5

        def rollout_fn(actions: torch.Tensor):
            rewards = - torch.mean(torch.abs(actions - ground_truth_best_a.unsqueeze(0) ** 2), dim=2)
            return rewards, None, {}

        actions, dist,i_winners, R_winners, rollout_data = ce_planner.plan(rollout_fn=rollout_fn,
                                                                n_rollouts=d_batch,
                                                                d_dist=n_action, n_plan_steps=n_plan_steps,
                                                                n_evolution_steps=n_opt_steps, winning_perc=winning_perc,
                                                                discount=discount, act_noise=act_noise)

        champion_a = actions[i_winners[0]]
        champion_dist_mode = dist.mean[i_winners[0]]
        all_dist_mode = dist.mean[i_winners].mean(dim=0)

        print(torch.mean(torch.abs(ground_truth_best_a - champion_a)))
        print(torch.mean(torch.abs(ground_truth_best_a - champion_dist_mode)))
        print(torch.mean(torch.abs(ground_truth_best_a - all_dist_mode)))

    def test_planning_with_groundtruth(self):
        d_batch = 128
        n_opt_steps = 10
        n_plan_steps = 100
        winning_perc = 0.25
        discount = 0.99
        act_noise = 0.001

        batched_envs = [Gridworld.from_cleartext(here() / 'testmap.mapdata') for _ in range(d_batch)]
        ce_planner = CrossentropyPlanner(DistributionType.CATEGORICAL)

        def rollout_fn(actions: torch.Tensor):
            rewards = torch.zeros(d_batch, n_plan_steps)
            terminal_flags = torch.ones(d_batch, n_plan_steps)
            for i_env, env in enumerate(batched_envs):
                env.reset()
                for i_a, a in enumerate(actions[i_env]):
                    s, r, done, info = env.step(a.detach().cpu().numpy())
                    rewards[i_env, i_a] = r
                    terminal_flags[i_env, i_a] = done
                    if done:
                        break
            #terminal_flags = None
            return rewards, terminal_flags, {}

        actions, dist, i_winners, R_winners, rollout_data = ce_planner.plan(rollout_fn=rollout_fn,
                                                                 d_dist=batched_envs[0].action_space.n,
                                                                 n_rollouts=d_batch,
                                                                 n_plan_steps=n_plan_steps,
                                                                 n_evolution_steps=n_opt_steps,
                                                                 winning_perc=winning_perc,
                                                                 discount=discount,
                                                                 act_noise=act_noise)
        actions = actions[i_winners[0]]
        eval_env = batched_envs[0]
        eval_env.reset()
        for a in actions:
            eval_env.render()
            a, r, done, info = eval_env.step(a.detach().cpu().numpy())
            time.sleep(0.5)
            if done:
                break



if __name__ == '__main__':
    unittest.main()
