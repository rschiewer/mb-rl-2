import time
import unittest

import torch

from mdm.planning.cem_planner import CrossentropyPlanner
from mdm.gridworld.gridworld import Gridworld
from mdm.utils.utils import here, DistributionType


class CrossentropyMethodPlanner(unittest.TestCase):

    def setUp(self) -> None:
        self.n_start_states = 3
        self.d_state = 5

    def test_plan_categorical_1(self):
        n_envs = 1
        d_batch = 64
        n_action = 10
        n_opt_steps = 15
        n_plan_steps = 20
        winning_perc = 0.35
        discount = 1
        act_noise = 0.01
        r_good, r_bad = torch.tensor(1.0), torch.tensor(0.0)
        dist_type = DistributionType.CATEGORICAL

        ce_planner = CrossentropyPlanner(type=dist_type, d_dist=n_action, n_evolution_steps=n_opt_steps,
                                         winning_perc=winning_perc, discount=discount, act_noise=act_noise)
        ground_truth_best_a = torch.randint(0, n_action, (n_envs, n_plan_steps))

        def rollout_fn(actions: torch.Tensor):
            _n_envs, _d_batch, _d_time = actions.shape[:3]
            rewards = torch.zeros(_n_envs, _d_batch, _d_time)
            for t in range(_d_time):
                rewards[:, :, t] = torch.where(actions[:, :, t] == ground_truth_best_a[:, t], r_good, r_bad)
            return rewards, None, {}

        actions, dist, i_winners, R_winners, rollout_data = ce_planner.plan(rollout_fn=rollout_fn, n_rollouts=d_batch,
                                                                            n_plan_steps=n_plan_steps, n_envs=n_envs)

        champion_a = ce_planner.get_winner_actions(actions, dist, i_winners, resample=False)
        champion_dist_mode = torch.argmax(dist.probs[torch.arange(n_envs), i_winners[:, 0]], dim=-1)
        all_dist_mode = torch.argmax(dist.probs[torch.arange(n_envs).unsqueeze(1), i_winners].mean(dim=1), dim=-1)

        print(len(torch.nonzero(ground_truth_best_a - champion_a)))
        print(len(torch.nonzero(ground_truth_best_a - champion_dist_mode)))
        print(len(torch.nonzero(ground_truth_best_a - all_dist_mode)))

    def test_plan_normal_1(self):
        n_envs = 1
        d_batch = 256
        n_action = 10
        n_opt_steps = 20
        n_plan_steps = 20
        winning_perc = 0.1
        discount = 0.99
        act_noise = 0.001
        dist_type = DistributionType.NORMAL

        ce_planner = CrossentropyPlanner(type=dist_type, d_dist=n_action, n_evolution_steps=n_opt_steps,
                                         winning_perc=winning_perc, discount=discount, act_noise=act_noise)
        ground_truth_best_a = 10 * torch.rand(n_envs, n_plan_steps, n_action) - 5

        def rollout_fn(actions: torch.Tensor):
            rewards = - torch.mean(torch.abs(actions - ground_truth_best_a.unsqueeze(0) ** 2), dim=2)
            return rewards, None, {}

        actions, dist, i_winners, R_winners, rollout_data = ce_planner.plan(rollout_fn=rollout_fn, n_rollouts=d_batch,
                                                                            n_plan_steps=n_plan_steps, n_envs=n_envs)

        champion_a = ce_planner.get_winner_actions(actions, dist, i_winners, resample=False)
        champion_dist_mode = dist.loc[torch.arange(n_envs), i_winners[:, 0]]
        all_dist_mode = dist.loc[torch.arange(n_envs).unsqueeze(1), i_winners].mean(dim=1)

        print(torch.mean(torch.abs(ground_truth_best_a - champion_a)))
        print(torch.mean(torch.abs(ground_truth_best_a - champion_dist_mode)))
        print(torch.mean(torch.abs(ground_truth_best_a - all_dist_mode)))

    #@unittest.skip
    def test_planning_with_groundtruth(self):
        d_batch = 128
        n_opt_steps = 10
        n_plan_steps = 50
        winning_perc = 0.25
        discount = 0.99
        act_noise = 0.000
        dist_type = DistributionType.CATEGORICAL

        batch_env = [Gridworld.from_cleartext(here() / 'testmap.mapdata') for _ in range(d_batch)]

        def rollout_fn(actions: torch.Tensor):
            actions = actions[0]  # remove "envs" dimension since we've only one environment
            rewards = torch.zeros(d_batch, n_plan_steps)
            terminal_flags = torch.ones(d_batch, n_plan_steps)
            for i_env, env in enumerate(batch_env):
                env.reset()
                for i_a, a in enumerate(actions[i_env]):
                    s, r, term, trunc, info = env.step(a.detach().cpu().numpy())
                    rewards[i_env, i_a] = r
                    terminal_flags[i_env, i_a] = term
                    if term or trunc:
                        break
            # add "envs" dimension to rewards and terminal flags
            rewards = rewards.unsqueeze(0)
            terminal_flags = terminal_flags.unsqueeze(0)
            return rewards, terminal_flags, {}

        ce_planner = CrossentropyPlanner(type=dist_type, d_dist=batch_env[0].action_space.n,
                                         n_evolution_steps=n_opt_steps, winning_perc=winning_perc, discount=discount,
                                         act_noise=act_noise)

        actions, dist, i_winners, R_winners, rollout_data = ce_planner.plan(rollout_fn=rollout_fn, n_rollouts=d_batch,
                                                                 n_plan_steps=n_plan_steps, n_envs=1)
        actions = ce_planner.get_winner_actions(actions, dist, i_winners, resample=False)
        actions = actions[0]  # remove redundant "envs" dimension
        eval_env = batch_env[0]
        eval_env.reset()
        for a in actions:
            eval_env.render()
            a, r, term, trunc, info = eval_env.step(a.detach().cpu().numpy())
            time.sleep(0.1)
            if term or trunc:
                break

    def test_planning_with_groundtruth_multi_env(self):
        n_envs = 2
        d_batch = 128
        n_opt_steps = 10
        n_plan_steps = 50
        winning_perc = 0.25
        discount = 0.99
        act_noise = 0.000
        dist_type = DistributionType.CATEGORICAL

        self.assertTrue(n_envs % 2 == 0)
        batch_envs = [[Gridworld.from_cleartext(here() / 'testmap.mapdata') for _ in range(d_batch)]
                      for _ in range(n_envs // 2)]
        batch_envs += [[Gridworld.from_cleartext(here() / 'testmap2.mapdata') for _ in range(d_batch)]
                       for _ in range(n_envs // 2)]

        def rollout_fn(actions: torch.Tensor):
            rewards = torch.zeros(n_envs, d_batch, n_plan_steps)
            terminal_flags = torch.ones(n_envs, d_batch, n_plan_steps)
            for i_env, batch_env in enumerate(batch_envs):
                for i_batch, env in enumerate(batch_env):
                    env.reset()
                    for i_a, a in enumerate(actions[i_env, i_batch]):
                        s, r, term, trunc, info = env.step(a.detach().cpu().numpy())
                        rewards[i_env, i_batch, i_a] = r
                        terminal_flags[i_env, i_batch, i_a] = term
                        if term or trunc:
                            break
            # add "envs" dimension to rewards and terminal flags
            rewards = rewards
            terminal_flags = terminal_flags
            return rewards, terminal_flags, {}

        ce_planner = CrossentropyPlanner(type=dist_type, d_dist=batch_envs[0][0].action_space.n,
                                         n_evolution_steps=n_opt_steps, winning_perc=winning_perc, discount=discount,
                                         act_noise=act_noise)

        actions, dist, i_winners, R_winners, rollout_data = ce_planner.plan(rollout_fn=rollout_fn, n_rollouts=d_batch,
                                                                            n_plan_steps=n_plan_steps, n_envs=n_envs)
        actions = ce_planner.get_winner_actions(actions, dist, i_winners, resample=False)
        for i_env, batch_env in enumerate(batch_envs):
            cur_env_actions = actions[i_env]  # choose correct "envs" dimension
            eval_env = batch_env[0]
            eval_env.reset()
            for a in cur_env_actions:
                eval_env.render()
                a, r, term, trunc, info = eval_env.step(a.detach().cpu().numpy())
                time.sleep(0.1)
                if term or trunc:
                    break


if __name__ == '__main__':
    unittest.main()
