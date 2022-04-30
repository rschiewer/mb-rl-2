import torch
import numpy as np
from tqdm import tqdm

from mdm.gridworld.gridworld import Gridworld
from mdm.models.multiscale_model import MultiscaleDynamicsModel
from mdm.planning.cem_planner import CrossentropyPlanner, DistributionType
from mdm.utils.utils import here, normalize_obs, one_hot_actions
from mdm.utils.torch_tools import extract_sub_distribution
from mdm.memory.trajectory_memory import flatten_and_unsqueeze, TrajectoryMemory


if __name__ == '__main__':
    env = Gridworld.from_cleartext(here() / '../../mdm/gridworld/8x8_v0.mapdata')
    mdl: MultiscaleDynamicsModel = torch.load(here() / 'model.ptmdl')
    planner = CrossentropyPlanner(DistributionType.CATEGORICAL, device=mdl.device)
    planner_abstract = CrossentropyPlanner(DistributionType.NORMAL, device=mdl.device)

    n_episodes = 10
    store_result_trajectories = False
    pln_d_batch = 8000
    pln_n_optim_steps_macro = 20
    pln_winning_perc = 0.05
    pln_discount = 0.99
    pln_act_noise = 0.01
    pln_n_abstract_steps = 100

    def _rollout_init_fn(start_states: torch.Tensor, actions: torch.Tensor):
        start_states = normalize_obs(start_states, env)
        actions = one_hot_actions(actions, n_classes=mdl.d_action)
        #start_states = flatten_and_unsqueeze(start_states)
        #start_states = start_states.float() / torch.tensor((env.grid_h - 1, env.grid_w - 1), device=mdl.device ) - 0.5
        #actions = torch.nn.functional.one_hot(actions, num_classes=mdl.d_action)
        predictions_ss = mdl.rollout_single_step(start_states, actions)
        return predictions_ss['r'].squeeze(), predictions_ss['term'].squeeze(), predictions_ss

    def _rollout_abstract_fn(macro_start_state: torch.Tensor, macro_actions: torch.Tensor):
        #macro_actions = torch.nn.functional.one_hot(macro_actions, num_classes=mdl.d_macro_action)
        predictions = mdl.rollout_abstract(macro_start_state, macro_actions)
        macro_s_prior, macro_s_prior_dist, macro_r_prior, macro_r_prior_dist = predictions
        return macro_r_prior.squeeze(), None, {'macro_s': macro_s_prior, 'macro_s_dist': macro_s_prior_dist,
                                               'macro_r': macro_r_prior, 'macro_r_dist': macro_r_prior_dist}

    def init_macro_s(s: torch.Tensor):
        s_batch = torch.tile(s, dims=(pln_d_batch, 1))  # copy same starting observation along batch
        s_batch = s_batch.unsqueeze(1)  # add time dimension of 1
        actions, act_dist, i_winners, rollout_data = planner.plan(rollout_fn=_rollout_init_fn,
                                                                  start_states=s_batch,
                                                                  d_dist=env.action_space.n,
                                                                  n_plan_steps=mdl.macro_step_size,
                                                                  n_evolution_steps=pln_n_optim_steps_macro,
                                                                  winning_perc=pln_winning_perc,
                                                                  discount=pln_discount,
                                                                  act_noise=pln_act_noise)
        i_best = i_winners[0]
        best_h = rollout_data['h'][0][:, i_best].unsqueeze(1), rollout_data['h'][1][:, i_best].unsqueeze(1)
        best_a = torch.nn.functional.one_hot(actions[i_best], num_classes=mdl.d_action).float()
        best_s = rollout_data['s'][i_best]
        zero_macro_s = torch.zeros(1, mdl.d_macro_state, device=mdl.device)
        zero_macro_a = torch.zeros(1, mdl.d_macro_action, device=mdl.device)
        #macro_a = mdl.macro_action_model(best_a.unsqueeze(0))
        predictions = mdl.macro_next_posterior(zero_macro_s, zero_macro_a, best_h)
        macro_s_next_post, macro_s_next_post_dist, macro_r_next_post, macro_r_next_post_dist = predictions

        return macro_s_next_post, macro_s_next_post_dist, macro_r_next_post, macro_r_next_post_dist, best_a, best_s

    def plan_section(s: torch.Tensor, macro_s: torch.Tensor, macro_a: torch.Tensor,
                     macro_r: torch.Tensor, macro_s_next: torch.Tensor):
        s_batch = torch.tile(s, dims=(pln_d_batch, 1))  # copy same starting observation along batch
        s_batch = s_batch.unsqueeze(1)  # add time dimension of 1
        macro_s_batch = torch.tile(macro_s, dims=(pln_d_batch, 1))
        macro_a_batch = torch.tile(macro_a, dims=(pln_d_batch, 1))
        macro_r_batch = torch.tile(macro_r, dims=(pln_d_batch, 1))
        macro_s_next_batch = torch.tile(macro_s_next, dims=(pln_d_batch, 1))

        # use closure to bind macro_x arguments inside the function to the above defined ones
        def _rollout_detailed_fn(start_states: torch.Tensor, actions: torch.Tensor):
            start_states = flatten_and_unsqueeze(start_states)
            start_states = start_states.float() / torch.tensor((env.grid_h - 1, env.grid_w - 1), device=mdl.device) - 0.5
            actions = torch.nn.functional.one_hot(actions, num_classes=mdl.d_action)
            predictions_ss = mdl.rollout_single_step(start_states, actions, macro_s_batch, macro_a_batch)
            predictons_ms = mdl.macro_next_posterior(macro_s_batch, macro_a_batch, predictions_ss['h'])
            macro_s_next_post, macro_s_next_post_dist, macro_r_next_post, macro_r_next_post_dist = predictons_ms
            #overlap = torch.distributions.kl_divergence(macro_s_next_post_dist,
            #                                            macro_s_next.expand((pln_d_batch, mdl.d_macro_state)))
            #overlap = - overlap.abs().sum(dim=1, keepdim=True)
            overlap = macro_s_next_post_dist.log_prob(macro_s_next_batch).sum(dim=1)
            #overlap = -torch.sum(torch.abs(macro_s_next_post - macro_s_next_batch), dim=1)
            return overlap, None, predictions_ss

        actions, act_dist, i_winners, rollout_data = planner.plan(rollout_fn=_rollout_detailed_fn,
                                                                  start_states=s_batch,
                                                                  d_dist=env.action_space.n,
                                                                  n_plan_steps=mdl.macro_step_size,
                                                                  n_evolution_steps=10,
                                                                  winning_perc=pln_winning_perc,
                                                                  discount=1.0,
                                                                  act_noise=0.001)
        i_best = i_winners[0]
        best_a = torch.nn.functional.one_hot(actions[i_best], num_classes=mdl.d_action).float()
        best_s = rollout_data['s'][i_best]

        return best_s, best_a

    ### use model for planning ###

    mem = TrajectoryMemory()
    succeeded = 0
    n_steps = []
    action_stats = np.zeros(env.action_space.n)
    for i_ep in tqdm(range(n_episodes)):
        s_mem, a_mem, r_mem, term_mem = [], [], [], []

        s = env.reset()
        s_mem.append(s)

        s_torch = torch.from_numpy(s).to(mdl.device)
        macro_s_1, macro_s_1_dist, macro_r_0, macro_r_0_dist, as_, ss = init_macro_s(s_torch)  # macro_s_0 is always 0
        macro_s_1_dist = extract_sub_distribution(macro_s_1_dist, 0)  # remove time dim

        # abstract level rollout
        macro_s_batch = macro_s_1_dist.sample(sample_shape=(pln_d_batch,))  # sample some possible start states
        #macro_s_batch = torch.tile(macro_s_1.loc, dims=(pln_d_batch, 1))  # time dim required
        #macro_s_batch = torch.tile(macro_s_1, dims=(pln_d_batch, 1))  # time dim required
        macro_s_batch = macro_s_batch.unsqueeze(1)  # add time dimension of 1
        macro_actions, act_dist, i_winners, rollout_data = planner_abstract.plan(rollout_fn=_rollout_abstract_fn,
                                                                        start_states=macro_s_batch,
                                                                        d_dist=mdl.d_macro_action,
                                                                        n_plan_steps=pln_n_abstract_steps,
                                                                        n_evolution_steps=pln_n_optim_steps_macro,
                                                                        winning_perc=pln_winning_perc,
                                                                        discount=pln_discount,
                                                                        act_noise=pln_act_noise)
        i_top_cand = i_winners[0]
        #best_macro_as = torch.nn.functional.one_hot(macro_actions[i_top_cand], num_classes=mdl.d_macro_action).float()
        best_macro_as = macro_actions[i_top_cand]
        # extract best performer for each time step
        #best_macro_ss = [extract_sub_distribution(d, i_top_cand) for d in rollout_data['macro_s']]
        #best_macro_rs = [extract_sub_distribution(d, i_top_cand) for d in rollout_data['macro_r']]
        best_macro_ss = rollout_data['macro_s'][i_top_cand]
        best_macro_rs = rollout_data['macro_r'][i_top_cand]

        # assemble macro trajectories out of initial data and rollout results
        macro_s_traj = torch.concat([macro_s_1, best_macro_ss], dim=0)
        macro_a_traj = best_macro_as
        macro_r_traj = best_macro_rs

        # act out the details
        actions = as_
        for t in range(pln_n_abstract_steps):
            s_torch = torch.zeros_like(ss[-1])
            new_ss, new_as = plan_section(s_torch, macro_s_traj[t], macro_a_traj[t], macro_r_traj[t], macro_s_traj[t + 1])
            actions = torch.concat([actions, new_as], dim=0)

        action_iter = iter(actions.detach().cpu().numpy())
        terminal = False

        i_step = 0
        while not terminal:
            env.render()
            i_step += 1
            try:
                a_one_hot = next(action_iter)
                a = np.argmax(a_one_hot, axis=-1)
                s_, r, terminal, info = env.step(a)

                s_mem.append(s_)
                a_mem.append(a)
                r_mem.append(r)
                term_mem.append(terminal)

                action_stats[a] += 1
                if terminal and r > 0:
                    succeeded += 1
            except StopIteration:
                break

        n_steps.append(i_step)
        mem.push(s_mem, a_mem, r_mem, term_mem)

    if store_result_trajectories:
        TrajectoryMemory.store(mem, 'test_rollouts.samples.samples')

    # final debug output
    action_stats /= action_stats.sum()
    print(succeeded/n_episodes)
    print(n_steps)
    print(action_stats)
