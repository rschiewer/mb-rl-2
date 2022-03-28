import torch
import numpy as np
from tqdm import tqdm

from mdm.gridworld.gridworld import Gridworld
from mdm.models.multiscale_model import MultiscaleDynamicsModel
from mdm.planning.cem_planner import CrossentropyPlanner, DistributionType
from mdm.utils.utils import here
from mdm.memory.trajectory_memory import flatten_and_unsqueeze


if __name__ == '__main__':
    env = Gridworld.from_cleartext(here() / '../../mdm/gridworld/8x8_v0.mapdata')
    mdl: MultiscaleDynamicsModel = torch.load(here() / 'model.ptmdl')
    planner = CrossentropyPlanner(DistributionType.CATEGORICAL, device=mdl.device)

    n_episodes = 10

    pln_d_batch = 2048
    pln_n_optim_steps = 10
    pln_winning_perc = 0.1
    pln_discount = 1.0
    pln_act_noise = 0.01
    pln_n_abstract_steps = 50

    h_mem = []
    c_mem = []

    def _rollout_init_fn(start_states: torch.Tensor, actions: torch.Tensor):
        start_states = flatten_and_unsqueeze(start_states)
        start_states = start_states.float() / torch.tensor((env.grid_h - 1, env.grid_w - 1), device=mdl.device)
        actions = torch.nn.functional.one_hot(actions, num_classes=mdl.d_action)
        s, s_dist, r, r_dist, h = mdl.rollout_single_step(start_states, actions)
        r = r.squeeze()
        return r, {'h': h, 's': s}

    def _rollout_abstract_fn(macro_start_state: torch.Tensor, macro_actions: torch.Tensor):
        macro_actions = torch.nn.functional.one_hot(macro_actions, num_classes=mdl.d_macro_action)
        macro_s_prior, macro_r_prior = mdl.rollout_abstract(macro_start_state, macro_actions)
        macro_s = torch.stack([s.sample() for s in macro_s_prior], dim=1)
        macro_r = torch.stack([r.sample() for r in macro_r_prior], dim=1)
        macro_r = macro_r.squeeze()
        return macro_r, {'macro_s': macro_s, 'macro_r': macro_r}

    def init_macro_s(s: torch.Tensor):
        s_batch = torch.tile(s, dims=(pln_d_batch, 1))  # copy same starting observation along batch
        s_batch = s_batch.unsqueeze(1)  # add time dimension of 1
        actions, act_dist, i_winners, rollout_data = planner.plan(rollout_fn=_rollout_init_fn,
                                                                  start_states=s_batch,
                                                                  d_dist=env.action_space.n,
                                                                  n_plan_steps=mdl.abstract_step_size,
                                                                  n_evolution_steps=pln_n_optim_steps,
                                                                  winning_perc=pln_winning_perc,
                                                                  discount=pln_discount,
                                                                  act_noise=pln_act_noise)
        i_best = i_winners[0]
        best_h = rollout_data['h'][0][:, i_best].unsqueeze(1), rollout_data['h'][1][:, i_best].unsqueeze(1)
        best_a = torch.nn.functional.one_hot(actions[i_best], num_classes=mdl.d_action).float()
        best_s = rollout_data['s'][i_best]
        zero_macro_s = torch.zeros(1, mdl.d_macro_state, device=mdl.device)
        macro_a = mdl.macro_action_model(best_a.unsqueeze(0))
        macro_s_next_posterior, macro_r_posterior = mdl.macro_next_posterior(zero_macro_s, macro_a, best_h,
                                                                             return_samples=True)

        return macro_s_next_posterior, macro_a, macro_r_posterior, best_a, best_s

    def plan_section(s: torch.Tensor, macro_s: torch.Tensor, macro_a: torch.Tensor, macro_r: torch.Tensor,
                     macro_s_next: torch.Tensor):
        s_batch = torch.tile(s, dims=(pln_d_batch, 1))  # copy same starting observation along batch
        s_batch = s_batch.unsqueeze(1)  # add time dimension of 1
        macro_s_batch = torch.tile(macro_s, dims=(pln_d_batch, 1))
        macro_a_batch = torch.tile(macro_a, dims=(pln_d_batch, 1))
        macro_r_batch = torch.tile(macro_r, dims=(pln_d_batch, 1))
        macro_s_next_batch = torch.tile(macro_s_next, dims=(pln_d_batch, 1))

        # use closure to bind macro_x arguments inside the function to the above defined ones
        def _rollout_detailed_fn(start_states: torch.Tensor, actions: torch.Tensor):
            start_states = flatten_and_unsqueeze(start_states)
            start_states = start_states.float() / torch.tensor((env.grid_h - 1, env.grid_w - 1), device=mdl.device)
            actions = torch.nn.functional.one_hot(actions, num_classes=mdl.d_action)
            s, s_dist, r, r_dist, h = mdl.rollout_single_step(start_states, actions, macro_s_batch, macro_a_batch,
                                                              macro_r_batch, macro_s_next_batch)
            h_mem.append(h[0].detach().cpu().numpy())
            c_mem.append(h[1].detach().cpu().numpy())
            theoretical_macro_s_next, theoretical_macro_r = mdl.macro_next_posterior(macro_s_batch, macro_a_batch, h,
                                                                                     return_samples=True)
            #overlap = [torch.distributions.kl_divergence(t_macro_s, macro_s)
            #           for t_macro_s, macro_s in zip(theoretical_macro_s_next, macro_s_next_batch)]
            #overlap = -torch.stack(overlap).sum()
            overlap = - torch.sum(torch.abs(theoretical_macro_s_next - macro_s_next_batch), dim=1, keepdim=True)
            return overlap, {'r': r, 'h': h, 's': s}

        actions, act_dist, i_winners, rollout_data = planner.plan(rollout_fn=_rollout_detailed_fn,
                                                                  start_states=s_batch,
                                                                  d_dist=env.action_space.n,
                                                                  n_plan_steps=mdl.abstract_step_size,
                                                                  n_evolution_steps=pln_n_optim_steps,
                                                                  winning_perc=pln_winning_perc,
                                                                  discount=pln_discount,
                                                                  act_noise=pln_act_noise)
        i_best = i_winners[0]
        best_a = torch.nn.functional.one_hot(actions[i_best], num_classes=mdl.d_action).float()
        best_s = rollout_data['s'][i_best]

        return best_s, best_a

    ### use model for planning ###

    succeeded = 0
    action_stats = np.zeros(env.action_space.n)
    for i_ep in tqdm(range(n_episodes)):
        s = torch.from_numpy(env.reset()).to(mdl.device)
        macro_s_1, macro_a_0, macro_r_0, as_, ss = init_macro_s(s)  # macro_s_0 is always fixed at 0

        # abstract level rollout
        macro_s_batch = torch.tile(macro_s_1, dims=(pln_d_batch, 1))  # copy same starting observation along batch
        macro_s_batch = macro_s_batch.unsqueeze(1)  # add time dimension of 1
        macro_actions, act_dist, i_winners, rollout_data = planner.plan(rollout_fn=_rollout_abstract_fn,
                                                                        start_states=macro_s_batch,
                                                                        d_dist=mdl.d_macro_action,
                                                                        n_plan_steps=pln_n_abstract_steps,
                                                                        n_evolution_steps=pln_n_optim_steps,
                                                                        winning_perc=pln_winning_perc,
                                                                        discount=pln_discount,
                                                                        act_noise=pln_act_noise)
        i_top_cand = i_winners[0]
        best_macro_as = torch.nn.functional.one_hot(macro_actions[i_top_cand], num_classes=mdl.d_macro_action).float()
        best_macro_ss = rollout_data['macro_s'][i_top_cand]
        best_macro_rs = rollout_data['macro_r'][i_top_cand]

        # assemble macro trajectories out of initial data and rollout results
        macro_s_traj = torch.concat([macro_s_1, best_macro_ss], dim=0)
        macro_a_traj = best_macro_as
        macro_r_traj = best_macro_rs

        # act out the details
        actions = as_
        for t in range(pln_n_abstract_steps):
            s = torch.zeros_like(ss[-1])
            new_ss, new_as = plan_section(s, macro_s_traj[t], macro_a_traj[t], macro_r_traj[t], macro_s_traj[t+1])
            actions = torch.concat([actions, new_as], dim=0)

        action_iter = iter(actions.detach().cpu().numpy())
        terminal = False

        while not terminal:
            env.render()
            #video.write(np.random.randint(0, 255, (500, 500, 3), np.uint8))
            try:
                a_one_hot = next(action_iter)
                a = np.argmax(a_one_hot, axis=-1)
                s_, r, terminal, info = env.step(a)
                s = s_
                action_stats[a] += 1
                if terminal:
                    succeeded += 1
            except StopIteration:
                break

    action_stats /= action_stats.sum()
    print(succeeded/n_episodes)
    print(action_stats)

    #h_mem = np.stack(h_mem, axis=0).reshape((len(h_mem), -1))
    #c_mem = np.stack(c_mem, axis=0).reshape((len(c_mem), -1))

    #print(np.std(h_mem, axis=0))
    #print(np.std(c_mem, axis=0))

