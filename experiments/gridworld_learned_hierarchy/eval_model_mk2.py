import torch
import numpy as np
from tqdm import tqdm

from mdm.gridworld.gridworld import Gridworld
from mdm.models.multiscale_model_mk2 import MultiscaleDynamicsModelMK2
from mdm.planning.cem_planner import CrossentropyPlanner, DistributionType
from mdm.utils.utils import here, gen_macro_state_map, infer_position, transform_macro_s_init_history, gen_video
from mdm.utils.planning_tools_mk2 import init_abstr_s, plan_section, plan_abstract
from mdm.memory.trajectory_memory import TrajectoryMemory


if __name__ == '__main__':
    env = Gridworld.from_cleartext(here() / '../../mdm/gridworld/8x8_v0.mapdata')
    mdl: MultiscaleDynamicsModelMK2 = torch.load(here() / 'model.ptmdl')
    mdl = mdl.to('cuda')
    planner_prim = CrossentropyPlanner(DistributionType.CATEGORICAL, device=mdl.device)
    planner_abstr = CrossentropyPlanner(DistributionType.NORMAL, device=mdl.device)

    n_episodes = 10
    store_result_trajectories = False
    pln_d_batch = 5000
    pln_n_optim_steps_abstr = 30
    pln_n_optim_steps_prim = 20
    pln_winning_perc = 0.01
    pln_discount = 0.90
    pln_act_noise_abstr = 0.01
    pln_act_noise_prim = 0.001
    pln_n_abstract_steps = 100

    #macro_s_init_mean, macro_s_init_std, macro_s_init_per_state_per_action = gen_macro_state_map(env, mdl, 3)
    #macro_ss_list, loc_list, act_seq_list = transform_macro_s_init_history(macro_s_init_per_state_per_action)
    #macro_s_lookup = np.stack([v for k, v in macro_s_init_mean.items()])
    #positions_lookup = np.stack([k for k, v in macro_s_init_mean.items()])

    mem = TrajectoryMemory()
    succeeded = 0
    n_steps = []
    action_stats = np.zeros(env.action_space.n)
    for i_ep in tqdm(range(n_episodes)):
        s_mem, a_mem, r_mem, term_mem = [], [], [], []

        o = env.reset()
        s_mem.append(o)

        o_start = torch.from_numpy(o).to(mdl.device)
        plan_init = init_abstr_s(model=mdl, planner=planner_prim, env=env, o_start=o_start, n_rollouts=pln_d_batch,
                                 n_evolution_steps=pln_n_optim_steps_abstr, winning_perc=pln_winning_perc,
                                 discount=pln_discount, act_noise=pln_act_noise_abstr)

        plan_abstr = plan_abstract(model=mdl, planner=planner_abstr, abstr_s_start=plan_init['abstr_s'],
                                   abstr_s_start_dist=plan_init['abstr_s_post'],
                                   n_plan_steps=pln_n_abstract_steps, n_rollouts=pln_d_batch,
                                   n_evolution_steps=pln_n_optim_steps_abstr, winning_perc=pln_winning_perc,
                                   discount=pln_discount, act_noise=pln_act_noise_abstr)
        # assemble macro trajectory out of initial data and rollout results
        best_abstr_s = torch.concat([plan_init['abstr_s'], plan_abstr['abstr_s']], dim=0)

        actions = plan_init['prim_a']
        for t in range(pln_n_abstract_steps):
            plan_detail = plan_section(model=mdl, planner=planner_prim, env=env, abstr_s=best_abstr_s[t],
                                       abstr_a=plan_abstr['abstr_a'][t], abstr_s_next=best_abstr_s[t + 1],
                                       n_rollouts=pln_d_batch, n_evolution_steps=pln_n_optim_steps_prim,
                                       winning_perc=pln_winning_perc, discount=pln_discount,
                                       act_noise=pln_act_noise_prim)
            actions = torch.concat([actions, plan_detail['prim_a']], dim=0)

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

        n_macro_steps = np.ceil(i_step  / mdl.abstract_step_size).astype(np.int32)
        best_macro_terms = plan_abstr['abstr_term']
        #plot_mats = infer_position(env, best_macro_ss[:n_macro_steps], best_macro_terms[:n_macro_steps], macro_ss_list, loc_list, act_seq_list)
        #ani = gen_video(plot_mats, 1000, 2000)
        #ani.save(f'animation_{i_ep}.mp4')

    if store_result_trajectories:
        TrajectoryMemory.store(mem, 'test_rollouts.samples.samples')

    # final debug output
    action_stats /= action_stats.sum()
    print(succeeded/n_episodes)
    print(n_steps)
    print(action_stats)
