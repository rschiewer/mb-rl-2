import torch
import numpy as np
from tqdm import tqdm

from mdm.gridworld.gridworld import Gridworld
from mdm.models.multiscale_model import MultiscaleDynamicsModel
from mdm.planning.cem_planner import CrossentropyPlanner
from mdm.utils.utils import here, infer_position, gen_video, \
    DistributionType
from mdm.utils.planning_tools import init_macro_s, plan_section, plan_abstract
from mdm.memory.trajectory_memory import TrajectoryMemory


if __name__ == '__main__':
    env = Gridworld.from_cleartext(here() / '../../mdm/gridworld/8x8_v0.mapdata')
    mdl: MultiscaleDynamicsModel = torch.load(here() / 'model.ptmdl')

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

    planner_prim = CrossentropyPlanner(DistributionType.CATEGORICAL, device=mdl.device, d_dist=env.action_space.n,
                                       n_evolution_steps=pln_n_optim_steps_prim, winning_perc=pln_winning_perc,
                                       discount=pln_discount, act_noise=pln_act_noise_prim)
    planner_abstr = CrossentropyPlanner(DistributionType.NORMAL, device=mdl.device, d_dist=mdl.d_macro_action,
                                        n_evolution_steps=pln_n_optim_steps_abstr, winning_perc=pln_winning_perc,
                                        discount=pln_discount, act_noise=pln_act_noise_abstr)

    #macro_s_lookup = np.stack([v for k, v in macro_s_init_mean.items()])
    #positions_lookup = np.stack([k for k, v in macro_s_init_mean.items()])

    mem = TrajectoryMemory()
    succeeded = 0
    n_steps = []
    action_stats = np.zeros(env.action_space.n)
    for i_ep in tqdm(range(n_episodes)):
        s_mem, a_mem, r_mem, term_mem = [], [], [], []

        s = env.reset()
        s_mem.append(s)

        s_start = torch.from_numpy(s).to(mdl.device)
        plan_init = init_macro_s(model=mdl, planner=planner_prim, env=env, s_start=s_start, n_rollouts=pln_d_batch,
                                 n_evolution_steps=pln_n_optim_steps_abstr, winning_perc=pln_winning_perc,
                                 discount=pln_discount, act_noise=pln_act_noise_abstr)  # macro_s_0 is always 0

        plan_abstr = plan_abstract(model=mdl, planner=planner_abstr, macro_s_start=plan_init['macro_s_next'],
                                   macro_s_start_dist=plan_init['macro_s_next_post'],
                                   n_plan_steps=pln_n_abstract_steps, n_rollouts=pln_d_batch,
                                   n_evolution_steps=pln_n_optim_steps_abstr, winning_perc=pln_winning_perc,
                                   discount=pln_discount, act_noise=pln_act_noise_abstr)
        # assemble macro trajectory out of initial data and rollout results
        best_macro_ss = torch.concat([plan_init['macro_s_next'], plan_abstr['macro_ss']], dim=0)

        actions = plan_init['as']
        for t in range(pln_n_abstract_steps):
            plan_detail = plan_section(model=mdl, planner=planner_prim, env=env, macro_s=best_macro_ss[t],
                                       macro_a=plan_abstr['macro_as'][t], macro_s_next=best_macro_ss[t+1],
                                       n_rollouts=pln_d_batch, n_evolution_steps=pln_n_optim_steps_prim,
                                       winning_perc=pln_winning_perc, discount=pln_discount,
                                       act_noise=pln_act_noise_prim)
            actions = torch.concat([actions, plan_detail['as']], dim=0)

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

        n_macro_steps = np.ceil(i_step  / mdl.macro_step_size).astype(np.int32)
        best_macro_terms = plan_abstr['macro_terms']
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
