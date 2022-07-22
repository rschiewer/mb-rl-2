import argparse
from math import ceil

import torch
import numpy as np
from tqdm import tqdm


from mdm.gridworld.gridworld import Gridworld
from mdm.models.multiscale_model_mk2 import MultiscaleDynamicsModelMK2
from mdm.planning.cem_planner import CrossentropyPlanner, DistributionType
from mdm.utils.utils import (here, load_yaml, gen_macro_state_map, infer_position, transform_macro_s_init_history,
                             gen_video)
from mdm.utils.planning_tools_mk2 import init_s_abstr, plan_section, plan_abstract
from mdm.utils.torch_tools import add_time_dim
from mdm.memory.trajectory_memory import TrajectoryMemory
from mdm.logging.neptune_logger import NeptuneLogger
from mdm.logging.logger import Scope


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Provide neptune run_id for loading the correct model')
    parser.add_argument('id', type=str, nargs=1)
    parser.add_argument('-render', action='store_true')
    parser.add_argument('-log', action='store_true')
    args = parser.parse_args()

    cfg = load_yaml(here() / 'model_mk2.yaml')
    neptune_cfg = load_yaml(here() / cfg['neptune_cfg'])

    if len(args.id) == 0:
        model_path = f'{cfg["final_model_path"]}.ptmdl'
    else:
        model_path = f'{cfg["final_model_path"]}_{args.id[0]}.ptmdl'

    env = Gridworld.from_cleartext(here() / '../../mdm/gridworld/8x8_v0.mapdata')
    mdl: MultiscaleDynamicsModelMK2 = torch.load(here() / model_path)
    mdl = mdl.to('cuda')

    if args.log:
        logger = NeptuneLogger(neptune_cfg['PROJECT_NAME'], api_token=neptune_cfg['NEPTUNE_API_TOKEN'],
                               run_id=args.id[0])
        logger.start_session()
    else:
        logger = None

    n_episodes = 10
    store_result_trajectories = False
    pln_d_batch = 2048
    pln_n_optim_steps_abstr = 30
    pln_n_optim_steps_prim = 20
    pln_winning_perc = 0.2
    pln_discount = 0.90
    pln_act_noise_abstr = 0.001
    pln_act_noise_prim = 0.001
    pln_n_abstract_steps = ceil(100 / mdl.abstract_step_size)

    if logger:
        logger.log({'pln_d_batch': pln_d_batch,
                    'pln_n_optim_steps_abstr': pln_n_optim_steps_abstr,
                    'pln_n_optim_steps_prim': pln_n_optim_steps_prim,
                    'pln_winning_perc': pln_winning_perc,
                    'pln_discount': pln_discount,
                    'pln_act_noise_abstr': pln_act_noise_abstr,
                    'pln_act_noise_prim': pln_act_noise_prim,
                    'pln_n_abstract_steps': pln_n_abstract_steps}, Scope.HYPERPARAMETERS() / 'plan')

    #macro_s_init_mean, macro_s_init_std, macro_s_init_per_state_per_action = gen_macro_state_map(env, mdl, 3)
    #macro_ss_list, loc_list, act_seq_list = transform_macro_s_init_history(macro_s_init_per_state_per_action)
    #macro_s_lookup = np.stack([v for k, v in macro_s_init_mean.items()])
    #positions_lookup = np.stack([k for k, v in macro_s_init_mean.items()])

    mem = TrajectoryMemory()
    succeeded = 0
    n_steps = []
    action_stats = np.zeros(env.action_space.n)
    for i_ep in tqdm(range(n_episodes)):
        planner_prim = CrossentropyPlanner(DistributionType.CATEGORICAL, device=mdl.device)
        planner_abstr = CrossentropyPlanner(DistributionType.NORMAL, device=mdl.device)
        o_mem, a_mem, r_mem, term_mem = [], [], [], []

        o_start = env.reset()
        o_mem.append(o_start)

        o_start = torch.from_numpy(o_start).to(mdl.device)
        plan_init = init_s_abstr(model=mdl, planner=planner_prim, env=env, o_start=o_start,
                                 n_rollouts=pln_d_batch, n_evolution_steps=pln_n_optim_steps_abstr,
                                 winning_perc=pln_winning_perc, discount=pln_discount,
                                 act_noise=pln_act_noise_abstr)

        plan_abstr = plan_abstract(model=mdl, planner=planner_abstr, abstr_s_start=plan_init['abstr_s'],
                                   abstr_rnn_state_start=plan_init['abstr_rnn_state'],
                                   abstr_r_start=plan_init['abstr_r'], abstr_term_start=plan_init['abstr_term'],
                                   n_plan_steps=pln_n_abstract_steps, n_rollouts=pln_d_batch,
                                   n_evolution_steps=pln_n_optim_steps_abstr, winning_perc=pln_winning_perc,
                                   discount=pln_discount, act_noise=pln_act_noise_abstr)

        # assemble macro trajectory out of initial data and rollout results
        best_abstr_s = torch.concat([add_time_dim(plan_init['abstr_s']), plan_abstr['abstr_s']], dim=0)
        best_abstr_rnn_state = [plan_init['abstr_rnn_state']] + plan_abstr['abstr_rnn_state']
        best_abstr_r = torch.concat([add_time_dim(plan_init['abstr_r']), plan_abstr['abstr_r']], dim=0)
        best_abstr_term = torch.concat([add_time_dim(plan_init['abstr_term']), plan_abstr['abstr_term']], dim=0)
        best_abstr_o = torch.concat([add_time_dim(plan_init['abstr_o']), plan_abstr['abstr_o']], dim=0)

        actions = plan_init['prim_a']
        prim_s = plan_init['prim_s']
        prim_rnn_state = plan_init['prim_rnn_state']
        init_prim_o = plan_init['prim_o']
        init_prim_r = plan_init['prim_r']
        init_prim_term = plan_init['prim_term']
        for t in range(pln_n_abstract_steps):
            #prim_s, prim_rnn_state = mdl.unfuse_state(best_abstr_o[t])  # use abstract model predictions for state inits
            plan_detail = plan_section(model=mdl, planner=planner_prim, env=env,
                                       init_prim_o=init_prim_o, init_prim_r=init_prim_r,
                                       init_prim_term=init_prim_term, prim_s=prim_s, prim_rnn_state=prim_rnn_state,
                                       subtraj_hist_target=best_abstr_o[t + 1],
                                       abstr_s=best_abstr_s[t], abstr_rnn_state=best_abstr_rnn_state[t],
                                       abstr_s_next=best_abstr_s[t + 1],
                                       abstr_r=best_abstr_r[t], abstr_term=best_abstr_term[t],
                                       n_rollouts=pln_d_batch, n_evolution_steps=pln_n_optim_steps_prim,
                                       winning_perc=pln_winning_perc, act_noise=pln_act_noise_prim)
            actions = torch.concat([actions, plan_detail['prim_a']], dim=0)
            init_prim_o = plan_detail['prim_o']
            init_prim_r = plan_detail['prim_r']
            init_prim_term = plan_detail['prim_term']
            prim_s = plan_detail['prim_s']
            prim_rnn_state = plan_detail['prim_rnn_state']

        action_iter = iter(actions.detach().cpu().numpy().argmax(axis=-1))
        terminal = False

        i_step = 0
        while not terminal:
            if args.render:
                env.render()
            i_step += 1
            try:
                a = next(action_iter)
                s_, r, terminal, info = env.step(a)

                o_mem.append(s_)
                a_mem.append(a)
                r_mem.append(r)
                term_mem.append(terminal)

                action_stats[a] += 1
                if terminal and r > 0:
                    succeeded += 1
            except StopIteration:
                break

        n_steps.append(i_step)
        mem.push(o_mem, a_mem, r_mem, term_mem)

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

    if logger:
        logger.log({'success_rate': succeeded/n_episodes,
                    'n_steps': n_steps,
                    'action_stats': action_stats}, Scope.TEST())
        logger.teardown()
