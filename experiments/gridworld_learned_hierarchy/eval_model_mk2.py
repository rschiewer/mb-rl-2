import argparse
from math import ceil

import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm


from mdm.gridworld.gridworld import Gridworld
from mdm.models.multiscale_model_mk2 import MultiscaleDynamicsModelMK2
from mdm.planning.cem_planner import CrossentropyPlanner, DistributionType
from mdm.utils.utils import (here, load_yaml, gen_macro_state_map, infer_position, transform_macro_s_init_history,
                             gen_video, gen_prim_state_map)
from mdm.utils.planning_tools_mk2 import *
from mdm.utils.torch_tools import add_time_dim
from mdm.memory.trajectory_memory import TrajectoryMemory
from mdm.logging.neptune_logger import NeptuneLogger
from mdm.logging.not_logger import NotLogger
from mdm.logging.logger import Scope


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Provide neptune run_id for loading the correct model')
    parser.add_argument('id', type=str, nargs=1)
    parser.add_argument('-render', action='store_true')
    parser.add_argument('-log', action='store_true')
    args = parser.parse_args()

    cfg = load_yaml(here() / 'model_cfg.yaml')
    neptune_cfg = load_yaml(here() / cfg['neptune_cfg'])
    planning_cfg = load_yaml(here() / 'planning_cfg.yaml')

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
        logger = NotLogger()

    """
    trajectory_history = {'o': [], 'r': [], 'term': [], 'prim_o': [], 'prim_o_dist': [], 'prim_r': [],
                          'prim_r_dist': [], 'prim_term': [], 'prim_s': [], 'prim_s_dist': [], 'prim_rnn_state': [],
                          'abstr_o': [], 'abstr_o_dist': [], 'abstr_r': [], 'abstr_r_dist': [], 'abstr_term': [],
                          'abstr_s': [], 'abstr_s_dist': [], 'abstr_rnn_state': []}
    """

    planning_cfg['n_abstract_steps'] = ceil(100 / mdl.abstract_step_size) - planning_cfg['n_warmup_abstr']
    logger.log(planning_cfg, Scope.HYPERPARAMETERS() / 'plan')

    #macro_s_init_mean, macro_s_init_std, macro_s_init_per_state_per_action = gen_macro_state_map(env, mdl, 3)
    #macro_ss_list, loc_list, act_seq_list = transform_macro_s_init_history(macro_s_init_per_state_per_action)
    #macro_s_lookup = np.stack([v for k, v in macro_s_init_mean.items()])
    #positions_lookup = np.stack([k for k, v in macro_s_init_mean.items()])

    mem = TrajectoryMemory()
    succeeded = 0
    n_steps = []
    action_stats = np.zeros(env.action_space.n)
    abstract_rewards = []
    abstract_terminals = []
    for i_ep in tqdm(range(planning_cfg['n_episodes'])):
        planner_prim = CrossentropyPlanner(DistributionType.CATEGORICAL, device=mdl.device)
        planner_abstr = CrossentropyPlanner(DistributionType.NORMAL, device=mdl.device)

        #ret = init_model(mdl, env, mdl.n_warmup_prim, mdl.n_warmup_abstr, planner_prim, planning_cfg['n_rollouts'],
        #                 planning_cfg['n_optim_steps_prim'], planning_cfg['winning_perc'],
        #                 planning_cfg['discount'], planning_cfg['act_noise_prim'])

        trajectory_history = init_prim_s(mdl, env, planning_cfg['n_warmup_prim'])
        trajectory_history = init_abstr_s(mdl, trajectory_history, planning_cfg['n_warmup_abstr'], planner_prim,
                                          planning_cfg['n_rollouts'],
                                          planning_cfg['n_optim_steps_prim'], planning_cfg['winning_perc'],
                                          planning_cfg['discount'],
                                          planning_cfg['act_noise_prim'])
        #print(f'After prim init: {trajectory_history["prim_o"].shape[1]}')
        trajectory_history = plan_abstract(mdl, trajectory_history, planner_abstr, planning_cfg['n_abstract_steps'],
                                           planning_cfg['n_rollouts'],
                                           planning_cfg['n_optim_steps_abstr'], planning_cfg['winning_perc'],
                                           planning_cfg['discount'],
                                           planning_cfg['act_noise_abstr'])
        #print(f'After abstr init: {trajectory_history["prim_o"].shape[1]}')
        for i_sec in range(planning_cfg['n_abstract_steps']):
            trajectory_history = plan_section(mdl, trajectory_history, planner_prim,
                                              planning_cfg['n_rollouts'],
                                              planning_cfg['n_optim_steps_prim'], planning_cfg['winning_perc'],
                                              planning_cfg['discount'],
                                              planning_cfg['act_noise_prim'])
            #print(f'After section {i_sec}: {trajectory_history["prim_o"].shape[1]}')
        actions = trajectory_history['prim_a'][0]
        quit()

        # plan_init_prim = init_s_prim(mdl, env, planning_cfg['n_warmup_prim'], trajectory_history)
        #
        # plan_init_abstr = init_s_abstr(model=mdl,
        #                                planner=planner_prim,
        #                                env=env,
        #                                s_start=plan_init_prim['s'],
        #                                rnn_state_start=plan_init_prim['rnn_state'],
        #                                n_rollouts=planning_cfg['n_rollouts'],
        #                                n_evolution_steps=planning_cfg['n_optim_steps_abstr'],
        #                                winning_perc=planning_cfg['winning_perc'],
        #                                discount=planning_cfg['discount'],
        #                                act_noise=planning_cfg['act_noise_abstr'])
        #
        # plan_abstr = plan_abstract(model=mdl,
        #                            planner=planner_abstr,
        #                            abstr_s_start=plan_init_abstr['abstr_s'],
        #                            abstr_rnn_state_start=plan_init_abstr['abstr_rnn_state'],
        #                            abstr_r_start=plan_init_abstr['abstr_r'],
        #                            abstr_term_start=plan_init_abstr['abstr_term'],
        #                            n_plan_steps=planning_cfg['n_abstract_steps'],
        #                            n_rollouts=planning_cfg['n_rollouts'],
        #                            n_evolution_steps=planning_cfg['n_optim_steps_abstr'],
        #                            winning_perc=planning_cfg['winning_perc'],
        #                            discount=planning_cfg['discount'],
        #                            act_noise=planning_cfg['act_noise_abstr'])
        #
        # # assemble macro trajectory out of initial data and rollout results
        # best_abstr_s = torch.concat([add_time_dim(plan_init_abstr['abstr_s']), plan_abstr['abstr_s']], dim=0)
        # best_abstr_rnn_state = [plan_init_abstr['abstr_rnn_state']] + plan_abstr['abstr_rnn_state']
        # best_abstr_r = torch.concat([add_time_dim(plan_init_abstr['abstr_r']), plan_abstr['abstr_r']], dim=0)
        # best_abstr_term = torch.concat([add_time_dim(plan_init_abstr['abstr_term']), plan_abstr['abstr_term']], dim=0)
        # best_abstr_o = torch.concat([add_time_dim(plan_init_abstr['abstr_o']), plan_abstr['abstr_o']], dim=0)
        #
        # abstract_rewards.append(best_abstr_r.detach().cpu().numpy().squeeze())
        # abstract_terminals.append(best_abstr_term.detach().cpu().numpy().squeeze())

        # TODO: move this to eval callback
        # test if abstract actions actually make a difference w.r.t. the output
        #d_batch_test = 128
        #rand_abstr_a = torch.rand(d_batch_test, 1, mdl.abstract_model.d_action, device=mdl.device)
        #abstr_s_start = broadcast_to_batch(best_abstr_s[0], d_batch_test)
        #abstr_rnn_state_start = broadcast_rnn_state_to_batch(best_abstr_rnn_state[0], mdl.abstract_model.rnn_type,
        #                                                     d_batch_test)
        #abstr_r_start = broadcast_to_batch(best_abstr_r[0], d_batch_test)
        #abstr_r_start = add_time_dim(abstr_r_start)
        #abstr_term_start = broadcast_to_batch(best_abstr_term[0], d_batch_test)
        #abstr_term_start = add_time_dim(abstr_term_start)
        #mem_, abstr_final_ = mdl.rollout_abstract(a=rand_abstr_a, r=abstr_r_start, term=abstr_term_start,
        #                                          s=abstr_s_start, rnn_state=abstr_rnn_state_start,
        #                                          mem=None, sample=False)
        #final_s = abstr_final_['s'].detach().cpu().numpy()
        #logger.log({'randomized_action_abstract_s_avg': final_s.mean(),
        #            'randomized_action_abstract_s_std': final_s.std()}, Scope.TEST())

        # actions = plan_init_abstr['prim_a']
        # prim_s = plan_init_abstr['prim_s']
        # prim_rnn_state = plan_init_abstr['prim_rnn_state']
        # init_prim_o = plan_init_abstr['prim_o']
        # init_prim_r = plan_init_abstr['prim_r']
        # init_prim_term = plan_init_abstr['prim_term']
        # for t in range(planning_cfg['n_abstract_steps']):
        #     #prim_s, prim_rnn_state = mdl.unfuse_state(best_abstr_o[t])  # use abstract model predictions for state inits
        #     plan_detail = plan_section(model=mdl, planner=planner_prim, env=env,
        #                                init_prim_o=init_prim_o, init_prim_r=init_prim_r,
        #                                init_prim_term=init_prim_term, prim_s=prim_s, prim_rnn_state=prim_rnn_state,
        #                                subtraj_hist_target=best_abstr_o[t + 1],
        #                                abstr_s=best_abstr_s[t], abstr_rnn_state=best_abstr_rnn_state[t],
        #                                abstr_s_next=best_abstr_s[t + 1],
        #                                abstr_r=best_abstr_r[t], abstr_term=best_abstr_term[t],
        #                                n_rollouts=planning_cfg['n_rollouts'], n_evolution_steps=planning_cfg['n_optim_steps_prim'],
        #                                winning_perc=planning_cfg['winning_perc'], act_noise=planning_cfg['act_noise_prim'])
        #     actions = torch.concat([actions, plan_detail['prim_a']], dim=0)
        #     init_prim_o = plan_detail['prim_o']
        #     init_prim_r = plan_detail['prim_r']
        #     init_prim_term = plan_detail['prim_term']
        #     prim_s = plan_detail['prim_s']
        #     prim_rnn_state = plan_detail['prim_rnn_state']

        action_iter = iter(actions.detach().cpu().numpy().argmax(axis=-1))
        terminal = False

        i_step = planning_cfg['n_warmup_prim']
        while not terminal:
            if args.render:
                env.render()
            i_step += 1
            try:
                a = next(action_iter)
                s_, r, terminal, info = env.step(a)

                action_stats[a] += 1
                if terminal and r > 0:
                    succeeded += 1
            except StopIteration:
                break

        n_steps.append(i_step)

        n_macro_steps = np.ceil(i_step  / mdl.abstract_step_size).astype(np.int32)
        #best_macro_terms = plan_abstr['abstr_term']
        #plot_mats = infer_position(env, best_macro_ss[:n_macro_steps], best_macro_terms[:n_macro_steps], macro_ss_list, loc_list, act_seq_list)
        #ani = gen_video(plot_mats, 1000, 2000)
        #ani.save(f'animation_{i_ep}.mp4')

    abstract_rewards = np.stack(abstract_rewards)
    abstract_terminals = np.stack(abstract_terminals)

    # final debug output
    action_stats /= action_stats.sum()
    print(succeeded/planning_cfg['n_episodes'])
    print(n_steps, f' mean: {np.mean(n_steps)}')
    print(action_stats)

    logger.log({'success_rate': succeeded/planning_cfg['n_episodes'],
                'n_steps': n_steps,
                'action_stats': action_stats,
                'abstract_reward_avg': abstract_rewards.mean(axis=0),
                'abstract_reward_std': abstract_rewards.std(axis=0),
                'abstract_terminal_avg': abstract_terminals.mean(axis=0),
                'abstract_terminal_std': abstract_terminals.std(axis=0)}, Scope.TEST())
    logger.stop_session()
