import argparse
from math import ceil

import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm


from mdm.gridworld.gridworld import Gridworld, FullyObservableGridworld
from mdm.models.multiscale_model_mk2 import MultiscaleDynamicsModelMK2
from mdm.planning.cem_planner import CrossentropyPlanner
from mdm.planning.gradient_planner import GradientPlanner
from mdm.utils.utils import (here, load_yaml, gen_video, DistributionType)
from mdm.utils.analysis_tools import gen_value_map_prim, visualize_plan, primitive_action_maps, infer_primitive_actions, \
    plot_plan
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
    #env = FullyObservableGridworld(env)
    mdl: MultiscaleDynamicsModelMK2 = torch.load(here() / model_path).to('cuda')
    mdl.eval()  # deactivate dropout in RNN

    if args.log:
        logger = NeptuneLogger(neptune_cfg['PROJECT_NAME'], api_token=neptune_cfg['NEPTUNE_API_TOKEN'],
                               run_id=args.id[0])
        logger.start_session()
    else:
        logger = NotLogger()

    planning_cfg['n_abstract_steps'] = ceil(50 / mdl.abstract_step_size) - planning_cfg['n_warmup_abstr']

    logger.log(planning_cfg, Scope.HYPERPARAMETERS() / 'plan')

    descriptions, canonical_abstr_a = primitive_action_maps(env, mdl)

    succeeded = 0
    action_stats = np.zeros(env.action_space.n)
    n_steps = []
    abstract_rewards = []
    abstract_terminals = []
    section_lengths = []
    for i_ep in tqdm(range(planning_cfg['n_episodes'])):
        if mdl.abstract_action_model.distribution_type in (None, 'normal'):
            abstr_dist_type = DistributionType.NORMAL
        else:
            abstr_dist_type = DistributionType.CATEGORICAL
        planner_abstr = CrossentropyPlanner(abstr_dist_type, d_dist=mdl.d_abstract_action, device=mdl.device,
                                            **planning_cfg['pln_abstr'])
        planner_prim = CrossentropyPlanner(DistributionType.CATEGORICAL, d_dist=mdl.d_action, device=mdl.device,
                                           **planning_cfg['pln_prim'])

        trajectory_history = mdl.gen_mem()
        trajectory_history = collect_groundtruth_data(mdl, trajectory_history, env, planning_cfg['n_warmup_prim'])
        trajectory_history = init_prim_s(mdl, trajectory_history)
        trajectory_history = init_abstr_s(mdl, trajectory_history, planner_prim, planning_cfg['n_warmup_abstr'],
                                          planning_cfg['n_rollouts'], allow_prim_imagination=True)
        trajectory_history = plan_abstract(mdl, trajectory_history, planner_abstr, planning_cfg['n_abstract_steps'],
                                           planning_cfg['n_rollouts'])

        plan_descr = infer_primitive_actions(trajectory_history, descriptions, canonical_abstr_a)
        logger.log({'most_prob_prim_a': plan_descr}, Scope.TEST(), i_ep)

        current_section_lengths = []
        for i_sec in range(planning_cfg['n_abstract_steps']):
            trajectory_history, sec_len = plan_section_flexible(mdl, trajectory_history, planner_prim, i_sec,
                                                       planning_cfg['n_rollouts'])
            current_section_lengths.append(sec_len)
        section_lengths.append(current_section_lengths)
        #visualize_plan(trajectory_history, env, mdl)

        actions = trajectory_history['prim_a'][0]
        actions = actions[planning_cfg['n_warmup_prim']:]  # remove the first default and the warmup actions

        abstract_rewards.append(trajectory_history['abstr_r'].detach().cpu().numpy())
        abstract_terminals.append(trajectory_history['abstr_term'].detach().cpu().numpy())

        #diffs = []
        #for i_step, abstr_a in enumerate(trajectory_history['abstr_a'][0].detach().cpu().numpy()):
        #    abstr_a = np.tile(abstr_a, (canonical_abstr_a.shape[0], 1))
        #    diff = np.mean((abstr_a - canonical_abstr_a) ** 2, axis=-1)
        #    diffs.append(diff)
        #diffs = np.stack(diffs)
        #plt.plot(diffs.min(axis=-1), linestyle='dashed', label='closest abstract a')
        #plt.plot(diffs.max(axis=-1), linestyle='dotted', label='most different abstract a')
        #plt.plot(diffs.std(axis=-1), label='std', alpha=0.5)
        #plt.legend()
        #plt.show()

        action_iter = iter(actions.detach().cpu().numpy().argmax(axis=-1))
        plan_checkpoints = (trajectory_history['abstr_o'][0].detach().cpu().numpy() + 0.5) * 7
        #plot_plan(env, trajectory_history, 20)
        i_step = planning_cfg['n_warmup_prim']
        terminal = False
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
    section_lengths = np.stack(section_lengths)

    # final debug output
    action_stats /= action_stats.sum()
    print(succeeded/planning_cfg['n_episodes'])
    print(n_steps, f' mean: {np.mean(n_steps)}')
    print(action_stats)
    print(section_lengths.mean(axis=0))
    print(section_lengths.std(axis=0))

    logger.log({'success_rate': succeeded/planning_cfg['n_episodes'],
                'n_steps': n_steps,
                'action_stats': action_stats,
                'abstract_reward_avg': abstract_rewards.mean(axis=0),
                'abstract_reward_std': abstract_rewards.std(axis=0),
                'abstract_terminal_avg': abstract_terminals.mean(axis=0),
                'abstract_terminal_std': abstract_terminals.std(axis=0),
                'section_lengths_avg': section_lengths.mean(axis=0),
                'section_lengths_std': section_lengths.std(axis=0)},
               Scope.TEST())
    logger.stop_session()
