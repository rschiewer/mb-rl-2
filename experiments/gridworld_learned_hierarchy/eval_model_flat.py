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
from mdm.utils.torch_tools import unpack_rnn_state
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

    logger.log(planning_cfg, Scope.HYPERPARAMETERS() / 'plan/flat')

    succeeded = 0
    action_stats = np.zeros(env.action_space.n)
    n_steps = []
    for i_ep in tqdm(range(planning_cfg['n_episodes'])):
        planner_prim = CrossentropyPlanner(DistributionType.CATEGORICAL, d_dist=mdl.d_action, device=mdl.device,
                                           **planning_cfg['pln_prim'])

        trajectory_history = mdl.gen_mem()
        trajectory_history = collect_groundtruth_data(mdl, trajectory_history, env, planning_cfg['n_warmup_prim'])
        trajectory_history = init_prim_s(mdl, trajectory_history)
        rnn_state = unpack_rnn_state(trajectory_history['prim_rnn_state'][:, -1])
        z = trajectory_history['prim_z'][:, -1]
        a, i_win = plan_prim_free(mdl, planner_prim, rnn_state, z, planning_cfg['n_plan_steps_prim'],
                                  planning_cfg['n_rollouts'])

        action_iter = iter(a.detach().cpu().numpy())

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

    # final debug output
    action_stats /= action_stats.sum()
    print(succeeded/planning_cfg['n_episodes'])
    print(n_steps, f' mean: {np.mean(n_steps)}')
    print(action_stats)

    logger.log({'success_rate': succeeded / planning_cfg['n_episodes'],
                'n_steps': n_steps,
                'action_stats': action_stats},
               Scope.TEST() / 'plan/flat')
    logger.stop_session()
