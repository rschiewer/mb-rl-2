import argparse
from math import ceil
import time

import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm


from mdm.gridworld.gridworld import Gridworld
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
    planning_cfg = load_yaml(here() / 'planning_cfg.yaml')
    neptune_cfg = load_yaml(here() / cfg['neptune_cfg'])

    if len(args.id) == 0:
        model_path = f'{cfg["final_model_path"]}.ptmdl'
    else:
        model_path = f'{cfg["final_model_path"]}_{args.id[0]}.ptmdl'

    env = Gridworld.from_cleartext(here() / '../../mdm/gridworld/8x8_v0.mapdata')
    mdl: MultiscaleDynamicsModelMK2 = torch.load(here() / model_path).to('cuda')
    mdl.eval()  # deactivate dropout in RNN

    if args.log:
        logger = NeptuneLogger(neptune_cfg['PROJECT_NAME'], api_token=neptune_cfg['NEPTUNE_API_TOKEN'],
                               run_id=args.id[0])
        logger.start_session()
    else:
        logger = NotLogger()

    perfect_actions = [0] + [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1]  # first one is default zero action

    trajectory_history = mdl.gen_mem()
    n_warmup = planning_cfg['n_warmup_prim']
    warmup_actions = perfect_actions[1:n_warmup]  # omit default zero action here
    trajectory_history = collect_groundtruth_data(mdl, trajectory_history, env, n_warmup, warmup_actions)
    trajectory_history = init_prim_s(mdl, trajectory_history)

    z = trajectory_history['prim_z'][-1]
    rnn_state = trajectory_history['prim_rnn_state'][-1]
    perfect_actions_torch = torch.tensor(perfect_actions[n_warmup:], dtype=torch.float32)
    perfect_actions_torch = perfect_actions_torch.to(mdl.device).unsqueeze(1)
    perfect_actions_torch = to_onehot(perfect_actions_torch, env.action_space.n)
    a_binned = bin_every_k_steps(perfect_actions_torch, mdl.abstract_step_size)
    #abstr_a = torch.stack([mdl.abstract_action_model(a_binned[i_chunk], sample=False)
    #                       for i_chunk in range(a_binned.shape[1])], dim=1)
    mem, prim_final = mdl.rollout_primitive(perfect_actions_torch, z=z, rnn_state=rnn_state, sample=False)
    mem = mdl.pack_mem(mem)
    #z = trajectory_history['prim_z'][:, -1]
    #rnn_state = unpack_rnn_state(trajectory_history['prim_rnn_state'][:, -1])
    #print(z)
    #print(rnn_state)

    rewards = []
    rollout_rewards = mem["prim_r"].detach().cpu().numpy().squeeze()
    for a in perfect_actions[1:]:
        o, r, term, _ = env.step(a)
        rewards.append(r)
    print(f'Rollout reward: {rollout_rewards.sum()}, real reward: {np.sum(rewards)}')
    print(rewards)
    print(rollout_rewards)
    quit()


    trajectory_history = init_abstr_s(mdl, trajectory_history, None, 0, 0)
    assert(torch.sum(abstr_a[:, 0] - trajectory_history['abstr_a'][:, 0]) < 0.0001)
    abstr_z_start = trajectory_history['abstr_z'][:, -1]
    abstr_rnn_state_start = unpack_rnn_state(trajectory_history['abstr_rnn_state'][:, -1])
    mem, abstr_final = mdl.rollout_abstract(abstr_a, z=abstr_z_start, rnn_state=abstr_rnn_state_start, sample=False,
                                                n_posterior_steps=0)
    mem = mdl.pack_mem(mem)
    trajectory_history['abstr_a'] = []
    trajectory_history = update_history(trajectory_history, mem)

    current_section_lengths = []
    for i_sec in range(abstr_a.shape[1]):
        trajectory_history, sec_len = plan_section_flexible(mdl, trajectory_history, None, i_sec, 0)
        current_section_lengths.append(sec_len)

    actions = trajectory_history['prim_a'][0].argmax(dim=-1)
    actions = perfect_actions[:3] + actions[3:].detach().cpu().numpy().tolist()
    print(perfect_actions)
    print(actions)
    print(trajectory_history['abstr_r'])

