import copy
import os.path
import random
import time
from pathlib import Path
import io
import argparse
import pickle

import gym.vector
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image

from mdm.utils.utils import *
from mdm.utils.torch_tools import to_tensors
from mdm.models.building_blocks import *
from mdm.models.hierarchical_rssm import HierarchicalRSSM
from mdm.models.rssm_cell import RSSMCell
from mdm.models.dynamics_model import DynamicsModel
from mdm.models.rnn_baseline import RnnBaselineModel
from mdm.training.dynamics_model_trainer import DynamicsModelTrainer
from mdm.planning.cem_planner import CrossentropyPlanner
from mdm.training.offline_rl_driver import OfflineRLDriver, SamplingType
from mdm.logging.neptune_logger import NeptuneLogger
from mdm.logging.not_logger import NotLogger
from mdm.logging.logger import Scope
from mdm.policies.random_policy import RandomPolicy
from mdm.policies.predefined_policy import PredefinedPolicy
from mdm.training.gym_driver import act_in_env, act_in_vector_env
from mdm.utils.gym_wrappers import CacheLastStepEnv, CacheLastStepVecEnv
from mdm.utils.torch_tools import extract_sub_distribution, pack_rnn_state, unpack_rnn_state, TensorIndex
from mdm.training.gym_driver import collect_data
from mdm.planning.planning_tools import plan_hierarchical

"""
def select_winners(mem: Dict[str, Union[torch.Tensor, torch.distributions.Distribution]],
                   i_win: torch.Tensor,
                   n_envs: int,
                   n_rollouts: int):
    assert i_win.ndim == 1

    offsets = torch.tensor([i * n_rollouts for i in range(n_envs)], dtype=torch.long, device=i_win.device)
    i_win_offs = i_win + offsets

    for k, v in mem.items():
        for t, x_t in enumerate(v):
            if isinstance(x_t, torch.Tensor):
                v[t] = x_t[i_win_offs]
            elif isinstance(x_t, torch.distributions.Distribution):
                v[t] = extract_sub_distribution(x_t, i_win_offs)
            elif 'rnn_state' in k:
                v[t] = unpack_rnn_state(pack_rnn_state(x_t)[i_win_offs])


def plan(model: HierarchicalRSSM,
         level: int,
         planner: CrossentropyPlanner,
         n_plan_steps: int,
         n_rollouts: int,
         n_warmup: int,
         env_data: Optional[Dict[str, torch.Tensor]] = None,
         model_state: Optional[Dict[str, torch.Tensor]] = None,
         goal_data: Optional[Dict[str, torch.Tensor]] = None):
    assert env_data is not None or model_state is not None, 'need at least warmup data or a model state'

    # repeat the starting data n_rollouts times per environment, so use repeat_interleave instead of repeat
    if env_data is None:
        assert model_state is not None
        assert n_warmup == 0
        n_envs = model_state['z'].shape[0]
        o_start_batch = torch.zeros(0, 0, 0, device=model.device)  # zero time, batch and data dim
        a_start_batch = torch.zeros(0, n_envs * n_rollouts, model.rssm_modules[level].d_a, device=model.device)
        r_start_batch = torch.zeros(0, 0, 0, device=model.device)
        term_start_batch = torch.zeros(0, 0, 0, device=model.device)
        z_rep = torch.repeat_interleave(model_state['z'], n_rollouts, dim=0)  # no time dimension here, so batch_dim=0
        rnn_rep = unpack_rnn_state(torch.repeat_interleave(pack_rnn_state(model_state['rnn_state']), n_rollouts, dim=0))
        model_state = {'z': z_rep, 'rnn_state': rnn_rep}
    else:
        assert model_state is None
        n_envs = env_data['o'].shape[1]
        o_start_batch = torch.repeat_interleave(env_data['o'], n_rollouts, dim=1)
        a_start_batch = torch.repeat_interleave(env_data['a'], n_rollouts, dim=1)
        r_start_batch = torch.repeat_interleave(env_data['r'], n_rollouts, dim=1)
        term_start_batch = torch.repeat_interleave(env_data['terminal'], n_rollouts, dim=1)

    if goal_data is None:
        def _calc_criterion(_mem):
            _criterion = torch.stack(_mem['r']).squeeze(-1).swapaxes(0, 1)
            # _criterion -= torch.stack([d.scale / 2 for d in _mem['r_dist']]).squeeze(-1).swapaxes(0, 1)
            _discount = torch.stack(_mem['terminal']).squeeze(-1).swapaxes(0, 1)
            # _discount = torch.where(_discount > 0.75, 1.0, 0.0)
            return _criterion, _discount
    else:
        goal_data = {
            k: torch.repeat_interleave(v, n_rollouts, dim=0)
            if v.ndim <= 2  # no time dim, so first is batch
            else torch.repeat_interleave(v, n_rollouts, dim=1)
            for k, v in goal_data.items()
        }

        def _calc_criterion(_mem):
            _criterion = torch.zeros(n_envs * n_rollouts, 1, device=model.device)
            _discount = torch.ones_like(_criterion)
            for k, v in goal_data.items():
                _criterion -= torch.mean((_mem[k][-1] - goal_data[k]) ** 2, dim=-1, keepdim=True)
            return _criterion, _discount

    def _rollout_fn(_a: torch.Tensor):
        # fold 'env' dimension into batch dimension for rollout
        _a = _a.reshape(_a.shape[0] * _a.shape[1], *_a.shape[2:])
        _a = _a.swapaxes(0, 1)  # swap batch and time dim since model is time-first but planner is batch-first
        _a = torch.cat([a_start_batch, _a], dim=0)
        _mem, _ = model(a=_a, o=o_start_batch, r=r_start_batch, terminal=term_start_batch,
                        n_warmup=n_warmup, level=level, start_state=model_state, sample_state=True, sample_output=True)
        _criterion, _discount = _calc_criterion(_mem)

        _criterion = _criterion.reshape(n_envs, n_rollouts, _criterion.shape[-1])
        _discount = _discount.reshape(n_envs, n_rollouts, _discount.shape[-1])
        return _criterion, _discount, _mem

    a, a_dist, i_win, R_win, data = planner.plan(rollout_fn=_rollout_fn, n_rollouts=n_rollouts,
                                                 n_plan_steps=n_plan_steps, n_envs=n_envs)

    # select winner batch item per per memory timestep
    select_winners(data, i_win[:, 0], n_envs, n_rollouts)
    # CAUTION: the actions from planner are without the already performed warmup acitons!
    a_win = planner.get_winner_actions(a, a_dist, i_win, resample=False)
    return a_win, R_win[:, 0], data


def run_model(model: DynamicsModel,
              level: int,
              env_data: Dict[str, torch.Tensor],
              discount: float = 1.0):
    mem, _ = model(**env_data, n_warmup=-1, level=level, sample_state=True, sample_output=True)
    step_rewards = torch.stack(mem['r'])
    disc_mat = torch.cumprod(torch.full_like(step_rewards, discount), dim=0)
    disc_mat = torch.roll(disc_mat, 1, dims=0)
    disc_mat[0, :, :] = 1
    R_win = torch.sum(step_rewards * disc_mat, dim=0).squeeze()
    a_win = env_data['a']
    return a_win, R_win, mem


def plan_hierarchical(model: HierarchicalRSSM,
                      env_data: Dict[str, torch.Tensor],
                      planners: List[CrossentropyPlanner],
                      n_plan_steps: int,
                      n_rollouts: List[int],
                      n_warmup: List[int]):
    n_groundtruth_steps, n_envs = env_data['o'].shape[:2]
    del env_data['truncated']
    del env_data['mask']

    min_init_steps = []
    for n_wu, stride in zip(n_warmup, model.strides):
        assert n_wu >= 1, 'every level needs at least one warmup step'
        min_init_steps.append(n_wu * stride)
    min_init_steps.append(n_warmup[-1])  # last hierarchy doesn't need to satisfy any requirements of above hierarchy
    assert n_groundtruth_steps >= min_init_steps.pop(0), 'not enough groundtruth data for lowest level'

    # min_init_steps now contains for every level the information how much model steps have to be taken in order to
    # satisfy the timestep requirements of the above hierarchy.

    # climb hierarchy and collect warmup data
    init_data = []
    inp_lvl = env_data
    for level in range(model.levels):
        filters = model.upwards_filters[level]
        filtered_inp_level = {k: filters[k](inp_lvl[k]) for k in inp_lvl}
        n_steps_available = filtered_inp_level['o'].shape[0]

        plan_steps_lvl = min_init_steps[level] - n_steps_available
        if plan_steps_lvl > 0:
            a_win, return_win, data = plan(model=model, level=level, planner=planners[level],
                                           n_plan_steps=plan_steps_lvl,
                                           n_rollouts=n_rollouts[level], n_warmup=n_warmup[level],
                                           env_data=filtered_inp_level)
        else:
            a_win, return_win, data = run_model(model=model, level=level, env_data=filtered_inp_level,
                                                discount=planners[level].discount)
        link = model.links[level]
        inp_lvl = {'o': torch.stack(data[link]), 'a': filtered_inp_level['a'], 'r': torch.stack(data['r']),
                   'terminal': torch.stack(data['terminal'])}
        # data['a'] = list(filtered_inp_level['a'].unbind(0))  # record actions as well; make list to match other buffers
        init_data.append(data)

    # plan top level to maximize reward
    i_top_lvl = model.levels - 1
    top_lvl_input_data = {k: torch.stack(init_data[i_top_lvl][k]) for k in ('o', 'a', 'r', 'terminal')}
    a_win_top, return_win_top, top_lvl_data = plan(model=model, level=i_top_lvl, planner=planners[i_top_lvl],
                                                   n_plan_steps=n_plan_steps, n_rollouts=n_rollouts[i_top_lvl],
                                                   n_warmup=n_warmup[i_top_lvl], env_data=top_lvl_input_data)

    # now plan top to bottom levels to maximize target similarity

    planning_data = [{}] * model.levels
    best_actions = [[]] * model.levels
    best_returns = [None] * model.levels

    # top level for all memories is already done, so fill it in
    planning_data[-1] = top_lvl_data
    best_actions[-1] = a_win_top  # 0 is env/batch dimension, 1 is time dimension, 2 is action dimension
    best_returns[-1] = return_win_top

    for level in reversed(range(model.levels - 1)):
        above_level = level + 1
        i_chunk_start = len(init_data[above_level]['o'])  # omit the warm up steps
        i_chunk_end = len(planning_data[above_level]['o'])
        n_plan_steps_level = model.strides[above_level]  # plan chunk by chunk
        a_win_level = []
        return_win_level = []

        # add init data portion of the trajectories
        for k, v in init_data[level].items():
            tmp = planning_data[level].get(k, [])
            tmp.extend(v)
            planning_data[level][k] = tmp

        # iterate through the chunks and do planning
        model_state = {'rnn_state': init_data[level]['rnn_state'][-1], 'z': init_data[level]['z'][-1]}
        for i_chunk in range(i_chunk_start, i_chunk_end):
            goal_data = {model.links[level]: planning_data[above_level]['o'][i_chunk]}
            a_win, return_win, data = plan(model=model, level=level, planner=planners[level],
                                           n_plan_steps=n_plan_steps_level, n_rollouts=n_rollouts[level],
                                           n_warmup=0, model_state=model_state,
                                           goal_data=goal_data)
            model_state = {'rnn_state': data['rnn_state'][-1], 'z': data['z'][-1]}

            # bookkeeping
            a_win_level.extend(list(a_win.unbind(1)))  # 0 is env dimension, 1 is time dimension, 2 is action dimension
            return_win_level.append(return_win)
            for k, v in data.items():
                tmp = planning_data[level].get(k, [])
                tmp.extend(v)
                planning_data[level][k] = tmp
        best_actions[level] = torch.stack(a_win_level, dim=1)
        best_returns[level] = torch.stack(return_win_level).sum(dim=0)

    best_lvl_0_actions = best_actions[0]
    return best_lvl_0_actions, best_returns, planning_data
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-log', default=False, action='store_true')
    args = parser.parse_args()

    cfg = load_yaml(here() / 'cfg_simple_rssm_train.yaml')
    planning_cfg = load_yaml(here() / 'cfg_rssm_plan.yaml')
    neptune_cfg = load_yaml(here() / cfg['neptune_cfg'])

    if args.log:
        logger = NeptuneLogger(neptune_cfg['PROJECT_NAME'], api_token=neptune_cfg['NEPTUNE_API_TOKEN'])
    else:
        logger = NotLogger()

    map_version = 'VeryEasy'
    env = gym.make(f'gym_nav2d:nav2d{map_version}-v0')
    env = CacheLastStepEnv(env)


    def make_env_fn():
        return gym.make(f'gym_nav2d:nav2d{map_version}-v0')


    # infer missing config values for RSSMs
    for i_module, module_args in enumerate(cfg['mdm']['rssm_modules']):
        d_state = module_args['d_z'] + module_args['d_h']
        if i_module == 0:
            module_args['d_a'] = env.action_space.shape[0]
            s_o = env.observation_space.shape
        else:
            if cfg['mdm']['links'][i_module - 1] == 'z':
                s_o = cfg['mdm']['rssm_modules'][i_module - 1]['d_z']
            elif cfg['mdm']['links'][i_module - 1] == 'h':
                s_o = cfg['mdm']['rssm_modules'][i_module - 1]['d_h']
            elif cfg['mdm']['links'][i_module - 1] == 's':
                s_o = cfg['mdm']['rssm_modules'][i_module - 1]['d_z'] + cfg['mdm']['rssm_modules'][i_module - 1]['d_h']
            elif cfg['mdm']['links'][i_module - 1] == 'o':
                s_o = cfg['mdm']['rssm_modules'][i_module - 1]['o_decoder']['s_x_orig']
            else:
                raise ValueError(f'Unknown link key: {cfg["mdm"]["links"][i_module - 1]}')

        module_args['o_encoder']['s_x_orig'] = s_o
        module_args['o_decoder']['s_x_orig'] = s_o
        module_args['o_decoder']['d_x_encoded'] = d_state
        module_args['r_decoder']['s_x_orig'] = 1
        module_args['r_decoder']['d_x_encoded'] = d_state
        module_args['term_decoder']['s_x_orig'] = 1
        module_args['term_decoder']['d_x_encoded'] = d_state

    for i_filter, filter_args in enumerate(cfg['mdm']['upwards_filters']):
        rssm = cfg['mdm']['rssm_modules'][i_filter]
        next_rssm = cfg['mdm']['rssm_modules'][i_filter + 1]
        filter_args['a']['d_x_orig'] = rssm['d_a']
        filter_args['a']['d_x_filtered'] = next_rssm['d_a']

    # log completed config
    logger.start_session()
    logger.log(cfg, Scope.HYPERPARAMETERS())
    logger.log(planning_cfg, Scope.HYPERPARAMETERS())

    # generate objects
    for i_module, module_args in enumerate(cfg['mdm']['rssm_modules']):
        for k, v in module_args.items():  # generate encoders and decoder objects for current RSSM
            if isinstance(v, dict) and 'class' in v:
                cls_name = v.pop('class')
                instance = globals()[cls_name](**v)
                module_args[k] = instance
        cfg['mdm']['rssm_modules'][i_module] = RSSMCell(**module_args)  # generate RSSM
    for i_filter, filter_args in enumerate(cfg['mdm']['upwards_filters']):  # generate filter objects
        for k, v in filter_args.items():
            cls_name = v.pop('class')
            instance = globals()[cls_name](**v)
            filter_args[k] = instance
    model = HierarchicalRSSM(**cfg['mdm']).to('cuda')
    model.training = True

    optim_type = cfg['optim'].pop('type')
    if optim_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), **cfg['optim'])
    elif optim_type == 'adamW':
        optimizer = torch.optim.AdamW(model.parameters(), **cfg['optim'])
    elif optim_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), **cfg['optim'])
    else:
        raise ValueError(f'Unknown optimizer type: {optim_type}')

    offline_mem = load_memory(here() / cfg['train_samples'])
    offline_driver = OfflineRLDriver(offline_mem, sampling_type=SamplingType.RANDOM)
    online_mem = []
    online_driver = OfflineRLDriver(online_mem, sampling_type=SamplingType.RANDOM)
    test_mem = load_memory(here() / cfg['test_samples'])
    test_driver = OfflineRLDriver(test_mem, sampling_type=SamplingType.RANDOM)

    d_batch = cfg['trainer']['d_batch']

    planners = [CrossentropyPlanner(DistributionType.NORMAL, d_dist=m.d_a, device=model.device, debug_env=None,
                                    **pln_cfg, a_min=-1.0, a_max=1.0)
                for pln_cfg, m in zip(planning_cfg['planners'], model.rssm_modules)]

    n_envs = cfg['trainer']['collect_envs']
    collect_env = gym.vector.AsyncVectorEnv([make_env_fn] * n_envs)
    collect_env = CacheLastStepVecEnv(collect_env)
    n_eval_envs = cfg['eval']['eval_envs']
    eval_env = gym.vector.AsyncVectorEnv([make_env_fn] * n_eval_envs)
    eval_env = CacheLastStepVecEnv(eval_env)

    n_wu_lvl_0 = planning_cfg['n_warmup'][0]
    n_wu_lvl_n = planning_cfg['n_warmup'][-1]


    def get_batch_train(i_step):
        if i_step % cfg['trainer']['collect_interval'] == 0:# and i_step > 0:
            collect_env.reset()
            #if random.random() > (i_step / cfg['trainer']['n_train_steps']):
            #    mem = collect_data(collect_env, -1, RandomPolicy(collect_env))
            #    current_factor = 1
            #    for i_level in range(1, model.levels):
            #        current_factor /= model.strides[i_level]
            #        for traj in mem:
            #            l = len(traj['o']) * current_factor
            #            rand_actions = (np.random.rand(l, model.rssm_modules[i_level].d_a) - 0.5) * 2
            #            traj[f'a_{i_level}'] = rand_actions
            #else:
            model.eval()
            warmup_data_trajectories = collect_data(collect_env, n_wu_lvl_0, RandomPolicy(collect_env))
            warmup_data = prepare_data(to_tensors(warmup_data_trajectories, model.device))
            a_win, return_levels, plan_data = plan_hierarchical(model, warmup_data, planners,
                                                                planning_cfg['n_plan_steps'] - n_wu_lvl_n,
                                                                planning_cfg['n_rollouts'],
                                                                planning_cfg['n_warmup'])
            collect_policy = PredefinedPolicy(collect_env, a_win.detach().cpu().numpy().swapaxes(0, 1))
            collected_data_trajectories = collect_data(collect_env, collect_policy.max_timestep, collect_policy)
            # on the environment level, we want to record the real data to the train memory
            mem = [{k: np.concatenate([wu[k], col[k]]) for k in wu}
                   for wu, col in zip(warmup_data_trajectories, collected_data_trajectories)]
            # for all other levels, only store the actions proposed by the planner
            for i_level, level in enumerate(plan_data[1:]):
                a_level = torch.stack(level['a']).detach().cpu().numpy()
                for i_traj, traj in enumerate(mem):
                    traj[f'a_{i_level + 1}'] = a_level[:, i_traj]
            for level, return_level in enumerate(return_levels):
                avg_score = return_level.mean().detach().cpu().numpy()
                logger.log({'top_planning_score': avg_score}, Scope.TRAIN() / f'planning/level_{level}', i_step)
            model.train()
            online_mem.extend(mem)

        if random.random() > 0.5 and len(online_mem) > 0:
            batch = online_driver.interact(d_batch)
            #if len(online_mem) == 4 * n_eval_envs:  # avoid old combinations of abstract and primitive actions in mem
            #    online_mem.clear()
        else:
            batch = offline_driver.interact(d_batch)
        batch = to_tensors(batch, model.device)
        batch = prepare_data(batch)
        #batch = subtrajectories(batch, 15)
        return batch


    def get_batch_test(i_step):
        batch = test_driver.interact(d_batch)
        batch = to_tensors(batch, model.device)
        batch = prepare_data(batch)
        return batch


    fig = plt.figure(figsize=(10, 10))


    def eval_callback(training_data, i_step: int):
        # record plots of reward/terminal predictions for expert dataset, we have an expectation how they should look
        model.eval()
        predictions = []
        for i_lvl in range(len(model.rssm_modules)):
            pred, _ = model(training_data['o'], training_data['a'], training_data['r'], training_data['terminal'],
                            cfg['eval']['warmup_steps'][i_lvl], sample_state=True, sample_output=True)
            predictions.append(pred)

            r_mean = torch.stack(pred['r']).mean(dim=1).squeeze().detach().cpu().numpy()
            r_std = torch.stack(pred['r']).std(dim=1).squeeze().detach().cpu().numpy()
            term_mean = torch.stack(pred['terminal']).mean(dim=1).squeeze().detach().cpu().numpy()
            term_std = torch.stack(pred['terminal']).std(dim=1).squeeze().detach().cpu().numpy()

            plt.plot(r_mean, label='mean')
            plt.plot(r_std, label='std')
            plt.legend()
            logger.log_plot(fig_to_img(fig), Scope.PARAMETERS() / f'model_stats/r_{i_lvl}', i_step)
            plt.plot(term_mean, label='mean')
            plt.plot(term_std, label='std')
            plt.legend()
            logger.log_plot(fig_to_img(fig), Scope.PARAMETERS() / f'model_stats/term_{i_lvl}', i_step)

        # do some planning and see how successfull the model is
        eval_env.reset()
        warmup_data_trajectories = collect_data(eval_env, n_wu_lvl_0, RandomPolicy(eval_env))
        warmup_data = prepare_data(to_tensors(warmup_data_trajectories, model.device))
        a_win, R_win, _ = plan_hierarchical(model, warmup_data, planners,
                                            planning_cfg['n_plan_steps'],
                                            planning_cfg['n_rollouts'],
                                            planning_cfg['n_warmup'])
        collect_policy = PredefinedPolicy(eval_env, a_win.detach().cpu().numpy().swapaxes(0, 1))
        collected_data_trajectories = collect_data(eval_env, collect_policy.max_timestep, collect_policy)
        mem = [{k: np.concatenate([wu[k], col[k]]) for k in wu}
               for wu, col in zip(warmup_data_trajectories, collected_data_trajectories)]

        ep_len = 0
        success = 0
        avg_return = 0
        for ep in mem:
            avg_return += np.stack(ep['r']).sum()
            if ep['terminal'].sum() == 1:
                ep_len += len(ep['terminal'])
                success += 1
            elif ep['terminal'].sum() > 1:
                raise RuntimeError('More than one terminal flag, there is something wrong!')
            else:
                ep_len += len(ep['terminal'])
        success /= n_eval_envs
        ep_len /= n_eval_envs
        avg_return /= n_eval_envs

        logger.log({'ep_len': ep_len, 'success': success, 'avg_return': avg_return},
                   Scope.PARAMETERS() / 'model_stats/eval_planning', i_step)
        model.train()


    trainer = DynamicsModelTrainer(model=model, optimizer=optimizer, get_batch_train=get_batch_train,
                                   get_batch_test=get_batch_test, logger=logger, eval_callback=eval_callback,
                                   **cfg['trainer'])
    trainer.train(n_train_steps=cfg['trainer']['n_train_steps'], progress_bar=True,
                  checkpoint_path=here() / cfg['checkpoint_path'])

    # store model and output run id
    p = here() / cfg['final_model_path'][:cfg['final_model_path'].rindex('/')]
    if not os.path.exists(p):
        os.makedirs(p)
    model_path = f'{cfg["final_model_path"]}_{logger.run_id}.ptmdl'
    torch.save(model, here() / model_path)
    logger.start_session()
    logger.log_file(here() / model_path, Scope.DATA() / 'final_weights')
    logger.stop_session()
    print(logger.run_id)
