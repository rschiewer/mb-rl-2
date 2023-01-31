import os.path
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
from mdm.models.hierarchical_rssm import HierarchicalRSSM, StandardRSSM
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


def select_batch_items(mem: Dict[str, Union[torch.Tensor, torch.distributions.Distribution]],
                       i: TensorIndex,
                       keepdim: bool = False):
    if keepdim and type(i) is not slice:
        if isinstance(i, torch.Tensor) and i.size() == 1:
            i = i.detach().cpu().numpy().item()
            i = slice(i, i + 1)

    ret = {name: [] for name in mem}
    for name, trajectory_data in mem.items():
        for x_t in trajectory_data:
            if x_t is None:
                ret[name].append(None)
            elif isinstance(x_t, torch.Tensor):
                ret[name].append(x_t[i])
            elif isinstance(x_t, torch.distributions.Distribution):
                ret[name].append(extract_sub_distribution(x_t, i))  # keepdim is handled by calling function
            elif 'rnn_state' in name:
                ret[name].append(unpack_rnn_state(pack_rnn_state(x_t)[i]))
            else:
                raise ValueError(f'Unknown memory content for key {name}: {x_t}')
    return ret


def plan(model: DynamicsModel,
         level: int,
         planner: CrossentropyPlanner,
         env_data: Dict[str, torch.Tensor],
         n_plan_steps: int,
         n_rollouts: int,
         n_warmup: int):
    n_envs = env_data['o'].shape[1]

    # repeat the starting data n_rollouts times per environment, so use repeat_interleave instead of repeat
    o_start_batch = torch.repeat_interleave(env_data['o'], n_rollouts, dim=1)
    a_start_batch = torch.repeat_interleave(env_data['a'], n_rollouts, dim=1)
    r_start_batch = torch.repeat_interleave(env_data['r'], n_rollouts, dim=1)
    term_start_batch = torch.repeat_interleave(env_data['terminal'], n_rollouts, dim=1)

    def _rollout_fn(_a: torch.Tensor):
        # fold 'env' dimension into batch dimension for rollout
        _a = _a.reshape(_a.shape[0] * _a.shape[1], *_a.shape[2:])
        _a = _a.swapaxes(0, 1)  # swap batch and time dim since model is time-first but planner is batch-first
        _a = torch.cat([a_start_batch, _a], dim=0)
        _mem, _ = model(a=_a, o=o_start_batch, r=r_start_batch, terminal=term_start_batch,
                        n_warmup=n_warmup, level=level, sample_state=True, sample_output=True)
        _criterion = torch.stack(_mem['r']).squeeze(-1).swapaxes(0, 1)
        # _criterion -= torch.stack([d.scale / 2 for d in _mem['r_dist']]).squeeze(-1).swapaxes(0, 1)
        _discount = torch.stack(_mem['terminal']).squeeze(-1).swapaxes(0, 1)
        # _discount = torch.where(_discount > 0.75, 1.0, 0.0)

        _criterion = _criterion.reshape(n_envs, n_rollouts, _criterion.shape[-1])
        _discount = _discount.reshape(n_envs, n_rollouts, _discount.shape[-1])
        return _criterion, _discount, _mem

    a, a_dist, i_win, R_win, data = planner.plan(rollout_fn=_rollout_fn, n_rollouts=n_rollouts,
                                                 n_plan_steps=n_plan_steps, n_envs=n_envs)

    # select winner batch item per per memory timestep
    data = select_batch_items(data, i_win[:, 0], keepdim=True)
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
    R_win = torch.sum(step_rewards * disc_mat, dim=0)
    a_win = mem['a']
    return a_win, R_win, mem


def plan_hierarchical(model: HierarchicalRSSM,
                      env_data: Dict[str, torch.Tensor],
                      planners: List[CrossentropyPlanner],
                      n_plan_steps: List[int],
                      n_rollouts: List[int],
                      n_warmup: List[int]):
    n_groundtruth_steps, n_envs = env_data['o'].shape[:2]
    del env_data['truncated']
    del env_data['mask']

    min_n_warmup = [n_warmup[-1]]
    for level in reversed(range(model.levels - 1)):
        from_above = min_n_warmup[-1] * model.strides[level + 1]
        wu = max(from_above, n_warmup[level])
        min_n_warmup.append(wu)
    min_n_warmup.reverse()

    assert min_n_warmup[0] == n_groundtruth_steps

    # climb hierarchy and collect warmup data
    inp_lvl = env_data
    for level in range(model.levels):
        filters = model.upwards_filters[level]
        filtered_inp_level = {k: filters[k](inp_lvl[k]) for k in inp_lvl}

        plan_steps_lvl = min_n_warmup[level] - n_warmup[level]
        if plan_steps_lvl > 0:
            a_win, R_win, data = plan(model=model, level=level, planner=planners[level], env_data=filtered_inp_level,
                                      n_plan_steps=plan_steps_lvl, n_rollouts=n_rollouts[level],
                                      n_warmup=n_warmup[level])
        else:
            a_win, R_win, data = run_model(model=model, level=level, env_data=filtered_inp_level,
                                           discount=planners[level].discount)
        link = model.links[level]
        inp_lvl = {'o': torch.stack(data[link]), 'a': filtered_inp_level['a'], 'r': torch.stack(data['r']),
                   'terminal': torch.stack(data['terminal'])}

    # plan top level
    top_level = model.levels
    a_win, R_win, data = plan(model=model, level=top_level, planner=planners[top_level], env_data=filtered_inp_level,
                              n_plan_steps=n_plan_steps[top_level], n_rollouts=n_rollouts[top_level],
                              n_warmup=n_warmup[level])

    # plan top to bottom
    for level in reversed(range(model.levels)):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-log', default=False, action='store_true')
    args = parser.parse_args()

    cfg = load_yaml(here() / 'cfg_rssm_train.yaml')
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

    # generate objects
    for i_module, module_args in enumerate(cfg['mdm']['rssm_modules']):
        for k, v in module_args.items():  # generate encoders and decoder objects for current RSSM
            if isinstance(v, dict) and 'class' in v:
                cls_name = v.pop('class')
                instance = globals()[cls_name](**v)
                module_args[k] = instance
        cfg['mdm']['rssm_modules'][i_module] = StandardRSSM(**module_args)  # generate RSSM
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

    train_mem = load_memory(here() / cfg['train_samples'])
    train_driver = OfflineRLDriver(train_mem, sampling_type=SamplingType.RANDOM)
    test_mem = load_memory(here() / cfg['test_samples'])
    test_driver = OfflineRLDriver(test_mem, sampling_type=SamplingType.RANDOM)

    d_batch = cfg['trainer']['d_batch']

    planner_0 = CrossentropyPlanner(DistributionType.NORMAL, d_dist=2,
                                    device=model.device, debug_env=None, **planning_cfg['pln_prim'],
                                    a_min=-1.0, a_max=1.0)
    planner_1 = CrossentropyPlanner(DistributionType.NORMAL, d_dist=model.rssm_modules[1].d_a,
                                    device=model.device, debug_env=None, **planning_cfg['pln_prim'],
                                    a_min=-1.0, a_max=1.0)

    n_envs = cfg['trainer']['collect_envs']
    collect_env = gym.vector.AsyncVectorEnv([make_env_fn] * n_envs)
    collect_env = CacheLastStepVecEnv(collect_env)
    n_eval_envs = cfg['eval']['eval_envs']
    eval_env = gym.vector.AsyncVectorEnv([make_env_fn] * n_eval_envs)
    eval_env = CacheLastStepVecEnv(eval_env)


    def get_batch_train(i_step):
        # collect live data using current model for planning
        if i_step % cfg['trainer']['collect_interval'] == 0:
            model.eval()
            collect_env.reset()
            warmup_data_trajectories = collect_data(collect_env, planning_cfg['n_warmup_prim'],
                                                    RandomPolicy(collect_env))
            warmup_data = prepare_data(**to_tensors(warmup_data_trajectories, model.device))
            a_win, R_win, _ = plan(model, 0, planner_0, warmup_data, planning_cfg['n_plan_steps_prim'],
                                   planning_cfg['n_rollouts'], planning_cfg['n_warmup_prim'])
            a_win, R_win, _ = plan_hierarchical(model, warmup_data, [planner_0, planner_1],
                                                [planning_cfg['n_plan_steps_prim'], planning_cfg['n_plan_steps_abstr']],
                                                [planning_cfg['n_rollouts'], planning_cfg['n_rollouts']],
                                                [planning_cfg['n_warmup_prim'], planning_cfg['n_warmup_abstr']])
            collect_policy = PredefinedPolicy(collect_env, a_win.detach().cpu().numpy().swapaxes(0, 1))
            remaining_steps = planning_cfg['n_plan_steps_prim'] - planning_cfg['n_warmup_prim'] - 1
            collected_data_trajectories = collect_data(collect_env, remaining_steps, collect_policy)
            mem = [{k: np.concatenate([wu[k], col[k]]) for k in wu}
                   for wu, col in zip(warmup_data_trajectories, collected_data_trajectories)]
            train_mem.extend(mem)
            logger.log({'highest_planning_reward': R_win.mean().detach().cpu().numpy()}, Scope.TRAIN(), i_step)
            model.train()

        batch = train_driver.interact(d_batch)
        batch = to_tensors(batch, model.device)
        batch = prepare_data(**batch)
        return batch['o'], batch['a'], batch['r'], batch['terminal'], batch['truncated'], batch['mask']


    def get_batch_test(i_step):
        # collect_env.reset()
        # warmup_data_trajectories = collect_data(collect_env, n_warmup, RandomPolicy(collect_env))
        # warmup_data = prepare_data(**to_tensors(warmup_data_trajectories, model.device))
        # a_win, R_win, i_win = plan_with_warmup(model, planner_prim, warmup_data, n_plan_steps, n_rollouts, n_warmup)
        # collect_policy = PredefinedPolicy(collect_env, a_win.detach().cpu().numpy().swapaxes(0, 1))
        # collected_data_trajectories = collect_data(collect_env, n_plan_steps - n_warmup - 1, collect_policy)
        # mem = [{k: np.concatenate([wu[k], col[k]]) for k in wu}
        #       for wu, col in zip(warmup_data_trajectories, collected_data_trajectories)]
        # test_mem.extend(mem)

        batch = test_driver.interact(d_batch)
        batch = to_tensors(batch, model.device)
        batch = prepare_data(**batch)
        return batch['o'], batch['a'], batch['r'], batch['terminal'], batch['truncated'], batch['mask']


    model_path = f'{cfg["final_model_path"]}.ptmdl'

    # train
    fig = plt.figure(figsize=(10, 10))


    def eval_callback(o: torch.Tensor, a: torch.Tensor, r: torch.Tensor, term: torch.Tensor, mask, i_step: int):
        model.eval()
        predictions = []
        for i_lvl in range(len(model.rssm_modules)):
            pred, _ = model(o, a, r, term, cfg['eval']['warmup_steps'][i_lvl], sample_state=True, sample_output=True)
            predictions.append(pred)

        lvl_0_r_mean = torch.stack(predictions[0]['r']).mean(dim=1).squeeze().detach().cpu().numpy()
        lvl_0_r_std = torch.stack(predictions[0]['r']).std(dim=1).squeeze().detach().cpu().numpy()
        lvl_0_term_mean = torch.stack(predictions[0]['terminal']).mean(dim=1).squeeze().detach().cpu().numpy()
        lvl_0_term_std = torch.stack(predictions[0]['terminal']).std(dim=1).squeeze().detach().cpu().numpy()

        plt.plot(lvl_0_r_mean, label='mean')
        plt.plot(lvl_0_r_std, label='std')
        plt.legend()
        logger.log_plot(fig_to_img(fig), Scope.PARAMETERS() / 'model_stats/prim_r', i_step)
        plt.plot(lvl_0_term_mean, label='mean')
        plt.plot(lvl_0_term_std, label='std')
        plt.legend()
        logger.log_plot(fig_to_img(fig), Scope.PARAMETERS() / 'model_stats/prim_term', i_step)

        # losses = model.calc_loss(pred, o, r, term, mask)
        # losses = {k: v.detach().cpu().numpy() for k, v in losses.items()}
        # logger.log(losses, Scope.TEST() / 'with_warmup', i_step)

        eval_env.reset()
        warmup_data_trajectories = collect_data(eval_env, planning_cfg['n_warmup_prim'], RandomPolicy(eval_env))
        warmup_data = prepare_data(**to_tensors(warmup_data_trajectories, model.device))
        a_win, R_win, _ = plan(model, 0, planner_0, warmup_data, planning_cfg['n_plan_steps_prim'],
                               planning_cfg['n_rollouts'], planning_cfg['n_warmup_prim'])
        collect_policy = PredefinedPolicy(eval_env, a_win.detach().cpu().numpy().swapaxes(0, 1))
        remaining_steps = planning_cfg['n_plan_steps_prim'] - planning_cfg['n_warmup_prim'] - 1
        collected_data_trajectories = collect_data(eval_env, remaining_steps, collect_policy)
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
