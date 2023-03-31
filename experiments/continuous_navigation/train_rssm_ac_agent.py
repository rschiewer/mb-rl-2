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
from mdm.planning.planning_tools import plan_hierarchical
from mdm.policies.actor_critic_agent import ActorCriticAgent
from mdm.policies.agent_policy import *


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

    agent_lvl = 0
    agent = ActorCriticAgent(level=agent_lvl, link='z', s_a=(model.rssm_modules[agent_lvl].d_a,),
                             s_o=(model.rssm_modules[agent_lvl].d_z,), min_a=(-1.0, -1.0), max_a=(1.0, 1.0), eps=0.0,
                             eps_mul=0.999, ema_reg=True, entropy_exploration=False,
                             model_novelty_exploration=True, use_ema_world_model=False)
    agent = agent.to('cuda')
    actor_optimizer = torch.optim.Adam(agent.actor_net.parameters(), lr=0.001)
    critic_optimizer = torch.optim.Adam(agent.critic_net.parameters(), lr=0.01)

    def get_batch_train(i_step):
        if i_step % cfg['trainer']['collect_interval'] == 0:
            agent.eval()
            collect_env.reset()
            policy = LatentAgentPolicy(agent, model)
            #policy = AgentPolicy(agent)
            collected_data_trajectories = collect_data(collect_env, 25, policy)
            #online_mem.extend(collected_data_trajectories)
            offline_mem.extend(collected_data_trajectories)
            avg_score = np.mean([traj['r'].mean() for traj in collected_data_trajectories])
            logger.log({'top_planning_score': avg_score}, Scope.TRAIN() / f'planning/level_0', i_step)

        #if random.random() > 0.3 and len(online_mem) > 0:
        #    batch = online_driver.interact(d_batch)
        #else:
        #    batch = offline_driver.interact(d_batch)

        batch = offline_driver.interact(d_batch)
        batch = to_tensors(batch, model.device)
        batch = prepare_data(batch)
        if i_step % cfg['trainer']['agent_train_interval'] == 0:
            agent.train()
            init_data_agent = {k: v[:n_wu_lvl_0] for k, v in batch.items()}
            agent_loss = agent.sim_train_step(sim_env=model, init_data=init_data_agent, n_steps=25,
                                              actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer)
            agent.update_exploration()
            logger.log(agent_loss, Scope.TRAIN() / 'agent')
        #batch = valid_subtrajectories(batch, 15)
        batch = subtrajectories(batch, 15)
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
        agent.eval()
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
        policy = LatentAgentPolicy(agent, model)
        #policy = AgentPolicy(agent)
        mem = collect_data(eval_env, 25, policy)
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

        logger.log({'ep_len': ep_len, 'success': success, 'avg_return': avg_return, 'exploration': agent.eps},
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
