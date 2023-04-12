import copy
import os.path
import random
import time
from pathlib import Path
import io
import argparse
import pickle
from typing import Any

import gym.vector
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from mdm.utils.utils import *
from mdm.utils.torch_tools import to_tensors
from mdm.models.building_blocks import *
from mdm.models.hierarchical_rssm import HierarchicalRSSM, RSSMCell
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


def _to_np(data_dict: Dict[str, Union[torch.Tensor, Dict]]):
    np_data_dict = {}
    for k, v in data_dict.items():
        if isinstance(v, dict):
            np_data_dict[k] = _to_np(v)
        elif isinstance(v, torch.Tensor):
            np_data_dict[k] = v.detach().cpu().numpy()
        else:
            raise ValueError(f'Unsupported type: {type(k)}')
    return np_data_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-log', default=False, action='store_true')
    args = parser.parse_args()

    cfg = load_yaml(here() / 'cfg_rssm_train.yaml')
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
        cfg['mdm']['rssm_modules'][i_module] = RSSMCell(**module_args)  # generate RSSM
    for i_filter, filter_args in enumerate(cfg['mdm']['upwards_filters']):  # generate filter objects
        for k, v in filter_args.items():
            cls_name = v.pop('class')
            instance = globals()[cls_name](**v)
            filter_args[k] = instance

    def gen_agent_fn(level: int, goal_seeking: bool) -> (ActorCriticAgent, torch.optim.Optimizer, torch.optim.Optimizer):
        agent = ActorCriticAgent(level=level, observation_key='z', d_a=cfg['mdm']['rssm_modules'][level].d_a,
                                 d_o=cfg['mdm']['rssm_modules'][level].d_z, min_a=(-1.0, -1.0), max_a=(1.0, 1.0),
                                 ema_coeff=0.99, trust_region_policy_update_beta=0.5, eps_exploration=0.0,
                                 eps_exploration_mul=0.0, action_entropy_exploration=0.00,
                                 model_novelty_exploration=0.1, use_ema_world_model=False, goal_seeking=goal_seeking)
        agent = agent.to('cuda')
        actor_optimizer = torch.optim.Adam(agent.actor_net.parameters(), lr=0.001)
        critic_optimizer = torch.optim.Adam(agent.critic_net.parameters(), lr=0.01)
        return agent, actor_optimizer, critic_optimizer

    r_max_agents = []
    goal_seeking_agents = []
    for agent_lvl in range(len(cfg['mdm']['rssm_modules'])):
        r_max_agents.append(gen_agent_fn(agent_lvl, False))
        goal_seeking_agents.append(gen_agent_fn(agent_lvl, True))
    goal_seeking_agents[-1] = None  # no homing agent needed on last level


    model = HierarchicalRSSM(**cfg['mdm'], r_max_agents=r_max_agents, goal_seeking_agents=goal_seeking_agents).to('cuda')
    model.training = True

    optim_type = cfg['optim'].pop('type')
    if optim_type == 'adam':
        opt_model = torch.optim.Adam(model.parameters(), **cfg['optim'])
    elif optim_type == 'adamW':
        opt_model = torch.optim.AdamW(model.parameters(), **cfg['optim'])
    elif optim_type == 'sgd':
        opt_model = torch.optim.SGD(model.parameters(), **cfg['optim'])
    else:
        raise ValueError(f'Unknown optimizer type: {optim_type}')

    mem = load_memory(here() / cfg['train_samples'])
    train_driver = OfflineRLDriver(mem, sampling_type=SamplingType.RANDOM)
    test_mem = load_memory(here() / cfg['test_samples'])
    test_driver = OfflineRLDriver(test_mem, sampling_type=SamplingType.RANDOM)

    d_batch = cfg['trainer']['d_batch']
    n_envs = cfg['trainer']['collect_envs']
    collect_env = gym.vector.AsyncVectorEnv([make_env_fn] * n_envs)
    collect_env = CacheLastStepVecEnv(collect_env)
    n_eval_envs = cfg['eval']['eval_envs']
    eval_env = gym.vector.AsyncVectorEnv([make_env_fn] * n_eval_envs)
    eval_env = CacheLastStepVecEnv(eval_env)


    def collect_wrapper():
        agent = r_max_agents[0][0]
        agent.eval()
        collect_env.reset()
        policy = LatentAgentPolicy(agent, model)
        collected_data_trajectories = collect_data(collect_env, 25, policy)
        mem.extend(collected_data_trajectories)
        avg_score = np.mean([traj['r'].mean() for traj in collected_data_trajectories])
        logger.log({'top_planning_score': avg_score}, Scope.TRAIN() / f'planning/level_0', i_step)


    def get_batch_train(i_step):
        batch = train_driver.interact(d_batch)
        batch = to_tensors(batch, model.device)
        batch = prepare_data(batch)

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
        # policy = AgentPolicy(agent)
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

    # start training ---------------------------------------------------------------------------------------------------

    logger.start_session()
    model.prepare_for_training()
    for i_step in tqdm(range(cfg['trainer']['n_train_steps']), desc='Training Progress'):
        if i_step % cfg['trainer']['collect_interval'] == 0:
            collect_wrapper()
        batch = get_batch_train(i_step)

        # train model
        model.train()
        #model_batch = valid_subtrajectories(batch, 15)
        model_batch = subtrajectories(batch, 15)
        train_agents = i_step % cfg['trainer']['agent_train_interval'] == 0
        train_losses = model.train_step(model_batch, opt_model, train_agents=train_agents)
        logger.log(_to_np(train_losses), Scope.TRAIN(), i_step)

        # train agents
        continue
        if i_step % cfg['trainer']['agent_train_interval'] == 0:

            # r_max agents
            for agent_lvl in range(model.levels):
                agent, act_opt, crit_opt = r_max_agents[agent_lvl]
                n_wu = cfg['trainer']['agent_world_model_warmup'][agent_lvl]
                n_t = cfg['trainer']['agent_sim_steps'][agent_lvl]

                agent.train()
                init_data_agent = {k: v[:n_wu] for k, v in batch.items()}
                interact_data = agent.act_in_sim(init_data=init_data_agent, n_steps=n_t, sim_env=model)
                agent_loss = agent.sim_train_step(**interact_data, actor_optimizer=act_opt, critic_optimizer=crit_opt)
                agent.update_exploration()
                logger.log(agent_loss, Scope.TRAIN() / f'agent_{agent_lvl}')

                # take subtrajectory out of interact_data and train homing agent on that part
                if agent_lvl < model.levels - 1:
                    agent, act_opt, crit_opt = goal_seeking_agents[agent_lvl]

            # homing agents
            for agent_lvl in range(model.levels - 1):
                agent = r_max_agents[agent_lvl + 1]

                agent, act_opt, crit_opt = goal_seeking_agents[agent_lvl]
                n_wu = cfg['trainer']['agent_world_model_warmup'][agent_lvl]
                n_t = cfg['trainer']['agent_sim_steps'][agent_lvl]

                agent.train()
                init_data_agent = {k: v[:n_wu] for k, v in batch.items()}
                agent_loss = agent.sim_train_step(sim_env=model, init_data=init_data_agent, n_steps=n_t,
                                                  actor_optimizer=act_opt, critic_optimizer=crit_opt)
                agent.update_exploration()
                logger.log(agent_loss, Scope.TRAIN() / f'agent_{agent_lvl}')
        continue
        # eval
        if cfg['trainer']['eval_interval'] is not None and i_step % cfg['trainer']['eval_interval'] == 0:
            batch = get_batch_test(i_step)
            model.eval()
            eval_losses = model.eval_step(batch)
            logger.log(_to_np(eval_losses), Scope.TEST(), i_step)
            eval_callback(batch, i_step)

    # training done ----------------------------------------------------------------------------------------------------

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
