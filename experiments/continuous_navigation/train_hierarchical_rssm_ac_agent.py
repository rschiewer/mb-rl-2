import copy
import os.path
import argparse

import gym.vector
from tqdm import tqdm

from mdm.utils.utils import *
from mdm.models.hierarchical_rssm import HierarchicalRSSM
from mdm.models.building_blocks import RSSMCell
from mdm.training.offline_rl_driver import OfflineRLDriver, SamplingType
from mdm.logging.neptune_logger import NeptuneLogger
from mdm.logging.not_logger import NotLogger
from mdm.logging.logger import Scope
from mdm.models.building_blocks import *
from mdm.training.gym_driver import collect_data
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

    env_name = 'gym_nav2d:nav2dVeryEasy-v0'
    env = gym.make(env_name)
    env = CacheLastStepEnv(env)


    def make_env_fn():
        return gym.make(env_name)


    cfg = cfg_infer_missing_values(cfg, env)  # fill in missing config values
    logger.start_session()
    logger.log(cfg, Scope.HYPERPARAMETERS())  # log complete config
    cfg = build_rssms(cfg)  # generate RSSM cells and upwards filters


    def gen_agent_fn(level: int, goal_seeking: bool) -> (
            ActorCriticAgent, torch.optim.Optimizer, torch.optim.Optimizer):
        mu = 0.01 if goal_seeking else 0.1
        beta = 0.02 if goal_seeking else 0.2
        agent = ActorCriticAgent(level=level, observation_key='z', d_a=cfg['mdm']['rssm_modules'][level].d_a,
                                 d_o=cfg['mdm']['rssm_modules'][level].d_z, min_a=(-1.0, -1.0), max_a=(1.0, 1.0),
                                 ema_coeff=0.95, trust_region_policy_update_beta=beta, eps_exploration=0.0,
                                 eps_exploration_mul=0.0, action_entropy_exploration=0.01,
                                 model_novelty_exploration=mu, use_ema_world_model=False, goal_seeking=goal_seeking)
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

    model = HierarchicalRSSM(**cfg['mdm'], r_max_agents=r_max_agents, goal_seeking_agents=goal_seeking_agents)
    model = model.to('cuda')
    model.training = True
    # model = torch.load(here() / 'trained_models/model_MBRL-2422.ptmdl').to('cuda')

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

    collect_env = gym.vector.AsyncVectorEnv([make_env_fn] * cfg['trainer']['collect_envs'])
    collect_env = CacheLastStepVecEnv(collect_env)
    eval_env = gym.vector.AsyncVectorEnv([make_env_fn] * cfg['eval']['eval_envs'])
    eval_env = CacheLastStepVecEnv(eval_env)


    def collect_simple():
        agent = r_max_agents[0][0]
        agent.eval()
        collect_env.reset()
        policy = LatentAgentPolicy(agent, model)
        collected_data_trajectories = collect_data(collect_env, 25, policy)
        mem.extend(collected_data_trajectories)


    def collect():
        collect_env.reset()
        policy = HierarchicalLatentAgentPolicy(model)
        collected_data_trajectories = collect_data(collect_env, 25, policy)
        mem.extend(collected_data_trajectories)


    # start training ---------------------------------------------------------------------------------------------------

    logger.start_session()
    model.prepare_for_training()
    for i_step in tqdm(range(cfg['trainer']['n_train_steps']), desc='Training Progress'):
        batch = train_driver.interact(cfg['trainer']['d_batch'])
        batch = to_tensors(batch, model.device)
        batch = prepare_data(batch)

        # train model
        model.train()
        model_batch = subtrajectories(batch, 15)
        train_losses = model.train_step(model_batch, opt_model)
        logger.log(_to_np(train_losses), Scope.TRAIN(), i_step)

        # train agent
        if i_step % cfg['trainer']['agent_train_interval'] == 0:
            for a in r_max_agents + goal_seeking_agents:
                if a is None: continue
                a[0].train()

            agent_batch = valid_subtrajectories(batch, 1)  # for agents avoid subtrajectories that contain padding
            # NOTE: lvl 0 needs warmup of at least 1 to make sure that agent sees first observation from environment
            warmup_steps = [1] + [1 for _ in range(model.levels - 1)]  # only lvl 0 warmup steps is relevant
            agent_steps = [20, 10, 5]  # arbitrary, test various values
            _, _, _, r_ag_mem, goal_ag_mem, _ = model.forward_all(agent_batch, warmup_steps, agent_steps,
                                                                  agent_training=True)
            if random.random() < 0.5:  # can only propagate through model once so decide which agent gets training
                for i_lvl, data_lvl in enumerate(r_ag_mem):
                    agent, opt_act, opt_crit = r_max_agents[i_lvl]
                    losses = agent.train_step(**data_lvl, actor_optimizer=opt_act, critic_optimizer=opt_crit)
                    losses['action_entropy'] = torch.stack([d.entropy() for d in data_lvl['a_dist']]).mean()
                    losses['obtained_reward'] = torch.stack(data_lvl['r']).mean()
                    logger.log(_to_np(losses), Scope.TRAIN() / f'r_max_agent/{i_lvl}/', i_step)
            else:
                for i_lvl, data_lvl in enumerate(goal_ag_mem):
                    agent, opt_act, opt_crit = goal_seeking_agents[i_lvl]
                    losses = agent.train_step(**data_lvl, actor_optimizer=opt_act, critic_optimizer=opt_crit)
                    losses['action_entropy'] = torch.stack([d.entropy() for d in data_lvl['a_dist']]).mean()
                    losses['obtained_reward'] = torch.stack(data_lvl['r']).mean()
                    logger.log(_to_np(losses), Scope.TRAIN() / f'goal_seeking_agent/{i_lvl}/', i_step)

        if i_step % cfg['trainer']['collect_interval'] == 0:
            # collect_simple()
            collect()

        # eval
        if cfg['trainer']['eval_interval'] is not None and i_step % cfg['trainer']['eval_interval'] == 0:
            for a in r_max_agents + goal_seeking_agents:
                if a is None: continue
                a[0].eval()
            model.eval()

            # model
            batch = test_driver.interact(cfg['trainer']['d_batch'])
            batch = to_tensors(batch, model.device)
            batch = prepare_data(batch)
            model_batch = subtrajectories(batch, 15)
            eval_losses = model.eval_step(model_batch)
            logger.log(_to_np(eval_losses), Scope.TEST(), i_step)
            # hierarchical agent
            eval_env.reset()
            policy = HierarchicalLatentAgentPolicy(model)
            eval_mem = collect_data(eval_env, 25, policy)
            logger.log(trajectory_statistics(eval_mem), Scope.TEST() / 'hierarchical_agent/', i_step)
            # flat agent
            eval_env.reset()
            policy = LatentAgentPolicy(r_max_agents[0][0], model)
            eval_mem = collect_data(eval_env, 25, policy)
            logger.log(trajectory_statistics(eval_mem), Scope.TEST() / 'flat_agent/', i_step)

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
