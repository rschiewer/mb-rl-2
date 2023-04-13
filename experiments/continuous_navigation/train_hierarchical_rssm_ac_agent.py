import copy
import os.path
import argparse

import gym.vector
from tqdm import tqdm

from mdm.utils.utils import *
from mdm.models.hierarchical_rssm import HierarchicalRSSM, RSSMCell
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


    def gen_agent_fn(level: int, goal_seeking: bool) -> (
    ActorCriticAgent, torch.optim.Optimizer, torch.optim.Optimizer):
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

    model = HierarchicalRSSM(**cfg['mdm'], r_max_agents=r_max_agents, goal_seeking_agents=goal_seeking_agents).to(
        'cuda')
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


    def collect_simple():
        agent = r_max_agents[0][0]
        agent.eval()
        collect_env.reset()
        policy = LatentAgentPolicy(agent, model)
        collected_data_trajectories = collect_data(collect_env, 25, policy)
        mem.extend(collected_data_trajectories)
        #avg_score = np.mean([traj['r'].mean() for traj in collected_data_trajectories])
        #logger.log({'average_collected_reward': avg_score}, Scope.TRAIN() / 'agent/', i_step)


    def collect():
        """
        1.  Observe first env data time step from env initialization
        2.  From lower to higher level:
        3.    Use current level r_max agent to make decision and collect new time step from env
        4.    If not enough time steps to escalate to next level, GOTO 3
        5.  Use r_max agent on highest level to make decision
        6.  From higher to lower level:
        7.    While above level goals are available:
        8.      Use current level goal_seeking agent to find proposed goal from above
        9.      Store achieved model latent states in every step
        10.   Filter out every k-th step as goals for lower level
        11. Execute lowest level actions in real world
        """


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
        """
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
        """

        # do some planning and see how successfull the model is
        agent = r_max_agents[0][0]
        agent.eval()
        eval_env.reset()
        policy = LatentAgentPolicy(agent, model)
        # policy = AgentPolicy(agent)
        eval_mem = collect_data(eval_env, 25, policy)
        ep_len = 0
        success = 0
        avg_return = 0
        for ep in eval_mem:
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
                   Scope.TEST() / 'flat_agent/', i_step)
        model.train()


    # start training ---------------------------------------------------------------------------------------------------

    logger.start_session()
    model.prepare_for_training()
    for i_step in tqdm(range(cfg['trainer']['n_train_steps']), desc='Training Progress'):
        batch = get_batch_train(i_step)

        # train model
        model.train()
        model_batch = subtrajectories(batch, 15)
        # train_agents = i_step % cfg['trainer']['agent_train_interval'] == 0
        # train_losses = model.train_step(model_batch, opt_model, train_agents=train_agents)
        train_losses = model.train_step(model_batch, opt_model)
        logger.log(_to_np(train_losses), Scope.TRAIN(), i_step)

        # train agent
        if i_step % cfg['trainer']['agent_train_interval'] == 0:
            agent_batch = valid_subtrajectories(batch, 1)  # for agents avoid subtrajectories that contain padding
            # NOTE: lvl 0 needs warmup of 1 to make sure that the agent sees the first observation from the environment
            warmup_steps = [1] + [1 for _ in range(model.levels - 1)]  # only lvl 0 warmup steps is relevant
            agent_steps = [20, 10, 5]  # arbitrary, test various values
            _, _, _, r_ag_mem, goal_ag_mem, _ = model.forward_all(agent_batch, warmup_steps, agent_steps)
            if random.random() < 0.5:  # can only propagate through model once so decide which agent gets training
                for i_lvl, data_lvl in enumerate(r_ag_mem):
                    agent, opt_act, opt_crit = r_max_agents[i_lvl]
                    losses = agent.train_step(**data_lvl, actor_optimizer=opt_act, critic_optimizer=opt_crit)
                    logger.log(_to_np(losses), Scope.TRAIN() / f'r_max_agent/{i_lvl}/', i_step)
            else:
                for i_lvl, data_lvl in enumerate(goal_ag_mem):
                    agent, opt_act, opt_crit = goal_seeking_agents[i_lvl]
                    losses = agent.train_step(**data_lvl, actor_optimizer=opt_act, critic_optimizer=opt_crit)
                    logger.log(_to_np(losses), Scope.TRAIN() / f'goal_seeking_agent/{i_lvl}/', i_step)

        if i_step % cfg['trainer']['collect_interval'] == 0:
            collect_simple()

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
