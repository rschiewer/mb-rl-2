import os.path
import argparse

import gym.vector

from mdm.training.train import train_model, agent_train_mode, agent_eval_mode
from mdm.utils.utils import *
from mdm.training.offline_rl_driver import OfflineRLDriver, SamplingType
from mdm.logging.neptune_logger import NeptuneLogger
from mdm.logging.not_logger import NotLogger
from mdm.logging.logger import Scope
from mdm.training.gym_driver import collect_data, GymEpisodeDriver
from mdm.policies.agent_policy import *



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-log', default=False, action='store_true')
    args = parser.parse_args()

    cfg = load_yaml(here() / 'cfg_rssm_train.yaml')
    neptune_cfg = load_yaml(here() / cfg['neptune_cfg'])

    if args.log:
        logger = NeptuneLogger(neptune_cfg['PROJECT_NAME'], api_token=neptune_cfg['NEPTUNE_API_TOKEN'])
    else:
        logger = NotLogger()

    env = gym.make(cfg['env_name'])
    env = CacheLastStepEnv(env)

    def make_env_fn():
        return gym.make(cfg['env_name'])

    cfg = cfg_infer_missing_values(cfg, env)  # fill in missing config values
    logger.start_session()
    logger.log(cfg, Scope.HYPERPARAMETERS())  # log complete config
    cfg = build_rssms(cfg)  # generate RSSM cells and upwards filters
    r_max_agents, goal_seeking_agents = build_agents(cfg, env, 'cuda')

    model = HierarchicalRSSM(**cfg['mdm'], r_max_agents=r_max_agents, goal_seeking_agents=goal_seeking_agents)
    model = model.to('cuda')
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

    collect_env = gym.vector.AsyncVectorEnv([make_env_fn] * cfg['trainer']['collect_envs'])
    collect_env = CacheLastStepVecEnv(collect_env)
    eval_env = gym.vector.AsyncVectorEnv([make_env_fn] * cfg['eval']['eval_envs'])
    eval_env = CacheLastStepVecEnv(eval_env)

    train_mem = []
    match cfg['prefill_memory']:
        case 'offline-dataset':
            print('loading offline data to prefill training memory...')
            train_mem = load_memory(here() / cfg['train_samples'])
        case 'random':
            print('collecting initial random trajectories...')
            collect_driver = GymEpisodeDriver(collect_env, lambda *x: collect_env.action_space.sample())
            collect_driver.interact(cfg['prefill_episodes'], train_mem)
        case False:
            print('starting with empty training memory...')

    # fig, ani = visualize_trajectory(train_mem[0])
    # gif = anim_to_gif(ani)
    # plt.show()
    # fig = plot_trajectory_stats(train_mem, 20)
    # plt.show()

    train_driver = OfflineRLDriver(train_mem, sampling_type=SamplingType.RANDOM)
    test_mem = load_memory(here() / cfg['test_samples'])
    test_driver = OfflineRLDriver(test_mem, sampling_type=SamplingType.RANDOM)

    def simple_collect_fn():
        agent = r_max_agents[0][0]
        agent.eval()
        collect_env.reset()
        policy = LatentAgentPolicy(agent, model)
        collected_data_trajectories = collect_data(collect_env, 50, policy)
        train_mem.extend(collected_data_trajectories)

    def collect_fn():
        collect_env.reset()
        agent_eval_mode(r_max_agents + goal_seeking_agents)
        policy = HierarchicalLatentAgentPolicy(model)
        collected_data_trajectories = collect_data(collect_env, 50, policy)
        # visualize_trajectory(collected_data_trajectories[0])
        train_mem.extend(collected_data_trajectories)

    train_model(cfg, model, opt_model, r_max_agents, goal_seeking_agents, collect_fn, eval_env, test_driver,
                train_driver, logger)

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


if __name__ == '__main__':
    main()
