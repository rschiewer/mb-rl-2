import os.path
import argparse
from warnings import simplefilter

import gym.vector
import neptune

from mdm.training.train import train_model, agent_eval_mode, build_rssms, build_agents, build_model_opt
from mdm.utils.utils import *
from mdm.training.offline_rl_driver import OfflineRLDriver, SamplingType
from mdm.logging.neptune_logger import NeptuneLogger
from mdm.logging.not_logger import NotLogger
from mdm.logging.logger import Scope, GlobalLogger
from mdm.training.gym_driver import collect_data, GymEpisodeDriver
from mdm.policies.agent_policy import *


def main():
    simplefilter(action='ignore', category=DeprecationWarning)  # numpy deprecation warning from outdated gym lib
    parser = argparse.ArgumentParser()
    parser.add_argument('-log', default=False, action='store_true')
    parser.add_argument('-d_batch', type=int)
    parser.add_argument('-n_collect', type=int)
    args = parser.parse_args()

    cfg = load_yaml(here() / 'cfg_rssm_train.yaml')
    neptune_cfg = load_yaml(here() / cfg['neptune_cfg'])

    if args.log:
        logger = NeptuneLogger(**neptune_cfg)
    else:
        logger = NotLogger()

    # for debugging
    GlobalLogger.bind(logger, {'_mask_model': 50,
                               '_mask_latent_overshooting': 50,
                               '_mask_agent': 50,
                               'simulated_ground_truth_goal_distance': 50,
                               '_sanity_check_goal_computation': 50})

    if args.d_batch:
        cfg['trainer']['d_batch'] = args.d_batch
    if args.n_collect:
        cfg['prefill_episodes'] = args.n_collect

    env = gym.make(cfg['env_name'])
    env = CacheLastStepEnv(env)

    def make_env_fn():
        return gym.make(cfg['env_name'])

    cfg = cfg_infer_missing_values(cfg, env)  # fill in missing config values
    logger.start_session()
    logger.log(cfg, Scope.HYPERPARAMETERS())  # log complete config

    cfg = build_rssms(cfg)
    r_max_agents, goal_seeking_agents = build_agents(cfg, env, 'cuda')
    model = HierarchicalRSSM(**cfg['mdm'], r_max_agents=r_max_agents, goal_seeking_agents=goal_seeking_agents)

    # load model if necessary
    if cfg['pretrained_model'] is None:
        print('Starting training from scratch')
    else:
        print(f'Using pretrained model {cfg["pretrained_model"]}')
        mdl_path = here() / Path(f'trained_models/model_{cfg["pretrained_model"]}.ptmdl')
        if not mdl_path.exists():
            tmp_run = neptune.init_run(**neptune_cfg, with_id=cfg['pretrained_model'])
            tmp_run[f'{Scope.DATA()}/weights/final_weights'].download(str(mdl_path))
            tmp_run.stop()
        pretrained_model = torch.load(here() / mdl_path)
        copy_params(pretrained_model, model)
    model = model.to('cuda')

    opt_model = build_model_opt(model, cfg)

    # temporary hack to only train parts of the model

    #params = chain.from_iterable([model.rssm_modules[0].r_decoder.parameters(),
    #                              model.rssm_modules[0].term_decoder.parameters(),
    #                              model.rssm_modules[1].r_decoder.parameters(),
    #                              model.rssm_modules[1].term_decoder.parameters()])
    #opt_model = torch.optim.Adam(params, **cfg['optim'])

    # temporary hack end

    collect_env = gym.vector.AsyncVectorEnv([make_env_fn] * cfg['trainer']['collect_envs'])
    collect_env = CacheLastStepVecEnv(collect_env)
    eval_env = gym.vector.AsyncVectorEnv([make_env_fn] * cfg['eval']['eval_envs'])
    eval_env = CacheLastStepVecEnv(eval_env)

    train_mem = []
    match cfg['prefill_memory']:
        case 'offline-dataset':
            print('loading offline data to prefill training memory...', flush=True)
            train_mem = load_memory(here() / cfg['train_samples'])
        case 'random':
            print('collecting initial random trajectories...', flush=True)
            collect_driver = GymEpisodeDriver(collect_env, lambda *x: collect_env.action_space.sample())
            collect_driver.interact(cfg['prefill_episodes'], train_mem)
        case False:
            print('starting with empty training memory...', flush=True)

    #fig, ani = visualize_trajectory(train_mem[0])
    #gif = anim_to_gif(ani)
    #plt.show()
    #fig = plot_trajectory_stats(train_mem, 20)
    #plt.show()

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

    #with torch.autograd.detect_anomaly(check_nan=True):
    train_model(cfg, model, opt_model, r_max_agents, goal_seeking_agents, collect_fn, eval_env, test_driver,
                train_driver, logger)

    # store model and output run id
    p = here() / cfg['final_model_path'][:cfg['final_model_path'].rindex('/')]
    if not os.path.exists(p):
        os.makedirs(p)
    model_path = here() / f'{cfg["final_model_path"]}_{logger.run_id}.ptmdl'
    torch.save(model, model_path)
    model_weights = InMemoryFile(model_path, name='final_weights')
    logger.start_session()
    logger.log_file(model_weights, Scope.DATA() / 'weights')
    logger.stop_session()
    print(logger.run_id)


if __name__ == '__main__':
    main()
