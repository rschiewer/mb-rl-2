import os.path
import argparse
from warnings import simplefilter

import neptune

from mdm.training.train import train_model
from mdm.utils.build_models import build_model_opt, build_rssms, build_agents, cfg_infer_missing_values
from mdm.utils.utils import *
from mdm.training.offline_rl_driver import OfflineRLDriver, SamplingType
from mdm.logging.neptune_logger import NeptuneLogger
from mdm.logging.not_logger import NotLogger
from mdm.logging.logger import Scope, GlobalLogger
from mdm.policies.agent_policy import *
from mdm.policies.expert_policies import *
from mdm.utils.gym_wrappers import vec_env_worker_no_auto_reset


# from tqdm import tqdm
# tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


def main():
    simplefilter(action='ignore', category=DeprecationWarning)  # numpy deprecation warning from outdated gym lib
    parser = argparse.ArgumentParser()
    parser.add_argument('-log', default=False, action='store_true')
    parser.add_argument('-d_batch', type=int)
    parser.add_argument('-n_collect', type=int)
    args = parser.parse_args()

    cfg = load_yaml(here() / 'cfg_rssm_train.yaml')
    neptune_cfg = load_yaml(here() / cfg['neptune_cfg'])

    if args.d_batch:
        cfg['trainer']['d_batch'] = args.d_batch
    if args.n_collect:
        cfg['prefill_episodes'] = args.n_collect
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

    def make_env_fn():
        _env = gym.make(cfg['env_name'])
        _env = gym.wrappers.RescaleAction(_env, min_action=-1.0, max_action=1.0)
        # if isinstance(_env.observation_space, gym.spaces.dict.Dict):
        #    _env = gym.wrappers.FlattenObservation(_env)
        return _env

    env = make_env_fn()
    cfg = cfg_infer_missing_values(cfg, env)  # fill in missing config values
    logger.start_session()
    logger.log(cfg, Scope.HYPERPARAMETERS())  # log complete config

    cfg = build_rssms(cfg)
    r_max_agents, goal_seeking_agents = build_agents(cfg, env, torch.device('cuda'))

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

    # model = torch.compile(model, disable=disable_torch_compile)

    # temporary hack to only train parts of the model
    # params = chain.from_iterable([model.rssm_modules[0].r_decoder.parameters(),
    #                              model.rssm_modules[0].term_decoder.parameters(),
    #                              model.rssm_modules[1].r_decoder.parameters(),
    #                              model.rssm_modules[1].term_decoder.parameters()])
    # opt_model = torch.optim.Adam(params, **cfg['optim'])
    # temporary hack end

    collect_env = gym.vector.AsyncVectorEnv([make_env_fn] * cfg['trainer']['collect_envs'],
                                            worker=vec_env_worker_no_auto_reset)
    collect_env = CacheLastStepVecEnv(collect_env)
    #collect_env = CacheLastStepEnv(make_env_fn())
    eval_env = gym.vector.AsyncVectorEnv([make_env_fn] * cfg['eval']['eval_envs'],
                                         worker=vec_env_worker_no_auto_reset)
    eval_env = CacheLastStepVecEnv(eval_env)
    #eval_env = CacheLastStepEnv(make_env_fn())
    video_env = gym.make(cfg['env_name'], render_mode='rgb_array')
    video_env = gym.wrappers.RescaleAction(video_env, min_action=-1.0, max_action=1.0)

    #def make_video_env_fn():
    #    _env = gym.make(cfg['env_name'], render_mode='rgb_array')
    #    _env = gym.wrappers.RescaleAction(_env, min_action=-1.0, max_action=1.0)
    #    return _env
    #video_env = gym.vector.AsyncVectorEnv([make_video_env_fn] * cfg['eval']['eval_envs'])
    video_env = gym.wrappers.RecordVideo(video_env, video_folder='videos', name_prefix=f'{os.getpid()}',
                                         disable_logger=True)
    video_env = CacheLastStepEnv(video_env)

    train_mem = []
    match cfg['prefill_memory']:
        case 'offline-dataset':
            print('loading offline data to prefill training memory...', flush=True)
            train_mem = load_memory(here() / cfg['train_samples'])
        case 'random':
            print('collecting initial random trajectories...', flush=True)
            collect_driver = GymEpisodeDriver(collect_env, lambda *x: collect_env.action_space.sample())
            collect_driver.interact(cfg['prefill_episodes'], train_mem, progress_bar=True)
        case False:
            print('starting with empty training memory...', flush=True)
    train_driver = OfflineRLDriver(train_mem, sampling_type=SamplingType.RANDOM)

    test_mem = []
    if cfg['test_samples']:
        test_mem = load_memory(here() / cfg['test_samples'])
    else:
        policy = get_expert_policy(cfg['env_name'], fallback_policy=lambda *x: collect_env.action_space.sample())
        collect_driver = GymEpisodeDriver(collect_env, policy)
        collect_driver.interact(100, test_mem, progress_bar=True)
    test_driver = OfflineRLDriver(test_mem, sampling_type=SamplingType.RANDOM)

    # fig, ani = visualize_trajectory(train_mem[0])
    # gif = anim_to_gif(ani)
    # plt.show()
    # fig = plot_trajectory_stats(train_mem, 20)
    # plt.show()

    def simple_collect_fn(explore: bool):
        agent = r_max_agents[0][0]
        agent.eval()
        collect_env.reset()
        policy = LatentAgentPolicy(agent, model, explore=explore)
        d = GymEpisodeDriver(collect_env, policy)
        d.interact(10, train_mem)
        #collected_data_trajectories = collect_data(collect_env, -1, policy)
        #train_mem.extend(collected_data_trajectories)

    def collect_fn(explore: bool):
        agent = r_max_agents[0][0]
        agent.eval()
        collect_env.reset()
        policy = LatentAgentPolicy(agent, model, explore=explore)
        #collected_data_trajectories = collect_data(collect_env, -1, policy)
        #train_mem.extend(collected_data_trajectories)
        #train_mem.extend(collected_data_trajectories)
        d = GymEpisodeDriver(collect_env, policy)
        d.interact(10, train_mem)

        #collect_env.reset()
        #agent_eval_mode(r_max_agents + goal_seeking_agents)
        #policy = HierarchicalLatentAgentPolicy(model, explore=explore)
        #collected_data_trajectories = collect_data(collect_env, -1, policy)
        #train_mem.extend(collected_data_trajectories)
        # visualize_trajectory(collected_data_trajectories[0])

        """
        n_plots = model.levels + 1
        with TempFigure(figsize=(5 * n_plots, 6)) as fig:
            for l, flight_record_l in enumerate(policy.flight_record):
                states_lvl = torch.stack(flight_record_l['z_post'])[:, :, :, 0]
                time_steps_lvl = torch.stack(flight_record_l['time_step'])
                time_steps_lvl = 1 - (time_steps_lvl / time_steps_lvl.max())
                d_time, d_batch = states_lvl.shape[:2]

                states_lvl = states_lvl.reshape(d_time * d_batch, -1).detach().cpu().numpy()
                # time_steps_lvl = time_steps_lvl.reshape(d_time * d_batch, -1).detach().cpu().numpy()
                time_steps_lvl = time_steps_lvl.detach().cpu().numpy()

                pca = PCA(n_components=3)
                states_trans = pca.fit_transform(states_lvl)
                states_trans = states_trans.reshape((d_time, d_batch, -1))
                colors = np.concatenate([time_steps_lvl, np.zeros((d_time, d_batch, 2))], axis=-1)

                ax = fig.add_subplot(100 + n_plots * 10 + (l + 1), projection='3d')
                ax.set_title(f'Total explained variance level {l}: {np.sum(pca.explained_variance_ratio_):.3f}')
                ax.scatter(states_trans[:, 0, 0], states_trans[:, 0, 1], states_trans[:, 0, 2], c=colors[:, 0])
                ax.set_xlabel(f'PCA 1 ({pca.explained_variance_ratio_[0]:.3f})')
                ax.set_ylabel(f'PCA 2 ({pca.explained_variance_ratio_[1]:.3f})')
                ax.set_zlabel(f'PCA 3 ({pca.explained_variance_ratio_[2]:.3f})')
            obs = torch.stack(policy.flight_record[0]['o']).detach().cpu().numpy()
            d_time, d_batch = obs.shape[:2]
            time_steps_lvl = torch.stack(policy.flight_record[0]['time_step']).detach().cpu().numpy()
            time_steps_lvl = 1 - (time_steps_lvl / time_steps_lvl.max())
            colors = np.concatenate([time_steps_lvl, np.zeros((d_time, d_batch, 2))], axis=-1)
            ax = fig.add_subplot(100 + n_plots * 10 + n_plots)
            ax.scatter(obs[:, 0, 0], obs[:, 0, 1], label='agent position', c=colors[:, 0])
            ax.scatter(obs[:, 0, 2], obs[:, 0, 3], label='goal position')
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            plt.tight_layout()
            #plt.show()
            logger.log_plot(fig_to_img(fig), Scope.TEST() / f'model/latent_state_pca')
        """


    print('Starting Training')
    # with torch.autograd.detect_anomaly(check_nan=True):
    train_model(cfg, model, opt_model, r_max_agents, goal_seeking_agents, collect_fn, eval_env, test_driver,
                train_driver, logger, log_videos=False, video_env=video_env)
    # with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
    #    train_model(cfg, model, opt_model, r_max_agents, goal_seeking_agents, collect_fn, eval_env, test_driver,
    #        train_driver, logger, profile=profiling_run, log_videos=True)
    # print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))

    collect_env.close()
    eval_env.close()
    # store model and output run id
    p = here() / cfg['final_model_path'][:cfg['final_model_path'].rindex('/')]
    if not os.path.exists(p):
        os.makedirs(p)
    model_path = here() / f'{cfg["final_model_path"]}_{logger.run_id}.ptmdl'
    torch.save(model.state_dict(), model_path)
    model_weights = InMemoryFile(model_path, name='final_weights')
    logger.start_session()
    logger.log_file(model_weights, Scope.DATA() / 'weights')
    logger.stop_session()
    print(logger.run_id)


if __name__ == '__main__':
    main()
