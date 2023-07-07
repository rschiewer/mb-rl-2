import argparse
from itertools import chain
import math

import gym.vector
from tqdm import tqdm

from mdm.training.train import train_model, agent_eval_mode, build_rssms, build_agents
from mdm.utils.utils import *
from mdm.utils.torch_tools import to_np
from mdm.utils.analysis_tools import plot_trajectory_stats
from mdm.training.offline_rl_driver import OfflineRLDriver, SamplingType
from mdm.logging.neptune_logger import NeptuneLogger
from mdm.logging.not_logger import NotLogger
from mdm.logging.logger import Scope, GlobalLogger
from mdm.training.gym_driver import collect_data, GymEpisodeDriver
from mdm.policies.agent_policy import *


def expert_collect_policy(env: CacheLastStepVecEnv):
    o = env.last_o
    agent_pos = o[:, :2]
    goal_pos = o[:, 2:4]
    distance = o[:, 4]
    adjacent = (goal_pos[:, 1] - agent_pos[:, 1])
    disjacent = (goal_pos[:, 0] - agent_pos[:, 0])
    angle = np.arctan2(adjacent, disjacent) + math.pi * 1.5

    dist_a = np.where(distance > 0.05, 1.0, 0.1)
    angle = np.where(angle > 2 * math.pi, angle - 2 * math.pi, angle)
    angle_a = angle / (2 * math.pi) * 2 - 1
    angle_a += (np.random.random() - 0.5) * 0.5
    angle_a = np.clip(angle_a, -1.0, 1.0)
    a = np.stack([angle_a, dist_a], axis=1).astype(np.float32)

    return a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-log', default=False, action='store_true')
    parser.add_argument('-d_batch', type=int)
    parser.add_argument('-n_collect', type=int)
    args = parser.parse_args()

    cfg = load_yaml(here() / 'cfg_model.yaml')
    neptune_cfg = load_yaml(here() / cfg['neptune_cfg'])

    if args.log:
        logger = NeptuneLogger(**neptune_cfg)
    else:
        logger = NotLogger()
    GlobalLogger.bind(logger, {'mask_model': 25, 'mask_latent_overshooting': 25})  # for debugging

    env = gym.make(cfg['env_name'])
    env = CacheLastStepEnv(env)

    def make_env_fn():
        return gym.make(cfg['env_name'])

    cfg = cfg_infer_missing_values(cfg, env)  # fill in missing config values
    logger.start_session()
    logger.log(cfg, Scope.HYPERPARAMETERS())  # log complete config

    cfg = build_rssms(cfg)
    model = HierarchicalRSSM(**cfg['mdm'], r_max_agents=[], goal_seeking_agents=[])

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

    collect_env = gym.vector.AsyncVectorEnv([make_env_fn] * cfg['trainer']['collect_envs'])
    collect_env = CacheLastStepVecEnv(collect_env)

    train_mem = []
    rand_driver = GymEpisodeDriver(collect_env, lambda *x: collect_env.action_space.sample())
    rand_driver.interact(cfg['random_episodes'], train_mem)
    expert_driver = GymEpisodeDriver(collect_env, expert_collect_policy)
    expert_driver.interact(cfg['expert_episodes'], train_mem)

    # fig, ani = visualize_trajectory(train_mem[0])
    # gif = anim_to_gif(ani)
    # plt.show()
    # fig = plot_trajectory_stats(train_mem, 20)
    # plt.show()

    random.shuffle(train_mem)
    i_split = len(train_mem) // 5
    train_mem = train_mem[i_split:]
    test_mem = train_mem[:i_split]

    train_driver = OfflineRLDriver(train_mem, sampling_type=SamplingType.RANDOM)
    test_driver = OfflineRLDriver(test_mem, sampling_type=SamplingType.RANDOM)

    # start training
    logger.start_session()
    model.prepare_for_training()
    for i_step in tqdm(range(cfg['trainer']['n_train_steps']), desc='Training Progress'):
        batch = train_driver.interact(cfg['trainer']['d_batch'])
        batch = to_tensors(batch, model.device)
        batch = prepare_data(batch)

        model.train()
        # model_batch = valid_subtrajectories(batch, cfg['trainer']['subtrajectory_len'])
        train_losses, pred, targets, states_below = model.train_step(batch, opt_model,
                                                                     model_steps=cfg['trainer']['model_train_steps'])
        logger.log(to_np(train_losses), Scope.TRAIN(), i_step)

        if cfg['trainer']['eval_interval'] is not None and i_step % cfg['trainer']['eval_interval'] == 0:
            model.eval()
            eval_losses, pred, _, _ = model.eval_step(batch, model_steps=[-1], sample_state=True,
                                                      sample_output=True, force_warmup=[-1])
            eval_losses_deter, pred_deter, _, _ = model.eval_step(batch, model_steps=[-1], sample_state=True,
                                                                  sample_output=False, force_warmup=[-1])
            logger.log(to_np(eval_losses), Scope.TEST(), i_step)

            trajs_orig_pad = trajectories_from_simulation(batch)  # do this to get padded versions of orig trajectories
            trajs_sim = trajectories_from_simulation(pred[0])
            fig, anim = visualize_overlaid_trajectories(trajs_sim[0], trajs_orig_pad[0])
            vid_sampled = anim_to_vid(anim)
            vid_sampled.name = 'model_sim_sampled'
            plt.close(fig)  # explicitly close to avoid memory leak
            trajs_sim = trajectories_from_simulation(pred_deter[0])
            fig, anim = visualize_overlaid_trajectories(trajs_sim[0], trajs_orig_pad[0])
            vid_deter = anim_to_vid(anim)
            vid_deter.name = 'model_sim_deter'
            plt.close(fig)  # explicitly close to avoid memory leak

            logger.log({'sampled': vid_sampled, 'deterministic': vid_deter},
                       Scope.TEST() / 'model_prediction_video/', i_step)

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
