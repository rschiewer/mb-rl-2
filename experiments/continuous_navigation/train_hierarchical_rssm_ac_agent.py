import os.path
import argparse
from typing import Any

import gym.vector
from tqdm import tqdm

from mdm.utils.torch_tools import to_np
from mdm.utils.utils import *
from mdm.training.offline_rl_driver import OfflineRLDriver, SamplingType
from mdm.logging.neptune_logger import NeptuneLogger
from mdm.logging.not_logger import NotLogger
from mdm.logging.logger import Scope
from mdm.training.gym_driver import collect_data, GymEpisodeDriver
from mdm.policies.agent_policy import *


def agent_train_mode(agents):
    for a in agents:
        if a is None: continue
        a[0].train()


def agent_eval_mode(agents):
    for a in agents:
        if a is None: continue
        a[0].eval()


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

    mem = []
    match cfg['prefill_memory']:
        case 'offline-dataset':
            print('loading offline data to prefill training memory...')
            mem = load_memory(here() / cfg['train_samples'])
        case 'random':
            print('collecting initial random trajectories...')
            collect_driver = GymEpisodeDriver(env, lambda *x: env.action_space.sample())
            collect_driver.interact(cfg['prefill_episodes'], True, mem)
        case False:
            print('starting with empty training memory...')

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
        agent_eval_mode(r_max_agents + goal_seeking_agents)
        policy = HierarchicalLatentAgentPolicy(model)
        collected_data_trajectories = collect_data(collect_env, 25, policy)
        # visualize_trajectory(collected_data_trajectories[0])
        mem.extend(collected_data_trajectories)

    # start training ---------------------------------------------------------------------------------------------------

    logger.start_session()
    model.prepare_for_training()
    for i_step in tqdm(range(cfg['trainer']['n_train_steps']), desc='Training Progress'):
        batch = train_driver.interact(cfg['trainer']['d_batch'])
        batch = to_tensors(batch, model.device)
        batch = prepare_data(batch)

        # train model normal
        model.train()
        agent_eval_mode(r_max_agents + goal_seeking_agents)
        model_batch = valid_subtrajectories(batch, cfg['trainer']['subtrajectory_len'])
        #model_batch_2 = valid_subtrajectories_2(batch, cfg['trainer']['subtrajectory_len'])
        #for k, v in model_batch.items():
        #    assert torch.all(model_batch_2[k] == v)

        #model_batch = batch
        train_steps = [-1, cfg['trainer']['subtrajectory_len'] * model.strides[1]]
        # now = time.time()
        train_losses = model.train_step(model_batch, opt_model, model_steps=train_steps)
        # print(time.time() - now)
        logger.log(to_np(train_losses), Scope.TRAIN(), i_step)

        # train model in observation mode
        # train_losses = model.train_step(model_batch, opt_model, model_steps=train_steps, learn_states=True)
        # logger.log(_to_np(train_losses), Scope.TRAIN() / 'observation_mode', i_step)

        # train agents
        if i_step % cfg['trainer']['agent_train_interval'] == 0:
            agent_train_mode(r_max_agents + goal_seeking_agents)
            agent_model_steps = cfg['trainer']['agent_model_steps']
            agent_model_max_warmup_steps = cfg['trainer']['agent_model_max_warmup_steps']
            agent_model_warmup_steps = [random.randint(1, n_wu) for n_wu in agent_model_max_warmup_steps]

            #trajectory_below = batch
            trajectory_below = valid_subtrajectories(batch, agent_model_warmup_steps[0])
            for l in range(model.levels):
                # model.reset_debug_counter()
                n_wu = agent_model_warmup_steps[l]
                n_steps = agent_model_steps[l]

                if l == 0:
                    _, _, start_state_lvl = model.forward_static(trajectory_below, level=0, n_steps=n_wu, n_warmup=-1)
                else:
                    _, _, _, start_state_lvl, start_state_below = model.ground_level(trajectory_below=trajectory_below,
                                                                                     level=l)
                    # we don't have groundtruth trajectories with actions, so make actions up but don't use
                    # them for training.
                    # TODO: Can we use the warmup actions for training as well? Maybe two separate act_in_sim calls so
                    _, _, _, start_state_lvl = model.forward_dynamic(start_state=start_state_lvl,
                                                                     start_state_below=start_state_below, level=l,
                                                                     n_steps=n_wu - 1, n_warmup=-1)

                # prevent gradient flow into the start state
                start_state_lvl['z'] = start_state_lvl['z'].detach()
                start_state_lvl['rnn_state'] = (start_state_lvl['rnn_state'][0].detach(),  # assumes LSTM state
                                                  start_state_lvl['rnn_state'][1].detach())

                # r_max agent
                r_max_agent, r_max_actor_opt, r_max_critic_opt = model.r_max_agents[l]
                r_max_simulation = r_max_agent.act_in_sim(start_state_lvl, model, n_steps - n_wu)
                r_max_losses = r_max_agent.train_step(r_max_simulation['agent'], actor_optimizer=r_max_actor_opt,
                                                      critic_optimizer=r_max_critic_opt)
                r_max_losses['obtained_reward'] = torch.stack(r_max_simulation['agent']['r']).mean()
                logger.log(to_np(r_max_losses), Scope.TRAIN() / f'r_max_agent/{l}/', i_step)

                # goal_seeking agent
                # We start at the same spot as the r_max agent, namely at start_state_lvl. We then use every
                # k-th time step from the r_max agent's simulation as intermediate goal and train goal finding
                if l < model.levels - 1:
                    goal_agent, goal_actor_opt, goal_critic_opt = model.goal_seeking_agents[l]
                    total_steps = len(r_max_simulation['model']['z'])
                    chunk_size = model.strides[l + 1]
                    agent_mem = {}

                    state = start_state_lvl
                    # start at chunk_size - 1 because start_state_lvl is not recorded in r_max_simulation
                    for t in range(chunk_size - 1, total_steps, chunk_size):
                    #for t in range(0, total_steps, chunk_size):
                        goal = r_max_simulation['model']['z'][t]
                        goal_simulation = goal_agent.act_in_sim(state, model, chunk_size, goal, agent_memory=agent_mem)
                        # ground goal agent with r_max agent trajectory after every chunk
                        # state = {'z': r_max_simulation['model']['z'][t].detach(),
                        #         'rnn_state': (r_max_simulation['model']['rnn_state'][t][0].detach(),
                        #                       r_max_simulation['model']['rnn_state'][t][1].detach())}
                        state = goal_simulation['model_state']

                    goal_losses = goal_agent.train_step(agent_mem, actor_optimizer=goal_actor_opt,
                                                        critic_optimizer=goal_critic_opt)
                    goal_losses['obtained_reward'] = torch.stack(goal_simulation['agent']['r']).mean()
                    logger.log(to_np(goal_losses), Scope.TRAIN() / f'goal_seeking_agent/{l}/', i_step)

                # prepare grounding information for next level
                trajectory_below = r_max_simulation['model']

        if i_step % cfg['trainer']['collect_interval'] == 0:
            collect_simple()
            #collect()

        # eval
        if cfg['trainer']['eval_interval'] is not None and i_step % cfg['trainer']['eval_interval'] == 0:
            agent_eval_mode(r_max_agents + goal_seeking_agents)
            model.eval()

            # model
            batch = test_driver.interact(cfg['trainer']['d_batch'])
            batch = to_tensors(batch, model.device)
            batch = prepare_data(batch)

            eval_steps = [-1, 20, 10]
            eval_losses = model.eval_step(batch, model_steps=eval_steps)
            logger.log(to_np(eval_losses), Scope.TEST(), i_step)
            # hierarchical agent
            eval_env.reset()
            policy = HierarchicalLatentAgentPolicy(model)
            eval_mem = collect_data(eval_env, 50, policy)
            logger.log(trajectory_statistics(eval_mem), Scope.TEST() / 'hierarchical_agent/', i_step)
            # flat agent
            eval_env.reset()
            policy = LatentAgentPolicy(r_max_agents[0][0], model)
            eval_mem = collect_data(eval_env, 50, policy)
            logger.log(trajectory_statistics(eval_mem), Scope.TEST() / 'flat_agent/', i_step)

            # latent state distribution
            # warmup_steps = cfg['eval']['warmup_steps']
            # pred, _, targets = model.forward_all_levels(batch, model_steps=eval_steps, warmup_steps=warmup_steps)
            # for l in range(model.levels):
            #    mask = model.compute_mask(targets[l])
            #    mask = mask.detach().cpu().numpy().reshape(mask.shape[0] * mask.shape[1], -1)
            #    latent_states = torch.stack(pred[l]['z']).detach().cpu().numpy()
            #    latent_states = latent_states.reshape(latent_states.shape[0] * latent_states.shape[1], -1)
            #    #latent_states = np.stack([s for s, m in zip(latent_states, mask) if m < 0.9])
            #    rewards = torch.stack(pred[l]['r']).detach().cpu().numpy()
            #    rewards = rewards.reshape(rewards.shape[0] * rewards.shape[1], -1)
            #    #rewards = np.stack([r for r, m in zip(rewards, mask) if m < 0.9])

            #    fig = plt.figure(figsize=(10, 10))
            #    fig.suptitle(f'Latent States Level {l}')
            #    ax = fig.add_subplot(111)#, projection='3d')
            #    #sc = ax.scatter(latent_states[:, 0], latent_states[:, 1], latent_states[:, 2], c=rewards.ravel())
            #    sc = ax.scatter(latent_states[:, 0], latent_states[:, 1], c=rewards.ravel())
            #    fig.colorbar(sc)
            #    #plt.show()
            #    logger.log_plot(fig_to_img(fig), Scope.TEST() / f'latent_state_distribution/{l}', i_step)
            #    plt.close(fig)
            #    del fig

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


if __name__ == '__main__':
    main()
