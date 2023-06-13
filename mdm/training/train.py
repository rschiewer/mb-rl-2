from typing import List, Dict
import random
import os
import time

import torch
from torchviz import make_dot
import matplotlib.pyplot as plt
from tqdm import tqdm

from mdm.logging.logger import Scope
from mdm.policies.agent_policy import HierarchicalLatentAgentPolicy, LatentAgentPolicy
from mdm.training.gym_driver import collect_data
from mdm.utils.torch_tools import to_tensors, to_np
from mdm.utils.utils import prepare_data, valid_subtrajectories, trajectory_statistics, trajectories_from_simulation, \
    visualize_overlaid_trajectories, anim_to_vid, get_dist_params, rssm_states_seq_to_batch, log_params
from mdm.models.building_blocks import RSSMCell


def agent_train_mode(agents):
    for a in agents:
        if a is None: continue
        a[0].train()


def agent_eval_mode(agents):
    for a in agents:
        if a is None: continue
        a[0].eval()


def train_model(cfg, model, opt_model, r_max_agents, goal_seeking_agents, collect_fn, eval_env, test_driver,
                train_driver, logger):
    model_train_steps = [-1, 10]
    logger.start_session()
    model.prepare_for_training()
    for i_step in tqdm(range(cfg['trainer']['n_train_steps']), desc='Training Progress'):
        batch = train_driver.interact(cfg['trainer']['d_batch'])
        batch = to_tensors(batch, model.device)
        batch = prepare_data(batch)
        # for easier diagnosis and debugging

        # train model normal
        model.train()
        agent_eval_mode(r_max_agents + goal_seeking_agents)
        #model_batch = valid_subtrajectories(batch, cfg['trainer']['subtrajectory_len'])
        model_batch = batch
        # model_batch_2 = valid_subtrajectories_2(batch, cfg['trainer']['subtrajectory_len'])
        # for k, v in model_batch.items():
        #    assert torch.all(model_batch_2[k] == v)

        # model_batch = batch
        # now = time.time()
        train_losses, pred = model.train_step(model_batch, opt_model, model_steps=model_train_steps)
        #make_dot(pred[0]['r'][-1].mean(), dict(model.named_parameters())).view()
        # print(time.time() - now)
        logger.log(to_np(train_losses), Scope.TRAIN(), i_step)

        # train model in observation mode
        # train_losses = model.train_step(model_batch, opt_model, model_steps=model_train_steps, learn_states=True)
        # logger.log(_to_np(train_losses), Scope.TRAIN() / 'observation_mode', i_step)

        # train agents
        if i_step % cfg['trainer']['agent_train_interval'] == 0:
            agent_train_mode(r_max_agents + goal_seeking_agents)
            agent_model_steps = cfg['trainer']['agent_model_steps']

            for l in range(model.levels):
                # use all time steps of teacher forcing rollout from model as starting point
                start_state_lvl = rssm_states_seq_to_batch(pred[l], model.rssm_modules[l], i_end=-1)
                # prevent gradient flow into the start state
                start_state_lvl = model.rssm_modules[l].detach_state(start_state_lvl)

                # r_max agent
                r_max_agent, r_max_actor_opt, r_max_critic_opt = model.r_max_agents[l]
                r_max_simulation = r_max_agent.act_in_sim(start_state_lvl, model, agent_model_steps[l],
                                                          sample_model=True, sample_actions=True,
                                                          reconstruct=False)
                r_max_losses = r_max_agent.train_step(r_max_simulation['agent'], actor_optimizer=r_max_actor_opt,
                                                      critic_optimizer=r_max_critic_opt)
                r_max_losses['obtained_reward'] = torch.stack(r_max_simulation['agent']['r']).mean()
                logger.log(to_np(r_max_losses), Scope.TRAIN() / f'r_max_agent/{l}/', i_step)

                # if l == 0:
                #    trajs_sim = trajectories_from_simulation(r_max_simulation['model'])
                #    fig, ani = visualize_trajectory(trajs_sim[0])
                #    plt.show()

                # goal_seeking agent
                # We start at the same spot as the r_max agent, namely at start_state_lvl. We then use every
                # k-th time step from the r_max agent's simulation as intermediate goal and train goal finding
                if l < model.levels - 1:
                    goal_agent, goal_actor_opt, goal_critic_opt = model.goal_seeking_agents[l]
                    total_steps = len(r_max_simulation['model']['z'])
                    chunk_size = model.strides[l + 1]
                    agent_mem = {}

                    # TODO: Die for-loop stellt sicher, dass von start_state_lvl an alle chunk_size schritte ein Ziel
                    #       verfolgt wird. Ist der finale Abschnitt der Trajektorie jedoch kürzer als chunk_size,
                    #       wird das letzte Ziel nicht mehr verfolgt. Da es sich aber um den finalen reward handeln
                    #       könnte, sollte das dennoch passieren.

                    state = start_state_lvl
                    # start at chunk_size - 1 because start_state_lvl is not recorded in r_max_simulation
                    for t in range(chunk_size - 1, total_steps, chunk_size):
                        goal = r_max_simulation['model']['z'][t]
                        goal_simulation = goal_agent.act_in_sim(state, model, chunk_size, goal, agent_memory=agent_mem,
                                                                sample_model=True, sample_actions=True,
                                                                reconstruct=False)
                        # ground goal agent with r_max agent trajectory after every chunk
                        # state = {'z': r_max_simulation['model']['z'][t].detach(),
                        #         'rnn_state': (r_max_simulation['model']['rnn_state'][t][0].detach(),
                        #                       r_max_simulation['model']['rnn_state'][t][1].detach())}
                        state = goal_simulation['model_state']

                    goal_losses = goal_agent.train_step(agent_mem, actor_optimizer=goal_actor_opt,
                                                        critic_optimizer=goal_critic_opt)
                    goal_losses['obtained_reward'] = torch.stack(goal_simulation['agent']['r']).mean()
                    logger.log(to_np(goal_losses), Scope.TRAIN() / f'goal_seeking_agent/{l}/', i_step)

        if i_step % cfg['trainer']['collect_interval'] == 0:
            collect_fn()

        # eval
        if cfg['trainer']['eval_interval'] is not None and i_step % cfg['trainer']['eval_interval'] == 0:
            agent_eval_mode(r_max_agents + goal_seeking_agents)
            model.eval()

            # model
            batch = test_driver.interact(cfg['trainer']['d_batch'])
            batch = to_tensors(batch, model.device)
            batch = prepare_data(batch)
            eval_steps = [-1, 20, 10]
            eval_losses, pred = model.eval_step(batch, model_steps=eval_steps, sample_state=False, sample_output=False)
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

            # model l0 simulation plot, works only for nav2d env
            warmup_steps = model.maybe_sample_warmup_steps(training_data=batch, model_steps=model_train_steps,
                                                           warmup_steps=model.warmup_steps)
            pred, pred_ema, targets = model.forward_all_levels(ground_truth_trajectory=batch,
                                                               warmup_steps=warmup_steps,
                                                               model_steps=model_train_steps,
                                                               sample_state=False,
                                                               sample_output=False)
            trajs_orig = trajectories_from_simulation(batch)
            trajs_sim = trajectories_from_simulation(pred[0])
            fig, anim = visualize_overlaid_trajectories(trajs_sim[0], trajs_orig[0])
            vid = anim_to_vid(anim)
            logger.log({'live_model': vid}, Scope.TEST() / 'model_prediction_video/', i_step)
            #os.remove(vid)
            plt.close(fig)

            # log model and agent params
            log_params(model, logger, Scope.PARAMETERS() / 'model', time_step=i_step)
            for i_agent, agent in enumerate(r_max_agents):
                if agent is None:
                    continue
                agent = agent[0]
                log_params(agent, logger, Scope.PARAMETERS() / f'agent/r_max_agent_{i_agent}', time_step=i_step)
            for i_agent, agent in enumerate(goal_seeking_agents):
                if agent is None:
                    continue
                agent = agent[0]
                log_params(agent, logger, Scope.PARAMETERS() / f'agent/goal_seeking_agent_{i_agent}', time_step=i_step)

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

        if i_step % cfg['trainer']['checkpoint_interval'] == 0:
            timestamp = time.time_ns()
            pid = os.getpid()
            model_path = f'.checkpoint_model_weights_{pid}_{timestamp}_{logger.run_id}.ptmdl'
            torch.save(model, model_path)
            logger.log_file(model_path, Scope.DATA() / 'weights')
            os.remove(model_path)

