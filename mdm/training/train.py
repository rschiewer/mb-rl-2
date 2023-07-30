import sys
from typing import List, Dict
import random
import os
import time

import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from tqdm import tqdm
import gym
import moviepy.editor as mp

from mdm.logging.logger import Scope, GlobalLogger
from mdm.policies.agent_policy import HierarchicalLatentAgentPolicy, LatentAgentPolicy
from mdm.policies.actor_critic_agent import ActorCriticAgent
from mdm.models.building_blocks import *
from mdm.training.gym_driver import collect_data
from mdm.utils.torch_tools import to_tensors, to_np, compute_mask
from mdm.utils.utils import (prepare_data, TempFigure, trajectory_statistics, trajectories_from_simulation,
                             anim_to_vid, rssm_states_seq_to_batch, log_params, InMemoryFile,
                             fig_to_img, valid_subtrajectories, valid_subtrajectories_unbiased,
                             valid_subtrajectories_unbiased_fast)
from mdm.utils.gym_nav2d_tools import gen_regular_grid_trajectories, visualize_overlaid_trajectories


def train_model(cfg, model, opt_model, r_max_agents, goal_seeking_agents, collect_fn, eval_env, test_driver,
                train_driver, logger, log_videos: bool = True, video_env = None):
    model_train_steps = cfg['trainer']['model_train_steps']
    freeze_model = cfg['trainer']['freeze_model'] if cfg['trainer']['freeze_model'] > 0 else sys.maxsize
    stop_collect = cfg['trainer']['stop_collect'] if cfg['trainer']['stop_collect'] > 0 else sys.maxsize
    logger.start_session()
    model.prepare_for_training()

    # for evaluation of latent space
    #grid_trajs = gen_regular_grid_trajectories(eval_env.env_fns[0](), trajs_vert=10, trajs_horiz=10, step_size=1.0)
    #grid_trajs = to_tensors(grid_trajs, model.device, padding='repeat')
    #grid_trajs = prepare_data(grid_trajs)

    for i_step in tqdm(range(cfg['trainer']['n_train_steps']), desc='Training Progress'):
        batch = train_driver.interact(cfg['trainer']['d_batch'])
        batch = to_tensors(batch, model.device, padding='repeat')
        batch = prepare_data(batch)
        # for easier diagnosis and debugging

        # train model normal
        model.train()
        agent_eval_mode(r_max_agents + goal_seeking_agents)
        if cfg['trainer']['subtrajectory_len'] > 0:
            # model_batch = valid_subtrajectories(batch, cfg['trainer']['subtrajectory_len'])
            # model_batch = valid_subtrajectories_unbiased(batch, 15)
            model_batch = valid_subtrajectories_unbiased_fast(batch, cfg['trainer']['subtrajectory_len'])
        else:
            model_batch = batch

        if i_step < freeze_model:
            train_losses, pred, targets, states_below = model.train_step(model_batch, opt_model,
                                                                         model_steps=model_train_steps)
        else:
            train_losses, pred, targets, states_below = model.eval_step(model_batch, model_steps=model_train_steps,
                                                                        force_warmup=[-1 for _ in range(model.levels)])

            print('freezing model')

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
                start_state_lvl, start_state_mask = rssm_states_seq_to_batch(pred[l], model.rssm_modules[l])
                # prevent gradient flow into the start state
                start_state_lvl = model.rssm_modules[l].detach_state(start_state_lvl)

                # mask to eliminate training steps that start after an episode has already ended

                # r_max agent
                r_max_agent, r_max_actor_opt, r_max_critic_opt = model.r_max_agents[l]
                abstract_level = l > 0
                r_max_simulation = r_max_agent.act_in_sim(start_state_lvl, model, agent_model_steps[l],
                                                          sample_model=True, sample_actions=True,
                                                          reconstruct=abstract_level)
                r_max_simulation['agent']['first_step_mask'] = start_state_mask.unsqueeze(0)
                r_max_losses = r_max_agent.train_step(r_max_simulation['agent'], actor_optimizer=r_max_actor_opt,
                                                      critic_optimizer=r_max_critic_opt)
                r_max_losses['obtained_reward'] = torch.stack(r_max_simulation['agent']['r']).mean()
                logger.log(to_np(r_max_losses), Scope.TRAIN() / f'r_max_agent/{l}/', i_step)

                if GlobalLogger.can_log('sanity_check_goal_computation', i_step):
                    if l < model.levels - 1:
                        achieved_goals = model.filter_up(o=r_max_simulation['agent']['o_env_next'],
                                                         r=r_max_simulation['agent']['r'],
                                                         terminal=r_max_simulation['agent']['terminal'], level=l + 1,
                                                         respect_terminal_flag=False)
                        chunk_size = model.strides[l + 1]

                        similarities = torch.zeros(len(r_max_simulation['agent']['o_env_next']),
                                                   len(r_max_simulation['agent']['o_env_next']), device=model.device,
                                                   dtype=torch.float32)
                        for i, g in enumerate(r_max_simulation['agent']['o_env_next']):
                            for j, g_other in enumerate(r_max_simulation['agent']['o_env_next']):
                                #similarities[i, j] = torch.nn.functional.cosine_similarity(g, g_other, dim=-1).mean()
                                similarities[i, j] = torch.mean(torch.abs(g - g_other))

                        with TempFigure(dpi=60) as fig:
                            plt.matshow(similarities.detach().cpu().numpy(), fignum=fig)
                            plt.tight_layout()
                            plt.colorbar()
                            logger.log_plot(fig_to_img(fig, minimize_size=False),
                                            Scope.TRAIN() / f'r_max_agent_fake_goals/{l}/trajectory_step_similarities',
                                            i_step)

                        with TempFigure(dpi=60) as fig:
                            o_next = torch.stack(r_max_simulation['agent']['o_env_next'])
                            z_diff = torch.mean(torch.abs(o_next[:-1] - o_next[1:]), dim=(1, 2))
                            plt.plot(z_diff.detach().cpu().numpy().squeeze())
                            logger.log_plot(fig_to_img(fig),
                                            Scope.TRAIN() / f'r_max_agent_fake_goals/{l}/state_distances',
                                            i_step)

                        r_max_simulation['agent']['r'] = []
                        for i_goal, goal in enumerate(achieved_goals['o'][:-1]):
                            for t_chunk in range(chunk_size):
                                t = i_goal * chunk_size + t_chunk
                                r_her = r_max_agent.build_step_reward(r_max_simulation['agent']['o_env_next'][t],
                                                                      r_max_simulation['agent']['r_raw'][t], goal,
                                                                      use_goal_reward=True)
                                r_max_simulation['agent']['r'].append(r_her)
                        obtained_step_reward = torch.stack(r_max_simulation['agent']['r']).mean(dim=1)
                        terminals = torch.stack(r_max_simulation['agent']['terminal']).mean(dim=1)
                        with TempFigure(dpi=60) as fig:
                            plt.plot(terminals.detach().cpu().numpy().squeeze(), marker='.', color='lightgray')
                            plt.scatter(range(len(obtained_step_reward)),
                                        obtained_step_reward.detach().cpu().numpy().squeeze(), marker='o')
                            plt.suptitle(f'L{l} R_max Agent Fake Goal Step Rewards')
                            plt.tight_layout()
                            logger.log_plot(fig_to_img(fig), Scope.TRAIN() / f'r_max_agent_fake_goals/{l}/her_step_reward',
                                            i_step)

                if l < model.levels - 1:
                    # update model's stats about how distant goals are on average
                    goals = model.filter_up(o=r_max_simulation['model']['z'], r=r_max_simulation['model']['r'],
                                            terminal=r_max_simulation['model']['terminal'], level=l + 1,
                                            respect_terminal_flag=False)
                    goals_mask = compute_mask(goals['terminal'])


                    """
                    Version A: 
                    We start at the same spot as the r_max agent, namely at start_state_lvl. We then use the observation
                    k steps ahead in the  r_max agent's simulation as goal and train goal finding.
                    We train only one chunk do avoid accumulating errors and difficult credit assignment

                    goal_agent, goal_actor_opt, goal_critic_opt = model.goal_seeking_agents[l]
                    goal = model.filter_up(o=r_max_simulation['model']['z'], r=r_max_simulation['model']['r'],
                                           terminal=r_max_simulation['model']['terminal'], level=l + 1,
                                           respect_terminal_flag=True, n_steps=1)
                    goal = goal['o'][0]
                    chunk_size = model.strides[l + 1]

                    goal_simulation = goal_agent.act_in_sim(start_state_lvl, model, chunk_size, goal,
                                                            sample_model=True, sample_actions=True, reconstruct=False)

                    goal_simulation['agent']['first_step_mask'] = start_state_mask.unsqueeze(0)
                    goal_losses = goal_agent.train_step(goal_simulation['agent'], actor_optimizer=goal_actor_opt,
                                                        critic_optimizer=goal_critic_opt)
                    goal_losses['obtained_reward'] = torch.stack(goal_simulation['agent']['r']).mean()
                    logger.log(to_np(goal_losses), Scope.TRAIN() / f'goal_seeking_agent/{l}/', i_step)
                    """
                    """
                    Version A2:
                    train just a single chunk but go one time step beyond that to have meaningful training targets 
                    for value function and policy

                    for each time step of r_max trajectory and each batch item, let agent do one chunk
                    goal_simulation = goal_agent.act_in_sim(state, model, chunk_size, goals['o'][0],
                                                           agent_memory=agent_mem, sample_model=True,
                                                           sample_actions=True, disable_exploration=False,
                                                           reconstruct=False)
                    go one step further to get a meaningful bootstrap for last step of first chunk
                    goal_simulation = goal_agent.act_in_sim(goal_simulation['model_state'], model, 1,
                                                           goals['o'][1], agent_memory=agent_mem, sample_model=True,
                                                           sample_actions=True, disable_exploration=False,
                                                           reconstruct=False)
                    """

                    """
                    Version B:
                    goal_seeking agent on goals made from current level r_max agent trajectory
                    We start at the same spot as the r_max agent, namely at start_state_lvl. We then use every
                    k-th time step from the r_max agent's simulation as intermediate goal and train goal finding
                    """
                    goal_agent, goal_actor_opt, goal_critic_opt = model.goal_seeking_agents[l]
                    goals = model.filter_up(o=r_max_simulation['model']['z'], r=r_max_simulation['model']['r'],
                                            terminal=r_max_simulation['model']['terminal'], level=l + 1,
                                            respect_terminal_flag=True)
                    chunk_size = model.strides[l + 1]
                    agent_mem = {}
                    state = start_state_lvl
                    for i_goal, goal in enumerate(goals['o']):
                        """
                        Variation B1:
                        give one additional step in last chunk
                        
                        if i_goal == len(goals['o']) - 1:
                           n_steps = chunk_size + 1
                        else:
                           n_steps = chunk_size
                        """
                        n_steps = chunk_size
                        goal_simulation = goal_agent.act_in_sim(state, model, n_steps, goal, agent_memory=agent_mem,
                                                                sample_model=True, sample_actions=True,
                                                                disable_exploration=False, reconstruct=False)
                        state = goal_simulation['model_state']
                        """
                        Variation B2:
                        ground goal agent with r_max agent trajectory after every chunk
                        
                        state = {'z': r_max_simulation['model']['z'][t].detach(),
                                'rnn_state': (r_max_simulation['model']['rnn_state'][t][0].detach(),
                                              r_max_simulation['model']['rnn_state'][t][1].detach())}
                        """

                    """
                    Version C:
                    goals from current level r_max agent but with half of the goals replaced by noisy trajectories
                    that are freshly generated from the model using noisy versions of the r_mas agent's actions

                    noisy_actions = torch.stack(r_max_simulation['agent']['a'])
                    a_space_midpoint = (r_max_agent.max_a + r_max_agent.min_a) / 2
                    a_space_span = r_max_agent.max_a - r_max_agent.min_a
                    act_noise = (torch.rand_like(noisy_actions[:, ::2]) - 0.5 + a_space_midpoint) * a_space_span
                    noisy_actions[:, ::2] += act_noise  # every 2nd trajectory gets noisy actions
                    noisy_actions = torch.clamp(noisy_actions, r_max_agent.min_a, r_max_agent.max_a)
                    noisy_simulation, _, _ = model.forward_static({'o': None, 'a': noisy_actions}, level=l, n_warmup=0,
                                                                 start_state=start_state_lvl, sample_state=True,
                                                                 sample_output=False, reconstruct=False)
                    goals = model.filter_up(o=noisy_simulation['z'], r=noisy_simulation['r'],
                                           terminal=noisy_simulation['terminal'], level=l + 1,
                                           respect_terminal_flag=True)
                    chunk_size = model.strides[l + 1]
                    agent_mem = {}
                    state = start_state_lvl
                    for i_goal, goal in enumerate(goals['o']):
                        n_steps = chunk_size
                        goal_simulation = goal_agent.act_in_sim(state, model, n_steps, goal, agent_memory=agent_mem,
                                                                sample_model=True, sample_actions=True,
                                                                disable_exploration=False, reconstruct=False)
                        state = goal_simulation['model_state']
                    """
                    """
                    Version D:
                    perform hindsight experience replay by replacing initial goals with the achieved goals, this means
                    the agent needs to be trained with standard policy gradients method and can't be trained by 
                    backpropagating through dynamics model anymore
                    
                    achieved_goals = model.filter_up(o=agent_mem['o_env_next'], r=agent_mem['r'],
                                                     terminal=agent_mem['terminal'], level=l + 1,
                                                     respect_terminal_flag=True)
                    agent_mem['r_old'] = [x.clone() for x in agent_mem['r']]
                    for i_goal, goal in enumerate(achieved_goals['o']):
                        for t_chunk in range(n_steps):
                            t = i_goal * n_steps + t_chunk
                            r_her = goal_agent.build_step_reward(agent_mem['o_env_next'][t],
                                                                 agent_mem['r_raw'][t], goal,
                                                                 use_goal_reward=True)
                            agent_mem['r'][t][::2] = r_her[::2]
                    """
                    """
                    Version E:
                    goal seeking agent on l-1 on current level's r_max agent trajectory

                    n_goals = 5
                    mem_below = {**states_below[l], 'terminal': pred[l]['terminal']}  # need terminals for mask
                    mem_below['z_post'] = mem_below['z_prior']  # both are not needed but z_post contains None elements
                    start_states_lvl_below, start_state_mask_below = rssm_states_seq_to_batch(mem_below,
                                                                                              model.rssm_modules[l - 1],
                                                                                              i_end=-n_goals)
                    start_states_lvl_below = model.rssm_modules[l - 1].detach_state(start_states_lvl_below)

                    goal_agent, goal_actor_opt, goal_critic_opt = model.goal_seeking_agents[l - 1]

                    goals = torch.stack(pred[l]['o'])
                    goals_batched = []
                    i_first = 1  # leave out first goal since we lack agent start state for it
                    i_last = len(goals) - n_goals  # latest time step where we have n future goals
                    for t in range(i_first, i_last + 1):
                        goals_batched.append(goals[t: t + n_goals])
                    goals_batched = torch.concat(goals_batched, dim=1)

                    chunk_size = model.strides[l]
                    agent_mem = {}

                    state = start_states_lvl_below
                    for goal in goals_batched:
                        goal_simulation = goal_agent.act_in_sim(state, model, chunk_size, goal, agent_memory=agent_mem,
                                                                sample_model=False, sample_actions=False,
                                                                disable_exploration=False,
                                                                reconstruct=False)
                        state = goal_simulation['model_state']
                    """

                    agent_mem['first_step_mask'] = start_state_mask.unsqueeze(0)
                    goal_losses = goal_agent.train_step(agent_mem, actor_optimizer=goal_actor_opt,
                                                        critic_optimizer=goal_critic_opt)
                    obtained_step_reward = torch.stack(agent_mem['r'])
                    goal_losses['obtained_reward'] = obtained_step_reward.mean()

                    if GlobalLogger.can_log('simulated_ground_truth_goal_distance', i_step):
                        obtained_step_reward = obtained_step_reward.mean(dim=1)
                        with TempFigure(dpi=60) as fig:
                            plt.plot(obtained_step_reward.detach().cpu().numpy().squeeze(), marker='o')
                            plt.scatter(np.arange(chunk_size - 1, agent_model_steps[l] + chunk_size - 1, chunk_size),
                                        obtained_step_reward.detach().cpu().numpy().squeeze()[
                                        chunk_size - 1::chunk_size],
                                        marker='o', s=100)
                            plt.suptitle(f'L{l} Goal Seeking Agent Step Rewards')
                            plt.tight_layout()
                            logger.log_plot(fig_to_img(fig), Scope.TRAIN() / f'goal_seeking_agent/{l}/step_reward',
                                            i_step)

                    logger.log(to_np(goal_losses), Scope.TRAIN() / f'goal_seeking_agent/{l}/', i_step)


        if i_step % cfg['trainer']['collect_interval'] == 0 and i_step < stop_collect:
            collect_fn()

        # eval =========================================================================================================
        if cfg['trainer']['eval_interval'] is not None and i_step % cfg['trainer']['eval_interval'] == 0:
            agent_eval_mode(r_max_agents + goal_seeking_agents)
            model.eval()

            # model
            trajs_orig = test_driver.interact(cfg['trainer']['d_batch'])
            batch = to_tensors(trajs_orig, model.device, padding='repeat')
            batch = prepare_data(batch)
            eval_steps = [-1] + cfg['trainer']['model_train_steps'][1:]
            eval_losses, pred, targets, _ = model.eval_step(batch, model_steps=eval_steps, sample_state=True,
                                                            sample_output=True,
                                                            force_warmup=[-1 for _ in range(model.levels)])
            logger.log(to_np(eval_losses), Scope.TEST(), i_step)

            for l, pred_l in enumerate(pred):
                o_predicted = torch.stack(pred_l['o']).detach().cpu().numpy()
                r_predicted = torch.stack(pred_l['r']).detach().cpu().numpy()
                term_predicted = torch.stack(pred_l['terminal']).detach().cpu().numpy()
                #with TempFigure() as fig:
                #    plt.plot(o_predicted[:, 0], marker='o', label='observation')
                #    plt.plot(r_predicted[:, 0], marker='+', label='reward')
                #    plt.plot(term_predicted[:, 0], marker='x', label='terminal')
                #    plt.legend()
                #    plt.suptitle(f'Observations and terminal flags level {l}')
                #    logger.log_plot(fig_to_img(fig), Scope.TEST() / f'model/predicted_o_{l}', i_step)

            # hierarchical agent
            eval_env.reset()
            policy = HierarchicalLatentAgentPolicy(model)
            eval_mem_hierarchical = collect_data(eval_env, cfg['eval']['eval_steps'], policy)
            logger.log(trajectory_statistics(eval_mem_hierarchical), Scope.TEST() / 'hierarchical_agent/', i_step)

            if video_env:
                # record an episode
                video_env.reset()
                video_env.start_video_recorder()
                _ = collect_data(video_env, cfg['eval']['eval_steps'], policy)
                video_env.close_video_recorder()

                # make video smaller
                video_name = f'{video_env.name_prefix}-episode-{video_env.episode_id}.mp4'
                video_path = os.path.join(video_env.video_folder, video_name)

                timestamp = time.time_ns()
                pid = os.getpid()
                tmp_file_name = f'.{pid}_{timestamp}_agent_video.mp4'

                clip = mp.VideoFileClip(video_path)
                clip = clip.resize(width=64)
                clip.write_videofile(tmp_file_name, preset='veryslow', verbose=False, logger=None)

                # upload
                video = InMemoryFile.consume_file(tmp_file_name)
                logger.log({'hierarchical_agent': video}, Scope.TEST() / 'agent_action_videos/', i_step)

            # flat agent
            eval_env.reset()
            policy = LatentAgentPolicy(r_max_agents[0][0], model)
            eval_mem_flat = collect_data(eval_env, cfg['eval']['eval_steps'], policy)
            logger.log(trajectory_statistics(eval_mem_flat), Scope.TEST() / 'flat_agent/', i_step)

            if log_videos:
                # print trajectories, works only for nav2d env
                fig, anim = visualize_overlaid_trajectories(eval_mem_flat[0])
                vid_flat = anim_to_vid(anim)
                vid_flat.name = 'flat_agent_acting'
                plt.close(fig)  # explicitly close to avoid memory leak
                del fig
                fig, anim = visualize_overlaid_trajectories(eval_mem_hierarchical[0])
                vid_hierarchical = anim_to_vid(anim)
                vid_hierarchical.name = 'hierarchical_agent_acting'
                plt.close(fig)  # explicitly close to avoid memory leak
                del fig
                logger.log({'flat_agent': vid_flat, 'hierarchical_agent': vid_hierarchical},
                           Scope.TEST() / 'agent_action_videos/', i_step)

                # model l0 simulation plot, works only for nav2d env
                warmup_steps = model.maybe_sample_warmup_steps(training_data=batch, model_steps=model_train_steps,
                                                               warmup_steps=model.warmup_steps)
                pred, pred_ema, _, _ = model.forward_all_levels(ground_truth_trajectory=batch,
                                                                warmup_steps=warmup_steps,
                                                                model_steps=model_train_steps,
                                                                sample_state=True,
                                                                sample_output=False)
                trajs_orig_pad = trajectories_from_simulation(batch)  # do this to get padded versions of orig trajectories
                trajs_sim = trajectories_from_simulation(pred[0])
                fig, anim = visualize_overlaid_trajectories(trajs_sim[0], trajs_orig_pad[0])
                vid = anim_to_vid(anim)
                vid.name = 'model_sim'
                logger.log({'live_model': vid}, Scope.TEST() / 'model_prediction_video/', i_step)
                plt.close(fig)  # explicitly close to avoid memory leak
                del fig

            # log model and agent params
            log_params(model, logger, Scope.PARAMETERS() / 'model', time_step=i_step)
            for i_ag, ag in enumerate(r_max_agents):
                if ag is None: continue
                log_params(ag[0], logger, Scope.PARAMETERS() / f'agent/r_max_agent_{i_ag}', time_step=i_step)
            for i_ag, ag in enumerate(goal_seeking_agents):
                if ag is None: continue
                log_params(ag[0], logger, Scope.PARAMETERS() / f'agent/goal_seeking_agent_{i_ag}', time_step=i_step)

            """
            _, pred_grid, _, _ = model.eval_step(grid_trajs, model_steps=eval_steps, sample_state=False,
                                                 sample_output=False,
                                                 force_warmup=[-1 for _ in range(model.levels)])

            # latent sate PCA 3d plot
            states = torch.stack(pred_grid[0]['z'])
            states = states.detach().cpu().numpy()
            d_time, d_batch, d_z = states.shape
            pca = PCA(n_components=3)
            states_trans = pca.fit_transform(states.reshape(d_time * d_batch, d_z))
            xy_positions = (grid_trajs['o'].detach().cpu().numpy()[:, :, :2].reshape(d_time * d_batch, 2) + 1.0) / 2.0
            colors = np.concatenate([xy_positions, np.zeros((d_time * d_batch, 1), dtype=float)], axis=-1)
            with TempFigure() as fig:
                ax = fig.add_subplot(projection='3d')
                ax.set_title(f'Total explained variance: {np.sum(pca.explained_variance_ratio_):.3f}')
                ax.scatter(states_trans[:, 0], states_trans[:, 1], states_trans[:, 2], c=colors)
                ax.set_xlabel(f'PCA 1 ({pca.explained_variance_ratio_[0]:.3f})')
                ax.set_ylabel(f'PCA 2 ({pca.explained_variance_ratio_[1]:.3f})')
                ax.set_zlabel(f'PCA 3 ({pca.explained_variance_ratio_[2]:.3f})')
                plt.tight_layout()
                logger.log_plot(fig_to_img(fig), Scope.TEST() / f'model/{0}/latent_state_pca', i_step)
            # sanity check to confirm that coloring based on xy positions makes sense
            # plt.scatter(xy_positions[:, 0], xy_positions[:, 1], c=colors)
            # plt.show()
            """

        if i_step % cfg['trainer']['checkpoint_interval'] == 0:
            timestamp = time.time_ns()
            pid = os.getpid()
            model_path = f'.checkpoint_model_weights_{pid}_{timestamp}_{logger.run_id}.ptmdl'
            torch.save(model, model_path)
            cpt_file = InMemoryFile.consume_file(model_path, new_name='checkpoint')
            logger.log_file(cpt_file, Scope.DATA() / 'weights')

        # print(torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())


def agent_train_mode(agents):
    for a in agents:
        if a is None: continue
        a[0].train()


def agent_eval_mode(agents):
    for a in agents:
        if a is None: continue
        a[0].eval()


def build_model_opt(model: torch.nn.Module, cfg: dict):
    optim_type = cfg['optim'].pop('type')
    params = model.parameters()

    if optim_type == 'adam':
        opt_model = torch.optim.Adam(params, **cfg['optim'])
    elif optim_type == 'adamW':
        opt_model = torch.optim.AdamW(params, **cfg['optim'])
    elif optim_type == 'sgd':
        opt_model = torch.optim.SGD(params, **cfg['optim'])
    else:
        raise ValueError(f'Unknown optimizer type: {optim_type}')

    return opt_model


def build_rssms(cfg: dict):
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
    return cfg


def build_agents(cfg: dict,
                 env: gym.Env,
                 device: torch.device):
    if not isinstance(env.action_space, gym.spaces.Box):
        raise ValueError('Only enviornments with continuous action space are supported')

    def gen_agent_fn(level: int, goal_seeking: bool, cfg) -> (
            ActorCriticAgent, torch.optim.Optimizer, torch.optim.Optimizer):
        agent = ActorCriticAgent(level=level, observation_key='z', goal_seeking=goal_seeking, **cfg)
        agent = agent.to(device)
        actor_optimizer = torch.optim.Adam(agent.actor_net.parameters(), lr=cfg['lr_actor'])
        critic_optimizer = torch.optim.Adam(agent.critic_net.parameters(), lr=cfg['lr_critic'])
        return agent, actor_optimizer, critic_optimizer

    r_max_agents = []
    goal_seeking_agents = []
    for agent_lvl in range(len(cfg['mdm']['rssm_modules'])):
        cfg_r_max = cfg['agents']['r_max'][agent_lvl]
        r_max_agents.append(gen_agent_fn(agent_lvl, False, cfg_r_max))

        if agent_lvl < len(cfg['mdm']['rssm_modules']) - 1:
            cfg_goal_seeking = cfg['agents']['goal_seeking'][agent_lvl]
            goal_seeking_agents.append(gen_agent_fn(agent_lvl, True, cfg_goal_seeking))
    # goal_seeking_agents.append(None)  # no homing agent needed on last level

    return r_max_agents, goal_seeking_agents
