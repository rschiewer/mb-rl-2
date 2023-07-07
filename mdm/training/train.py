import sys
from typing import List, Dict
import random
import os
import time

import torch
from torchviz import make_dot
import matplotlib.pyplot as plt
from tqdm import tqdm
import gym

from mdm.logging.logger import Scope
from mdm.policies.agent_policy import HierarchicalLatentAgentPolicy, LatentAgentPolicy
from mdm.policies.actor_critic_agent import ActorCriticAgent
from mdm.models.building_blocks import *
from mdm.training.gym_driver import collect_data
from mdm.utils.torch_tools import to_tensors, to_np
from mdm.utils.utils import prepare_data, valid_subtrajectories, trajectory_statistics, trajectories_from_simulation, \
    visualize_overlaid_trajectories, anim_to_vid, get_dist_params, rssm_states_seq_to_batch, log_params, InMemoryFile, \
    fig_to_img
from mdm.models.building_blocks import RSSMCell


def agent_train_mode(agents):
    for a in agents:
        if a is None: continue
        a[0].train()


def agent_eval_mode(agents):
    for a in agents:
        if a is None: continue
        a[0].eval()


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
        if agent_lvl == 0:
            cfg_r_max['min_a'] = tuple(env.action_space.low)
            cfg_r_max['max_a'] = tuple(env.action_space.high)
        cfg_r_max['d_a'] = cfg['mdm']['rssm_modules'][agent_lvl].d_a
        cfg_r_max['d_o'] = cfg['mdm']['rssm_modules'][agent_lvl].d_z
        r_max_agents.append(gen_agent_fn(agent_lvl, False, cfg_r_max))

        if agent_lvl < len(cfg['mdm']['rssm_modules']) - 1:
            cfg_goal_seeking = cfg['agents']['goal_seeking'][agent_lvl]
            if agent_lvl == 0:
                cfg_goal_seeking['min_a'] = tuple(env.action_space.low)
                cfg_goal_seeking['max_a'] = tuple(env.action_space.high)
            cfg_goal_seeking['d_a'] = cfg['mdm']['rssm_modules'][agent_lvl].d_a
            cfg_goal_seeking['d_o'] = cfg['mdm']['rssm_modules'][agent_lvl].d_z
            goal_seeking_agents.append(gen_agent_fn(agent_lvl, True, cfg_goal_seeking))
    # goal_seeking_agents.append(None)  # no homing agent needed on last level

    return r_max_agents, goal_seeking_agents


def train_model(cfg, model, opt_model, r_max_agents, goal_seeking_agents, collect_fn, eval_env, test_driver,
                train_driver, logger):
    model_train_steps = cfg['trainer']['model_train_steps']
    freeze_model = cfg['trainer']['freeze_model'] if cfg['trainer']['freeze_model'] > 0 else sys.maxsize
    stop_collect = cfg['trainer']['stop_collect'] if cfg['trainer']['stop_collect'] > 0 else sys.maxsize
    logger.start_session()
    model.prepare_for_training()
    for i_step in tqdm(range(cfg['trainer']['n_train_steps']), desc='Training Progress'):
        batch = train_driver.interact(cfg['trainer']['d_batch'])
        batch = to_tensors(batch, model.device, padding='repeat')
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
        if i_step < freeze_model:
            train_losses, pred, targets, states_below = model.train_step(model_batch, opt_model,
                                                                         model_steps=model_train_steps)
        else:
            train_losses, pred, targets, states_below = model.eval_step(model_batch, model_steps=model_train_steps,
                                                                        force_warmup=[-1 for _ in range(model.levels)])
            print('freezing model')
        logger.log(to_np(train_losses), Scope.TRAIN(), i_step)
        # make_dot(pred[0]['r'][-1].mean(), dict(model.named_parameters())).view()
        # print(time.time() - now)

        # train model in observation mode
        # train_losses = model.train_step(model_batch, opt_model, model_steps=model_train_steps, learn_states=True)
        # logger.log(_to_np(train_losses), Scope.TRAIN() / 'observation_mode', i_step)

        # train agents
        if i_step % cfg['trainer']['agent_train_interval'] == 0:
            agent_train_mode(r_max_agents + goal_seeking_agents)
            agent_model_steps = cfg['trainer']['agent_model_steps']

            for l in range(model.levels):
                # use all time steps of teacher forcing rollout from model as starting point
                start_state_lvl, start_state_mask = rssm_states_seq_to_batch(pred[l], model.rssm_modules[l], i_end=-1)
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

                # if l == 0:
                #    trajs_sim = trajectories_from_simulation(r_max_simulation['model'])
                #    fig, ani = visualize_trajectory(trajs_sim[0])
                #    plt.show()

                """
                # We start at the same spot as the r_max agent, namely at start_state_lvl. We then use the observation
                # k steps ahead in the  r_max agent's simulation as goal and train goal finding.
                # We train only one chunk do avoid accumulating errors and difficult credit assignment
                if l < model.levels - 1:
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
                # goal_seeking agent on goals made from current level r_max agent trajectory
                # We start at the same spot as the r_max agent, namely at start_state_lvl. We then use every
                # k-th time step from the r_max agent's simulation as intermediate goal and train goal finding
                if l < model.levels - 1:
                    goal_agent, goal_actor_opt, goal_critic_opt = model.goal_seeking_agents[l]
                    goals = model.filter_up(o=r_max_simulation['model']['z'], r=r_max_simulation['model']['r'],
                                            terminal=r_max_simulation['model']['terminal'], level=l + 1,
                                            respect_terminal_flag=True)
                    chunk_size = model.strides[l + 1]
                    agent_mem = {}

                    state = start_state_lvl
                    # start at chunk_size - 1 because start_state_lvl is not recorded in r_max_simulation
                    # for t in range(chunk_size - 1, total_steps, chunk_size):
                    for goal in goals['o']:
                        # for t in range(chunk_size, total_steps, chunk_size):
                        # goal = r_max_simulation['model']['z'][t]
                        goal_simulation = goal_agent.act_in_sim(state, model, chunk_size, goal, agent_memory=agent_mem,
                                                                sample_model=True, sample_actions=True,
                                                                disable_exploration=False, reconstruct=False)
                        # ground goal agent with r_max agent trajectory after every chunk
                        # state = {'z': r_max_simulation['model']['z'][t].detach(),
                        #         'rnn_state': (r_max_simulation['model']['rnn_state'][t][0].detach(),
                        #                       r_max_simulation['model']['rnn_state'][t][1].detach())}
                        state = goal_simulation['model_state']

                    agent_mem['first_step_mask'] = start_state_mask.unsqueeze(0)
                    goal_losses = goal_agent.train_step(agent_mem, actor_optimizer=goal_actor_opt,
                                                        critic_optimizer=goal_critic_opt)
                    goal_losses['obtained_reward'] = torch.stack(agent_mem['r']).mean()
                    logger.log(to_np(goal_losses), Scope.TRAIN() / f'goal_seeking_agent/{l}/', i_step)

                # goal seeking agent on l-1 on current level's r_max agent trajectory
                n_goals = 5
                if l > 0:
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

                    agent_mem['first_step_mask'] = start_state_mask_below.unsqueeze(0)
                    # goal_losses = goal_agent.train_step(agent_mem, actor_optimizer=goal_actor_opt,
                    #                                    critic_optimizer=goal_critic_opt)
                    goal_losses = goal_agent.eval_step(**agent_mem)
                    goal_losses['obtained_reward'] = torch.stack(goal_simulation['agent']['r']).mean()
                    logger.log(to_np(goal_losses), Scope.TRAIN() / f'goal_seeking_agent/hierarchical_goals/{l}/',
                               i_step)

        if i_step % cfg['trainer']['collect_interval'] == 0 and i_step < stop_collect:
            collect_fn()

        # eval
        if cfg['trainer']['eval_interval'] is not None and i_step % cfg['trainer']['eval_interval'] == 0:
            agent_eval_mode(r_max_agents + goal_seeking_agents)
            model.eval()

            # model
            trajs_orig = test_driver.interact(cfg['trainer']['d_batch'])
            batch = to_tensors(trajs_orig, model.device, padding='repeat')
            batch = prepare_data(batch)
            eval_steps = [-1] + cfg['trainer']['model_train_steps'][1:]
            eval_losses, pred, targets, _ = model.eval_step(batch, model_steps=eval_steps, sample_state=True,
                                                      sample_output=True, force_warmup=[-1 for _ in range(model.levels)])
            logger.log(to_np(eval_losses), Scope.TEST(), i_step)

            for l, pred_l in enumerate(pred):
                o_predicted = torch.stack(pred_l['o']).detach().cpu().numpy()
                r_predicted =  torch.stack(pred_l['r']).detach().cpu().numpy()
                term_predicted = torch.stack(pred_l['terminal']).detach().cpu().numpy()
                fig = plt.figure()
                plt.plot(o_predicted[:, 0], marker='o', label='observation')
                plt.plot(r_predicted[:, 0], marker='+', label='reward')
                plt.plot(term_predicted[:, 0], marker='x', label='terminal')
                plt.legend()
                plt.suptitle(f'Observations and terminal flags level {l}')
                logger.log_plot(fig_to_img(fig), Scope.TEST() / f'model/predicted_o_{l}', i_step)
                plt.close(fig)
                del fig

            # hierarchical agent
            eval_env.reset()
            policy = HierarchicalLatentAgentPolicy(model)
            eval_mem_hierarchical = collect_data(eval_env, 50, policy)
            logger.log(trajectory_statistics(eval_mem_hierarchical), Scope.TEST() / 'hierarchical_agent/', i_step)

            # flat agent
            eval_env.reset()
            policy = LatentAgentPolicy(r_max_agents[0][0], model)
            eval_mem_flat = collect_data(eval_env, 50, policy)
            logger.log(trajectory_statistics(eval_mem_flat), Scope.TEST() / 'flat_agent/', i_step)

            # print trajectories, works only for nav2d env
            fig, anim = visualize_overlaid_trajectories(eval_mem_flat[0])
            vid_flat = anim_to_vid(anim)
            vid_flat.name = 'flat_agent_acting'
            plt.close(fig)  # explicitly close to avoid memory leak
            fig, anim = visualize_overlaid_trajectories(eval_mem_hierarchical[0])
            vid_hierarchical = anim_to_vid(anim)
            vid_hierarchical.name = 'hierarchical_agent_acting'
            plt.close(fig)  # explicitly close to avoid memory leak
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
            cpt_file = InMemoryFile.consume_file(model_path, new_name='checkpoint')
            logger.log_file(cpt_file, Scope.DATA() / 'weights')

        # print(torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())
