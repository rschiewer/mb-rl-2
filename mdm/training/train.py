import math
from math import ceil

import gym_nav2d.envs
import gymnasium_robotics
import matplotlib.pyplot as plt
from gymnasium_robotics.envs.maze import PointMazeEnv
from gymnasium_robotics.envs.maze.maze_v4 import MazeEnv
from sklearn.decomposition import PCA
from matplotlib.colors import hsv_to_rgb, to_rgba
from matplotlib.patches import Rectangle
from tqdm import tqdm
import moviepy.editor as mp

from mdm.logging.logger import Scope, GlobalLogger
from mdm.policies.actor_critic_agent import ActorCriticAgent
from mdm.policies.agent_policy import HierarchicalLatentAgentPolicy, LatentAgentPolicy
from mdm.policies.predefined_policy import PredefinedPolicy
from mdm.training.gym_driver import collect_data, GymEpisodeDriver
from mdm.utils.gym_wrappers import CacheLastStepEnv
from mdm.utils.torch_tools import to_tensors, to_np, masked_mean, FreezeParameters
from mdm.utils.utils import *
from mdm.utils.gym_nav2d_tools import *


def train_model(cfg, model, opt_model, r_max_agents, goal_seeking_agents, collect_fn, eval_env, test_driver,
                train_driver, logger, log_videos: bool = True, video_env=None):
    model_train_steps = cfg['trainer']['model_train_steps']
    freeze_model = cfg['trainer']['freeze_model'] if cfg['trainer']['freeze_model'] > 0 else sys.maxsize
    stop_collect = cfg['trainer']['stop_collect'] if cfg['trainer']['stop_collect'] > 0 else sys.maxsize
    logger.start_session()
    model.prepare_for_training()

    # actor_params = np.sum(
    #    record_parameters(model.r_max_agents[0][0].actor_net, reduction_fn=lambda x: x.ravel().mean()))
    logger.n_log_calls = 0

    for i_step in tqdm(range(cfg['trainer']['n_train_steps']), desc='Training Progress'):
        batch = train_driver.interact(cfg['trainer']['d_batch'])
        batch = to_tensors(batch, device='cuda')
        batch = prepare_data(batch)

        # _new_actor_params = np.sum(
        #    record_parameters(model.r_max_agents[0][0].actor_net, reduction_fn=lambda x: x.ravel().mean()))
        # print(actor_params - _new_actor_params)
        # actor_params = _new_actor_params

        if cfg['trainer']['subtrajectory_len'] > 0:
            # model_batch = valid_subtrajectories(batch, cfg['trainer']['subtrajectory_len'])
            # model_batch = valid_subtrajectories_unbiased(batch, 15)
            # model_batch = valid_subtrajectories_unbiased_fast(batch, cfg['trainer']['subtrajectory_len'])
            model_batch = valid_subtrajectories_2(batch, cfg['trainer']['subtrajectory_len'])
        else:
            model_batch = batch

        model.train()
        agent_eval_mode(r_max_agents + goal_seeking_agents)
        if i_step < freeze_model:
            train_losses, pred, targets = model.train_step(model_batch, opt_model,
                                                           model_steps=model_train_steps,
                                                           logger=logger)
        else:
            train_losses, pred, targets = model.eval_step(model_batch, model_steps=model_train_steps,
                                                          force_warmup=[-1 for _ in
                                                                        range(model.levels)])
            print('freezing model')

        # all_losses = torch.stack([v for v in train_losses.values()]).mean()
        # make_dot(train_losses['o_0'], dict(model.named_parameters())).view()
        # quit()
        # make_dot(train_losses['term_0'], dict(model.named_parameters())).view()

        logger.log(to_np(train_losses), Scope.TRAIN(), i_step)

        # log latent state and observation distances
        if i_step % cfg['trainer']['eval_interval'] == 0:
            for level, pred_level in enumerate(pred):
                plot_latent_state_differences(pred_level, logger, i_step, major_scope=f'model/{level}')

        # train model in observation mode
        # train_losses = model.train_step(model_batch, opt_model, model_steps=model_train_steps, learn_states=True)
        # logger.log(_to_np(train_losses), Scope.TRAIN() / 'observation_mode', i_step)

        # p0 = torch.sum(torch.stack([p.mean() for p in model.parameters()]))
        # print(f'Model: {logger.n_log_calls}')
        logger.n_log_calls = 0

        # train agents
        sample_model = cfg['trainer']['sample_model_during_agent_training']
        sample_agents = cfg['trainer']['sample_agent_during_agent_training']
        if i_step % cfg['trainer']['agent_train_interval'] == 0:
            agent_train_mode(r_max_agents + goal_seeking_agents)
            agent_model_steps = cfg['trainer']['agent_model_steps']

            for level in range(model.levels):
                # pessimistic_model_training(model, opt_model, pred, targets, level,
                #                           1, sample_model, sample_agents,
                #                           i_step, logger)

                train_rmax_agent(agent_model_steps, cfg, eval_env, i_step, level, logger, model, pred,
                                 sample_agents, sample_model, targets)

                # GSA training
                if level < model.levels - 1:
                    # train_goal_seeking_agent_rand(model, level, pred, targets, eval_env, cfg, i_step, logger,
                    #                              sample_model, sample_agents)

                    # pred_mode, _, targets_mode = model.forward_all_levels(model_batch,
                    #                                                      warmup_steps=[-1 for _ in model.rssm_modules],
                    #                                                      model_steps=[-1 for _ in model.rssm_modules],
                    #                                                      sample_state=False, sample_output=False)
                    # train_goal_seeking_agent_same_level(model, level, pred_mode, targets_mode, eval_env, cfg, i_step,
                    #                                    logger, sample_agents)
                    # train_goal_seeking_agent_goals_above(model, level, pred, targets, eval_env, cfg, i_step, logger,
                    #                                     sample_agents)
                    train_goal_seeking_agent_goals_above_with_agent(model, level, pred, targets, eval_env, cfg, i_step,
                                                                    logger, sample_agents)
                    # train_goal_seeking_agent_one_step(model, level, pred, targets, eval_env, cfg, i_step, logger,
                    #                                  sample_agents)

                if level > 0:
                    pass
                    # imitation_learning(model, pred, targets, level, cfg['trainer']['agent_model_steps'][level],
                    #                  sample_model, sample_agents, i_step, logger)
                    # decoding_err_diff(model, pred, targets, level, cfg['trainer']['agent_model_steps'][level],
                    #                  sample_model, sample_agents, i_step, logger)

        # p1 = torch.sum(torch.stack([p.mean() for p in model.parameters()]))
        # assert np.isclose((p0 - p1).detach().cpu().numpy(), 0), 'Model parameters changed during agent training!'

        if i_step % cfg['trainer']['collect_interval'] == 0 and i_step < stop_collect:
            collect_fn(explore=True)

        # eval =========================================================================================================
        if i_step % cfg['trainer']['eval_interval'] == 0:
            with torch.no_grad():
                agent_eval_mode(r_max_agents + goal_seeking_agents)
                model.eval()

                # model
                trajs_orig = test_driver.interact(cfg['trainer']['d_batch'])
                # trajs_orig = add_no_ops(trajs_orig, model.strides[1])
                batch = to_tensors(trajs_orig, model.device)
                batch = prepare_data(batch)
                eval_steps = [-1] + cfg['trainer']['model_train_steps'][1:]
                eval_losses, pred, targets = model.eval_step(batch, model_steps=eval_steps, sample_state=False,
                                                             sample_output=False,
                                                             force_warmup=[-1 for _ in range(model.levels)])
                logger.log(to_np(eval_losses), Scope.TEST(), i_step)
                log_prediction_error_plot(batch, cfg, i_step, logger, model)

                # flat agent
                eval_mem_flat = []
                eval_env.reset()
                flat_policy = LatentAgentPolicy(r_max_agents[0][0], model, explore=False)
                d = GymEpisodeDriver(eval_env, flat_policy)
                d.interact(10, eval_mem_flat)
                # eval_mem_flat += collect_data(eval_env, cfg['eval']['eval_steps'], flat_policy)
                logger.log(trajectory_statistics(eval_mem_flat), Scope.TEST() / 'flat_agent/', i_step)

                # hierarchical agent
                eval_mem_hierarchical = []
                eval_env.reset()
                hierarchical_policy = HierarchicalLatentAgentPolicy(model, explore=False)
                d = GymEpisodeDriver(eval_env, hierarchical_policy)
                d.interact(10, eval_mem_hierarchical)
                # eval_mem_hierarchical += collect_data(eval_env, cfg['eval']['eval_steps'], hierarchical_policy)
                logger.log(trajectory_statistics(eval_mem_hierarchical), Scope.TEST() / 'hierarchical_agent/', i_step)

                # print(global_data_storage['n_resets'])
                # quit()

                # record video with hierarchical policy
                if video_env:
                    record_episode(cfg, i_step, logger, model, video_env)

                if len(model.goal_seeking_agents) > 0 and env_class_is(eval_env, Nav2dEnv):
                    pass
                    # probe L0 goal seeking agent
                    # nav2d_gsa_plot(cfg, eval_env, eval_steps, i_step, logger, model, train_driver)

                if env_class_is(eval_env, Nav2dEnv):
                    latent_state_pca_plot(model, eval_env, eval_steps, i_step, logger)

                if env_class_is(eval_env, Nav2dEnv):
                    returns_flat = [t['r'].sum() for t in eval_mem_flat]
                    worst_flat = np.argmin(returns_flat)
                    returns_hierarchical = [t['r'].sum() for t in eval_mem_hierarchical]
                    worst_hierac = np.argmin(returns_hierarchical)
                    with TempFigure() as fig:
                        fig, anim = visualize_overlaid_trajectories(eval_mem_flat[worst_flat], figure=fig)
                        vid_flat = anim_to_vid(anim)
                        vid_flat.name = 'flat_agent_acting'
                    with TempFigure() as fig:
                        fig, anim = visualize_overlaid_trajectories(eval_mem_hierarchical[worst_hierac], figure=fig)
                        vid_hierarchical = anim_to_vid(anim)
                        vid_hierarchical.name = 'hierarchical_agent_acting'
                    logger.log({'flat_agent': vid_flat, 'hierarchical_agent': vid_hierarchical},
                               Scope.TEST() / 'agent_action_videos/', i_step)

                    # model l0 simulation plot, works only for nav2d env
                    warmup_steps = model.maybe_sample_warmup_steps(training_data=batch, model_steps=model_train_steps,
                                                                   warmup_steps=cfg['eval']['warmup_steps'])
                    pred, pred_ema, _ = model.forward_all_levels(ground_truth_trajectory=batch,
                                                                 warmup_steps=warmup_steps,
                                                                 model_steps=model_train_steps,
                                                                 sample_state=True,
                                                                 sample_output=False)
                    trajs_orig_pad = trajectories_from_simulation(batch)  # to get padded version of orig trajectories
                    trajs_sim = trajectories_from_simulation(pred[0])
                    with TempFigure() as fig:
                        fig, anim = visualize_overlaid_trajectories(trajs_sim[0], trajs_orig_pad[0], figure=fig)
                        vid = anim_to_vid(anim)
                        vid.name = 'model_sim'
                    logger.log({'model': vid}, Scope.TEST() / 'model_prediction_video/0', i_step)

                    if model.levels > 1:
                        trajs_sim = trajectories_from_simulation(pred[1], model, 1)
                        # truncate to original trajectory length
                        l_traj_orig = trajs_orig_pad[0]['o'].shape[0]
                        trajs_sim = [{k: v[:l_traj_orig] for k, v in traj.items()} for traj in trajs_sim]
                        with TempFigure() as fig:
                            fig, anim = visualize_overlaid_trajectories(trajs_sim[0], trajs_orig_pad[0], figure=fig)
                            vid = anim_to_vid(anim)
                            vid.name = 'model_sim'
                        logger.log({'model': vid}, Scope.TEST() / 'model_prediction_video/1', i_step)
                # log model and agent params
                # log_params(model, logger, Scope.PARAMETERS() / 'model', time_step=i_step)
                # for i_ag, ag in enumerate(r_max_agents):
                #    if ag is None: continue
                #    log_params(ag[0], logger, Scope.PARAMETERS() / f'agent/r_max_agent_{i_ag}', time_step=i_step)
                # for i_ag, ag in enumerate(goal_seeking_agents):
                #    if ag is None: continue
                #    log_params(ag[0], logger, Scope.PARAMETERS() / f'agent/goal_seeking_agent_{i_ag}', time_step=i_step)

                # for debugging: check that reward location is always the same
                # r_pos = []
                # for traj in train_driver.memory:
                #    r_pos.append(traj['o'][:, 2:4])
                # r_pos = np.concatenate(r_pos, axis=0)
                # print(f'{r_pos.mean(axis=0)}({r_pos.std(axis=0)})')

        if i_step % cfg['trainer']['checkpoint_interval'] == 0:
            timestamp = time.time_ns()
            pid = os.getpid()
            model_path = f'.checkpoint_model_weights_{pid}_{timestamp}_{logger.run_id}.ptmdl'
            torch.save(model.state_dict(), model_path)
            cpt_file = InMemoryFile.consume_file(model_path, new_name='checkpoint')
            logger.log_file(cpt_file, Scope.DATA() / 'weights')


def plot_latent_state_differences(pred, logger, i_step, major_scope):
    def neg_mse(a, b):
        return -torch.mean((a - b) ** 2, dim=-1, keepdim=True)

    sim_fns = [neg_mse, ActorCriticAgent.goal_similarity]
    n_rows = 1
    n_cols = len(sim_fns)
    obs = pred['o']
    latents = pred['s_embedding']

    with TempFigure(figsize=(4 * n_cols, 3)) as fig:
        for i_plot, sim_fn in enumerate(sim_fns):
            ax = fig.add_subplot(n_rows, n_cols, i_plot + 1)
            diff_o, diff_latent = [], []
            for t in range(len(obs)):
                diff_o.append(sim_fn(obs[t], obs[-1]))
                diff_latent.append(sim_fn(latents[t], latents[-1]))
            diff_o = torch.stack(diff_o).detach().cpu().numpy().mean(axis=(1, 2))
            diff_latent = torch.stack(diff_latent).detach().cpu().numpy().mean(axis=(1, 2))

            ax.plot(diff_o, label='obs difference')
            ax.plot(diff_latent, label='latent difference')
            ax.set_title(f'{sim_fn.__qualname__}')
        ax.legend()  # legend only on last plot
        logger.log_plot(fig_to_img(fig), Scope.TRAIN() / f'{major_scope}/time_step_differences', i_step)


def decoding_err_diff(model, pred, targets, level, n_steps, sample_model, sample_agents, i_step, logger):
    rma, rma_act_opt, rma_crit_opt = model.r_max_agents[level]
    gsa, gsa_act_opt, gsa_crit_opt = model.goal_seeking_agents[level - 1]
    # omit first time step as we don't get a starting and end state for below model
    start_state_lvl, start_state_mask = filter_mem_state_seq_to_batch(pred[level], targets[level]['mask'])
    start_state_lvl = rssm_detach_state(*start_state_lvl)
    # take every k-th model state from below trajectory
    flt = PickOneUpwardsFilter(model.strides[level], offset=-1)
    # pick relevant keys and stack lists so upwards filter can process them
    pred_flt_below = {k: torch.stack(v) for k, v in pred[level - 1].items() if k in rssm_state_keys()}
    # filter relevant time steps
    pred_flt_below = {k: flt(v) for k, v in pred_flt_below.items()}
    mask_flt_below = flt(targets[level - 1]['mask'])
    # unstack output of filtering process into lists since we need this format for rssm_state_seq_to_batch() function
    pred_flt_below = {k: list(v.unbind(0)) for k, v in pred_flt_below.items()}
    # fold time into batch dim to do rollout starting from every batch item and time step at once
    start_state_below, start_state_below_mask = filter_mem_state_seq_to_batch(pred_flt_below, mask_flt_below)
    start_state_below = rssm_detach_state(*start_state_below)
    # do one step rollout for abstract agent
    rma_simulation = rma.act_in_sim(start_state_lvl, sim_env=model, n_steps=n_steps,
                                    sample_states=sample_model, sample_actions=sample_agents,
                                    explore=False, reconstruct=True)

    assert len(rma_simulation['model']['o']) == n_steps

    # get goals from the simulation and to a gsa simulation on level below
    gsa_model_mem, gsa_agent_mem = {}, {}
    state = start_state_below
    for g in rma_simulation['model']['o']:
        gsa_simulation = gsa.act_in_sim(env_start_state=state, sim_env=model, n_steps=model.strides[level],
                                        explore=False, goal=g,
                                        sample_states=sample_model, sample_actions=sample_agents,
                                        env_memory=gsa_model_mem, agent_memory=gsa_agent_mem,
                                        reconstruct=True)
        state = gsa_simulation['model_state']

    # We should prefer terminals collected by gsa over the ones from abstract rma since the gsa is one level closer
    # to ground truth. We compute the mask starting with the start_state_below_mask as first time step.
    a_gsa_mask = compute_mask(gsa_model_mem['terminal'], first_step_mask=start_state_below_mask)
    a_filtered = model.upwards_filters[level]['a'](torch.stack(gsa_agent_mem['a']), mask=a_gsa_mask)
    # a_filtered = model.actions_up(gsa_agent_mem['a'], level=level, mask=a_gsa_mask)

    assert a_filtered.shape[0] == n_steps

    rma_a = torch.stack(rma_simulation['agent']['a'])
    uncertainty_abstract = model.upwards_filters[level]['a'].decoding_uncertainty(rma_a, a_gsa_mask)
    uncertainty_filtered = model.upwards_filters[level]['a'].decoding_uncertainty(a_filtered, a_gsa_mask)

    message = {'uncertainty_abstract': uncertainty_abstract.mean(), 'uncertainty_filtered': uncertainty_filtered.mean()}
    logger.log(to_np(message), Scope.TRAIN() / f'r_max_agent/{level}/', i_step)

    """
    # use mask of filtered actions to compute loss mask as again it should be more reliable (see comment above)
    loss_mask = model.upwards_filters[level]['mask'](a_gsa_mask)
    # TODO: test if a_filtered should be detached or not
    # measure difference between original abstract rma action and filtered up one
    diff = (torch.stack(rma_simulation['agent']['a']) - a_filtered) ** 2
    diff = torch.sum(diff * (1 - loss_mask))
    # do train step
    rma_act_opt.zero_grad(set_to_none=True)
    diff.backward()
    torch.nn.utils.clip_grad_norm_(rma.parameters(), 100.0)
    rma_act_opt.step()
    # logging
    message = {'imitation_learning_loss': diff}
    logger.log(to_np(message), Scope.TRAIN() / f'r_max_agent/{level}/', i_step)
    """


def imitation_learning(model, pred, targets, level, n_steps, sample_model, sample_agents, i_step, logger):
    rma, rma_act_opt, rma_crit_opt = model.r_max_agents[level]
    gsa, gsa_act_opt, gsa_crit_opt = model.goal_seeking_agents[level - 1]
    # omit first time step as we don't get a starting and end state for below model
    start_state_lvl, start_state_mask = filter_mem_state_seq_to_batch(pred[level], targets[level]['mask'])
    start_state_lvl = rssm_detach_state(*start_state_lvl)
    # take every k-th model state from below trajectory
    flt = PickOneUpwardsFilter(model.strides[level], offset=-1)
    # pick relevant keys and stack lists so upwards filter can process them
    pred_flt_below = {k: torch.stack(v) for k, v in pred[level - 1].items() if k in rssm_state_keys()}
    # filter relevant time steps
    pred_flt_below = {k: flt(v) for k, v in pred_flt_below.items()}
    mask_flt_below = flt(targets[level - 1]['mask'])
    # unstack output of filtering process into lists since we need this format for rssm_state_seq_to_batch() function
    pred_flt_below = {k: list(v.unbind(0)) for k, v in pred_flt_below.items()}
    # fold time into batch dim to do rollout starting from every batch item and time step at once
    start_state_below, start_state_below_mask = filter_mem_state_seq_to_batch(pred_flt_below, mask_flt_below)
    start_state_below = rssm_detach_state(*start_state_below)
    # do one step rollout for abstract agent
    rma_simulation = rma.act_in_sim(start_state_lvl, sim_env=model, n_steps=n_steps,
                                    sample_states=sample_model, sample_actions=sample_agents,
                                    explore=False, reconstruct=True)

    assert len(rma_simulation['model']['o']) == n_steps

    # get goals from the simulation and to a gsa simulation on level below
    gsa_model_mem, gsa_agent_mem = {}, {}
    state = start_state_below
    for g in rma_simulation['model']['o']:
        gsa_simulation = gsa.act_in_sim(env_start_state=state, sim_env=model, n_steps=model.strides[level],
                                        explore=False, goal=g,
                                        sample_states=sample_model, sample_actions=sample_agents,
                                        env_memory=gsa_model_mem, agent_memory=gsa_agent_mem,
                                        reconstruct=True)
        state = gsa_simulation['model_state']

    # We should prefer terminals collected by gsa over the ones from abstract rma since the gsa is one level closer
    # to ground truth. We compute the mask starting with the start_state_below_mask as first time step.
    a_filtered_mask = compute_mask(gsa_model_mem['terminal'], first_step_mask=start_state_below_mask)
    a_filtered = model.upwards_filters[level]['a'](torch.stack(gsa_agent_mem['a']), mask=a_filtered_mask)
    # a_filtered = model.actions_up(gsa_agent_mem['a'], level=level, mask=a_filtered_mask)

    assert a_filtered.shape[0] == n_steps

    # use mask of filtered actions to compute loss mask as again it should be more reliable (see comment above)
    loss_mask = model.upwards_filters[level]['mask'](a_filtered_mask)
    # TODO: test if a_filtered should be detached or not
    # measure difference between original abstract rma action and filtered up one
    diff = (torch.stack(rma_simulation['agent']['a']) - a_filtered) ** 2
    diff = torch.sum(diff * (1 - loss_mask))
    # do train step
    rma_act_opt.zero_grad(set_to_none=True)
    diff.backward()
    torch.nn.utils.clip_grad_norm_(rma.parameters(), 100.0)
    rma_act_opt.step()
    # logging
    message = {'imitation_learning_loss': diff}
    logger.log(to_np(message), Scope.TRAIN() / f'r_max_agent/{level}/', i_step)


def pessimistic_model_training(model, opt_model, pred, targets, level, n_steps, sample_model, sample_agents, i_step,
                               logger):
    rma, rma_act_opt, rma_crit_opt = model.r_max_agents[level]
    world = model.rssm_modules[level]

    # fold batch dimension into time dimension to start simulation for all time steps in parallel
    start_state, start_state_mask = filter_mem_state_seq_to_batch(pred[level], targets[level]['mask'])
    start_state = rssm_detach_state(*start_state)

    model_mem = {}
    last_state = start_state
    with FreezeParameters([rma]):  # freeze agent parameters as we want to train the model
        for t in range(n_steps):
            # prepare current RSSM state for agent and sample action
            agent_o = rma.o_from_state(last_state)
            a_dist, a = rma(agent_o, sample=sample_agents, explore=True)
            # get next RSSM state
            current_state = world(a=a, last_state=last_state, use_posterior=False, sample_state=sample_model)
            # predict reward and temrinal flag of current state
            pred = world.decode(current_state[-1], sample=True, reconstruct_observation=False)
            # increment state and store predictions
            last_state = current_state
            append_memory(model_mem, **pred)
    model_mem = {k: torch.stack(v) for k, v in model_mem.items()}

    # rewards from agent actions are always assumed to be negative in this training routine
    pessimistic_r_target = torch.full_like(model_mem['r'], 0.0)
    # compute mask from simulated terminal flags and terminal flag of start state
    valid = (1 - compute_mask(model_mem['terminal'], first_step_mask=start_state_mask))
    # compute loss
    r_dist = world.r_decoder.dist(model_mem['r_dist'])
    loss = 0.1 * model._neg_log_prob(r_dist, pessimistic_r_target, valid)

    # update parameters
    opt_model.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    opt_model.step()

    # logging
    message = {'pessimistic_model_training_r': loss}
    logger.log(to_np(message), Scope.TRAIN() / f'model/{level}/', i_step)


def train_rmax_agent(agent_model_steps, cfg, eval_env, i_step, level, logger, model, pred, sample_agents, sample_model,
                     targets):
    r_max_agent, r_max_actor_opt, r_max_critic_opt = model.r_max_agents[level]
    # use all time steps of teacher forcing rollout from model as starting point
    start_state_lvl, start_state_mask = filter_mem_state_seq_to_batch(pred[level], targets[level]['mask'])
    # prevent gradient flow into the start state
    start_state_lvl = rssm_detach_state(*start_state_lvl)
    abstract_level = level > 0
    r_max_simulation = r_max_agent.act_in_sim(start_state_lvl, model, agent_model_steps[level],
                                              sample_states=sample_model, sample_actions=sample_agents,
                                              explore=True, expl_noise=0.0, reconstruct=True)
    r_max_losses = r_max_agent.update_step(r_max_simulation['agent'],
                                           first_step_mask=start_state_mask.unsqueeze(0),
                                           actor_optimizer=r_max_actor_opt,
                                           critic_optimizer=r_max_critic_opt,
                                           logger=logger)
    logger.log(to_np(r_max_losses), Scope.TRAIN() / f'r_max_agent/{level}/', i_step)
    # debugging and inspection
    if i_step % cfg['trainer']['eval_interval'] == 0 and env_class_is(eval_env, Nav2dEnv):
        plot_value_function(eval_env, model, r_max_agent, level, logger, i_step)
        plot_rewards(eval_env, model, level, logger, i_step)
    if level > 0 and i_step % cfg['trainer']['eval_interval'] == 0 and env_class_is(eval_env, Nav2dEnv):
        plot_goals(r_max_simulation, model, logger, i_step, level)
    if GlobalLogger.can_log('sanity_check_goal_computation', i_step) and level < model.levels - 1:
        plot_goals_fancy(i_step, level, logger, model, r_max_agent, r_max_simulation)

    if level < model.levels - 1:
        pass
        # update_model_chunk_distance(i_step, level, logger, model, r_max_simulation)


def plot_goals_fancy(i_step, l, logger, model, r_max_agent, r_max_simulation):
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
            # similarities[i, j] = torch.nn.functional.cosine_similarity(g, g_other, dim=-1).mean()
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
        logger.log_plot(fig_to_img(fig),
                        Scope.TRAIN() / f'r_max_agent_fake_goals/{l}/her_step_reward',
                        i_step)


def abstract_model_training_static(abstract_train_driver, model, optimizer):
    # get batch
    batch = abstract_train_driver.interact(128)
    batch = to_tensors(batch, 'cuda')
    batch = {k.replace('_abstract', ''): v for k, v in batch.items() if k.endswith('_abstract')}
    pred_upper, _, _ = model.forward_static(batch, start_state=None, level=1)
    # compute loss
    mask_abstract = compute_mask(batch['terminal'])
    loss_abstract = model.rssm_loss(pred_upper, pred_upper, batch, mask_abstract, 1.0, level=1)
    # update model
    optimizer.zero_grad(set_to_none=True)
    loss_abstract['total'].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    loss_abstract = {f'abstract_{k}': v for k, v in loss_abstract.items()}
    return loss_abstract


def abstract_model_training(abstract_train_driver, model, optimizer, cfg):
    # get batch
    batch = abstract_train_driver.interact(cfg['trainer']['d_batch'])
    batch = to_tensors(batch, 'cuda')
    # batch = valid_subtrajectories_2(batch, cfg['trainer']['subtrajectory_len'] * model.strides[1])
    # generate up to date latent states from lower level
    pred, _, _ = model.forward_static(batch, start_state=None, level=0)
    # use batch data and up to date world model states to generate inputs for higher level
    batch_upper = model.filter_up(o=pred['s_embedding'],
                                  r=batch['r'],
                                  terminal=batch['terminal'],
                                  level=1)
    # this is the whole reason for this training routine: replace upper level actions with recorded ones from agent
    batch_upper['a'] = batch['a_abstract']
    # commence rollout generation
    pred_upper, _, _ = model.forward_static(batch_upper, start_state=None, level=1)
    # compute loss
    mask_abstract = compute_mask(batch_upper['terminal'])
    loss_abstract = model.rssm_loss(pred_upper, pred_upper, batch_upper, mask_abstract, 1.0, level=1)
    # update model
    optimizer.zero_grad(set_to_none=True)
    loss_abstract['total'].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    loss_abstract = {f'abstract_{k}': v for k, v in loss_abstract.items()}
    return loss_abstract


def train_goal_seeking_agent_one_step(model, level, pred, targets, eval_env, cfg, i_step, logger, sample_agents):
    n_goals = 3
    chunk_size = model.strides[level + 1]
    goal_agent, goal_actor_opt, goal_critic_opt = model.goal_seeking_agents[level]
    trajectories = pred[level]
    mask = targets[level]['mask']

    if len(trajectories['a']) <= chunk_size * n_goals:
        return

    # get starting states, leave enough states out at the end to collect goals
    start_state, start_state_mask = filter_mem_state_seq_to_batch(trajectories, mask, i_start=0, i_end=-chunk_size)
    start_state = rssm_detach_state(*start_state)
    # i_end == len(sequence)
    goal_state, goal_state_mask = filter_mem_state_seq_to_batch(trajectories, mask, i_start=chunk_size)
    goal_state = rssm_detach_state(*goal_state)

    explore = random.random() > 0.5
    if explore:
        exploration_noise = random.random() * 0.3
        sample_model = random.random() > 0.5
        print(f'explore ', end=': ')
    else:
        exploration_noise = 0.0
        sample_model = False
        print(f'        ', end=': ')

    goal = goal_agent.o_from_state(goal_state)
    goal_simulation = goal_agent.act_in_sim(env_start_state=start_state, sim_env=model, n_steps=chunk_size,
                                            explore=explore, goal=goal, sample_states=sample_model,
                                            sample_actions=sample_agents, expl_noise=exploration_noise,
                                            reconstruct=True)
    loss = goal_agent.update_step(goal_simulation['agent'],
                                  first_step_mask=start_state_mask,
                                  actor_optimizer=goal_actor_opt,
                                  critic_optimizer=goal_critic_opt)

    logger.log(to_np(loss), Scope.TRAIN() / f'goal_seeking_agent/{level}/', i_step)

    if i_step % cfg['trainer']['eval_interval'] == 0:
        # add start state and goal state to comparison
        goal_simulation['model']['s_embedding'].insert(0, start_state[5])
        goal_simulation['model']['s_embedding'].append(goal)
        # same for observation
        start_state_obs = seq_to_batch(trajectories, i_start=0, i_end=-chunk_size)['o']
        goal_state_obs = seq_to_batch(trajectories, i_start=chunk_size)['o']
        goal_simulation['model']['o'].insert(0, start_state_obs)
        goal_simulation['model']['o'].append(goal_state_obs)

        plot_latent_state_differences(goal_simulation['model'], logger, i_step,
                                      major_scope=f'goal_seeking_agent/{level}')


def train_goal_seeking_agent_goals_above_with_agent(model, level, pred, targets, eval_env, cfg, i_step, logger,
                                                    sample_agents):
    n_groundtruth_goals = 2
    n_generated_goals = 2
    chunk_size = model.strides[level + 1]
    n_goals_total = len(targets[level + 1]['o'])
    gsa, actor_opt, critic_opt = model.goal_seeking_agents[level]
    rma, _, _ = model.r_max_agents[level + 1]

    if len(pred[level]['a']) <= chunk_size * n_groundtruth_goals + 1:
        return

    # cut out a window of gsa start states for first chunk, for n_groundtruth_goals=2 and chunk_size=2 this means:
    # B0: [a0] a1 [a2] a3 [a4] a5 a6 a7 a8        [a0] [a2] [a4] a6 a8        [a0] [a2] [a4]
    # B1: [b0] b1 [b2] b3 [b4] b5 b6 b7 b8   ->   [b0] [b2] [b4] b6 b8   ->   [b0] [b2] [b4]
    # B2: [c0] c1 [c2] c3 [c4] c5 c6 c7 c8        [c0] [c2] [c4] c6 c8        [c0] [c2] [c4]
    # filter out every k-th step from current level predictions (for agent start states) and from targets (for masking)
    flt = PickOneUpwardsFilter(window_size=chunk_size, offset=-1)
    pred_flt = {k: flt(torch.stack(v)) for k, v in pred[level].items()}
    targets_flt = {k: flt(v) for k, v in targets[level].items()}
    # cut off last n_groundtruth_goals steps as they are needed as goals
    pred_flt = {k: v[:-n_groundtruth_goals] for k, v in pred_flt.items()}
    targets_flt = {k: v[:-n_groundtruth_goals] for k, v in targets_flt.items()}
    # fold time into batch dim, this yields one time step with shape (1, BxT, ...)
    # [a0]
    # [b0]
    # [c0]
    # [a2]
    # [b2]
    # [c2]
    # ...
    pred_flt = {k: v.reshape(v.shape[0] * v.shape[1], *v.shape[2:]) for k, v in pred_flt.items()}
    targets_flt = {k: v.reshape(v.shape[0] * v.shape[1], *v.shape[2:]) for k, v in targets_flt.items()}
    # filter out relevant keys that belong to RSSM state
    first_start_state = {k: v for k, v in pred_flt.items() if k in rssm_state_keys()}
    # remove keys to obtain tuple format again
    first_start_state = rssm_remove_labels(first_start_state)
    first_step_mask = targets_flt['mask']

    # goals are taken from above level's model predictions
    goals = []
    for i in range(n_groundtruth_goals):
        t_start = i + 1  # for each chunk, goals are 1 index after start states
        t_end = n_goals_total - n_groundtruth_goals + i + 1  # indexing is exclusive t_end, so add 1 at the end
        # pick goals from above level
        goal = torch.stack(pred[level + 1]['o'][t_start:t_end])
        # fold time into batch dimension
        goal = goal.reshape(goal.shape[0] * goal.shape[1], *goal.shape[2:])
        goals.append(goal.detach())

    if n_generated_goals > 0:
        # generate rollout to produce diverse in distribution goals from the higher level agent beyond groundtruth goals
        # use state belonging to last goal to start agent rollout from
        # t_start = n_groundtruth_goals
        # t_end = n_goals_total
        rollout_state = {k: torch.stack(v[t_start:t_end]) for k, v in pred[level + 1].items() if k in rssm_state_keys()}
        # fold time into batch dimension
        rollout_state = {k: v.reshape(v.shape[0] * v.shape[1], *v.shape[2:]) for k, v in rollout_state.items()}
        rollout_state = rssm_remove_labels(rollout_state)
        # produce additional goals
        simulation = rma.act_in_sim(env_start_state=rollout_state, sim_env=model, n_steps=n_generated_goals,
                                    explore=True, expl_noise=0.3)
        # add simulation goals to the goal memory
        goals += [goal.detach() for goal in simulation['model']['o']]

    gsa_agent_mem, gsa_model_mem, gsa_losses = {}, {}, {}
    start_state = first_start_state
    start_state_mask = first_step_mask
    for i_goal, goal in enumerate(goals):
        # vary step count to give the agent some slack sometimes
        n_steps = random.randint(math.ceil(0.5 * chunk_size), 3 * chunk_size)
        # n_steps = random.randint(chunk_size, chunk_size + 1)
        #n_steps = chunk_size

        explore = random.random() > 0.75
        if explore:
            expl_noise = random.random() * 0.5
            sample_model = random.random() > 0.33
            print('explore ', end='')
        else:
            expl_noise = 0.0
            sample_model = False
            print('        ', end='')

        start_state = rssm_detach_state(*start_state)  # prevent gradients to flow into previous chunk
        goal_simulation = gsa.act_in_sim(env_start_state=start_state, sim_env=model, n_steps=n_steps,
                                         explore=explore, goal=goal, sample_states=sample_model,
                                         sample_actions=sample_agents, expl_noise=expl_noise, reconstruct=True)
        agent_mem = goal_simulation['agent']
        loss = gsa.update_step(agent_mem,
                               first_step_mask=start_state_mask,
                               actor_optimizer=actor_opt,
                               critic_optimizer=critic_opt)
        # loss = {f'{k}_{i_goal}': v for i, (k, v) in enumerate(loss.items())}
        # gsa_losses.update(loss)
        for k in loss:
            val = gsa_losses.get(k, 0.0)
            gsa_losses[k] = val + loss[k]

        extend_memory(gsa_agent_mem, goal_simulation['agent'])
        extend_memory(gsa_model_mem, goal_simulation['model'])

        # set new start state
        start_state = goal_simulation['model_state']
        # remember all trajectories that had invalid start states or became invalid throughout current chunk
        mask = torch.stack([start_state_mask] + goal_simulation['model']['terminal'])
        # use similar technique as in compute_mask function, but don't shift terminal states
        valid = torch.cumprod(1.0 - mask.to(torch.float32), dim=0)
        start_state_mask = (1.0 - valid)[-1]

    gsa_losses = {k: v / n_groundtruth_goals for k, v in gsa_losses.items()}

    logger.log(to_np(gsa_losses), Scope.TRAIN() / f'goal_seeking_agent/{level}/', i_step)

    if i_step % cfg['trainer']['eval_interval'] == 0 and env_class_is(eval_env, MazeEnv):
        # choose random batch indices to plot
        indices = [random.randint(0, first_start_state[-1].shape[0] - 1) for _ in range(2)]

        # get trajectory observations
        obs = []
        # first time step is extra as it's not recorded in agent simulation memory
        pred = model.rssm_modules[level].decode(first_start_state[-1], sample=False, reconstruct_observation=True)
        obs.append(pred['o'])
        obs += gsa_model_mem['o']
        obs = torch.stack(obs).detach().cpu().numpy().swapaxes(0, 1)[indices].swapaxes(0, 1)

        # decode goal state observations
        goal_obs = []
        for i_g, g in enumerate(goals):
            pred = model.rssm_modules[level].decode(g, sample=False, reconstruct_observation=True)
            goal_obs.append(pred['o'])
        goal_obs = torch.stack(goal_obs).detach().cpu().numpy().swapaxes(0, 1)[indices].swapaxes(0, 1)

        with TempFigure(figsize=(10, 10)) as fig:
            plot_maze_env(eval_env, obs, goal_obs, fig.gca())
            logger.log_plot(fig_to_img(fig),
                            Scope.TRAIN() / f'goal_seeking_agent/{level}/goal_seeking_train_performance', i_step)

    if i_step % cfg['trainer']['eval_interval'] == 0 and level == 0 and env_class_is(eval_env, Nav2dEnv):
        plot_goal_conditioned_value_function(eval_env, model, gsa, level, logger, i_step, 3, 3)
        plot_goal_seeking_performance(gsa, gsa_model_mem, gsa_agent_mem, goals, first_start_state, model,
                                      [chunk_size for _ in goals], logger, i_step, level)


def train_goal_seeking_agent_goals_above(model, level, pred, targets, eval_env, cfg, i_step, logger, sample_agents):
    n_goals = 3
    chunk_size = model.strides[level + 1]
    goal_agent, goal_actor_opt, goal_critic_opt = model.goal_seeking_agents[level]
    n_goals_total = len(targets[level + 1]['o'])

    if len(pred[level]['a']) <= chunk_size * n_goals + 1:
        return

    # filter out every k-th step from current level predictions (for agent start states) and targets (for masking)
    flt = PickOneUpwardsFilter(window_size=chunk_size, offset=-1)
    pred_flt = {k: flt(torch.stack(v)) for k, v in pred[level].items()}
    targets_flt = {k: flt(v) for k, v in targets[level].items()}
    # cut off last n_goals steps as they are needed as goals
    pred_flt = {k: v[:-n_goals] for k, v in pred_flt.items()}
    targets_flt = {k: v[:-n_goals] for k, v in targets_flt.items()}
    # fold time into batch dim
    pred_flt = {k: v.reshape(v.shape[0] * v.shape[1], *v.shape[2:]) for k, v in pred_flt.items()}
    targets_flt = {k: v.reshape(v.shape[0] * v.shape[1], *v.shape[2:]) for k, v in targets_flt.items()}
    # filter out relevant keys that belong to RSSM state
    first_start_state = {k: v for k, v in pred_flt.items() if k in rssm_state_keys()}
    # remove keys to obtain tuple format again
    first_start_state = rssm_remove_labels(first_start_state)
    first_step_mask = targets_flt['mask']

    # goals are taken from above level's model predictions
    goals = []
    for i in range(n_goals):
        t_start = i + 1  # 0th time step is start state, so goals start at 1st time step
        t_end = n_goals_total - n_goals + i + 1  # indexing is exclusive t_end, so add 1 at the end
        # pick goals from above level
        goal = torch.stack(pred[level + 1]['o'][t_start:t_end])
        # fold time into batch dimension
        goal = goal.reshape(goal.shape[0] * goal.shape[1], -1)
        goals.append(goal.detach())

    gsa_agent_mem, gsa_model_mem, gsa_losses = {}, {}, {}
    start_state = first_start_state
    start_state_mask = first_step_mask
    for i_goal, goal in enumerate(goals):
        # vary step count to give the agent some slack sometimes
        n_steps = random.randint(math.ceil(0.5 * chunk_size), 5 * chunk_size)
        # n_steps = random.randint(chunk_size, chunk_size + 1)
        # n_steps = chunk_size

        start_state = rssm_detach_state(*start_state)  # prevent gradients to flow into previous chunk
        goal_simulation = goal_agent.act_in_sim(env_start_state=start_state, sim_env=model, n_steps=n_steps,
                                                explore=True, goal=goal, sample_states=False,
                                                sample_actions=sample_agents, expl_noise=0.1, reconstruct=True)
        agent_mem = goal_simulation['agent']
        loss = goal_agent.update_step(agent_mem,
                                      first_step_mask=start_state_mask,
                                      actor_optimizer=goal_actor_opt,
                                      critic_optimizer=goal_critic_opt)
        # loss = {f'{k}_{i_goal}': v for i, (k, v) in enumerate(loss.items())}
        # gsa_losses.update(loss)
        for k in loss:
            val = gsa_losses.get(k, 0.0)
            gsa_losses[k] = val + loss[k]

        extend_memory(gsa_agent_mem, goal_simulation['agent'])
        extend_memory(gsa_model_mem, goal_simulation['model'])

        # set new start state
        start_state = goal_simulation['model_state']
        # remember all trajectories that had invalid start states or became invalid throughout current chunk
        mask = torch.stack([start_state_mask] + goal_simulation['model']['terminal'])
        # use similar technique as in compute_mask function, but don't shift terminal states
        valid = torch.cumprod(1.0 - mask.to(torch.float32), dim=0)
        start_state_mask = (1.0 - valid)[-1]

    gsa_losses = {k: v / n_goals for k, v in gsa_losses.items()}

    logger.log(to_np(gsa_losses), Scope.TRAIN() / f'goal_seeking_agent/{level}/', i_step)

    if i_step % cfg['trainer']['eval_interval'] == 0 and level == 0 and env_class_is(eval_env, Nav2dEnv):
        plot_goal_conditioned_value_function(eval_env, model, goal_agent, level, logger, i_step, 3, 3)
        plot_goal_seeking_performance(goal_agent, gsa_model_mem, gsa_agent_mem, goals, first_start_state, model,
                                      [chunk_size for _ in goals], logger, i_step, level)


def train_goal_seeking_agent_same_level(model, level, pred, targets, eval_env, cfg, i_step, logger, sample_agents,
                                        perturb=False):
    n_goals = 3
    chunk_size = model.strides[level + 1]
    goal_agent, goal_actor_opt, goal_critic_opt = model.goal_seeking_agents[level]
    trajectories = pred[level]
    mask = targets[level]['mask']
    train_traj_len = len(mask) - (1 + n_goals * chunk_size)

    if len(trajectories['a']) <= chunk_size * n_goals + 1:
        return

    if perturb:
        pass
        # TODO: re-sample all states from distributions

    # get starting states, leave enough states out at the end to collect goals
    first_start_state, first_step_mask = filter_mem_state_seq_to_batch(trajectories, mask,
                                                                       i_end=-(1 + n_goals * chunk_size))
    first_start_state = rssm_detach_state(*first_start_state)

    # get goal states
    goal_states = []
    for i in range(1, n_goals + 1):
        start_offset = chunk_size * i
        start_offset += 1  # very first state is start state, from there on we need to go chunk_size steps to next goal
        end_offset = start_offset + train_traj_len
        g, _ = filter_mem_state_seq_to_batch(trajectories, mask, i_start=start_offset, i_end=end_offset)
        g = rssm_detach_state(*g)
        goal_states.append(g)

    goal_mem = []
    gsa_agent_mem, gsa_model_mem, gsa_losses = {}, {}, {}
    start_state = first_start_state
    start_state_mask = first_step_mask
    for goal_state in goal_states:
        goal = goal_agent.o_from_state(goal_state)
        goal_mem.append(goal)

        # vary step count to give the agent some slack sometimes
        n_steps = random.randint(math.ceil(0.5 * chunk_size), 3 * chunk_size)
        # n_steps = random.randint(chunk_size, chunk_size + 1)
        # n_steps = chunk_size

        goal_simulation = goal_agent.act_in_sim(env_start_state=start_state, sim_env=model, n_steps=n_steps,
                                                explore=True, goal=goal, sample_states=False,
                                                sample_actions=sample_agents, expl_noise=0.5, reconstruct=True)
        agent_mem = goal_simulation['agent']
        loss = goal_agent.update_step(agent_mem,
                                      first_step_mask=start_state_mask,
                                      actor_optimizer=goal_actor_opt,
                                      critic_optimizer=goal_critic_opt)
        # loss = {f'{k}_{i_goal}': v for i, (k, v) in enumerate(loss.items())}
        # gsa_losses.update(loss)
        for k in loss:
            val = gsa_losses.get(k, 0.0)
            gsa_losses[k] = val + loss[k]

        extend_memory(gsa_agent_mem, goal_simulation['agent'])
        extend_memory(gsa_model_mem, goal_simulation['model'])

        start_state = rssm_detach_state(*goal_simulation['model_state'])
        # remember all trajectories that had invalid start states or became invalid throughout current chunk
        mask = torch.stack([start_state_mask] + goal_simulation['model']['terminal'])
        # use same technique as in compute_mask function, but don't cut last step as an end state that is invalid should
        # be marked drectly as such
        valid = torch.cumprod(1.0 - mask.to(torch.float32), dim=0)
        start_state_mask = (1.0 - valid)[-1]

    gsa_losses = {k: v / n_goals for k, v in gsa_losses.items()}

    logger.log(to_np(gsa_losses), Scope.TRAIN() / f'goal_seeking_agent/{level}/', i_step)

    if i_step % cfg['trainer']['eval_interval'] == 0 and level == 0 and env_class_is(eval_env, Nav2dEnv):
        plot_goal_conditioned_value_function(eval_env, model, goal_agent, level, logger, i_step, 3, 3)
        plot_goal_seeking_performance(goal_agent, gsa_model_mem, gsa_agent_mem, goal_mem, first_start_state, model,
                                      [chunk_size for _ in goal_states], logger, i_step, level)


def train_goal_seeking_agent_rand(model, level, pred, targets, eval_env, cfg, i_step, logger, sample_model,
                                  sample_agents):
    # alternative training methods
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
                    with record_function(f'goal_seeking_agent_training_{l}'):
                        goal_agent, goal_actor_opt, goal_critic_opt = model.goal_seeking_agents[l]
                        goals = model.filter_up(o=r_max_simulation['model']['z'], r=r_max_simulation['model']['r'],
                                                terminal=r_max_simulation['model']['terminal'], level=l + 1,
                                                respect_terminal_flag=True)
                        chunk_size = model.strides[l + 1]
                        agent_mem = {}
                        state = start_state_lvl
                        # TODO: Compute the amount of required steps individually per batch item as each trajectory could
                        #       differ in length and thus the number of required steps in the final chunk. This probably
                        #       requires the adaption of act_in_sim code.
                        #       Idea: n_steps has d_batch dim and where n_steps have been done, the terminal flag of the
                        #       env is overwritten with True to invalidate all further efforts of the goal seeking agent
                        #       to reach the goal.
                        for i_goal, goal in enumerate(goals['o']):
                            # give one additional step in last chunk
                            if i_goal == len(goals['o']) - 1:
                                n_steps = chunk_size + 1
                            else:
                                n_steps = chunk_size

                            # goal = goal + torch.rand_like(goal) * 0.1
                            goal_simulation = goal_agent.act_in_sim(state, model, n_steps, goal,
                                                                    agent_memory=agent_mem, sample_states=True,
                                                                    sample_actions=True, disable_exploration=False,
                                                                    reconstruct=False)
                            state = goal_simulation['model_state']

                            # Variation B-1:
                            # ground goal agent with r_max agent trajectory after every chunk
                            #
                            # state = {'z': r_max_simulation['model']['z'][t].detach(),
                            #        'rnn_state': (r_max_simulation['model']['rnn_state'][t][0].detach(),
                            #                      r_max_simulation['model']['rnn_state'][t][1].detach())}

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
    # take completely random start and goal states for the GSA to train on
    chunk_size = model.strides[level + 1]
    goal_agent, goal_actor_opt, goal_critic_opt = model.goal_seeking_agents[level]
    gsa_start_states, gsa_first_step_mask = filter_mem_state_seq_to_batch(pred[level], targets[level]['mask'])
    gsa_start_states = rssm_detach_state(*gsa_start_states)
    # this avoids start states if the mask value is high
    probs = 1 - gsa_first_step_mask.squeeze()
    start_state_idx = torch.multinomial(probs, gsa_first_step_mask.shape[0], replacement=True)
    randomized_gsa_start_states = [x[start_state_idx] for x in gsa_start_states]
    randomized_gsa_start_state_mask = gsa_first_step_mask[start_state_idx]
    n_goals = 3
    n_steps = [random.randint(chunk_size, 3 * chunk_size) for _ in range(n_goals)]
    gsa_agent_mem, gsa_model_mem, gsa_losses = {}, {}, {}
    goal_mem = []
    start_state = randomized_gsa_start_states
    start_state_mask = randomized_gsa_start_state_mask
    for i_goal in range(n_goals):
        goal_state_idx = torch.randperm(gsa_start_states[0].shape[0])
        gsa_goal_states = [x[goal_state_idx] for x in gsa_start_states]
        # 10% random perturbation
        # gsa_start_states = [x + 0.1 * (torch.rand_like(x) - 0.5) * x for x in gsa_start_states]
        # gsa_goal_states = [x + 0.1 * (torch.rand_like(x) - 0.5) * x for x in gsa_goal_states]
        gsa_goals = goal_agent.o_from_state(gsa_goal_states)
        goal_mem.append(gsa_goals)

        goal_simulation = goal_agent.act_in_sim(start_state, model, n_steps[i_goal], gsa_goals,
                                                sample_states=sample_model, sample_actions=sample_agents,
                                                explore=True, expl_noise=0.1, reconstruct=True)
        agent_mem = goal_simulation['agent']
        # agent_mem['o'].append(torch.zeros_like(agent_mem['o'][0]))
        # agent_mem['r'].append(torch.zeros_like(agent_mem['r'][0]))
        # agent_mem['terminal'].append(torch.zeros_like(agent_mem['terminal'][0]))
        # agent_mem['a'].append(torch.zeros_like(agent_mem['a'][0]))
        # agent_mem['a_dist'].append(torch.zeros_like(agent_mem['a_dist'][0]))
        # make last reward 0 to avoid goal seeking agent getting reward from beyond goal
        # agent_mem['r'][-1] = torch.zeros_like(agent_mem['r'][-1])
        # make last time step terminal to avoid goal seeking agent getting bootstraps from beyond goals
        # agent_mem['terminal'][-1] = torch.ones_like(agent_mem['terminal'][-1])
        # agent_mem['terminal'][-2] = torch.ones_like(agent_mem['terminal'][-2])
        loss = goal_agent.update_step(agent_mem,
                                      first_step_mask=start_state_mask,
                                      actor_optimizer=goal_actor_opt,
                                      critic_optimizer=goal_critic_opt)
        # loss = {f'{k}_{i_goal}': v for i, (k, v) in enumerate(loss.items())}
        # gsa_losses.update(loss)

        for k in loss:
            val = gsa_losses.get(k, 0.0)
            gsa_losses[k] = val + loss[k]

        extend_memory(gsa_agent_mem, goal_simulation['agent'])
        extend_memory(gsa_model_mem, goal_simulation['model'])

        start_state = rssm_detach_state(*goal_simulation['model_state'])
        start_state_mask = torch.stack([start_state_mask] + goal_simulation['model']['terminal'])
        start_state_mask = (1.0 - torch.cumprod(1.0 - start_state_mask, dim=0))[-1]

    gsa_losses = {k: v / n_goals for k, v in gsa_losses.items()}

    logger.log(to_np(gsa_losses), Scope.TRAIN() / f'goal_seeking_agent/{level}/', i_step)

    if (i_step % cfg['trainer']['eval_interval'] == 0
            and level == 0
            and isinstance(eval_env.unwrapped.env_fns[0]().unwrapped, gym_nav2d.envs.Nav2dEnv)):
        plot_goal_conditioned_value_function(eval_env, model, goal_agent, level, logger, i_step, 3, 3)
        plot_goal_seeking_performance(goal_agent, gsa_model_mem, gsa_agent_mem, goal_mem,
                                      randomized_gsa_start_states, model, n_steps, logger, i_step, level)


def record_episode(cfg, i_step, logger, model, video_env):
    # record an episode
    video_env.reset()
    video_env.get_wrapper_attr('start_video_recorder')()
    policy = HierarchicalLatentAgentPolicy(model, explore=False)
    # policy = LatentAgentPolicy(model.r_max_agents[0][0], model, explore=False)
    _ = collect_data(video_env, cfg['eval']['eval_steps'], policy)
    video_env.get_wrapper_attr('close_video_recorder')()
    # make video smaller
    # video_name = f'{video_env.name_prefix}-episode-{video_env.episode_id-1}.mp4'
    # video_path = os.path.join(video_env.video_folder, video_name)
    video_path = video_env.env.video_recorder.path
    metadata_path = video_path[:video_path.rindex('.')] + '.meta.json'
    timestamp = time.time_ns()
    pid = os.getpid()
    tmp_file_name = f'.{pid}_{timestamp}_agent_video.mp4'
    clip = mp.VideoFileClip(video_path)
    # clip = clip.resize(width=80)
    clip.write_videofile(tmp_file_name, preset='veryslow', verbose=False, logger=None)
    try:
        os.remove(video_path)  # delete original video file
        os.remove(metadata_path)
    except FileNotFoundError:
        print('Failed to delete original video data')
    # upload
    video = InMemoryFile.consume_file(tmp_file_name)
    logger.log({'hierarchical_agent': video}, Scope.TEST() / 'agent_action_videos/', i_step)


def plot_value_function(eval_env, model, agent, level, logger, i_step):
    grid_trajs = gen_regular_grid_trajectories(get_env_instance(eval_env), trajs_vert=30, trajs_horiz=30,
                                               step_size=1.0)
    grid_trajs = to_tensors(grid_trajs, model.device)
    grid_trajs = prepare_data(grid_trajs)
    _, pred_grid, targets_grid, = model.eval_step(grid_trajs, model_steps=[-1 for _ in range(model.levels)],
                                                  sample_state=False, sample_output=False,
                                                  force_warmup=[-1 for _ in range(model.levels)])
    values = agent.critic_net(torch.stack(pred_grid[level]['s_embedding'])).detach().cpu().numpy()
    obs = grid_trajs['o']
    for upsampling_stage in range(level + 1):
        obs = model.upwards_filters[upsampling_stage]['o'](obs)
    obs = obs.detach().cpu().numpy()
    values = values.reshape(-1, 1)
    obs = obs.reshape(-1, obs.shape[-1])
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.plot_trisurf(obs[:, 0], obs[:, 1], values[:, 0], antialiased=True, cmap=plt.cm.viridis)

    logger.log_plot(fig_to_img(fig),
                    Scope.TRAIN() / f'r_max_agent/{level}/value_function_plot',
                    i_step)
    plt.close(fig)
    del fig


def plot_rewards(eval_env, model, level, logger, i_step):
    grid_trajs = gen_regular_grid_trajectories(get_env_instance(eval_env), trajs_vert=30, trajs_horiz=30,
                                               step_size=1.0)
    grid_trajs = to_tensors(grid_trajs, model.device)
    grid_trajs = prepare_data(grid_trajs)
    _, pred_grid, targets_grid = model.eval_step(grid_trajs, model_steps=[-1 for _ in range(model.levels)],
                                                 sample_state=False, sample_output=False,
                                                 force_warmup=[-1 for _ in range(model.levels)])
    rewards = torch.stack(pred_grid[level]['r']).detach().cpu().numpy()
    obs = grid_trajs['o']
    for upsampling_stage in range(level + 1):
        obs = model.upwards_filters[upsampling_stage]['o'](obs)
    obs = obs.detach().cpu().numpy()
    rewards = rewards.reshape(-1, 1)
    obs = obs.reshape(-1, obs.shape[-1])
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.plot_trisurf(obs[:, 0], obs[:, 1], rewards[:, 0], antialiased=True, cmap=plt.cm.viridis)

    logger.log_plot(fig_to_img(fig),
                    Scope.TRAIN() / f'model/{level}/reward_plot',
                    i_step)
    plt.close(fig)
    del fig


def plot_goal_conditioned_value_function(eval_env, model, agent, level, logger, i_step, n_rows, n_cols):
    eval_env = get_env_instance(eval_env)
    grid_trajs = gen_regular_grid_trajectories(eval_env, trajs_vert=30, trajs_horiz=30, step_size=1.0)
    grid_trajs = to_tensors(grid_trajs, model.device)
    grid_trajs = prepare_data(grid_trajs)
    _, pred_grid, _ = model.eval_step(grid_trajs, model_steps=[-1 for _ in range(model.levels)],
                                      sample_state=False, sample_output=False,
                                      force_warmup=[-1 for _ in range(model.levels)])
    s_embeddings = torch.stack(pred_grid[level]['s_embedding']).flatten(start_dim=0, end_dim=-2)
    fig, axes = plt.subplots(n_rows, n_cols, subplot_kw={"projection": "3d", 'computed_zorder': False},
                             figsize=(10, 10))
    for ax in axes.ravel():
        i_goal = np.random.randint(s_embeddings.shape[0])
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])

        goal = s_embeddings[i_goal].unsqueeze(0).repeat((s_embeddings.shape[0], 1))
        fused_embeddings = torch.concat([s_embeddings, goal], dim=-1)
        values = agent.critic_net(fused_embeddings).detach().cpu().numpy()

        obs = grid_trajs['o'].flatten(start_dim=0, end_dim=-2).detach().cpu().numpy()
        ax.plot_trisurf(obs[:, 0], obs[:, 1], values[:, 0], antialiased=True, zorder=0, cmap=plt.cm.viridis)

        goal_world_coords = (obs[i_goal] + 1) * 255 / 2
        goal_dist_to_reward = math.sqrt(pow((eval_env.goal_x - goal_world_coords[0]), 2)
                                        + pow(eval_env.goal_y - goal_world_coords[1], 2))
        if goal_dist_to_reward <= eval_env.eps:
            c = 'magenta'
        else:
            c = 'red'
        ax.scatter(obs[i_goal, 0], obs[i_goal, 1], values[i_goal, 0], zorder=1, linewidth=5, color=c, depthshade=True)
    plt.tight_layout()
    # plt.show()
    logger.log_plot(fig_to_img(fig),
                    Scope.TRAIN() / f'goal_seeking_agent/{level}/value_function_plot',
                    i_step)
    plt.close(fig)
    del fig


def plot_goals(r_max_simulation, model, logger, i_step, l):
    abstract_obs = torch.stack(r_max_simulation['model']['o'])
    goal_obs = model.rssm_modules[l - 1].decode(abstract_obs, sample=False, reconstruct_observation=True)['o']
    goal_obs = goal_obs.detach().cpu().numpy()
    terminals = torch.stack(r_max_simulation['model']['terminal']).detach().cpu().numpy()

    fig, axes = plt.subplots(5, 5, figsize=(10, 10), sharex=True, sharey=True)
    indices = np.random.randint(low=0, high=goal_obs[0].shape[0], size=(25,))

    for i_ax, ax in enumerate(axes.ravel()):
        # preparations
        idx = indices[i_ax]
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])

        alphas = 1 - terminals[:, idx, 0]
        for t in list(range(goal_obs.shape[0])):
            ax.scatter(goal_obs[t, idx, 0], goal_obs[t, idx, 1], c='C0', s=50, alpha=alphas)
            ax.scatter(goal_obs[t, idx, 0], goal_obs[t, idx, 1], marker=f'${t + 1}$', c='white', s=30,
                       alpha=min(0.5, alphas[t]))

        # add reward location for reference
        ax.scatter(0, 0, label='reward', marker='x', c='red', s=45, alpha=0.2)
    plt.tight_layout()
    # plt.show()

    logger.log_plot(fig_to_img(fig), Scope.TRAIN() / f'r_max_agent/{l}/goal_seeking_train_performance', i_step)
    plt.close(fig)
    del fig


def nav2d_gsa_plot(cfg, eval_env, eval_steps, i_step, logger, model, train_driver):
    trajs_orig = train_driver.interact(cfg['trainer']['d_batch'])
    # trajs_orig = add_no_ops(trajs_orig, model.strides[1])
    batch = to_tensors(trajs_orig, model.device)
    batch = prepare_data(batch)
    eval_steps = [-1] + cfg['trainer']['model_train_steps'][1:]
    eval_losses, pred, targets = model.eval_step(batch, model_steps=eval_steps, sample_state=False,
                                                 sample_output=False,
                                                 force_warmup=[-1 for _ in range(model.levels)])
    gsa = model.goal_seeking_agents[0][0]
    states = {k: torch.stack(v) for k, v in pred[0].items() if k in rssm_state_keys()}
    gsa_goals = gsa.o_from_state(rssm_remove_labels(states))
    goals = model.filter_up(o=gsa_goals[1:], terminal=targets[0]['terminal'][1:], level=1)['o']
    goal_obs = model.filter_up(o=targets[0]['o'][1:], terminal=targets[0]['terminal'][1:], level=1)['o']
    goal_terminals = model.filter_up(o=targets[0]['terminal'][1:], terminal=targets[0]['terminal'][1:],
                                     level=1)['o']
    goal_distances = []
    for i in range(len(goals)):
        goal_distances.append([])
        for j in range(len(goals)):
            d = gsa.goal_similarity(goals[i], goals[j])[0].detach().cpu().numpy()
            goal_distances[-1].append(d)
    goal_distances = np.array(goal_distances).squeeze(-1)
    chunk_size = model.strides[1]
    gsa_agent_mem = {}
    total_steps_remaining = len(batch['o']) - 1  # first step is start state
    state = {k: pred[0][k][0] for k in rssm_state_keys()}
    state = rssm_remove_labels(state)
    for i_goal, goal in enumerate(goals):
        # i_state = i_goal * chunk_size
        # state = rssm_remove_labels({k: pred[0][k][i_state] for k in rssm_state_keys()})
        n_steps = np.minimum(total_steps_remaining, chunk_size)
        simulation = gsa.act_in_sim(env_start_state=state, sim_env=model, n_steps=n_steps, goal=goal,
                                    sample_actions=False, explore=False, sample_states=False,
                                    agent_memory=gsa_agent_mem)
        state = simulation['model_state']
        total_steps_remaining -= chunk_size
    a_gsa_sim = torch.stack(gsa_agent_mem['a'])[:, 0].detach().cpu().numpy()
    env = CacheLastStepEnv(eval_env.unwrapped.env_fns[0]())
    init_obs = targets[0]['o'][0][0].detach().cpu().numpy()
    init_obs_unnorm = env.unwrapped.unnormalize_observation(init_obs)
    env_init_options = {'agent_x': init_obs_unnorm[0], 'agent_y': init_obs_unnorm[1]}
    gsa_executed_actions_mem = collect_data(env, a_gsa_sim.shape[0] + 1, PredefinedPolicy(a_gsa_sim),
                                            options=env_init_options)
    orig_obs = targets[0]['o'][1:].detach().cpu().numpy()
    valid_steps = 1 - compute_mask(targets[0]['terminal'][1:]).detach().cpu().numpy()
    agent_obs = gsa_executed_actions_mem[0]['o'][1:]
    goal_obs = goal_obs.detach().cpu().numpy()
    valid_goals = 1 - compute_mask(goal_terminals).detach().cpu().numpy()
    hsv_values = np.linspace(0.9, 0.2, num=np.maximum(len(agent_obs), len(orig_obs)))
    with TempFigure(figsize=(10, 5)) as fig:
        ax_0 = fig.add_subplot(1, 2, 1)
        c = [to_rgba(hsv_to_rgb((0.05, 1.0, hsv_values[t])), valid_goals[t, 0, 0]) for t in
             range(len(goal_obs))]
        ax_0.scatter(goal_obs[:, 0, 0], goal_obs[:, 0, 1], label='goal pos', figure=fig, c=c, s=100)
        c = [to_rgba(hsv_to_rgb((0.15, 1.0, hsv_values[t])), valid_steps[t, 0, 0]) for t in
             range(len(orig_obs))]
        ax_0.scatter(orig_obs[:, 0, 0], orig_obs[:, 0, 1], label='original pos', figure=fig, c=c, s=60)
        c = [to_rgba(hsv_to_rgb((0.55, 1.0, hsv_values[t])), valid_steps[t, 0, 0]) for t in
             range(len(agent_obs))]
        ax_0.scatter(agent_obs[:, 0], agent_obs[:, 1], label='agent pos', figure=fig, c=c, marker="P")
        ax_0.set_title(f'{valid_steps[:, 0, 0].sum().astype(int)} steps')
        ax_0.scatter(init_obs[0], init_obs[1], marker='p', label='start pos', c='green')
        ax_0.legend()
        fig.axes[0].add_patch(Rectangle((-1, -1), 2, 2, fill=False, edgecolor='grey', linestyle='--'))
        ax_1 = fig.add_subplot(1, 2, 2)
        ax_1.set_title('Euclidean goal distances')
        mat = ax_1.matshow(goal_distances)
        fig.colorbar(mat, ax=ax_1)
        plt.tight_layout()
        # plt.show()
        logger.log_plot(fig_to_img(fig), Scope.TRAIN() / f'goal_seeking_agent/{0}/goal_seeking_plot',
                        i_step)
    return batch, eval_steps


def latent_state_pca_plot(model, eval_env, eval_steps, i_step, logger):
    eval_env = get_env_instance(eval_env)
    grid_trajs = gen_regular_grid_trajectories(eval_env, trajs_vert=30, trajs_horiz=30, step_size=1.0)
    grid_trajs = to_tensors(grid_trajs, model.device)
    grid_trajs = prepare_data(grid_trajs)
    _, pred_pca, _ = model.eval_step(grid_trajs, model_steps=eval_steps, sample_state=False,
                                     sample_output=False,
                                     force_warmup=[-1 for _ in range(model.levels)])
    obs = grid_trajs['o']

    # latent sate PCA 3d plot
    for quantity in ['s_embedding', 'z', 'h']:
        for level in range(model.levels):
            states = torch.stack(pred_pca[level][quantity]).detach().cpu().numpy()
            d_time, d_batch, d_z = states.shape

            pca = PCA(n_components=3)
            states_trans = pca.fit_transform(states.reshape(d_time * d_batch, d_z))

            obs_level = obs
            for upsampling_stage in range(level + 1):
                obs_level = model.upwards_filters[upsampling_stage]['o'](obs_level)
            obs_level = obs_level.reshape(-1, obs_level.shape[-1]).detach().cpu().numpy()
            xy_positions = (obs_level[:, :2] + 1.0) / 2.0
            colors = np.concatenate([xy_positions, np.zeros((d_time * d_batch, 1), dtype=float)], axis=-1)

            with TempFigure() as fig:
                ax = fig.add_subplot(projection='3d')
                ax.view_init(elev=50, azim=45)
                ax.set_title(f'Total explained variance: {np.sum(pca.explained_variance_ratio_):.3f}')
                ax.scatter(states_trans[:, 0], states_trans[:, 1], states_trans[:, 2], c=colors)
                ax.set_xlabel(f'PCA 1 ({pca.explained_variance_ratio_[0]:.3f})')
                ax.set_ylabel(f'PCA 2 ({pca.explained_variance_ratio_[1]:.3f})')
                ax.set_zlabel(f'PCA 3 ({pca.explained_variance_ratio_[2]:.3f})')
                plt.tight_layout()
                # plt.show()
                logger.log_plot(fig_to_img(fig), Scope.TRAIN() / f'model/{level}/pca_{quantity}', i_step)
                # sanity check to confirm that coloring based on xy positions makes sense
                # plt.scatter(xy_positions[:, 0], xy_positions[:, 1], c=colors)
                # plt.show()


def plot_goal_seeking_performance(goal_agent, model_mem, agent_mem, gsa_goals, gsa_start_states, model,
                                  n_steps_per_chunk, logger, i_step, l):
    chunk_colors = ['lightblue', 'deepskyblue', 'dodgerblue', 'cornflowerblue', 'royalblue', 'blue']
    markers = ['^', 's', 'p', 'h']
    if not isinstance(gsa_goals, list):
        gsa_goals = [gsa_goals]
    start_obs = model.rssm_modules[0].decode(goal_agent.o_from_state(gsa_start_states),
                                             sample=False, reconstruct_observation=True)['o']
    start_obs = start_obs.detach().cpu().numpy()
    goal_reconstr = [model.rssm_modules[0].decode(g, sample=False, reconstruct_observation=True) for g in gsa_goals]
    goal_obs = [g['o'].detach().cpu().numpy() for g in goal_reconstr]
    traj_obs = torch.stack(model_mem['o']).detach().cpu().numpy()
    traj_terminals = torch.stack(agent_mem['terminal']).detach().cpu().numpy()

    fig, axes = plt.subplots(5, 5, figsize=(10, 10), sharex=True, sharey=True)
    indices = np.random.randint(low=0, high=goal_obs[0].shape[0], size=(25,))
    for i_ax, ax in enumerate(axes.ravel()):
        # preparations
        idx = indices[i_ax]
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])

        # plot observations of agent steps
        current_step = 0
        for i_chunk, n_steps in enumerate(n_steps_per_chunk):
            t_s, t_e = current_step, current_step + n_steps
            a = 1 - traj_terminals[t_s:t_e, idx, 0]
            ax.scatter(traj_obs[t_s:t_e, idx, 0], traj_obs[t_s:t_e, idx, 1], c=chunk_colors[i_chunk],
                       label='steps', alpha=a)
            current_step += n_steps

        # plot start and goals
        ax.scatter(start_obs[idx, 0], start_obs[idx, 1], c='orange', label='start', s=45)
        ax.scatter(start_obs[idx, 0], start_obs[idx, 1], marker='$s$', c='white', s=25, alpha=0.5)
        for i_goal, g in enumerate(goal_obs):
            ax.scatter(g[idx, 0], g[idx, 1], c='green', label=f'goal {i_goal + 1}', s=45)
            ax.scatter(g[idx, 0], g[idx, 1], marker=f'${i_goal + 1}$', c='white', s=25, alpha=0.5)

        # add reward location for reference
        ax.scatter(0, 0, label='reward', marker='x', c='red', s=45, alpha=0.2)
    # ax.legend()
    plt.tight_layout()

    logger.log_plot(fig_to_img(fig), Scope.TRAIN() / f'goal_seeking_agent/{l}/goal_seeking_train_performance', i_step)
    plt.close(fig)
    del fig
    # plt.show()


def update_model_chunk_distance(i_step, l, logger, model, r_max_simulation):
    # update model's stats about how distant goals are on average
    with torch.no_grad():
        chunk_size = model.strides[l + 1]
        goals = model.filter_up(o=r_max_simulation['model']['z'], r=r_max_simulation['model']['r'],
                                terminal=r_max_simulation['model']['terminal'], level=l + 1,
                                respect_mask=False)
        goals_mask = compute_mask(goals['terminal'])
        goals_mask = torch.maximum(goals_mask[:-1], goals_mask[1:])  # need valid start _and_ end goal
        goals_distance = torch.abs(goals['o'][:-1] - goals['o'][1:])
        # partition goal sequence in 3 equal parts if possible, if not make middle part longer
        i0 = floor(chunk_size / 3)
        i1 = ceil(2 * chunk_size / 3)
        model.avg_chunk_dist_early[l].update(goals_distance[:i0], mask=goals_mask[:i0])
        model.avg_chunk_dist_mid[l].update(goals_distance[i0:-i1], mask=goals_mask[i0:-i1])
        model.avg_chunk_dist_late[l].update(goals_distance[-i1:], mask=goals_mask[-i1:])
        chunk_dist_stats = {'early': model.avg_chunk_dist_early[l].mean.mean(),
                            'mid': model.avg_chunk_dist_mid[l].mean.mean(),
                            'late': model.avg_chunk_dist_late[l].mean.mean()}
        logger.log(to_np(chunk_dist_stats), Scope.TRAIN() / f'model/{l}/average_chunk_length/', i_step)


def log_prediction_error_plot(batch, cfg, i_step, logger, model):
    traj_len = 20
    rand_number = random.randint(0, sys.maxsize)
    random.seed(rand_number)
    trunc_test_batch = valid_subtrajectories_2(batch, traj_len)

    # testing
    random.seed(rand_number)
    trunc_test_batch_2 = valid_subtrajectories_2(batch, traj_len)
    random.seed(rand_number)
    trunc_test_batch_3 = valid_subtrajectories_debug(batch, traj_len)
    for k, v in trunc_test_batch.items():
        assert torch.all(v == trunc_test_batch_2[k])
        if k != 'terminal':
            assert torch.all(v == trunc_test_batch_3[k])
        else:
            assert not torch.all(v == trunc_test_batch_3[k])
    # testing end

    eval_steps = [-1] + cfg['trainer']['model_train_steps'][1:]
    measurements = [{'o': [], 'r': [], 'terminal': [], 'n_warmup': [], 'perc_valid': None} for _ in range(model.levels)]
    for n_wu in range(1, traj_len, 3):
        eval_losses, pred, targets = model.eval_step(trunc_test_batch, model_steps=eval_steps,
                                                     sample_state=False, sample_output=False,
                                                     force_warmup=[n_wu for _ in range(model.levels)])
        eval_losses2, pred2, targets2 = model.eval_step(trunc_test_batch_2, model_steps=eval_steps,
                                                        sample_state=False, sample_output=False,
                                                        force_warmup=[n_wu for _ in range(model.levels)])
        for k, v in pred[0].items():
            assert torch.all(torch.stack(v) == torch.stack(pred2[0][k])), f'mismatch in {k}'
        for k, v in targets[0].items():
            assert torch.all(v == targets2[0][k])

        for l in range(model.levels):
            mask = targets[l]['mask']

            o_diff = torch.abs(torch.stack(pred[l]['o']) - targets[l]['o'])
            o_diff = masked_mean(o_diff, mask, dim=[1, 2])
            r_diff = torch.abs(torch.stack(pred[l]['r']) - targets[l]['r'])
            r_diff = masked_mean(r_diff, mask, dim=[1, 2])
            term_diff = torch.abs(torch.stack(pred[l]['terminal']) - targets[l]['terminal'])
            term_diff = masked_mean(term_diff, mask, dim=[1, 2])

            measurements[l]['n_warmup'].append(n_wu)
            measurements[l]['o'].append(o_diff.detach().cpu().numpy())
            measurements[l]['r'].append(r_diff.detach().cpu().numpy())
            measurements[l]['terminal'].append(term_diff.detach().cpu().numpy())

    # get percentage of valid trajectories per time step
    for l in range(model.levels):
        perc_valid = (1 - targets[l]['mask']).mean(dim=1).squeeze().detach().cpu().numpy()
        measurements[l]['perc_valid'] = perc_valid

    for l in range(model.levels):
        warmup_steps = measurements[l]['n_warmup']
        o_diff = measurements[l]['o']
        r_diff = measurements[l]['r']
        term_diff = measurements[l]['terminal']
        with TempFigure(figsize=(13, 4)) as fig:
            for diff, title, pos in zip([o_diff, r_diff, term_diff], ('o', 'r', 'terminal'), (1, 2, 3)):
                time_steps = list(range(len(diff[0])))
                ax = fig.add_subplot(1, 3, pos, projection='3d', computed_zorder=False)
                ax.set_title(f'Prediction MAE {title}')
                ax.set_xlabel('Time')
                ax.set_ylabel('Warmup Steps')
                ax.set_zlabel('Prediction MAE')
                ax.set_xticks(time_steps[::2])
                ax.set_yticks(warmup_steps)
                ax.grid(which='major', linewidth=2.0)
                ax.grid(which='minor', linewidth=0.5)
                ax.invert_yaxis()
                # plot percentage of valid trajectories in the background first
                ax.bar(left=time_steps, height=np.ones_like(time_steps), zs=0, zdir='y', color='lightgrey', alpha=0.7)
                ax.bar(left=time_steps, height=measurements[l]['perc_valid'], zs=0, zdir='y', color='grey', alpha=0.7)
                for i_run, n_wu in enumerate(warmup_steps):
                    y_coord = [n_wu for _ in time_steps]
                    z_coord = diff[i_run]
                    ax.plot(time_steps, y_coord, np.zeros_like(z_coord), c=(0.9, 0.9, 0.9, 0.6), linewidth=5)
                    ax.plot(time_steps, y_coord, np.zeros_like(z_coord), c=(0.8, 0.8, 0.8, 0.5), linewidth=3)
                    ax.plot(time_steps, y_coord, np.zeros_like(z_coord), c=(0.6, 0.6, 0.6, 0.6), linewidth=1)
                for i_run, n_wu in enumerate(warmup_steps):
                    y_coord = [n_wu for _ in time_steps]
                    z_coord = diff[i_run]
                    ax.plot(time_steps, y_coord, z_coord, linewidth=2)
            plt.tight_layout()
            # plt.show()
            logger.log_plot(fig_to_img(fig), Scope.TRAIN() / f'model/{l}/prediction_error', i_step)


def plot_maze_env(env, observations: np.ndarray = None, goal_observations: np.ndarray = None, axis: plt.axis = None):
    assert env_class_is(env, MazeEnv)
    if axis is None:
        plt_target = plt
    else:
        plt_target = axis
    instance = get_env_instance(env).unwrapped
    x_center = instance.maze.x_map_center
    y_center = instance.maze.y_map_center
    length = instance.maze.map_length
    width = instance.maze.map_width
    start_locations = instance.maze.unique_reset_locations
    goal_locations = instance.maze.unique_goal_locations
    maze_map = instance.maze.maze_map
    # remove maze_map start and goal locations
    for row in maze_map:
        for x in range(len(row)):
            if type(row[x]) is str:
                row[x] = 0
    maze_map = np.array(maze_map)
    left = - width / 2
    right = width / 2
    bottom = - length / 2
    top = length / 2
    plt_target.imshow(maze_map, extent=(left, right, bottom, top))
    for loc in start_locations:
        plt_target.scatter(loc[0], loc[1], marker='.', c='red', s=500)
        plt_target.scatter(loc[0], loc[1], marker='$S$', c='white', s=45)
    for loc in goal_locations:
        plt_target.scatter(loc[0], loc[1], marker='.', c='green', s=500)
        plt_target.scatter(loc[0], loc[1], marker='$R$', c='white', s=45)

    # plot individual trajectories
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    goal_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S']
    if observations is not None:
        for i_traj in range(observations.shape[1]):
            c = colors[i_traj % len(colors)]
            for t in range(observations.shape[0]):
                plt_target.scatter(observations[t, i_traj, 4], observations[t, i_traj, 5], marker=f'${t}$', c=c, s=25)
    if goal_observations is not None:
        for i_traj in range(goal_observations.shape[1]):
            c = colors[i_traj % len(colors)]
            for t in range(goal_observations.shape[0]):
                plt_target.scatter(goal_observations[t, i_traj, 4] - 0.002, goal_observations[t, i_traj, 5] - 0.002,
                                   marker=f'${goal_names[t]}$', c='black', s=25)
                plt_target.scatter(goal_observations[t, i_traj, 4], goal_observations[t, i_traj, 5],
                                   marker=f'${goal_names[t]}$', c=c, s=25)


# print(torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())


def agent_train_mode(agents):
    for a in agents:
        if a is None: continue
        a[0].train()


def agent_eval_mode(agents):
    for a in agents:
        if a is None: continue
        a[0].eval()
