import gymnasium as gym
import torch
import matplotlib.pyplot as plt

from mdm.models.hierarchical_rssm import HierarchicalRSSM
from mdm.policies.expert_policies import get_expert_policy
from mdm.training.gym_driver import GymEpisodeDriver
from mdm.training.offline_rl_driver import OfflineRLDriver, SamplingType
from mdm.utils.build_models import cfg_infer_missing_values, build_rssms, build_agents, build_model_opt
from mdm.utils.gym_wrappers import CacheLastStepVecEnv
from mdm.utils.torch_tools import to_tensors
from mdm.utils.utils import load_yaml, here, prepare_env, load_model_params, prepare_data, numpyfy


def main():
    run_id = 'MBRL-5676'
    n_trajs_to_vis = 1
    init_steps = 5
    sample_state = True

    # load configs
    cfg = load_yaml(here() / '../continuous_navigation/cfg_simple_rssm_train.yaml')
    neptune_cfg = load_yaml(here() / cfg['neptune_cfg'])
    cfg['pretrained_model'] = run_id  # inject chosen run ID into config file

    # env generating function
    def make_env_fn():
        _env = gym.make(cfg['env_name'])
        _env = prepare_env(_env)
        return _env

    # build env, config and models
    env = make_env_fn()
    cfg = cfg_infer_missing_values(cfg, env)  # fill in missing config values

    cfg = build_rssms(cfg)
    r_max_agents, goal_seeking_agents = build_agents(cfg, env, torch.device('cuda'))
    model = HierarchicalRSSM(**cfg['mdm'], r_max_agents=r_max_agents, goal_seeking_agents=goal_seeking_agents)
    model = model.to('cuda')
    opt_model = build_model_opt(model, cfg)

    # set everything to eval mode
    model.eval()
    for a in r_max_agents + goal_seeking_agents: a[0].eval()

    # load existing pretrained model
    load_model_params(model, opt_model, cfg['model_save_path'], cfg['pretrained_model'], **neptune_cfg,
                      force_reload=True)

    # collect starting data
    train_mem = []
    collect_env = gym.vector.AsyncVectorEnv([make_env_fn] * 20)
    collect_env = CacheLastStepVecEnv(collect_env)
    collect_driver = GymEpisodeDriver(collect_env, lambda x: collect_env.action_space.sample())
    collect_driver.interact(200, train_mem, progress_bar=True)
    train_driver = OfflineRLDriver(train_mem, sampling_type=SamplingType.RANDOM)

    # get some starting points
    batch = train_driver.interact(n_trajs_to_vis)
    batch = to_tensors(batch, device='cuda')
    batch = prepare_data(batch)
    pred, _, targets, model_state = model.forward_all_levels(batch,
                                                             warmup_steps=[-1 for _ in model.rssm_modules],
                                                             model_steps=[init_steps for _ in model.rssm_modules],
                                                             sample_state=sample_state, sample_output=False,
                                                             reconstruct=True)
    last_obs = numpyfy(targets[0]['o'][-1])

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(n_trajs_to_vis)]
    act_magnitude = 1.0
    goal = None
    goal_obs = None

    def do_a_step(event):
        nonlocal model_state
        nonlocal last_obs
        nonlocal batch
        nonlocal sample_state
        nonlocal act_magnitude
        nonlocal goal
        nonlocal goal_obs
        nonlocal pred

        if event.key in ('up', 'down', 'left', 'right', ' '):
            a = torch.zeros_like(batch['a'][0:1])
            if event.key == 'up':
                a[:, :, 1] = act_magnitude
            elif event.key == 'left':
                a[:, :, 0] = -act_magnitude
            elif event.key == 'down':
                a[:, :, 1] = -act_magnitude
            elif event.key == 'right':
                a[:, :, 0] = act_magnitude
            else:
                a = (torch.rand_like(batch['a'][0:1]) - 0.5) * 2.0

            step_batch = {'o': torch.ones_like(batch['o'][0:1]),
                          'a': a,
                          'r': torch.zeros_like(batch['r'][0:1]),
                          'terminal': torch.zeros_like(batch['terminal'][0:1]),
                          'mask': torch.zeros_like(batch['mask'][0:1])}

            pred, _, targets, model_state = model.forward_all_levels(step_batch,
                                                                     warmup_steps=[0 for _ in model.rssm_modules],
                                                                     model_steps=[1 for _ in model.rssm_modules],
                                                                     model_state=model_state, sample_state=sample_state,
                                                                     sample_output=False, reconstruct=True)
            obs = numpyfy(pred[0]['o'][-1])
            # start updating the plot
            plt.clf()
            # draw arrows
            for i in range(n_trajs_to_vis):
                last_pos = last_obs[i, :2]
                dxdy = numpyfy(step_batch['a'][0, i]) * 10 / 255
                plt.arrow(last_pos[0], last_pos[1], dx=dxdy[0], dy=dxdy[1], color=colors[i])
            # plot last and new positions
            plt.scatter(last_obs[:, 0], last_obs[:, 1], c=colors, alpha=0.1)
            plt.scatter(obs[:, 0], obs[:, 1], c=colors)
            plt.scatter(obs[:, 2], obs[:, 3], c=colors, marker='+')
            for i in range(n_trajs_to_vis):
                r = pred[0]['r'][-1][i].detach().cpu().numpy()
                plt.text(-0.98, -0.98 + i * 0.1, str(r), c=colors[i], fontsize='x-small')
            if goal_obs is not None:
                plt.scatter(goal_obs[:, 0], goal_obs[:, 1], marker='$G$', c=colors)
            plt.ylim([-1.1, 1.1])
            plt.xlim([-1.1, 1.1])
            plt.draw()

            last_obs = obs

            if goal is not None:
                s_current = pred[0]['s_embedding'][-1]
                diff = torch.mean(torch.abs(s_current - goal), dim=-1)
                print(numpyfy(diff))

        elif event.key == 'r':
            batch = train_driver.interact(n_trajs_to_vis)
            batch = to_tensors(batch, device='cuda')
            batch = prepare_data(batch)
            pred, _, targets, model_state = model.forward_all_levels(batch,
                                                                     warmup_steps=[-1 for _ in model.rssm_modules],
                                                                     model_steps=[init_steps for _ in
                                                                                  model.rssm_modules],
                                                                     sample_state=sample_state, sample_output=False,
                                                                     reconstruct=True)
            last_obs = numpyfy(pred[0]['o'][-1])

            # start updating the plot
            plt.clf()
            # plot new positions
            plt.scatter(last_obs[:, 0], last_obs[:, 1], c=colors)
            plt.scatter(last_obs[:, 2], last_obs[:, 3], c=colors, marker='+')
            for i in range(n_trajs_to_vis):
                r = pred[0]['r'][-1][i].detach().cpu().numpy()
                plt.text(-0.98, -0.98 + i * 0.1, str(r), c=colors[i], fontsize='x-small')
            if goal_obs is not None:
                plt.scatter(goal_obs[:, 0], goal_obs[:, 1], marker='$G$', c=colors)
            plt.ylim([-1.1, 1.1])
            plt.xlim([-1.1, 1.1])
            plt.draw()
        elif event.key == 'y':
            sample_state = not sample_state
            print(f'sample states: {sample_state}')
        elif event.key == '1':
            act_magnitude = 0.1
            print(f'action magnitude: {act_magnitude}')
        elif event.key == '2':
            act_magnitude = 0.5
            print(f'action magnitude: {act_magnitude}')
        elif event.key == '3':
            act_magnitude = 1.0
            print(f'action magnitude: {act_magnitude}')
        elif event.key == 'g':
            batch = train_driver.interact(n_trajs_to_vis)
            batch = to_tensors(batch, device='cuda')
            batch = prepare_data(batch)
            pred_goal, _, _, _ = model.forward_all_levels(batch,
                                                          warmup_steps=[-1 for _ in model.rssm_modules],
                                                          model_steps=[init_steps for _ in
                                                                       model.rssm_modules],
                                                          sample_state=sample_state,
                                                          sample_output=False,
                                                          reconstruct=True)
            goal = pred_goal[0]['s_embedding'][0]
            goal_obs = numpyfy(pred_goal[0]['o'][0])

            plt.clf()
            # plot new positions
            plt.scatter(last_obs[:, 0], last_obs[:, 1], c=colors)
            plt.scatter(last_obs[:, 2], last_obs[:, 3], c=colors, marker='+')
            for i in range(n_trajs_to_vis):
                r = pred[0]['r'][-1][i].detach().cpu().numpy()
                plt.text(-0.98, -0.98 + i * 0.1, str(r), c=colors[i], fontsize='x-small')
            if goal_obs is not None:
                plt.scatter(goal_obs[:, 0], goal_obs[:, 1], marker='$G$', c=colors)
            plt.ylim([-1.1, 1.1])
            plt.xlim([-1.1, 1.1])
            plt.draw()

    # visualize env
    fig, ax = plt.subplots()
    ax.scatter(last_obs[:, 0], last_obs[:, 1], c=colors)
    plt.scatter(last_obs[:, 2], last_obs[:, 3], c=colors, marker='+')
    for i in range(n_trajs_to_vis):
        r = pred[0]['r'][-1][i].detach().cpu().numpy()
        plt.text(-0.98, -0.98 + i * 0.1, str(r), c=colors[i], fontsize='x-small')
    ax.set_ylim([-1.1, 1.1])
    ax.set_xlim([-1.1, 1.1])
    fig.canvas.mpl_connect('key_press_event', do_a_step)
    plt.show()


if __name__ == '__main__':
    main()
