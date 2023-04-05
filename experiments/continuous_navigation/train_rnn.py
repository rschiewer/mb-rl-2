import os.path
from pathlib import Path
import io
import argparse
import pickle

import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image

from mdm.gridworld.gridworld import Gridworld, FullyObservableGridworld
from mdm.utils.utils import *
from mdm.utils.torch_tools import get_mu, get_sigma, bin_every_k_steps, to_tensors
from mdm.models.building_blocks import *
from mdm.models.multiscale_model_mk2 import MultiscaleDynamicsModelMK2
from mdm.models.rnn_baseline import RnnBaselineModel
from mdm.training.dynamics_model_trainer import DynamicsModelTrainer
from mdm.memory.trajectory_memory import TrajectoryMemory
from mdm.training.gym_driver import GymEpisodeDriver
from mdm.planning.planning_policy import PlanningPolicy
from mdm.planning.cem_planner import CrossentropyPlanner
from mdm.training.offline_rl_driver import OfflineRLDriver, SamplingType
from mdm.logging.neptune_logger import NeptuneLogger
from mdm.logging.not_logger import NotLogger
from mdm.logging.logger import Scope
from mdm.policies.random_policy import RandomPolicy
from mdm.policies.predefined_policy import PredefinedPolicy
from mdm.training.gym_driver import act_in_env, act_in_vector_env
from mdm.utils.gym_wrappers import CacheLastStepEnv, CacheLastStepVecEnv
from mdm.utils.torch_tools import extract_sub_distribution, pack_rnn_state, unpack_rnn_state, TensorIndex
from mdm.training.gym_driver import collect_data


"""
def _collect_data(env: CacheLastStepVecEnv, n_steps: int, policy: callable = None):
    if policy is None:
        policy = RandomPolicy(env)

    traj_o, traj_a, traj_r, traj_term, traj_trunc, traj_mask = [], [], [], [], [], []
    act_in_vector_env(env, policy, n_steps, traj_o, traj_a, traj_r, traj_term, traj_trunc, traj_mask)
    o = np.stack(traj_o)
    a = np.stack(traj_a)
    r = np.stack(traj_r)
    term = np.stack(traj_term)
    trunc = np.stack(traj_trunc)
    mask = np.stack(traj_mask)

    if term.any() or trunc.any():
        print('!')

    mem = []
    l_trajs = (~mask).sum(axis=0).squeeze()
    for i_traj in range(o.shape[1]):
        l_traj = l_trajs[i_traj]
        traj = {'o': o[:l_traj, i_traj], 'a': a[:l_traj, i_traj], 'r': r[:l_traj, i_traj],
                'terminal': term[:l_traj, i_traj], 'truncated': trunc[:l_traj, i_traj]}
        mem.append(traj)

    return mem


def purge_invalid_trajectories(o: np.ndarray, a: np.ndarray, r: np.ndarray, terminal: np.ndarray,
                               truncated: np.ndarray, mask: np.ndarray):
    mem = []
    max_len, n_envs = o.shape[:2]
    l_trajs = (~mask).sum(axis=0).squeeze()
    for i_traj in range(n_envs):
        l_traj = l_trajs[i_traj]
        traj = {'o': o[:l_traj, i_traj].detach().cpu().numpy(),
                'a': a[:l_traj, i_traj].detach().cpu().numpy(),
                'r': r[:l_traj, i_traj].detach().cpu().numpy(),
                'terminal': terminal[:l_traj, i_traj].detach().cpu().numpy(),
                'truncated': truncated[:l_traj, i_traj].detach().cpu().numpy()}
        mem.append(traj)
    return mem
"""


def select_batch_items(mem: Dict[str, Union[torch.Tensor, torch.distributions.Distribution]],
                       i: TensorIndex,
                       keepdim: bool = False):
    if keepdim and type(i) is not slice:
        if isinstance(i, torch.Tensor) and i.size() == 1:
            i = i.detach().cpu().numpy().item()
            i = slice(i, i+1)

    ret = {}
    for name, val in mem.items():
        ret[name] = []
        for timestep in val:
            if timestep is None:
                ret[name].append(None)
            elif isinstance(timestep, torch.Tensor):
                ret[name].append(timestep[i])
            elif isinstance(timestep, torch.distributions.Distribution):
                ret[name].append(extract_sub_distribution(timestep, i))  # keepdim is handled by calling function
            elif 'rnn_state' in name:
                ret[name].append(unpack_rnn_state(pack_rnn_state(timestep)[i]))
            else:
                raise ValueError(f'Unknown memory content for key {name}: {timestep}')
    return ret


def plan_prim_with_warmup(model: MultiscaleDynamicsModelMK2,
                          planner_prim: CrossentropyPlanner,
                          env_data: Dict[str, torch.Tensor],
                          n_plan_steps: int,
                          n_rollouts: int,
                          n_warmup: int):
    n_envs = env_data['o'].shape[1]

    # repeat the starting data n_rollouts times per environment, so use repeat_interleave instead of repeat
    o_start_batch = torch.repeat_interleave(env_data['o'], n_rollouts, dim=1)
    a_start_batch = torch.repeat_interleave(env_data['a'], n_rollouts, dim=1)
    r_start_batch = torch.repeat_interleave(env_data['r'], n_rollouts, dim=1)
    term_start_batch = torch.repeat_interleave(env_data['terminal'], n_rollouts, dim=1)
    #o_start_batch = env_data['o'].repeat(1, n_rollouts, 1)
    #a_start_batch = env_data['a'].repeat(1, n_rollouts, 1)
    #r_start_batch = env_data['r'].repeat(1, n_rollouts, 1)
    #term_start_batch = env_data['term'].repeat(1, n_rollouts, 1)

    def _rollout_fn(_a: torch.Tensor):
        # fold 'env' dimension into batch dimension for rollout
        _a = _a.reshape(_a.shape[0] * _a.shape[1], *_a.shape[2:])
        _a = _a.swapaxes(0, 1)  # swap batch and time dim since model is time-first but planner is batch-first
        _a = torch.cat([a_start_batch, _a], dim=0)
        _mem = model(a=_a, o=o_start_batch, r=r_start_batch, term=term_start_batch,
                     n_warmup=n_warmup, sample=True)
        _criterion = _mem['prim_r'].squeeze(-1).swapaxes(0, 1)
        _discount = _mem['prim_term'].squeeze(-1).swapaxes(0, 1)

        _criterion = _criterion.reshape(n_envs, n_rollouts, _criterion.shape[-1])
        _discount = _discount.reshape(n_envs, n_rollouts, _discount.shape[-1])
        return _criterion, _discount, _mem

    a, a_dist, i_win, R_win, data = planner_prim.plan(rollout_fn=_rollout_fn, n_rollouts=n_rollouts,
                                                      n_plan_steps=n_plan_steps, n_envs=n_envs)

    # select winner batch item per per memory timestep
    data = select_batch_items(data, i_win[:, 0], keepdim=True)
    # CAUTION: the actions from planner are without the already performed warmup acitons!
    a_win = planner_prim.get_winner_actions(a, a_dist, i_win, resample=True)
    return a_win, data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-log', default=False, action='store_true')
    args = parser.parse_args()

    cfg = load_yaml(here() / 'cfg_rnn_train.yaml')
    planning_cfg = load_yaml(here() / 'cfg_rnn_plan.yaml')
    neptune_cfg = load_yaml(here() / cfg['neptune_cfg'])

    map_version = 'VeryEasy'
    env = gym.make(f'gym_nav2d:nav2d{map_version}-v0')
    env = CacheLastStepEnv(env)

    def make_env_fn():
        return gym.make(f'gym_nav2d:nav2d{map_version}-v0')

    # infer missing config values
    cfg['prim_mdl']['d_a'] = 2
    cfg['abstr_act_mdl']['d_a'] = 2
    cfg['abstr_act_mdl']['abstract_step_size'] = cfg['mdm']['abstract_step_size']
    cfg['abstr_act_mdl']['d_a_abstract'] = cfg['abstr_mdl']['d_a']

    # calculate missing model parameter dimensions
    #d_prim_s = cfg['prim_mdl']['d_h'] + cfg['prim_mdl']['d_z']
    d_prim_s = 32
    d_abstr_s = cfg['abstr_mdl']['d_h'] + cfg['abstr_mdl']['d_z']

    o_shape = env.observation_space.shape
    abstr_target = cfg['mdm']['abstract_pred_target']
    if abstr_target == 'prim_o':
        abstr_o_shape = o_shape
    elif abstr_target == 'prim_h':
        abstr_o_shape = cfg['prim_mdl']['d_h']
    elif abstr_target == 'prim_z':
        abstr_o_shape = cfg['prim_mdl']['d_z']
    else:
        raise ValueError(f'Unknown abstract o prediction target: {abstr_target}')

    # build model
    prim_obs_enc = MLPEncoder(s_x_orig=o_shape, **cfg['prim_o_enc'])
    prim_obs_dec = MLPDecoder(s_x_orig=o_shape, d_x_encoded=d_prim_s, final_activation='tanh', **cfg['prim_o_dec'])
    prim_r_dec = MLPDecoder(s_x_orig=1, d_x_encoded=d_prim_s, **cfg['prim_r_dec'])
    prim_term_dec = MLPDecoder(s_x_orig=1, d_x_encoded=d_prim_s, final_activation='sigmoid', **cfg['prim_term_dec'])
    prim_mdl = DeprecatedRSSM(obs_encoder=prim_obs_enc, obs_decoder=prim_obs_dec, r_decoder=prim_r_dec,
                              term_decoder=prim_term_dec, **cfg['prim_mdl'])

    #abstr_obs_enc = MLPEncoder(s_x_orig=abstr_o_shape, **cfg['abstr_o_enc'])
    #abstr_obs_dec = GaussianDecoder(s_x_orig=abstr_o_shape, d_x_encoded=d_abstr_s, **cfg['abstr_o_dec'])
    #abstr_r_dec = GaussianDecoder(s_x_orig=1, d_x_encoded=d_abstr_s, **cfg['abstr_r_dec'])
    #abstr_term_dec = BinomialDecoder(s_x_orig=1, d_x_encoded=d_abstr_s, **cfg['abstr_term_dec'])
    #abstr_mdl = RSSM(obs_encoder=abstr_obs_enc, obs_decoder=abstr_obs_dec, r_decoder=abstr_r_dec,
    #                 term_decoder=abstr_term_dec, **cfg['abstr_mdl'])

    #abstr_act_mdl = AbstractActionModel(**cfg['abstr_act_mdl'])

    #model = MultiscaleDynamicsModelMK2(primitive_model=prim_mdl, abstract_model=abstr_mdl,
    #                                   abstract_action_model=abstr_act_mdl, **cfg['mdm'])
    model = RnnBaselineModel(d_h=d_prim_s, d_a=cfg['prim_mdl']['d_a'], obs_encoder=prim_obs_enc,
                             obs_decoder=prim_obs_dec, r_decoder=prim_r_dec, term_decoder=prim_term_dec,
                             n_hidden_layers=3, train_multistep_predictions=False, multistep_prediction_stride=5,
                             random_warmup_training=True)
    model = model.to('cuda')
    model.training = True

    optim_type = cfg['optim'].pop('type')
    if optim_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), **cfg['optim'])
    elif optim_type == 'adamW':
        optimizer = torch.optim.AdamW(model.parameters(), **cfg['optim'])
    elif optim_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), **cfg['optim'])
    else:
        raise ValueError(f'Unknown optimizer type: {optim_type}')

    # build data pipeline
    # train_mem = TrajectoryMemory(step_count_environment)
    # random_driver = GymEpisodeDriver(env, lambda o, r, term, i_ep: env.action_space.sample())
    # planner_prim = CrossentropyPlanner(DistributionType.CATEGORICAL, d_dist=model.d_action,
    #                                   device=model.device, **planning_cfg['pln_prim'])
    # planner_abstr = CrossentropyPlanner(DistributionType.NORMAL, d_dist=model.d_abstract_action,
    #                                    device=model.device, **planning_cfg['pln_abstr'])
    # policy = PlanningPolicy(model=model, env=env, planner_prim=planner_prim, planner_abstr=planner_abstr,
    #                        n_rollouts=2048, plan_horizon_prim=30, plan_horizon_abstr=20, replan_interval_prim=3,
    #                        replan_interval_abstr=3, n_warmup_prim=3, n_warmup_abstr=1)
    # planning_driver = GymEpisodeDriver(env, policy)

    train_mem = load_memory(here() / cfg['train_samples'])
    train_driver = OfflineRLDriver(train_mem, sampling_type=SamplingType.RANDOM)
    test_mem = load_memory(here() / cfg['test_samples'])
    test_driver = OfflineRLDriver(test_mem, sampling_type=SamplingType.RANDOM)

    if args.log:
        logger = NeptuneLogger(neptune_cfg['PROJECT_NAME'], api_token=neptune_cfg['NEPTUNE_API_TOKEN'])
        logger.start_session()
        logger.log(cfg, Scope.HYPERPARAMETERS())
    else:
        logger = NotLogger()

    d_batch, pad = cfg['trainer']['d_batch'], cfg['trainer']['pad_last_terminal_flag']
    train_with_subtrajectories = cfg['trainer']['subtrajectory_len']

    n_evolution_steps = 40
    winning_perc = 0.2
    discount = 0.98
    act_noise = 0.0
    alpha = 1
    n_warmup = 2
    n_plan_steps = 50
    n_rollouts = 1024
    n_envs = 50

    planner_prim = CrossentropyPlanner(DistributionType.NORMAL, d_dist=2, act_noise=act_noise, discount=discount,
                                       n_evolution_steps=n_evolution_steps, winning_perc=winning_perc, alpha=alpha,
                                       device=model.device, debug_env=None,
                                       a_min=-1.0, a_max=1.0)

    collect_env = gym.vector.AsyncVectorEnv([make_env_fn] * n_envs)
    collect_env = CacheLastStepVecEnv(collect_env)

    def get_batch_train(i_step):
        # collect live data using current model for planning
        if i_step % 50 == 0:
            collect_env.reset()
            warmup_data_trajectories = collect_data(collect_env, n_warmup, RandomPolicy(collect_env))
            warmup_data = prepare_data(**to_tensors(warmup_data_trajectories, model.device))
            a_win, i_win = plan_prim_with_warmup(model, planner_prim, warmup_data, n_plan_steps, n_rollouts, n_warmup)
            collect_policy = PredefinedPolicy(collect_env, a_win.detach().cpu().numpy().swapaxes(0, 1))
            collected_data_trajectories = collect_data(collect_env, n_plan_steps - n_warmup - 1, collect_policy)
            mem = [{k: np.concatenate([wu[k], col[k]]) for k in wu}
                   for wu, col in zip(warmup_data_trajectories, collected_data_trajectories)]
            train_mem.extend(mem)

        batch = train_driver.interact(d_batch)
        batch = to_tensors(batch, model.device)
        batch = prepare_data(**batch)
        return batch['o'], batch['a'], batch['r'], batch['terminal'], batch['truncated'], batch['mask']

    def get_batch_test(i_step):
        collect_env.reset()
        warmup_data_trajectories = collect_data(collect_env, n_warmup, RandomPolicy(collect_env))
        warmup_data = prepare_data(**to_tensors(warmup_data_trajectories, model.device))
        a_win, i_win = plan_prim_with_warmup(model, planner_prim, warmup_data, n_plan_steps, n_rollouts, n_warmup)
        collect_policy = PredefinedPolicy(collect_env, a_win.detach().cpu().numpy().swapaxes(0, 1))
        collected_data_trajectories = collect_data(collect_env, n_plan_steps - n_warmup - 1, collect_policy)
        mem = [{k: np.concatenate([wu[k], col[k]]) for k in wu}
               for wu, col in zip(warmup_data_trajectories, collected_data_trajectories)]
        test_mem.extend(mem)

        batch = test_driver.interact(d_batch)
        batch = to_tensors(batch, model.device)
        batch = prepare_data(**batch)
        return batch['o'], batch['a'], batch['r'], batch['terminal'], batch['truncated'], batch['mask']

    model_path = f'{cfg["final_model_path"]}_{cfg["mdm"]["abstract_step_size"]}.ptmdl'

    # train
    fig = plt.figure(figsize=(10, 10))

    def eval_callback(o: torch.Tensor, a: torch.Tensor, r: torch.Tensor, term: torch.Tensor, mask, i_step: int):
        model.eval()
        pred = model(o, a, r, term, cfg['eval']['n_warmup_prim'], sample=cfg['prim_mdl']['stochastic_outputs'])
        model.train()

        prim_r_mean = pred['prim_r'].mean(dim=1).squeeze().detach().cpu().numpy()
        prim_r_std = pred['prim_r'].std(dim=1).squeeze().detach().cpu().numpy()
        prim_term_mean = pred['prim_term'].mean(dim=1).squeeze().detach().cpu().numpy()
        prim_term_std = pred['prim_term'].std(dim=1).squeeze().detach().cpu().numpy()

        plt.plot(prim_r_mean, label='mean')
        plt.plot(prim_r_std, label='std')
        plt.legend()
        logger.log_plot(fig_to_img(fig), Scope.PARAMETERS() / 'model_stats/prim_r', i_step)
        plt.plot(prim_term_mean, label='mean')
        plt.plot(prim_term_std, label='std')
        plt.legend()
        logger.log_plot(fig_to_img(fig), Scope.PARAMETERS() / 'model_stats/prim_term', i_step)

        losses = model.calc_loss(pred, o, r, term, mask)
        losses = {k: v.detach().cpu().numpy() for k, v in losses.items()}
        logger.log(losses, Scope.TEST() / 'with_warmup', i_step)

    trainer = DynamicsModelTrainer(model=model, optimizer=optimizer, get_batch_train=get_batch_train,
                                   get_batch_test=get_batch_test, logger=logger, eval_callback=eval_callback,
                                   **cfg['trainer'])
    trainer.train(n_train_steps=cfg['trainer']['n_train_steps'], progress_bar=True,
                  checkpoint_path=here() / cfg['checkpoint_path'])

    # check if output folder exists, if not create it
    p = here() / cfg['final_model_path'][:cfg['final_model_path'].rindex('/')]
    if not os.path.exists(p):
        os.makedirs(p)

    # query the run id from logger if neptune log is running
    if logger:
        model_path = f'{cfg["final_model_path"]}_{logger.run_id}.ptmdl'
    else:
        model_path = f'{cfg["final_model_path"]}.ptmdl'

    # store model and output run id if logger was active
    torch.save(model, here() / model_path)
    logger.start_session()
    logger.log_file(here() / model_path, Scope.DATA() / 'final_weights')
    logger.stop_session()

    if logger:
        print(logger.run_id)
