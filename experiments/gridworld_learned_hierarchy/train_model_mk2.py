from pathlib import Path
import io
import argparse

import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image

from mdm.gridworld.gridworld import Gridworld
from mdm.utils.utils import here, load_yaml, prepare_data, fill_placeholders, discrete_stats, compute_returns
from mdm.utils.torch_tools import get_mu, get_sigma, bin_every_k_steps
from mdm.models.building_blocks import RSSM, AbstractActionModel
from mdm.models.multiscale_model_mk2 import MultiscaleDynamicsModelMK2
from mdm.training.dynamics_model_trainer import DynamicsModelTrainer
from mdm.memory.trajectory_memory import TrajectoryMemory
from mdm.training.gym_driver import GymEpisodeDriver
from mdm.planning.planning_policy import PlanningPolicy
from mdm.planning.cem_planner import CrossentropyPlanner, DistributionType
from mdm.training.offline_rl_driver import OfflineRLDriver, SamplingType
from mdm.logging.neptune_logger import NeptuneLogger
from mdm.logging.not_logger import NotLogger
from mdm.logging.logger import Scope
from mdm.training.data_loader import ConcurrentDataLoader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-log', default=False, action='store_true')
    args = parser.parse_args()

    cfg = load_yaml(here() / 'model_cfg.yaml')
    env = Gridworld.from_cleartext(here() / cfg['env'])
    neptune_cfg = load_yaml(here() / cfg['neptune_cfg'])

    # infer missing config values
    cfg['prim_mdl']['d_o'] = env.observation_space.shape[0]
    cfg['prim_mdl']['d_a'] = env.action_space.n
    cfg['prim_mdl']['d_r'] = 1
    cfg['prim_mdl']['d_ctx_high_level'] = 0
    cfg['prim_mdl']['d_x_posterior'] = env.observation_space.shape[0] + 2  # observation, terminal flag and reward

    #prim_d_cell = 2 if cfg['prim_mdl']['rnn_type'] == 'lstm' else 1
    #cfg['abstr_mdl']['d_observation'] = cfg['prim_mdl']['d_state'] + cfg['prim_mdl']['d_hidden'] \
    #                                    * cfg['prim_mdl']['n_hidden_layers'] * prim_d_cell
    prim_mdl_key = cfg['mdm']['abstract_pred_target']
    prim_mdl_key = prim_mdl_key[prim_mdl_key.index('_') + 1:]
    prim_mdl_key = 'd_' + prim_mdl_key
    cfg['abstr_mdl']['d_o'] = cfg['prim_mdl'][prim_mdl_key]

    cfg['abstr_mdl']['d_ctx_high_level'] = 0
    #cfg['abstr_mdl']['d_x_posterior'] = cfg['abstr_mdl']['d_observation'] + cfg['abstr_mdl']['d_reward'] + \
    #                                    1 + cfg['prim_mdl']['d_hidden']
    cfg['abstr_mdl']['d_x_posterior'] = cfg['abstr_mdl']['d_o'] + cfg['abstr_mdl']['d_r'] + 1

    cfg['abstr_act_mdl']['d_a'] = env.action_space.n
    cfg['abstr_act_mdl']['abstract_step_size'] = cfg['mdm']['abstract_step_size']
    cfg['abstr_act_mdl']['d_a_abstract'] = cfg['abstr_mdl']['d_a']

    # build model
    prim_mdl = RSSM(**cfg['prim_mdl'])
    abstr_mdl = RSSM(**cfg['abstr_mdl'])
    abstr_act_mdl = AbstractActionModel(**cfg['abstr_act_mdl'])
    model = MultiscaleDynamicsModelMK2(primitive_model=prim_mdl, abstract_model=abstr_mdl,
                                       abstract_action_model=abstr_act_mdl, **cfg['mdm'])
    model = model.to('cuda')
    model.training = True
    optimizer = torch.optim.Adam(model.parameters(), **cfg['optim'])
    # optimizer = torch.optim.AdamW(model.parameters(), **cfg['optim'])

    # build data pipeline
    train_mem = TrajectoryMemory()
    random_driver = GymEpisodeDriver(env, lambda o, r, term, i_ep: env.action_space.sample())
    planner_prim = CrossentropyPlanner(DistributionType.CATEGORICAL, device=model.device)
    planner_abstr = CrossentropyPlanner(DistributionType.NORMAL, device=model.device)
    policy = PlanningPolicy(model=model, env=env, planner_prim=planner_prim, planner_abstr=planner_abstr,
                            n_rollouts=2048, n_plan_steps_abstr=30, n_optim_steps_prim=10,
                            n_optim_steps_abstr=10, winning_perc=0.2, discount=0.95, replan_interval=5,
                            act_noise_prim=0.01, act_noise_abstr=0.01, n_warmup_prim=model.n_warmup_prim,
                            n_warmup_abstr=model.n_warmup_abstr)
    planning_driver = GymEpisodeDriver(env, policy)
    d_batch, pad = cfg['trainer']['d_batch'], cfg['trainer']['pad_last_terminal_flag']

    train_mem = TrajectoryMemory.load(here() / cfg['train_samples'])
    compute_returns(train_mem)
    train_driver = OfflineRLDriver(train_mem, sampling_type=SamplingType.RANDOM)
    test_mem = TrajectoryMemory.load(here() / cfg['test_samples'])
    compute_returns(test_mem)
    test_driver = OfflineRLDriver(test_mem, sampling_type=SamplingType.RANDOM)

    if args.log:
        logger = NeptuneLogger(neptune_cfg['PROJECT_NAME'], api_token=neptune_cfg['NEPTUNE_API_TOKEN'])
        logger.start_session()
        logger.log(cfg, Scope.HYPERPARAMETERS())
    else:
        logger = NotLogger()

    def get_batch_train(i_step):
        s, a, r, terminal, w = train_driver.interact(d_batch).to_np_arrays(dtype=np.float32,
                                                                           pad_last_terminal_flag=pad)
        s, a, r, terminal = prepare_data(s, a, r, terminal, env)
        return s, a, r, terminal

    #def get_batch_train(i_step):
    #    global train_mem
    #    if i_step < 100:
    #        experience = random_driver.interact(10)
    #        train_mem += experience
    #    elif i_step % 100 == 0:
    #        experience = planning_driver.interact(1)
    #        train_mem += experience
    #    batch = train_mem.sample(d_batch, False)
    #    s, a, r, terminal, w, = batch.to_np_arrays(dtype=np.float32, pad_last_terminal_flag=pad)
    #    s, a, r, terminal = prepare_data(s, a, r, terminal, env)
    #    return s, a, r, terminal


    def get_batch_test(i_step):
        s, a, r, terminal, w = test_driver.interact(d_batch).to_np_arrays(dtype=np.float32, pad_last_terminal_flag=True)
        #experience = planning_driver.interact(1)
        #total_reward = experience[0]['r'].sum()
        #logger.log({'planning_r': total_reward}, Scope.TEST() / 'planning_reward', i_step)
        #s, a, r, terminal, w, = experience.to_np_arrays(dtype=np.float32, pad_last_terminal_flag=pad)
        s, a, r, terminal = prepare_data(s, a, r, terminal, env)
        return s, a, r, terminal


    #loader_train = ConcurrentDataLoader(get_batch_train, queue_len=10)
    #loader_test = ConcurrentDataLoader(get_batch_test, queue_len=10)
    #get_batch_train = loader_train.get_batch
    #get_batch_test = loader_test.get_batch
    model_path = f'{cfg["final_model_path"]}_{cfg["mdm"]["abstract_step_size"]}.ptmdl'

    # train

    fig = plt.figure(figsize=(10, 10))


    def eval_callback(o: torch.Tensor, a: torch.Tensor, r: torch.Tensor, term: torch.Tensor, i_step: int):
        if model.abstract_step_size <= 10:
            Y_mean, Y_std, Y_mae = discrete_stats(model.abstract_action_model, env.action_space.n,
                                                  model.abstract_step_size, 10)
            plt.matshow(Y_mae, fignum=1)
            plt.colorbar()
            buffer = io.BytesIO()
            fig.savefig(buffer)
            plt.clf()
            buffer.seek(0)
            logger.log_plot(Image.open(buffer), Scope.PARAMETERS() / 'abstr_a_stats/plots', i_step)
            logger.log({'abstr_a_mean': Y_mean.mean(), 'abstr_a_std': Y_std.mean()},
                       Scope.PARAMETERS() / 'abstr_a_stats', i_step)

        abstr_r = bin_every_k_steps(r, model.abstract_step_size, padding_val=0).sum(dim=2)
        abstr_term = bin_every_k_steps(term, model.abstract_step_size).max(dim=2).values
        pred = model(o, a, r, term, abstr_r, abstr_term, model.n_warmup_prim, model.n_warmup_abstr)
        for key in ('prim_o', 'prim_r', 'abstr_o', 'abstr_r'):
            full_key = key + '_dist'
            mu = get_mu(pred[full_key]).mean()
            sigma = get_sigma(pred[full_key]).mean()
            logger.log({full_key + '_mean': mu, full_key + '_sigma': sigma},
                       Scope.PARAMETERS() / 'model_stats', i_step)


    def train_callback(i_step: int):
        pass
        #logger.log({'n_warmup_prim': model.n_warmup_prim,
        #            'n_warmup_abstr': model.n_warmup_abstr},
        #           Scope.TRAIN(), i_step)


    trainer = DynamicsModelTrainer(model=model, optimizer=optimizer, get_batch_train=get_batch_train,
                                   get_batch_test=get_batch_test, logger=logger, train_callback=train_callback,
                                   eval_callback=eval_callback,
                                   **cfg['trainer'])
    trainer.train(n_train_steps=cfg['trainer']['n_train_steps'], progress_bar=True,
                  checkpoint_path=here() / cfg['checkpoint_path'])

    if logger:
        model_path = f'{cfg["final_model_path"]}_{logger.run_id}.ptmdl'
    else:
        model_path = f'{cfg["final_model_path"]}.ptmdl'

    torch.save(model, here() / model_path)
    logger.start_session()
    logger.log_file(here() / model_path, Scope.DATA() / 'final_weights')
    logger.stop_session()

    if logger:
        print(logger.run_id)
