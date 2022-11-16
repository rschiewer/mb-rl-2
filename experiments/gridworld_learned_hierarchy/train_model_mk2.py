from pathlib import Path
import io
import argparse

import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image

from mdm.gridworld.gridworld import Gridworld, FullyObservableGridworld
from mdm.utils.utils import here, load_yaml, prepare_data, discrete_stats, compute_returns, fig_to_img
from mdm.utils.torch_tools import get_mu, get_sigma, bin_every_k_steps
from mdm.models.building_blocks import RSSM, AbstractActionModel, OneHotDecoder, OneHotEncoder, GaussianDecoder, BinomialDecoder
from mdm.models.multiscale_model_mk2 import MultiscaleDynamicsModelMK2
from mdm.training.dynamics_model_trainer import DynamicsModelTrainer
from mdm.memory.trajectory_memory import TrajectoryMemory
from mdm.training.gym_driver import GymEpisodeDriver
from mdm.planning.planning_policy import PlanningPolicy
from mdm.planning.cem_planner import CrossentropyPlanner
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
    planning_cfg = load_yaml(here() / 'planning_cfg.yaml')
    neptune_cfg = load_yaml(here() / cfg['neptune_cfg'])

    env = Gridworld.from_cleartext(here() / cfg['env'])
    #env = FullyObservableGridworld(env)

    # infer missing config values
    cfg['prim_mdl']['d_a'] = env.action_space.n
    cfg['abstr_act_mdl']['d_a'] = env.action_space.n
    cfg['abstr_act_mdl']['abstract_step_size'] = cfg['mdm']['abstract_step_size']
    cfg['abstr_act_mdl']['d_a_abstract'] = cfg['abstr_mdl']['d_a']

    # calculate missing model parameter dimensions
    d_prim_s = cfg['prim_mdl']['d_h'] + cfg['prim_mdl']['d_z']
    d_abstr_s = cfg['abstr_mdl']['d_h'] + cfg['abstr_mdl']['d_z']

    o_shape = (*env.observation_space.shape, max(env.grid_h, env.grid_w))
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
    prim_obs_enc = OneHotEncoder(s_x_orig=o_shape, **cfg['prim_o_enc'])
    prim_obs_dec = OneHotDecoder(s_x_orig=o_shape, d_x_encoded=d_prim_s, **cfg['prim_o_dec'])
    prim_r_dec = GaussianDecoder(s_x_orig=1, d_x_encoded=d_prim_s, **cfg['prim_r_dec'])
    prim_term_dec = BinomialDecoder(s_x_orig=1, d_x_encoded=d_prim_s, **cfg['prim_term_dec'])
    prim_mdl = RSSM(obs_encoder=prim_obs_enc, obs_decoder=prim_obs_dec, r_decoder=prim_r_dec,
                    term_decoder=prim_term_dec, **cfg['prim_mdl'])

    abstr_obs_enc = OneHotEncoder(s_x_orig=abstr_o_shape, **cfg['abstr_o_enc'])
    abstr_obs_dec = OneHotDecoder(s_x_orig=abstr_o_shape, d_x_encoded=d_abstr_s, **cfg['abstr_o_dec'])
    abstr_r_dec = GaussianDecoder(s_x_orig=1, d_x_encoded=d_abstr_s, **cfg['abstr_r_dec'])
    abstr_term_dec = BinomialDecoder(s_x_orig=1, d_x_encoded=d_abstr_s, **cfg['abstr_term_dec'])
    abstr_mdl = RSSM(obs_encoder=abstr_obs_enc, obs_decoder=abstr_obs_dec, r_decoder=abstr_r_dec,
                     term_decoder=abstr_term_dec, **cfg['abstr_mdl'])

    abstr_act_mdl = AbstractActionModel(**cfg['abstr_act_mdl'])

    model = MultiscaleDynamicsModelMK2(primitive_model=prim_mdl, abstract_model=abstr_mdl,
                                       abstract_action_model=abstr_act_mdl, **cfg['mdm'])
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

    if False:
        def log_fn_prim(m, grad_input, grad_output):
            if model._current_train_step % 50 != 0:
                return
            if isinstance(m, torch.nn.Sequential):
                mod_name = next(m.named_children())[0]
                mod_name = mod_name[:mod_name.rindex('_')]
            else:
                mod_name = m.__class__.__name__
            scope = Scope.PARAMETERS() / 'gradients' / 'prim_mdl' / mod_name
            if grad_input is None or grad_input[0] is None:
                grad_input_norm = -1,
            else:
                grad_input_norm = grad_input[0].norm()
            if grad_output is None or grad_output[0] is None:
                grad_output_norm = -1,
            else:
                grad_output_norm = grad_output[0].norm()
            logger.log({'grad_in_norm': grad_input_norm, 'grad_out_norm': grad_output_norm}, scope)

        def log_fn_abstr(m, grad_input, grad_output):
            if model._current_train_step % 10 != 0:
                return
            if isinstance(m, torch.nn.Sequential):
                mod_name = next(m.named_children())[0]
                mod_name = mod_name[:mod_name.rindex('_')]
            else:
                mod_name = m.__class__.__name__
            scope = Scope.PARAMETERS() / 'gradients' / 'abstr_mdl' / mod_name
            if grad_input is None or grad_input[0] is None:
                grad_input_norm = -1,
            else:
                grad_input_norm = grad_input[0].norm()
            if grad_output is None or grad_output[0] is None:
                grad_output_norm = -1,
            else:
                grad_output_norm = grad_output[0].norm()
            logger.log({'grad_in_norm': grad_input_norm, 'grad_out_norm': grad_output_norm}, scope)

        for module in model.primitive_model.children():
            module.register_full_backward_hook(log_fn_prim)

        for module in model.abstract_model.children():
            module.register_full_backward_hook(log_fn_abstr)

    # build data pipeline
    # train_mem = TrajectoryMemory()
    # random_driver = GymEpisodeDriver(env, lambda o, r, term, i_ep: env.action_space.sample())
    # planner_prim = CrossentropyPlanner(DistributionType.CATEGORICAL, d_dist=model.d_action,
    #                                   device=model.device, **planning_cfg['pln_prim'])
    # planner_abstr = CrossentropyPlanner(DistributionType.NORMAL, d_dist=model.d_abstract_action,
    #                                    device=model.device, **planning_cfg['pln_abstr'])
    # policy = PlanningPolicy(model=model, env=env, planner_prim=planner_prim, planner_abstr=planner_abstr,
    #                        n_rollouts=2048, plan_horizon_prim=30, plan_horizon_abstr=20, replan_interval_prim=3,
    #                        replan_interval_abstr=3, n_warmup_prim=3, n_warmup_abstr=1)
    # planning_driver = GymEpisodeDriver(env, policy)

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

    d_batch, pad = cfg['trainer']['d_batch'], cfg['trainer']['pad_last_terminal_flag']

    def get_batch_train(i_step):
        s, a, r, terminal, w = train_driver.interact(d_batch).to_np_arrays(dtype=np.float32,
                                                                           pad_last_terminal_flag=pad,
                                                                           pad_last_reward=pad)
        s, a, r, terminal = prepare_data(s, a, r, terminal, env)
        return s, a, r, terminal


    #def get_batch_train(i_step):
    #    global train_mem
    #    if i_step < 10:
    #        experience = random_driver.interact(10)
    #        train_mem += experience
    #    elif i_step % 100 == 0:
    #        experience = planning_driver.interact(5)
    #        train_mem += experience
    #    batch = train_mem.sample(d_batch, False)
    #    s, a, r, terminal, w, = batch.to_np_arrays(dtype=np.float32, pad_last_terminal_flag=pad)
    #    s, a, r, terminal = prepare_data(s, a, r, terminal, env)
    #    return s, a, r, terminal


    def get_batch_test(i_step):
        s, a, r, terminal, w = test_driver.interact(d_batch).to_np_arrays(dtype=np.float32,
                                                                          pad_last_terminal_flag=pad,
                                                                          pad_last_reward=pad)
        #experience = planning_driver.interact(1)
        #total_reward = experience[0]['r'].sum()
        #logger.log({'planning_r': total_reward}, Scope.TEST() / 'planning_reward', i_step)
        #s, a, r, terminal, w, = experience.to_np_arrays(dtype=np.float32, pad_last_terminal_flag=pad)
        s, a, r, terminal = prepare_data(s, a, r, terminal, env)
        return s, a, r, terminal

    model_path = f'{cfg["final_model_path"]}_{cfg["mdm"]["abstract_step_size"]}.ptmdl'

    # train
    fig = plt.figure(figsize=(10, 10))

    def eval_callback(o: torch.Tensor, a: torch.Tensor, r: torch.Tensor, term: torch.Tensor, mask, i_step: int):
        if model.abstract_step_size <= 10:
            Y_mean, Y_std, Y_mae = discrete_stats(model.abstract_action_model, env.action_space.n,
                                                  model.abstract_step_size, 10, {'sample': False})
            plt.matshow(Y_mae, fignum=1)
            plt.colorbar()
            logger.log_plot(fig_to_img(fig), Scope.PARAMETERS() / 'abstr_a_stats/plots', i_step)
            logger.log({'abstr_a_mean': Y_mean.mean(), 'abstr_a_std': Y_std.mean()},
                       Scope.PARAMETERS() / 'abstr_a_stats', i_step)

        logger.log({'abstr_o_dec': model.abstract_model.obs_decoder.temperature,
                    'prim_o_dec': model.primitive_model.obs_decoder.temperature},
                   Scope.TEST() / 'RelaxedOneHotCategorical', i_step)

        abstr_r = model.calc_abstr_r_ground_truth(r)
        abstr_term = model.calc_abstr_term_ground_truth(term)
        model.eval()
        pred = model(o, a, r, term, abstr_r, abstr_term, cfg['eval']['n_warmup_prim'], cfg['eval']['n_warmup_abstr'])
        model.train()

        prim_r_mean = torch.stack([r_dist.loc for r_dist in pred['prim_r_dist']]).mean(dim=1).squeeze().detach().cpu().numpy()
        prim_r_std = torch.stack([r_dist.scale for r_dist in pred['prim_r_dist']]).mean(dim=1).squeeze().detach().cpu().numpy()
        prim_term_mean = torch.stack([term_dist.probs for term_dist in pred['prim_term_dist']]).mean(dim=1).squeeze().detach().cpu().numpy()
        prim_term_std = torch.stack([term_dist.probs for term_dist in pred['prim_term_dist']]).std(dim=1).squeeze().detach().cpu().numpy()
        abstr_r_mean = torch.stack([r_dist.loc for r_dist in pred['abstr_r_dist']]).mean(dim=1).detach().cpu().numpy()
        abstr_r_std= torch.stack([r_dist.scale for r_dist in pred['abstr_r_dist']]).mean(dim=1).detach().cpu().numpy()
        abstr_term_mean = torch.stack([term_dist.probs for term_dist in pred['abstr_term_dist']]).mean(dim=1).detach().cpu().numpy()
        abstr_term_std = torch.stack([term_dist.probs for term_dist in pred['abstr_term_dist']]).std(dim=1).detach().cpu().numpy()

        plt.plot(prim_r_mean, label='mean')
        plt.plot(prim_r_std, label='std')
        plt.legend()
        logger.log_plot(fig_to_img(fig), Scope.PARAMETERS() / 'model_stats/prim_r', i_step)
        plt.plot(prim_term_mean, label='mean')
        plt.plot(prim_term_std, label='std')
        plt.legend()
        logger.log_plot(fig_to_img(fig), Scope.PARAMETERS() / 'model_stats/prim_term', i_step)
        plt.plot(abstr_r_mean, label='mean')
        plt.plot(abstr_r_std, label='std')
        plt.legend()
        logger.log_plot(fig_to_img(fig), Scope.PARAMETERS() / 'model_stats/abstr_r', i_step)
        plt.plot(abstr_term_mean, label='mean')
        plt.plot(abstr_term_std, label='std')
        plt.legend()
        logger.log_plot(fig_to_img(fig), Scope.PARAMETERS() / 'model_stats/abstr_term', i_step)

        losses = model.calc_loss(pred, o, r, term, abstr_r, abstr_term, mask, 1)
        losses = {k: v.detach().cpu().numpy() for k, v in losses.items()}
        logger.log(losses, Scope.TEST() / 'with_warmup', i_step)

    def train_callback(o: torch.Tensor, a: torch.Tensor, r: torch.Tensor, term: torch.Tensor, mask, i_step: int):
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
