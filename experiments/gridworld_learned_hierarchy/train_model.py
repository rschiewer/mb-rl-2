from pathlib import Path
import time

import numpy as np

from mdm.training.dynamics_model_trainer import DynamicsModelTrainer
from mdm.memory.trajectory_memory import flatten_and_unsqueeze
from mdm.training.gym_driver import GymEpisodeDriver
from mdm.training.offline_rl_driver import OfflineRLDriver
from mdm.memory.trajectory_memory import TrajectoryMemory
from mdm.models.multiscale_model import *
from mdm.gridworld.gridworld import Gridworld
from mdm.utils.utils import here, load_yaml, np_one_hot, prepare_data
from mdm.logging.neptune_logger import NeptuneLogger


if __name__ == '__main__':
    env = Gridworld.from_cleartext(here() / '../../mdm/gridworld/8x8_v0.mapdata')
    neptune_cfg = load_yaml(here() / '../neptune_settings.yaml')

    mdl_d_state = env.observation_space.shape[0]
    mdl_d_action = env.action_space.n
    mdl_d_reward = 1
    mdl_d_macro_state = 4
    mdl_d_macro_action = 4
    mdl_d_macro_reward = 1
    mdl_n_abstract_steps = 2
    ss_mdl_n_rec_layers = 2
    ss_mdl_d_hidden = 64
    ss_mdl_s_lws = (128, 64)
    ss_mdl_r_lws = (64, 64)
    ss_mdl_term_lws = (64, 64)
    a_mdl_ff_lws = (128, 128)
    a_mdl_s_prior_lws = (128, 64)
    a_mdl_s_post_lws = (128, 64)
    a_mdl_r_prior_lws = (128, 64)
    a_mdl_r_post_lws = (128, 64)
    a_mdl_term_prior_lws = (128, 64)
    a_mdl_term_post_lws = (128, 64)

    trainer_d_batch = 1024
    trainer_n_warmup_steps = 1
    trainer_n_train_steps = 6000
    trainer_n_eval_interval = 100
    trainer_lr = 0.003
    trainer_betas = (0.90, 0.999)
    trainer_weight_decay = 0.02

    train_mem = TrajectoryMemory.load(here() / 'gridworld_train.samples').shuffle()
    #train_mem_experts = TrajectoryMemory.load(here() / 'test_rollouts.samples').shuffle()
    #for i in range(100):
    #    train_mem += train_mem_experts
    train_driver = OfflineRLDriver(train_mem)
    test_mem = TrajectoryMemory.load(here() / 'gridworld_test.samples').shuffle()
    test_driver = OfflineRLDriver(test_mem)

    # build model
    single_step_mdl = build_single_step_model(mdl_d_macro_state, mdl_d_macro_action, mdl_d_state,
                                              mdl_d_action, mdl_d_reward, ss_mdl_d_hidden, ss_mdl_n_rec_layers,
                                              ss_mdl_s_lws, ss_mdl_r_lws, ss_mdl_term_lws)
    abstract_mdl = build_abstract_model(mdl_d_macro_state, mdl_d_macro_action, mdl_d_macro_reward, ss_mdl_d_hidden,
                                        ss_mdl_n_rec_layers, a_mdl_ff_lws, a_mdl_s_prior_lws, a_mdl_s_post_lws,
                                        a_mdl_r_prior_lws, a_mdl_r_post_lws, a_mdl_term_prior_lws, a_mdl_term_post_lws)
    macro_action_mdl = MacroActionModel(mdl_d_action, mdl_n_abstract_steps, mdl_d_macro_action)
    multiscale_mdl = MultiscaleDynamicsModel(single_step_mdl, abstract_mdl, macro_action_mdl, mdl_n_abstract_steps,
                                             mdl_d_state, mdl_d_action, mdl_d_reward, mdl_d_macro_state,
                                             mdl_d_macro_action, mdl_d_macro_reward)

    #multiscale_mdl = torch.load(here() / 'model.ptmdl')
    multiscale_mdl = multiscale_mdl.to('cuda')
    #optimizer = torch.optim.AdamW(multiscale_mdl.parameters(), lr=trainer_lr, weight_decay=trainer_weight_decay)
    #optimizer = torch.optim.Adam(multiscale_mdl.parameters(), lr=trainer_lr, betas=trainer_betas)
    optimizer = torch.optim.Adam(multiscale_mdl.parameters(), lr=trainer_lr)

    # train model
    def get_batch_train():
        s, a, r, terminal = train_driver.interact(trainer_d_batch).to_np_arrays(dtype=np.float32, pad_last_terminal_flag=False)
        s, a, r, terminal = prepare_data(s, a, r, terminal, env)
        return s, a, r, terminal

    def get_batch_test():
        s, a, r, terminal = test_driver.interact(trainer_d_batch).to_np_arrays(dtype=np.float32, pad_last_terminal_flag=False)
        s, a, r, terminal = prepare_data(s, a, r, terminal, env)
        return s, a, r, terminal

    neptune_logger = NeptuneLogger(neptune_cfg['PROJECT_NAME'], api_token=neptune_cfg['NEPTUNE_API_TOKEN'])
    trainer = DynamicsModelTrainer(multiscale_mdl, optimizer, get_batch_train, get_batch_test, trainer_n_warmup_steps,
                                   trainer_n_eval_interval, logger=neptune_logger)

    trainer.train(trainer_n_train_steps, progress_bar=True, checkpoint_path=here() /'checkpoints/')

    torch.save(multiscale_mdl, Path(__file__).parent / 'model.ptmdl')






