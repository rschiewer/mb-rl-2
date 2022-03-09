from pathlib import Path

import numpy as np

from mdm.training.dynamics_model_trainer import DynamicsModelTrainer, np_one_hot
from mdm.memory.trajectory_memory import flatten_and_unsqueeze
from mdm.training.gym_driver import GymEpisodeDriver
from mdm.training.offline_rl_driver import OfflineRLDriver
from mdm.memory.trajectory_memory import TrajectoryMemory
from mdm.models.multiscale_model import *
from mdm.gridworld.gridworld import Gridworld
from mdm.utils.utils import here, load_yaml
from mdm.logging.NeptuneLogger import NeptuneLogger


if __name__ == '__main__':
    env = Gridworld.from_cleartext(here() / '../../mdm/gridworld/8x8_v0.mapdata')
    neptune_cfg = load_yaml(here() / '../neptune_settings.yaml')

    mdl_d_state = env.observation_space.shape[0]
    mdl_d_action = env.action_space.n
    mdl_d_reward = 1
    mdl_d_hidden = 128
    mdl_n_rec_layers = 2
    mdl_d_macro_state = 3
    mdl_d_macro_action = 8
    mdl_d_macro_reward = 1
    mdl_d_macro_hidden = 128
    mdl_n_macro_rec_layers = 2
    mdl_n_abstract_steps = 5

    trainer_d_batch = 512
    trainer_n_warmup_steps = 1
    trainer_n_train_steps = 8000
    trainer_n_eval_interval = 50

    train_mem = TrajectoryMemory.load(here() / 'gridworld_train.samples')
    train_driver = OfflineRLDriver(train_mem)
    test_mem = TrajectoryMemory.load(here() / 'gridworld_test.samples')
    test_driver = OfflineRLDriver(test_mem)

    # build model
    single_step_mdl = build_single_step_model(mdl_d_macro_state, mdl_d_macro_action, mdl_d_macro_reward, mdl_d_state,
                                              mdl_d_action, mdl_d_reward, mdl_d_hidden, mdl_n_rec_layers)
    abstract_mdl = build_abstract_model(mdl_d_macro_state, mdl_d_macro_action, mdl_d_macro_reward, mdl_d_macro_hidden,
                                        mdl_n_macro_rec_layers, mdl_d_hidden, mdl_n_rec_layers)
    macro_action_mdl = MacroActionModel(mdl_d_action, mdl_n_abstract_steps, mdl_d_macro_action)
    multiscale_mdl = MultiscaleDynamicsModel(single_step_mdl, abstract_mdl, macro_action_mdl, mdl_n_abstract_steps,
                                             mdl_d_state, mdl_d_action, mdl_d_reward, mdl_d_macro_state,
                                             mdl_d_macro_action, mdl_d_macro_reward)
    multiscale_mdl = multiscale_mdl.to('cuda')
    optimizer = torch.optim.Adam(multiscale_mdl.parameters(), lr=0.001, weight_decay=0.0001)

    # train model
    def get_batch_train():
        s, a, r, terminal = train_driver.interact(trainer_d_batch).to_np_arrays(dtype=np.float32)
        s, a, r, terminal = flatten_and_unsqueeze(s, a, r, terminal)
        s /= (env.grid_h - 1, env.grid_w - 1)
        a = np_one_hot(a.astype(np.int64), n_categories=mdl_d_action)
        return s, a, r, terminal

    def get_batch_test():
        s, a, r, terminal = test_driver.interact(trainer_d_batch).to_np_arrays(dtype=np.float32)
        s, a, r, terminal = flatten_and_unsqueeze(s, a, r, terminal)
        s /= (env.grid_h - 1, env.grid_w - 1)
        a = np_one_hot(a.astype(np.int64), n_categories=mdl_d_action)
        return s, a, r, terminal

    neptune_logger = NeptuneLogger(neptune_cfg['PROJECT_NAME'], api_token=neptune_cfg['NEPTUNE_API_TOKEN'])
    trainer = DynamicsModelTrainer(multiscale_mdl, optimizer, get_batch_train, get_batch_test, trainer_n_warmup_steps,
                                   trainer_n_eval_interval, logger=neptune_logger)

    trainer.train(trainer_n_train_steps, progress_bar=True)

    torch.save(multiscale_mdl, Path(__file__).parent / 'model.ptmdl')






