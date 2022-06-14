from pathlib import Path
import os

import torch
import numpy as np

from mdm.gridworld.gridworld import Gridworld
from mdm.utils.utils import here, load_yaml, prepare_data
from mdm.models.building_blocks import RSSM, AbstractActionModel
from mdm.models.multiscale_model_mk2 import MultiscaleDynamicsModelMK2
from mdm.training.dynamics_model_trainer import DynamicsModelTrainer
from mdm.memory.trajectory_memory import TrajectoryMemory
from mdm.training.offline_rl_driver import OfflineRLDriver
from mdm.logging.neptune_logger import NeptuneLogger
from mdm.logging.logger import Scope

if __name__ == '__main__':
    cfg = load_yaml(here() / 'model_mk2.yaml')
    env = Gridworld.from_cleartext(here() / cfg['env'])
    neptune_cfg = load_yaml(here() / cfg['neptune_cfg'])

    # infer missing config values
    cfg['prim_mdl']['d_observation'] = env.observation_space.shape[0]
    cfg['prim_mdl']['d_action'] = env.action_space.n
    cfg['prim_mdl']['d_reward'] = 1
    d_cell = 2 if cfg['prim_mdl']['rnn_type'] == 'lstm' else 1
    #cfg['prim_mdl']['d_ctx_high_level'] = cfg['abstr_mdl']['d_state'] + cfg['abstr_mdl']['d_hidden'] \
    #                                      * cfg['abstr_mdl']['n_hidden_layers'] * d_cell
    cfg['prim_mdl']['d_ctx_high_level'] = cfg['abstr_mdl']['d_hidden'] + cfg['abstr_mdl']['d_state']
    cfg['prim_mdl']['d_x_posterior'] = env.observation_space.shape[0] + 2  # observation, terminal flag and reward

    cfg['abstr_mdl']['d_observation'] = 0
    cfg['abstr_mdl']['o_lws'] = (0,)
    cfg['abstr_mdl']['d_ctx_high_level'] = 0
    d_cell = 2 if cfg['prim_mdl']['rnn_type'] == 'lstm' else 1
    #cfg['abstr_mdl']['d_x_posterior'] = cfg['prim_mdl']['n_hidden_layers'] * cfg['prim_mdl']['d_hidden'] * d_cell
    cfg['abstr_mdl']['d_x_posterior'] = cfg['prim_mdl']['d_hidden'] + cfg['prim_mdl']['d_state']

    cfg['abstr_act_mdl']['d_action'] = env.action_space.n
    cfg['abstr_act_mdl']['abstract_step_size'] = cfg['mdm']['abstract_step_size']
    cfg['abstr_act_mdl']['d_abstract_action'] = cfg['abstr_mdl']['d_action']

    # build model
    prim_mdl = RSSM(**cfg['prim_mdl'])
    abstr_mdl = RSSM(**cfg['abstr_mdl'])
    abstr_act_mdl = AbstractActionModel(**cfg['abstr_act_mdl'])
    model = MultiscaleDynamicsModelMK2(primitive_model=prim_mdl, abstract_model=abstr_mdl,
                                       abstract_action_model=abstr_act_mdl, **cfg['mdm'])
    model = model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), **cfg['optim'])
    #optimizer = torch.optim.AdamW(model.parameters(), **cfg['optim'])

    # build data pipeline
    train_mem = TrajectoryMemory.load(here() / cfg['train_samples']).shuffle()
    train_driver = OfflineRLDriver(train_mem)
    test_mem = TrajectoryMemory.load(here() / cfg['test_samples']).shuffle()
    test_driver = OfflineRLDriver(test_mem)
    d_batch, pad = cfg['trainer']['d_batch'], cfg['trainer']['pad_last_terminal_flag']
    def get_batch_train():
        s, a, r, terminal = train_driver.interact(d_batch).to_np_arrays(dtype=np.float32, pad_last_terminal_flag=pad)
        s, a, r, terminal = prepare_data(s, a, r, terminal, env)
        return s, a, r, terminal
    def get_batch_test():
        s, a, r, terminal = test_driver.interact(d_batch).to_np_arrays(dtype=np.float32, pad_last_terminal_flag=pad)
        s, a, r, terminal = prepare_data(s, a, r, terminal, env)
        return s, a, r, terminal

    # train
    if os.environ.get('LOG_RUN', 0):
        logger = NeptuneLogger(neptune_cfg['PROJECT_NAME'], api_token=neptune_cfg['NEPTUNE_API_TOKEN'])
        logger.setup()
        logger.log(cfg, Scope.PARAMETERS())
    else:
        logger = None

    trainer = DynamicsModelTrainer(model=model, optimizer=optimizer, get_batch_train=get_batch_train,
                                   get_batch_test=get_batch_test, logger=logger, **cfg['trainer'])
    trainer.train(n_train_steps=cfg['trainer']['n_train_steps'], progress_bar=True,
                  checkpoint_path=here() / cfg['checkpoint_path'])
    torch.save(model, Path(__file__).parent / cfg['final_model_path'])




