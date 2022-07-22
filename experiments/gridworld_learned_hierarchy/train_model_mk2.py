from pathlib import Path
import io
import argparse

import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image

from mdm.gridworld.gridworld import Gridworld
from mdm.utils.utils import here, load_yaml, prepare_data, fill_placeholders, discrete_stats
from mdm.models.building_blocks import RSSM, AbstractActionModel
from mdm.models.multiscale_model_mk2 import MultiscaleDynamicsModelMK2
from mdm.training.dynamics_model_trainer import DynamicsModelTrainer
from mdm.memory.trajectory_memory import TrajectoryMemory
from mdm.training.offline_rl_driver import OfflineRLDriver
from mdm.logging.neptune_logger import NeptuneLogger
from mdm.logging.logger import Scope
from mdm.training.data_loader import ConcurrentDataLoader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-log', default=False, action='store_true')
    args = parser.parse_args()

    cfg = load_yaml(here() / 'model_mk2.yaml')
    env = Gridworld.from_cleartext(here() / cfg['env'])
    neptune_cfg = load_yaml(here() / cfg['neptune_cfg'])

    # infer missing config values
    cfg['prim_mdl']['d_observation'] = env.observation_space.shape[0]
    cfg['prim_mdl']['d_action'] = env.action_space.n
    cfg['prim_mdl']['d_reward'] = 1
    prim_d_cell = 2 if cfg['prim_mdl']['rnn_type'] == 'lstm' else 1
    cfg['prim_mdl']['d_ctx_high_level'] = 0
    cfg['prim_mdl']['d_x_posterior'] = env.observation_space.shape[0] + 2  # observation, terminal flag and reward

    #cfg['abstr_mdl']['d_observation'] = cfg['prim_mdl']['d_state'] + cfg['prim_mdl']['d_hidden'] \
    #                                    * cfg['prim_mdl']['n_hidden_layers'] * prim_d_cell
    cfg['abstr_mdl']['d_observation'] = cfg['prim_mdl']['d_state']

    cfg['abstr_mdl']['d_ctx_high_level'] = 0
    cfg['abstr_mdl']['d_x_posterior'] = cfg['abstr_mdl']['d_observation'] + cfg['abstr_mdl']['d_reward'] + 1

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
    # optimizer = torch.optim.AdamW(model.parameters(), **cfg['optim'])

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

    #loader_train = ConcurrentDataLoader(get_batch_train, queue_len=3)
    #loader_test = ConcurrentDataLoader(get_batch_test, queue_len=1)
    model_path = f'{cfg["final_model_path"]}_{cfg["mdm"]["abstract_step_size"]}.ptmdl'

    # train
    if args.log:
        logger = NeptuneLogger(neptune_cfg['PROJECT_NAME'], api_token=neptune_cfg['NEPTUNE_API_TOKEN'])
        logger.start_session()
        logger.log(cfg, Scope.HYPERPARAMETERS())
    else:
        logger = None

    fig = plt.figure(figsize=(10, 10))

    def eval_callback(i_step: int):
        if model.abstract_step_size <= 10 and logger:
            Y_mean, Y_std, Y_mae = discrete_stats(model.abstract_action_model, env.action_space.n,
                                                  model.abstract_step_size, 1)
            plt.matshow(Y_mae, fignum=1)
            plt.colorbar()
            buffer = io.BytesIO()
            fig.savefig(buffer)
            plt.clf()
            buffer.seek(0)
            logger.log_plot(Image.open(buffer), Scope.PARAMETERS() / 'abstr_a_stats/plots', i_step)
            logger.log({'abstr_a_mean': Y_mean.mean(), 'abstr_a_std': Y_std.mean()},
                       Scope.PARAMETERS() / 'abstr_a_stats', i_step)


    trainer = DynamicsModelTrainer(model=model, optimizer=optimizer, get_batch_train=get_batch_train,
                                   get_batch_test=get_batch_test, logger=logger, eval_callback=eval_callback,
                                   **cfg['trainer'])
    trainer.train(n_train_steps=cfg['trainer']['n_train_steps'], progress_bar=True,
                  checkpoint_path=here() / cfg['checkpoint_path'])

    if logger:
        model_path = f'{cfg["final_model_path"]}_{logger.run_id}.ptmdl'
    else:
        model_path = f'{cfg["final_model_path"]}.ptmdl'

    torch.save(model, Path(__file__).parent / model_path)

    if logger:
        print(logger.run_id)
