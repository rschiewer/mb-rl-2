from pathlib import Path

import numpy as np

from mdm.training.dynamics_model_trainer import DynamicsModelTrainer, flatten_and_unsqueeze, one_hot
from mdm.training.gym_driver import GymEpisodeDriver
from mdm.models.multiscale_model import *
from mdm.gridworld.gridworld import Gridworld


if __name__ == '__main__':
    env = Gridworld.from_cleartext(Path(__file__).parent / '../mdm/gridworld/8x8_v0.mapdata')
    n_episodes = 1000

    mdl_d_state = env.observation_space.shape[0]
    mdl_d_action = env.action_space.n
    mdl_d_reward = 1
    mdl_d_hidden = 32
    mdl_n_rec_layers = 2
    mdl_d_macro_state = 3
    mdl_d_macro_action = 8
    mdl_d_macro_reward = 1
    mdl_n_abstract_steps = 2

    trainer_d_batch = 128
    trainer_n_warmup_steps = 3
    trainer_n_train_steps = 500
    trainer_n_eval_interval = 10

    def collect_policy(observation):
        return env.action_space.sample()

    collect_driver = GymEpisodeDriver(env, collect_policy)

    single_step_mdl = build_single_step_model(mdl_d_macro_state, mdl_d_macro_action, mdl_d_macro_reward, mdl_d_state,
                                              mdl_d_action, mdl_d_reward, mdl_d_hidden, mdl_n_rec_layers)
    abstract_mdl = build_abstract_model(mdl_d_macro_state, mdl_d_macro_action, mdl_d_macro_reward, mdl_d_hidden,
                                        mdl_n_rec_layers)
    macro_action_mdl = MacroActionModel(mdl_d_action, mdl_n_abstract_steps, mdl_d_macro_action)
    multiscale_mdl = MultiscaleDynamicsModel(single_step_mdl, abstract_mdl, macro_action_mdl, mdl_n_abstract_steps,
                                             mdl_d_state, mdl_d_action, mdl_d_reward, mdl_d_macro_state,
                                             mdl_d_macro_action, mdl_d_macro_reward)
    optimizer = torch.optim.Adam(multiscale_mdl.parameters(), lr=0.001)

    def get_batch_train():
        s, a, r, terminal = collect_driver.interact(trainer_d_batch).to_np_arrays(dtype=np.float32)
        s, a, r, terminal = flatten_and_unsqueeze(s, a, r, terminal)
        s /= (env.grid_h - 1, env.grid_w - 1)
        a = one_hot(a.astype(np.int64), n_categories=mdl_d_action)
        return s, a, r, terminal

    def get_batch_test():
        s, a, r, terminal = collect_driver.interact(trainer_d_batch).to_np_arrays(dtype=np.float32)
        s, a, r, terminal = flatten_and_unsqueeze(s, a, r, terminal)
        s /= (env.grid_h - 1, env.grid_w - 1)
        a = one_hot(a.astype(np.int64), n_categories=mdl_d_action)
        return s, a, r, terminal

    trainer = DynamicsModelTrainer(multiscale_mdl, optimizer, get_batch_train, get_batch_test, trainer_n_warmup_steps,
                                   trainer_n_eval_interval)


    trainer.train(trainer_n_train_steps, progress_bar=True)


