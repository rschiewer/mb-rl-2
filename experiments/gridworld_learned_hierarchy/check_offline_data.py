from tqdm import tqdm

from mdm.gridworld.gridworld import Gridworld, FullyObservableGridworld
from mdm.memory.trajectory_memory import TrajectoryMemory
from mdm.utils.utils import here, load_yaml


if __name__ == '__main__':
    cfg = load_yaml(here() / 'model_cfg.yaml')

    map_version = 'v2'
    env = Gridworld.from_cleartext(here() / f'../../mdm/gridworld/8x8_{map_version}.mapdata')
    #env = FullyObservableGridworld(env)

    train_mem = TrajectoryMemory.load(here() / cfg['train_samples'])
    test_mem = TrajectoryMemory.load(here() / cfg['test_samples'])

    train_mem.plot_stats(bins=20)
    test_mem.plot_stats(bins=20)

    for traj in tqdm(train_mem, desc='Checking train samples'):
        s, a, r, terminal, w = traj.values()
        env.enact_sequence(s, a, r, terminal, False, 0)

    for traj in tqdm(test_mem, desc='Checking test samples'):
        s, a, r, terminal, w = traj.values()
        env.enact_sequence(s, a, r, terminal, False, 0)


