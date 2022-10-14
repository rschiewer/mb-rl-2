from tqdm import tqdm

from mdm.gridworld.gridworld import Gridworld
from mdm.memory.trajectory_memory import TrajectoryMemory
from mdm.utils.utils import here


if __name__ == '__main__':
    map_version = 'v0'
    env = Gridworld.from_cleartext(here() / f'../../mdm/gridworld/8x8_{map_version}.mapdata')
    train_mem = TrajectoryMemory.load(here() / 'gridworld_train.samples')
    test_mem = TrajectoryMemory.load(here() / 'gridworld_test.samples')

    for traj in tqdm(train_mem, desc='Checking train samples'):
        s, a, r, terminal, w = traj.values()
        env.enact_sequence(s, a, r, terminal, False, 0)

    for traj in tqdm(test_mem, desc='Checking test samples'):
        s, a, r, terminal, w = traj.values()
        env.enact_sequence(s, a, r, terminal, False, 0)
