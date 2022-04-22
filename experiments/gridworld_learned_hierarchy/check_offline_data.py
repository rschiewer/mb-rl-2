from tqdm import tqdm

from mdm.gridworld.gridworld import Gridworld
from mdm.memory.trajectory_memory import TrajectoryMemory
from mdm.utils.utils import here


if __name__ == '__main__':
    env = Gridworld.from_cleartext(here() / '../../mdm/gridworld/8x8_v0.mapdata')
    train_mem = TrajectoryMemory.load(here() / 'gridworld_train.samples').shuffle()
    test_mem = TrajectoryMemory.load(here() / 'gridworld_test.samples').shuffle()

    for traj in tqdm(train_mem, desc='Checking train samples'):
        s, a, r, terminal = traj.values()
        env.enact_sequence(s, a, r, terminal, False, 0)

    for traj in tqdm(test_mem, desc='Checking test samples'):
        s, a, r, terminal = traj.values()
        env.enact_sequence(s, a, r, terminal, False, 0)
