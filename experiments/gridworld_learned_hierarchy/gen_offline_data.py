from mdm.training.gym_driver import GymEpisodeDriver
from mdm.memory.trajectory_memory import TrajectoryMemory
from mdm.gridworld.gridworld import Gridworld
from mdm.utils.utils import here


if __name__ == '__main__':
    env = Gridworld.from_cleartext(here() / '../../mdm/gridworld/8x8_v1.mapdata')
    n_episodes_train = 10000
    n_episodes_test = 1000

    def collect_policy(observation):
        return env.action_space.sample()

    collect_driver = GymEpisodeDriver(env, collect_policy)
    train_mem = collect_driver.interact(n_episodes_train, True)
    test_mem = collect_driver.interact(n_episodes_test, True)

    TrajectoryMemory.store(train_mem, here() / 'gridworld_train.samples')
    TrajectoryMemory.store(test_mem, here() / 'gridworld_test.samples')


