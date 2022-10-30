from mdm.training.gym_driver import GymEpisodeDriver
from mdm.memory.trajectory_memory import TrajectoryMemory
from mdm.gridworld.gridworld import Gridworld, FullyObservableGridworld
from mdm.utils.utils import here, compute_returns

if __name__ == '__main__':
    map_version = 'v0'
    env = Gridworld.from_cleartext(here() / f'../../mdm/gridworld/8x8_{map_version}.mapdata')
    #env = FullyObservableGridworld(env)

    actions = [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1]
    actions += [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0]
    actions += [0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
    actions += [0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1]
    actions += [0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1]
    actions += [1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0]
    actions += [1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1]
    a_iter = iter(actions)

    def collect_policy(observation, reward, terminal, i_ep):
        return next(a_iter)

    collect_driver = GymEpisodeDriver(env, collect_policy)
    mem = collect_driver.interact(5, True)
    compute_returns(mem, 1.0)

    for traj in mem:
        print(traj['w'])

    TrajectoryMemory.store(mem, here() / f'gridworld_{map_version}_good_trajectories.samples')
