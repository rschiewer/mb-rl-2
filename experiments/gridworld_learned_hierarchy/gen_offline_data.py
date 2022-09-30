from multiprocessing import Pool

from tqdm import tqdm

from mdm.training.gym_driver import GymEpisodeDriver
from mdm.memory.trajectory_memory import TrajectoryMemory
from mdm.gridworld.gridworld import Gridworld
from mdm.utils.utils import here


class TrajectoryIsUnique:
    def __init__(self, mem: TrajectoryMemory):
        self.mem = mem

    def __call__(self, curr_traj):
        for comp_traj in self.mem:
            if curr_traj is comp_traj:
                continue
            elif self.mem.cmp_trajectories(curr_traj, comp_traj):
                return False
        return True


def remove_duplicates(mem: TrajectoryMemory):
    uniques = []
    for curr_traj in mem:
        unique = True
        for comp_traj in mem:
            if curr_traj is comp_traj:
                continue
            elif mem.cmp_trajectories(curr_traj, comp_traj):
                unique = False
                break
        if unique:
            uniques.append(curr_traj)
    return TrajectoryMemory(uniques)


def remove_duplicates_filter(mem: TrajectoryMemory):
    def is_unique(curr_traj):
        for comp_traj in mem:
            if curr_traj is comp_traj:
                continue
            elif mem.cmp_trajectories(curr_traj, comp_traj):
                return False
        return True
    uniques = filter(is_unique, mem)
    return TrajectoryMemory(uniques)


def remove_duplicates_mp(mem: TrajectoryMemory, n_proc: int = 10):
    is_unique_check = TrajectoryIsUnique(mem)

    with Pool(n_proc) as p:
        unique_flags = tqdm(p.imap(is_unique_check, mem, chunksize=50), desc='Filtering out duplicates', total=len(mem))
        uniques = [traj for unique, traj in zip(unique_flags, mem) if unique]

    return TrajectoryMemory(uniques)


if __name__ == '__main__':
    env = Gridworld.from_cleartext(here() / '../../mdm/gridworld/8x8_v1.mapdata')
    n_episodes_train = 50000
    perc_test = 0.10

    def collect_policy(observation):
        return env.action_space.sample()

    collect_driver = GymEpisodeDriver(env, collect_policy)
    train_mem = collect_driver.interact(n_episodes_train, True)

    train_mem_cleaned = remove_duplicates_mp(train_mem, n_proc=14)
    print(f'Removed {len(train_mem) - len(train_mem_cleaned)} duplicate trajectories from sample memory.')

    # for testing only
    #train_mem_cleaned_2 = remove_duplicates(train_mem)
    #print(f'Removed {len(train_mem) - len(train_mem_cleaned_2)} duplicate trajectories from sample memory.')

    train_mem_cleaned = train_mem_cleaned.shuffle()
    n_episodes_test = round(len(train_mem_cleaned) * perc_test)
    train_mem = train_mem_cleaned[n_episodes_test:]
    test_mem = train_mem_cleaned[:n_episodes_test]

    #n_episodes_test = round(len(train_mem) * perc_test)
    #trai_mem = train_mem.shuffle()
    #train_mem = train_mem[n_episodes_test:]
    #test_mem = train_mem[:n_episodes_test]

    TrajectoryMemory.store(train_mem, here() / 'gridworld_train.samples')
    TrajectoryMemory.store(test_mem, here() / 'gridworld_test.samples')


