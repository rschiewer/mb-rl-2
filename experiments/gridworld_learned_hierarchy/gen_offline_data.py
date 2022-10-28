from multiprocessing import Pool

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from mdm.training.gym_driver import GymEpisodeDriver
from mdm.memory.trajectory_memory import TrajectoryMemory
from mdm.gridworld.gridworld import Gridworld
from mdm.utils.utils import here


class TabularQLearningPolicy:

    def __init__(self, grid_w: int, grid_h: int, n_actions: int, alpha: float, gamma:float, epsilon: float):
        self.q = np.zeros((grid_w, grid_h, n_actions), dtype=np.float32)
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def _update(self, s_t, a_t, r_t, term_t, s_tt):
        s_t = tuple(s_t)
        s_tt = tuple(s_tt)
        target = r_t
        if not term_t:  # unnecessary for q initialized to zeros
            target += self.gamma * self.q[s_tt].max()

        q_new = (1 - self.alpha) * self.q[s_t][a_t] + self.alpha * target
        delta = np.abs(self.q[s_t][a_t] - q_new)
        self.q[s_t][a_t] = q_new
        return delta

    def _update_vec(self, s_t, a_t, r_t, term_t, s_tt):
        target = r_t + self.gamma * self.q[s_tt[:, 0], s_tt[:, 1]].max(axis=-1)
        new_q = (1 - self.alpha) * self.q[s_t[:, 0], s_t[:, 1], a_t] + self.alpha * target
        diff = np.sum(np.abs(self.q[s_t[:, 0], s_t[:, 1], a_t] - new_q))
        self.q[s_t[:, 0], s_t[:, 1], a_t] = new_q
        return diff

    def train(self, mem: TrajectoryMemory, d_batch: int):
        if len(mem) == 0:
            return

        batch = mem.sample(d_batch, False)
        s, a, r, terminal, w = batch.to_np_arrays(dtype=(int, int, float, bool, float))
        delta = 0

        for s_t, a_t, r_t, term_t, s_tt in zip(s[:, :-1], a[:, 1:], r[:, 1:], terminal[:, 1:], s[:, 1:]):
            delta += self._update_vec(s_t, a_t, r_t, term_t, s_tt)

        return delta / d_batch

        for n in range(d_batch):
            for s_t, a_t, r_t, term_t, s_tt in zip(s[n, :-1], a[n, 1:], r[n, 1:], terminal[n, 1:], s[n, 1:]):
                delta += self._update(s_t, a_t, r_t, term_t, s_tt)
                if term_t:
                    break
        return delta / d_batch

    def decide(self, o):
        if np.random.rand() < self.epsilon:
            a = np.random.choice(range(self.n_actions))
        else:
            aa = self.q[tuple(o.astype(int))]
            a = np.argmax(np.random.random(aa.shape) * (aa == aa.max()))
        return a


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
    map_version = 'v2'
    env = Gridworld.from_cleartext(here() / f'../../mdm/gridworld/8x8_{map_version}.mapdata')
    n_episodes_train = 20000
    perc_test = 0.1
    disjunct_train_test = False
    expert_trajectories = True

    train_mem = TrajectoryMemory()
    if expert_trajectories:
        agent = TabularQLearningPolicy(env.grid_h, env.grid_w, env.action_space.n, 0.1, 0.99, 0.1)

        def collect_policy(o, r, term, i_ep):
            a = agent.decide(o)
            if i_ep % 100 == 0:
                last_err = agent.train(train_mem, 128)
                #print(last_err)
            return a
    else:
        agent = None

        def collect_policy(*args):
            return env.action_space.sample()

    collect_driver = GymEpisodeDriver(env, collect_policy)
    train_mem = collect_driver.interact(n_episodes_train, True, train_mem)

    if agent:
        plt.matshow(agent.q.max(axis=2))
        plt.show()

    if disjunct_train_test:
        train_mem = remove_duplicates_mp(train_mem, n_proc=14)
        print(f'Removed {len(train_mem) - len(train_mem)} duplicate trajectories from sample memory.')

        # for testing only
        #train_mem_cleaned_2 = remove_duplicates(train_mem)
        #print(f'Removed {len(train_mem) - len(train_mem_cleaned_2)} duplicate trajectories from sample memory.')

    n_episodes_test = round(len(train_mem) * perc_test)
    trai_mem = train_mem.shuffle()
    train_mem = train_mem[n_episodes_test:]
    test_mem = train_mem[:n_episodes_test]

    TrajectoryMemory.store(train_mem, here() / f'gridworld_{map_version}_train.samples')
    TrajectoryMemory.store(test_mem, here() / f'gridworld_{map_version}_test.samples')


