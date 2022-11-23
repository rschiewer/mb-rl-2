import random
import pickle
from multiprocessing import Pool

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from mdm.training.gym_driver import GymEpisodeDriver
from mdm.memory.trajectory_memory import TrajectoryMemory
from mdm.gridworld.gridworld import Gridworld, CellType, FullyObservableGridworld
from mdm.utils.utils import here, to_np_arrays, store_memory


class TabularQLearningPolicy:

    def __init__(self, grid_w: int, grid_h: int, n_actions: int, alpha: float, gamma:float, epsilon: float,
                 improvement_bound: float, patience: int):
        self.q = np.zeros((grid_w, grid_h, n_actions), dtype=np.float32)
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.improvement_bound = improvement_bound
        self.patience = patience
        self._current_patience = patience

    def _translate_obs(self, s):
        if np.shape(s)[-1] == 2:
            return s
        else:
            if np.ndim(s) == 1:
                s = np.reshape(s, self.q.shape[0: 2])
                s = np.argwhere(s == CellType.AGENT).squeeze()
            else:
                d_batch = np.shape(s)[0]
                s = np.reshape(s, (d_batch, *self.q.shape[0: 2]))
                s = np.argwhere(s == CellType.AGENT)[:, 1:].squeeze()
                if len(s) != d_batch:
                    pad = np.zeros((d_batch - len(s), *np.shape(s)[1:]), dtype=int)
                    s = np.concatenate([s, pad], axis=0)
        return s

    def _update_vec(self, s_t, a_t, r_t, term_t, s_tt):
        s_t = self._translate_obs(s_t)
        s_tt = self._translate_obs(s_tt)
        target = r_t + self.gamma * self.q[s_tt[:, 0], s_tt[:, 1]].max(axis=-1)
        new_q = (1 - self.alpha) * self.q[s_t[:, 0], s_t[:, 1], a_t] + self.alpha * target
        diff = np.sum(np.abs(self.q[s_t[:, 0], s_t[:, 1], a_t] - new_q))
        self.q[s_t[:, 0], s_t[:, 1], a_t] = new_q
        return diff

    def train(self, mem: list, d_batch: int):
        if len(mem) < d_batch: return
        if self._current_patience == 0: return

        batch = random.sample(mem, d_batch)
        s, a, r, terminal, truncated, _ = to_np_arrays(batch, dtypes=(int, int, float, bool, bool))
        #batch = mem.sample(d_batch, False)
        #s, a, r, terminal, w = batch.to_np_arrays(dtype=(int, int, float, bool, float))
        delta = 0

        for s_t, a_t, r_t, term_t, s_tt in zip(s[:, :-1], a[:, 1:], r[:, 1:], terminal[:, 1:], s[:, 1:]):
            delta += self._update_vec(s_t, a_t, r_t, term_t, s_tt)
        delta /= d_batch

        # test if we can stop training
        if delta < self.improvement_bound:
            self._current_patience -= 1
        else:
            self._current_patience = self.patience

        return delta

    def decide(self, o):
        if np.random.rand() < self.epsilon:
            a = np.random.choice(range(self.n_actions))
        else:
            o = self._translate_obs(o)
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
    map_version = 'v0'
    env = Gridworld.from_cleartext(here() / f'../../mdm/gridworld/8x8_{map_version}.mapdata')
    #env = FullyObservableGridworld(env)
    n_episodes_train = 50000
    perc_test = 0.1
    disjunct_train_test = False
    expert_trajectories = 0.80

    train_mem = []#TrajectoryMemory()
    if expert_trajectories > 0:
        agent = TabularQLearningPolicy(env.grid_h, env.grid_w, env.action_space.n, 0.1, 0.99, 0.1,
                                       improvement_bound=1e-4, patience=10)

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
    collect_driver.interact(round(n_episodes_train * expert_trajectories), True, train_mem)
    print(len(train_mem))

    rand_driver = GymEpisodeDriver(env, lambda *args: env.action_space.sample())
    rand_driver.interact(n_episodes_train - len(train_mem), True, train_mem)
    print(len(train_mem))

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
    random.shuffle(train_mem)
    train_mem = train_mem[n_episodes_test:]
    test_mem = train_mem[:n_episodes_test]

    store_memory(train_mem, here() / f'gridworld_{map_version}_train.samples')
    store_memory(test_mem, here() / f'gridworld_{map_version}_test.samples')
    #TrajectoryMemory.store(train_mem, here() / f'gridworld_{map_version}_train.samples')
    #TrajectoryMemory.store(test_mem, here() / f'gridworld_{map_version}_test.samples')


