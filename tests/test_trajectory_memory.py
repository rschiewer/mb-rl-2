import copy
import unittest
from random import shuffle

import numpy as np

from memory.trajectory_memory import TrajectoryMemory


class TrajectoryMemoryTest(unittest.TestCase):

    @staticmethod
    def _rand_traj(shapes, t_len):
        def _rand_tens(shape, tens_len):
            return np.random.default_rng().uniform(size=(tens_len, *shape))

        return {'s': _rand_tens(shapes['s'], t_len + 1), 'a': _rand_tens(shapes['a'], t_len),
                'r': _rand_tens(shapes['r'], t_len), 'terminal': _rand_tens(shapes['terminal'], t_len)}

    def setUp(self) -> None:
        n_trajectories = 100
        min_len, longest = np.random.default_rng().integers(10, 50, 2)
        s_shape, a_shape, r_shape, terminal_shape = [(32, 32, 3), (5,), (), ()]
        t_lens = np.random.default_rng().integers(min_len, min_len + longest, n_trajectories)

        self.mem = TrajectoryMemory()
        self.shapes = {'s': s_shape, 'a': a_shape, 'r': r_shape, 'terminal': terminal_shape}
        self.n_trajectories = n_trajectories
        self.trajectories = [self._rand_traj(self.shapes, t_len) for t_len in t_lens]

    def test_init_with_dict(self):
        mem = TrajectoryMemory(self.trajectories)

        for traj_mem, traj_orig in zip(mem, self.trajectories):
            for traj_mem_elem, traj_orig_elem in zip(traj_mem.values(), traj_orig.values()):
                self.assertTrue((traj_mem_elem == traj_orig_elem).all())

    def test_init_with_sequence(self):
        traj_list_format = [[traj['s'], traj['a'], traj['r'], traj['terminal']] for traj in self.trajectories]
        mem = TrajectoryMemory(traj_list_format)

        for traj_mem, traj_orig in zip(mem, self.trajectories):
            for traj_mem_elem, traj_orig_elem in zip(traj_mem.values(), traj_orig.values()):
                self.assertTrue((traj_mem_elem == traj_orig_elem).all())

    def test_push(self):
        for t in self.trajectories:
            self.mem.push(t['s'], t['a'], t['r'], t['terminal'])

        self.assertEqual(len(self.mem), self.n_trajectories)
        self.assertEqual(self.mem.shapes, self.shapes)

        indices = list(range(len(self.mem)))
        shuffle(indices)

        for i in indices:
            traj_mem = self.mem[i]
            traj_orig = self.trajectories[i]
            equal = all([np.all(lhs == rhs) for lhs, rhs in zip(traj_mem.values(), traj_orig.values())])
            self.assertTrue(equal)

    def test_push_shape(self):
        for t in self.trajectories:
            self.mem.push(t['s'], t['a'], t['r'], t['terminal'])

        for chosen_shape_key in self.shapes:
            new_dim, new_len = np.random.default_rng().integers(10, 50, 2)
            new_shapes = {k: s if k != chosen_shape_key else s + (new_dim,) for k, s in self.shapes.items()}
            new_traj = self._rand_traj(new_shapes, new_len)
            self.assertRaises(ValueError, self.mem.push, *new_traj.values())
            self.assertEqual(len(self.mem), self.n_trajectories)

    def test_push_len(self):
        for t in self.trajectories:
            self.mem.push(t['s'], t['a'], t['r'], t['terminal'])

        for k in ['s', 'a', 'r', 'terminal']:
            for traj in self.mem:
                t_malformed = copy.deepcopy(traj)
                len_mismatch = np.random.default_rng().integers(1, 500)
                shape_current = t_malformed[k].shape
                t_malformed[k] = t_malformed[k] = np.zeros((shape_current[0] + len_mismatch, *shape_current[1:]))
                self.assertRaises(ValueError, self.mem.push, *t_malformed.values())

    def test_to_numpy_arrays(self):
        for t in self.trajectories:
            self.mem.push(t['s'], t['a'], t['r'], t['terminal'])

        for fill_value in [0, 1, 42]:
            ss_, as_, rs_, terminals_ = self.mem.to_np_arrays(padding=fill_value)

            for x in [ss_, as_, rs_, terminals_]:
                self.assertEqual(x.shape[0], self.n_trajectories)
                self.assertEqual(x.shape[1], self.mem.longest_trajectory)

            for shp, x in zip(self.shapes.values(), [ss_, as_, rs_, terminals_]):
                self.assertEqual(x.shape[2:], shp)

            for i_t, traj in enumerate(self.trajectories):
                for traj_data, np_data in zip(traj.values(), [ss_[i_t], as_[i_t], rs_[i_t], terminals_[i_t]]):
                    l_orig = len(traj_data)
                    l_diff = np_data.shape[0] - l_orig
                    # non-padded part of current trajectory is the same as the original trajectory
                    self.assertTrue((np_data[:l_orig] == traj_data).all())
                    # padded part should all be the padding value
                    if l_diff > 0:
                        self.assertTrue((np_data[l_orig:] == fill_value).all())


if __name__ == '__main__':
    unittest.main()
