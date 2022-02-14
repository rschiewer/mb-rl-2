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


if __name__ == '__main__':
    unittest.main()
