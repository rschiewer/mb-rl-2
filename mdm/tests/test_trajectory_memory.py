import copy
import unittest
from random import shuffle
from typing import Dict

import numpy as np

from mdm.memory.trajectory_memory import TrajectoryMemory


class TrajectoryMemoryTest(unittest.TestCase):

    @staticmethod
    def _rand_traj(shapes, t_len, dtype=np.float32) -> Dict[str, np.ndarray]:
        def _rand_tens(shape, tens_len, dtype):
            size = (tens_len, *shape)
            if np.issubdtype(dtype, np.integer):
                return np.random.default_rng().integers(0, 10, size=size, dtype=dtype)
            elif np.issubdtype(dtype, np.bool):
                return np.random.default_rng().choice([True, False], size=size)
            else:
                return np.random.default_rng().uniform(size=size)

        init_a = np.zeros((1, *shapes['a']), dtype=dtype)
        init_r = np.zeros((1, *shapes['r']), dtype=dtype)
        init_term = np.zeros((1, *shapes['terminal']), dtype=dtype)

        s = _rand_tens(shapes['s'], t_len, dtype)
        a = np.concatenate([init_a, _rand_tens(shapes['a'], t_len - 1, dtype)], axis=0)
        r = np.concatenate([init_r, _rand_tens(shapes['r'], t_len - 1, dtype)], axis=0)
        terminal = np.concatenate([init_term, _rand_tens(shapes['terminal'], t_len - 1, dtype)], axis=0)
        return {'s': s, 'a': a, 'r': r, 'terminal': terminal}

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
            self.mem.push(**t)

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
            self.mem.push(**t)

        for chosen_shape_key in self.shapes:
            new_dim, new_len = np.random.default_rng().integers(10, 50, 2)
            new_shapes = {k: s if k != chosen_shape_key else s + (new_dim,) for k, s in self.shapes.items()}
            new_traj = self._rand_traj(new_shapes, new_len)
            self.assertRaises(ValueError, self.mem.push, **new_traj)
            self.assertEqual(len(self.mem), self.n_trajectories)

    def test_push_len(self):
        for t in self.trajectories:
            self.mem.push(**t)

        for k in ['s', 'a', 'r', 'terminal']:
            for traj in self.mem:
                t_malformed = copy.deepcopy(traj)
                len_mismatch = np.random.default_rng().integers(1, 500)
                shape_current = t_malformed[k].shape
                t_malformed[k] = t_malformed[k] = np.zeros((shape_current[0] + len_mismatch, *shape_current[1:]))
                self.assertRaises(ValueError, self.mem.push, **t_malformed)

    def test_push_dtype(self):
        for t in self.trajectories:
            self.mem.push(**t)

        t_malformed = self._rand_traj(self.shapes, 10, np.int32)
        self.assertRaises(ValueError, self.mem.push, **t_malformed)

        t_malformed = self._rand_traj(self.shapes, 10, self.mem.dtypes['s'])
        t_malformed['s'] = t_malformed['s'].astype(np.int32)
        self.assertRaises(ValueError, self.mem.push, **t_malformed)

        t_ok = self._rand_traj(self.shapes, 10, self.mem.dtypes['s'])  # TODO: check with mixed dtypes
        self.mem.push(**t_ok)

    def test_to_numpy_arrays_pad_all(self):
        for t in self.trajectories:
            self.mem.push(**t)

        for fill_value in [0, 1, 42]:
            ss_, as_, rs_, terminals_ = self.mem.to_np_arrays(padding=fill_value, pad_last_terminal_flag=False)

            for x in [ss_, as_, rs_, terminals_]:
                self.assertEqual(x.shape[0], self.n_trajectories)
                self.assertEqual(x.shape[1], self.mem.longest_trajectory)

            for shp, x in zip(self.shapes.values(), [ss_, as_, rs_, terminals_]):
                self.assertEqual(x.shape[2:], shp)

            for i_t, traj in enumerate(self.trajectories):
                for (name, traj_data), np_data in zip(traj.items(), [ss_[i_t], as_[i_t], rs_[i_t], terminals_[i_t]]):
                    l_orig = len(traj_data)
                    l_diff = np_data.shape[0] - l_orig
                    dt_orig = self.mem.dtypes[name]
                    # non-padded part of current trajectory is the same as the original trajectory
                    self.assertTrue((np_data[:l_orig] == traj_data).all())
                    # padded part should all be the padding value
                    if l_diff > 0:
                        self.assertTrue((np_data[l_orig:] == fill_value).all())

    def test_to_numpy_arrays_repeat_last_terminal(self):
        for t in self.trajectories:
            self.mem.push(**t)

        for fill_value in [0, 1, 42]:
            ss_, as_, rs_, terminals_ = self.mem.to_np_arrays(padding=fill_value, pad_last_terminal_flag=True)

            for x in [ss_, as_, rs_, terminals_]:
                self.assertEqual(x.shape[0], self.n_trajectories)
                self.assertEqual(x.shape[1], self.mem.longest_trajectory)

            for shp, x in zip(self.shapes.values(), [ss_, as_, rs_, terminals_]):
                self.assertEqual(x.shape[2:], shp)

            for i_t, traj in enumerate(self.trajectories):
                for (name, traj_data), np_data in zip(traj.items(), [ss_[i_t], as_[i_t], rs_[i_t], terminals_[i_t]]):
                    l_orig = len(traj_data)
                    l_diff = np_data.shape[0] - l_orig
                    # non-padded part of current trajectory is the same as the original trajectory
                    self.assertTrue((np_data[:l_orig] == traj_data).all())
                    # padded part should all be the padding value
                    if l_diff > 0:
                        cmp_val = traj_data[-1] if name == 'terminal' else fill_value
                        self.assertTrue((np_data[l_orig:] == cmp_val).all())

    def test_getitem_slice(self):
        for t in self.trajectories:
            self.mem.push(**t)

        # slices produce new TrajectoryMemory objects, indexes do not
        sub_mem = self.mem[-3:]
        self.assertEqual(sub_mem.__class__, self.mem.__class__)
        self.assertEqual(len(sub_mem), 3)
        single_element = self.mem[1]
        self.assertNotEqual(single_element.__class__, self.mem.__class__)

        # make sure sub_mem and self.mem share the same memory for common elements (i.e. avoid copies)
        traj_new = self._rand_traj(self.shapes, 10)
        self.mem[-1]['s'] = traj_new['s']
        self.assertTrue((sub_mem[-1]['s'] == traj_new['s']).all())

        # make sure sub_mem doesn't change if self.mem does
        self.assertEqual(sub_mem[-1], self.mem[-1])
        self.mem.push(**self.trajectories[0])
        self.assertEqual(sub_mem[-1], self.mem[-2])

    def test_get_view(self):
        empty_view = self.mem.get_view()

        for t in self.trajectories:
            self.mem.push(**t)

        full_view = self.mem.get_view()

        self.assertEqual(len(empty_view), 0)
        self.assertEqual(len(full_view), len(self.mem))

        # make sure sub_mem and self.mem share the same memory for common elements (i.e. avoid copies)
        traj_new = self._rand_traj(self.shapes, 10)
        self.mem[-1]['s'] = traj_new['s']
        self.assertTrue((full_view[-1]['s'] == traj_new['s']).all())

    def test_shuffle(self):
        for t in self.trajectories:
            self.mem.push(**t)

        shuffled = self.mem.shuffle()

        indices = list(range(len(self.mem)))
        shuffle(indices)

        mismatches = 0
        for i in indices:
            traj_mem = self.mem[i]
            traj_orig = self.trajectories[i]
            traj_shuffled = shuffled[i]
            equal = all([np.all(lhs == rhs) for lhs, rhs in zip(traj_mem.values(), traj_orig.values())])
            mismatches += np.sum([lhs.shape[0] != rhs.shape[0]  # trajectory lengths should differ in some cases
                                  for lhs, rhs in zip(traj_mem.values(), traj_shuffled.values())])
            self.assertTrue(equal)
        self.assertNotEqual(mismatches, 0)

    def test_cmp_trajectories(self):
        for t in self.trajectories:
            self.mem.push(**t)

        for t_mem, t_orig in zip(self.mem, self.trajectories):
            self.assertTrue(self.mem.cmp_trajectories(t_mem, t_orig))

        # if accidentally two consecutive trajectories are the same, this fails while the code might sill be ok
        for i_t, t_mem in enumerate(self.mem[:-1]):
            t_orig = self.trajectories[i_t + 1]
            self.assertFalse(self.mem.cmp_trajectories(t_mem, t_orig))

        # test if compare fails of only a subset of trajectory info differs
        traj = self.mem[0]
        traj_altered = copy.deepcopy(self.mem[0])
        traj_altered['s'][0] += 10
        self.assertFalse(self.mem.cmp_trajectories(traj, traj_altered))

    def test_add_matching(self):
        for t in self.trajectories:
            self.mem.push(**t)

        traj_new_second_mem = self._rand_traj(self.shapes, 10)
        other = TrajectoryMemory([traj_new_second_mem])
        added = self.mem + other

        self.assertEqual(len(added), len(self.mem) + 1)

        # modify samples from first memory and check if that affects added
        traj_new_first_mem = self._rand_traj(self.shapes, 10)
        self.mem[-1]['s'] = traj_new_first_mem['s']

        self.assertTrue((added[-1]['s'] == traj_new_second_mem['s']).all())
        self.assertTrue((added[-2]['s'] == traj_new_first_mem['s']).all())

        # modify sample in second memory and check if that affects added
        other[-1]['s'][:] = 0
        self.assertTrue((added[-1]['s'] == 0).all())

    def test_add_mismatching(self):
        self.mem.push(**self.trajectories[0])

        new_shapes = copy.deepcopy(self.mem.shapes)
        new_shapes['s'] = tuple(np.array(new_shapes['s']) + 1)
        traj_new = self._rand_traj(new_shapes, 10)
        other = TrajectoryMemory([traj_new])

        with self.assertRaises(ValueError):
            added = self.mem + other


if __name__ == '__main__':
    unittest.main()
