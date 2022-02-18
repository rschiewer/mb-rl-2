from collections import deque
from typing import Dict, List, Iterable, Sequence, Union

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import numpy as np


class TrajectoryMemory(Dataset):

    def __init__(self, init_mem: List = None):
        self._mem = deque()
        self.shapes = None
        self.dtypes = None
        self.longest_trajectory = 0

        if init_mem is not None:
            for elem in init_mem:
                try:
                    if type(elem) is dict:
                        self.push(elem['s'], elem['a'], elem['r'], elem['terminal'])
                    elif isinstance(elem, Sequence):
                        self.push(elem[0], elem[1], elem[2], elem[3])
                    else:
                        raise ValueError('Unknown content in init_mem, elements should have type dict, list or tuple '
                                         f'but are of type {type(elem)}')
                except KeyError:
                    raise ValueError('Expected dict keys of elements are "s", "a", "r" and "terminal", found: '
                                     f'{elem.keys()}')
                except IndexError:
                    raise ValueError(f'Expected length of elements is 4, found {len(elem)}')

    def __getitem__(self, index) -> T_co:
        return self._mem[index]

    def __len__(self) -> int:
        return len(self._mem)

    def push(self, s, a, r, terminal) -> None:
        if self.shapes is None:
            self.shapes = self._detect_shapes(s, a, r, terminal)
            self.dtypes = self._detect_dtypes(s, a, r, terminal)
        else:
            data_shapes = self._detect_shapes(s, a, r, terminal)
            data_lengths = self._detect_lengths(s, a, r, terminal)
            data_dtypes = self._detect_dtypes(s, a, r, terminal)
            if not self._shapes_match(self.shapes, data_shapes):
                raise ValueError(f'Input has incompatible shape, expected {self.shapes}, found {data_shapes}')
            if not self._lengths_match(data_lengths):
                raise ValueError(f'Input has incompatible lengths, expected a, r, terminal to have equal lengths and'
                                 f' s to have on additional element')
            if not self._dtypes_match(self.dtypes, data_dtypes):
                raise ValueError(f'Input has different dtypes than previously added content, expected {self.dtypes}, '
                                 f'found {data_dtypes}')

        if len(s) > self.longest_trajectory:
            self.longest_trajectory = len(s)

        self._mem.append({'s': np.array(s), 'a': np.array(a), 'r': np.array(r), 'terminal': np.array(terminal)})

    def to_np_arrays(self, padding: float = 0, dtype: Union[np.dtype, Iterable[np.dtype]] = None):
        if dtype is None:
            dtype = self.dtypes.values()
        elif isinstance(dtype, Iterable):
            if len(list(dtype)) != 4:
                raise ValueError(f'If dtype argument is an iterable, expected length is 4, got {len(list(dtype))}')
        else:
            dtype = [dtype for _ in range(4)]

        mem = {'s': [], 'a': [], 'r': [], 'terminal': []}
        for traj in self._mem:
            for (name, data), dt in zip(traj.items(), dtype):
                data = data.astype(dt)
                if len(data) != self.longest_trajectory:
                    diff = self.longest_trajectory - len(data)
                    padding_shape = (diff, *self.shapes[name])
                    data = np.concatenate([data, np.full(padding_shape, fill_value=padding, dtype=dt)], axis=0)
                mem[name].append(data)

        mem = [np.array(data) for data in mem.values()]
        return tuple(mem)

    #@staticmethod
    #def fuse_batch(s, a, r, terminal):
    #    batch_size, time_steps = np.shape(a)[:2]  # don't use s to detect number of time steps
    #    data_shapes = TrajectoryMemory._detect_shapes(s[0], a[0], r[0], terminal[0])
    #
    #    # s is one element longer than all other arrays, make them equal
    #    a = np.concatenate(a, np.zeros((batch_size, 1, *data_shapes[1])), axis=1)
    #    r = np.concatenate(r, np.zeros((batch_size, 1, *data_shapes[2])), axis=1)
    #    terminal = np.concatenate(terminal, np.zeros((batch_size, 1, *data_shapes[3])), axis=1)
    #
    #    fused = np.stack([s, a, r, terminal], axis=2).reshape((batch_size, time_steps + 1, ))

    @staticmethod
    def _detect_dtypes(s, a, r, terminal) -> Dict:
        return {k: np.array(x).dtype for k, x in zip(('s', 'a', 'r', 'terminal'), (s, a, r, terminal))}

    @staticmethod
    def _detect_shapes(s, a, r, terminal) -> Dict:
        return {k: np.shape(x)[1:] for k, x in zip(('s', 'a', 'r', 'terminal'), (s, a, r, terminal))}

    @staticmethod
    def _detect_lengths(s, a, r, terminal) -> Dict:
        return {k: np.shape(x)[0] for k, x in zip(('s', 'a', 'r', 'terminal'), (s, a, r, terminal))}

    @staticmethod
    def _dtypes_match(dt0: Dict, dt1: Dict):
        if dt0.keys() != dt1.keys():
            raise ValueError(f'Mismatch in data structures to compare: {dt0.keys()} vs. {dt1.keys()}')

        return all([pair[0] == pair[1] for pair in zip(dt0.values(), dt1.values())])

    @staticmethod
    def _shapes_match(s0: Dict, s1: Dict) -> bool:
        if s0.keys() != s1.keys():
            raise ValueError(f'Mismatch in data structures to compare: {s0.keys()} vs. {s1.keys()}')

        return all([pair[0] == pair[1] for pair in zip(s0.values(), s1.values())])

    @staticmethod
    def _lengths_match(lengths) -> bool:
        return lengths['a'] == lengths['r'] and lengths['a'] == lengths['terminal'] and lengths['s'] == lengths['a'] + 1

