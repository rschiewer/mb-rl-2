from __future__ import annotations
from collections import deque
from pathlib import Path
from typing import Dict, List, Iterable, Sequence, Union
from copy import copy, deepcopy
import random
import pickle

import torch
import numpy as np


class TrajectoryMemory:

    def __init__(self,
                 init_mem: Iterable = None):
        self._mem = list()#deque()
        self._shapes = None
        self._dtypes = None
        self._longest_trajectory = 0

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

    @property
    def shapes(self):
        return self._shapes

    @property
    def dtypes(self):
        return self._dtypes

    @property
    def longest_trajectory(self):
        return self._longest_trajectory

    def __getitem__(self,
                    index) -> Union[Dict, TrajectoryMemory]:
        if type(index) is slice:
            # note: this view will inherit the value of self.longest_trajectory even if it contains only shorter ones
            view = copy(self)
            view._mem = self._mem[index]
            return view
        return self._mem[index]

    def __len__(self) -> int:
        return len(self._mem)

    def __add__(self,
                other: TrajectoryMemory) -> TrajectoryMemory:
        shapes_match = TrajectoryMemory._shapes_match(self._shapes, other._shapes)
        dtypes_match = TrajectoryMemory._dtypes_match(self._dtypes, other._dtypes)
        if not shapes_match or not dtypes_match:
                raise ValueError(f'Trajectory memory {self} and {other} can\'t be added because of shape or dtype '
                                 f'mismatch')

        ret = self.get_view()
        ret._longest_trajectory = max(self._longest_trajectory, other._longest_trajectory)
        ret._mem += other._mem
        return ret

    def get_view(self) -> TrajectoryMemory:
        """
        Generates a new TrajectoryMemory object with an independent list of references to the original memory's samples.
        Adding content to this view will not change the original memory but add to the view's list of samples.
        :return: the view
        """
        view = copy(self)
        view._mem = self._mem[:]  # see https://docs.python.org/3/library/copy.html
        view._shapes = deepcopy(self._shapes)
        view._dtypes = deepcopy(self._dtypes)
        return view

    def shuffle(self) -> TrajectoryMemory:
        view = self.get_view()
        random.shuffle(view._mem)
        return view

    def push(self, s, a, r, terminal) -> None:
        if self._shapes is None:
            self._shapes = self._detect_shapes(s, a, r, terminal)
            self._dtypes = self._detect_dtypes(s, a, r, terminal)
        else:
            data_shapes = self._detect_shapes(s, a, r, terminal)
            data_lengths = self._detect_lengths(s, a, r, terminal)
            data_dtypes = self._detect_dtypes(s, a, r, terminal)
            if not self._shapes_match(self._shapes, data_shapes):
                raise ValueError(f'Input has incompatible shape, expected {self._shapes}, found {data_shapes}')
            if not self._lengths_match(data_lengths):
                raise ValueError(f'Input has incompatible lengths, expected a, r, terminal to have equal lengths and'
                                 f' s to have on additional element')
            if not self._dtypes_match(self._dtypes, data_dtypes):
                raise ValueError(f'Input has different dtypes than previously added content, expected {self._dtypes}, '
                                 f'found {data_dtypes}')

        if len(s) > self._longest_trajectory:
            self._longest_trajectory = len(s)

        self._mem.append({'s': np.array(s), 'a': np.array(a), 'r': np.array(r), 'terminal': np.array(terminal)})

    def to_np_arrays(self,
                     padding: float = 0,
                     pad_last_terminal_flag: bool = True,
                     dtype: Union[np.dtype, Iterable[np.dtype]] = None):
        if dtype is None:
            dtype = self._dtypes.values()
        elif isinstance(dtype, Iterable):
            if len(list(dtype)) != 4:
                raise ValueError(f'If dtype argument is an iterable, expected length is 4, got {len(list(dtype))}')
        else:
            dtype = [dtype for _ in range(4)]

        mem = {'s': [], 'a': [], 'r': [], 'terminal': []}
        for traj in self._mem:
            for (name, data), dt in zip(traj.items(), dtype):
                data = data.astype(dt)
                if len(data) != self._longest_trajectory:
                    diff = self._longest_trajectory - len(data)
                    padding_shape = (diff, *self._shapes[name])
                    if name == 'terminal' and pad_last_terminal_flag:
                        data = np.concatenate([data, np.full(padding_shape, fill_value=data[-1], dtype=dt)], axis=0)
                    else:
                        data = np.concatenate([data, np.full(padding_shape, fill_value=padding, dtype=dt)], axis=0)
                mem[name].append(data)

        mem = [np.array(data) for data in mem.values()]
        return tuple(mem)

    @staticmethod
    def cmp_trajectories(t1: Dict[str, np.ndarray],
                         t2: Dict[str, np.ndarray]):
        for k, v in t1.items():
            if np.any(v != t2[k]):
                return False
        return True

    @staticmethod
    def store(mem: TrajectoryMemory,
              path: Union[Path, str]):
        with open(Path(path), 'wb') as f:
            pickle.dump(mem, f)

    @staticmethod
    def load(path: Union[Path, str]) -> TrajectoryMemory:
        with open(Path(path), 'rb') as f:
            mem = pickle.load(f)
        return mem

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
    def _dtypes_match(dt0: Dict,
                      dt1: Dict):
        if dt0.keys() != dt1.keys():
            raise ValueError(f'Mismatch in data structures to compare: {dt0.keys()} vs. {dt1.keys()}')

        return all([pair[0] == pair[1] for pair in zip(dt0.values(), dt1.values())])

    @staticmethod
    def _shapes_match(s0: Dict,
                      s1: Dict) -> bool:
        if s0.keys() != s1.keys():
            raise ValueError(f'Mismatch in data structures to compare: {s0.keys()} vs. {s1.keys()}')

        return all([pair[0] == pair[1] for pair in zip(s0.values(), s1.values())])

    @staticmethod
    def _lengths_match(lengths) -> bool:
        return lengths['a'] == lengths['r'] and lengths['a'] == lengths['terminal'] and lengths['s'] == lengths['a'] + 1


def flatten_and_unsqueeze(*xs: Union[torch.Tensor, np.ndarray]):
    reshaped = []
    for x in xs:
        d_x = np.prod(x.shape[2:]) if len(x.shape) >= 3 else 1
        reshaped.append(x.reshape((*x.shape[:2], d_x)))

    if len(reshaped) > 1:
        reshaped = tuple(reshaped)
    else:
        reshaped = reshaped[0]

    return reshaped


def flatten_and_unsqueeze_old(s: np.array, a: np.array, r: np.array, terminal: np.array):
    # flatten data dimensions if multiple or add explicit 1-sized data dimension if there is none
    d_s = np.prod(s.shape[2:]) if len(s.shape) >= 3 else 1
    d_a = np.prod(a.shape[2:]) if len(a.shape) >= 3 else 1
    d_r = np.prod(r.shape[2:]) if len(r.shape) >= 3 else 1
    d_term = np.prod(terminal.shape[2:]) if len(terminal.shape) >= 3 else 1

    s = s.reshape((*s.shape[:2], d_s))
    a = a.reshape((*a.shape[:2], d_a))  # mind: there is one more state than actions, rewards and terminal flags
    r = r.reshape((*r.shape[:2], d_r))
    terminal = terminal.reshape((*terminal.shape[:2], d_term))

    return s, a, r, terminal