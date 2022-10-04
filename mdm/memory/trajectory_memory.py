from __future__ import annotations
from collections import deque
from pathlib import Path
from typing import Dict, List, Iterable, Sequence, Union, Tuple, Optional, Callable
from copy import copy, deepcopy
import random
import pickle

import torch
import numpy as np
import numpy.ma as ma


class TrajectoryMemory:

    def __init__(self,
                 init_mem: Iterable = None):
        self._mem = list()
        self._shapes = None
        self._dtypes = None
        self._longest_trajectory = 0

        if init_mem is not None:
            for elem in init_mem:
                try:
                    if type(elem) is dict:
                        self.push(**elem)
                    elif isinstance(elem, Sequence):
                        self.push(*elem)
                    else:
                        raise ValueError('Unknown content in init_mem, elements should have type dict, list or tuple '
                                         f'but are of type {type(elem)}')
                except KeyError:
                    raise ValueError('Expected dict keys of elements are "s", "a", "r" and "terminal", found: '
                                     f'{elem.keys()}')
                except IndexError:
                    raise ValueError(f'Expected length of elements is 5, found {len(elem)}')

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
            view = self.get_view()
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

    def push(self, s, a, r, terminal, w: float = 1.0) -> None:
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
                raise ValueError(f'Input has incompatible lengths, expected s, a, r, terminal to have equal lengths')
            if not self._dtypes_match(self._dtypes, data_dtypes):
                raise ValueError(f'Input has different dtypes than previously added content, expected {self._dtypes}, '
                                 f'found {data_dtypes}')

        if not np.all(a[0] == 0):
            raise ValueError('Expected first action to be zero by convention')
        if not np.all(r[0] == 0):
            raise ValueError('Expected first reward to be zero by convention')
        if not np.all(terminal[0] == 0):
            raise ValueError('Expected first terminal flag to be zero by convention')
        if not np.ndim(w) == 0:
            raise ValueError(f'Weight must be a float, but has {np.ndim(w)} dimensions')
        #if not type(w) is (int, float):
        #    raise ValueError(f'Weight must be of dtype float but is: {np.array(w).dtype}')
        w = float(w)

        if len(s) > self._longest_trajectory:
            self._longest_trajectory = len(s)

        self._mem.append({'s': np.array(s), 'a': np.array(a), 'r': np.array(r), 'terminal': np.array(terminal),
                          'w': w})

    def to_np_arrays(self,
                     padding: float = 0,
                     pad_last_terminal_flag: bool = True,
                     dtype: Union[np.dtype, Iterable[np.dtype]] = None):
        if dtype is None:
            dtype = self._dtypes
        elif isinstance(dtype, Iterable):
            if len(list(dtype)) != 5:
                raise ValueError(f'If dtype argument is an iterable, expected length is 5, got {len(list(dtype))}')
            if isinstance(dtype, (list, tuple)):
                dtype = {name: dt for name, dt in zip(self._dtypes.keys(), dtype)}
        else:
            dtype = {name: dtype for name in self._dtypes.keys()}

        mem = {'s': [], 'a': [], 'r': [], 'terminal': [], 'w': []}
        masks = {'s': [], 'a': [], 'r': [], 'terminal': [], 'w': []}

        for traj in self._mem:
            for name, data in traj.items():
                if name == 'w':
                    mem[name].append(np.dtype(dtype).type(data))
                    masks[name].append(False)
                else:
                    mask = np.zeros_like(data)
                    if len(data) != self._longest_trajectory:
                        diff = self._longest_trajectory - len(data)
                        padding_shape = (diff, *self._shapes[name])
                        if name == 'terminal' and pad_last_terminal_flag:
                            filler = np.full(padding_shape, fill_value=data[-1], dtype=dtype[name])
                        else:
                            filler = np.full(padding_shape, fill_value=padding, dtype=dtype[name])
                        #pad_values = [(0, diff)]
                        #pad_values += [(0, 0) for _ in range(data.ndim - 1)]
                        #data = np.pad(data, pad_values, 'constant', constant_values=0)
                        data = np.concatenate([data, filler], axis=0)
                        mask = np.concatenate([mask, np.ones_like(filler)], axis=0)
                    mem[name].append(data)
                    masks[name].append(mask)

        mem = [ma.array(data, mask=mask) for data, mask in zip(mem.values(), masks.values())]
        return tuple(mem)

    def sort(self, key: 'str' = 'w', reverse: bool = False):
        key_fn = lambda elem: elem[key]
        self._mem.sort(key=key_fn, reverse=reverse)

    @staticmethod
    def cmp_trajectories(t1: Dict[str, np.ndarray],
                         t2: Dict[str, np.ndarray]):
        for k, v in t1.items():
            if k == 'w': continue
            else:
                if v.shape != t2[k].shape or np.any(v != t2[k]):
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
        data_dtypes = {k: np.array(x).dtype for k, x in zip(('s', 'a', 'r', 'terminal'), (s, a, r, terminal))}
        return data_dtypes

    @staticmethod
    def _detect_shapes(s, a, r, terminal) -> Dict:
        data_shapes = {k: np.shape(x)[1:] for k, x in zip(('s', 'a', 'r', 'terminal'), (s, a, r, terminal))}
        return data_shapes


    @staticmethod
    def _detect_lengths(s, a, r, terminal) -> Dict:
        data_lengths = {k: np.shape(x)[0] for k, x in zip(('s', 'a', 'r', 'terminal'), (s, a, r, terminal))}
        data_lengths['w'] = 0
        return data_lengths

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
        return lengths['a'] == lengths['r'] and lengths['a'] == lengths['terminal'] and lengths['a'] == lengths['s']


def flatten_and_unsqueeze(*xs: Union[torch.Tensor, np.ndarray]) -> Union[Tuple[torch.Tensor], Tuple[np.ndarray],
                                                                         torch.Tensor, np.ndarray]:
    reshaped = []
    for x in xs:
        d_x = np.prod(x.shape[2:]) if len(x.shape) >= 3 else 1
        reshaped.append(x.reshape((*x.shape[:2], d_x)))

    if len(reshaped) > 1:
        reshaped = tuple(reshaped)
    else:
        reshaped = reshaped[0]

    return reshaped