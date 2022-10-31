from typing import Union, Tuple, Sequence, Dict
from itertools import product

import gym
import torch
import numpy as np

from mdm.models.multiscale_model_mk2 import MultiscaleDynamicsModelMK2
from mdm.models.building_blocks import RnnStateType
from mdm.planning.cem_planner import CrossentropyPlanner, TensorData
from mdm.utils.utils import normalize_obs, to_onehot, prepare_data, flatten_and_unsqueeze, sensitivity_analysis
from mdm.utils.torch_tools import add_time_dim, build_gaussian, bin_every_k_steps, pack_rnn_state, unpack_rnn_state


def update_history(history: Dict[str, torch.Tensor],
                   new_data: Dict[str, torch.Tensor]):
    for k, v in new_data.items():
        if len(v) > 0:
            if len(history[k]) > 0:
                history[k] = torch.concat([history[k], v], dim=1)
            else:
                history[k] = v
    return history


def plan_prim_free(model: MultiscaleDynamicsModelMK2,
                   planner_prim: object,
                   prim_rnn_state: RnnStateType,
                   prim_z: torch.Tensor,
                   n_plan_steps: int,
                   n_rollouts: int):
    rnn_state_start = unpack_rnn_state(pack_rnn_state(prim_rnn_state).repeat(n_rollouts, 1, 1, 1))
    z_start = prim_z.repeat(n_rollouts, 1)

    def _rollout_fn(_a: torch.Tensor):
        _a = to_onehot(_a, n_classes=model.d_action)
        _mem, _prim_final = model.rollout_primitive(a=_a, z=z_start, rnn_state=rnn_state_start, sample=False)
        _mem = model.pack_mem(_mem)

        _criterion = _mem['prim_r'].squeeze(-1)
        _discount = _mem['prim_term'].squeeze(-1)
        return _criterion, _discount, _mem

    a, a_dist, i_win, R_win, data = planner_prim.plan(rollout_fn=_rollout_fn, n_rollouts=n_rollouts,
                                                      n_plan_steps=n_plan_steps)

    data = {k: v[i_win[0], None] if len(v) > 0 else v for k, v in data.items()}
    return a[i_win[0]], data


def plan_prim_chunk(model: MultiscaleDynamicsModelMK2,
                    planner_prim: object,
                    prim_rnn_state: RnnStateType,
                    prim_z: torch.Tensor,
                    abstr_target: torch.Tensor,
                    abstr_r: torch.Tensor,
                    abstr_term: torch.Tensor,
                    n_plan_steps: Union[int, Sequence[int]],
                    n_rollouts: int):
    rnn_state_start = unpack_rnn_state(pack_rnn_state(prim_rnn_state).repeat(n_rollouts, 1, 1, 1))
    z_start = prim_z.repeat(n_rollouts, 1)
    target = abstr_target.repeat(n_rollouts, 1)
    r_goal = abstr_r.repeat(n_rollouts, 1)
    term_goal = abstr_term.repeat(n_rollouts, 1)

    if type(n_plan_steps) is int:
        n_plan_steps = (n_plan_steps, )

    R_best = -np.inf
    a_best = None
    data_best = None
    i_win_best = None
    for n_steps in n_plan_steps:
        def _rollout_fn(_a: torch.Tensor):
            _a = to_onehot(_a, n_classes=model.d_action)
            _mem, _prim_final = model.rollout_primitive(a=_a, z=z_start, rnn_state=rnn_state_start, sample=False)
            _mem = model.pack_mem(_mem)

            _abstr_r_rollout = model.calc_abstr_r_ground_truth(_mem['prim_r'])[:, -1]
            _abstr_term_rollout = model.calc_abstr_term_ground_truth(_mem['prim_term'])[:, -1]
            _abstr_target_rollout = _mem[model.abstr_pred_target][:, -1]
            _criterion = - torch.mean((_abstr_target_rollout - target) ** 2, dim=1, keepdim=True)
            _criterion -= torch.mean((_abstr_r_rollout - r_goal) ** 2, dim=1, keepdim=True)
            _criterion -= torch.mean((_abstr_term_rollout - term_goal) ** 2, dim=1, keepdim=True)
            return _criterion, None, _mem

        a, a_dist, i_win, R_win, data = planner_prim.plan(rollout_fn=_rollout_fn, n_rollouts=n_rollouts,
                                                          n_plan_steps=n_steps)
        if R_win[0] > R_best:
            a_best = a
            data_best = data
            i_win_best = i_win

    data_best = {k: v[i_win_best[0], None] if len(v) > 0 else v for k, v in data_best.items()}
    return a_best[i_win_best[0]], data_best


def plan_abstr(model: MultiscaleDynamicsModelMK2,
               planner_abstr: object,
               abstr_rnn_state: RnnStateType,
               abstr_z: torch.Tensor,
               n_plan_steps: int,
               n_rollouts: int):
    rnn_state_start = unpack_rnn_state(pack_rnn_state(abstr_rnn_state).repeat(n_rollouts, 1, 1, 1))
    z_start = abstr_z.repeat(n_rollouts, 1)

    def _rollout_fn(_a: torch.Tensor):
        if model.abstract_action_model.distribution_type is None:
            _a = torch.clamp(_a, -0.99, 0.99)  # limit action range to allowed values
        elif model.abstract_action_model.distribution_type == 'categorical':
            _a = to_onehot(_a, model.abstract_model.d_action)
        _mem, _abstr_final = model.rollout_abstract(a=_a, z=z_start, rnn_state=rnn_state_start,
                                                    n_posterior_steps=0, sample=False)
        _mem = model.pack_mem(_mem)

        _criterion = _mem['abstr_r'].squeeze(-1)
        _discount = _mem['abstr_term'].squeeze(-1)
        return _criterion, _discount, _mem

    a, a_dist, i_win, R_win, data = planner_abstr.plan(rollout_fn=_rollout_fn, n_rollouts=n_rollouts,
                                                      n_plan_steps=n_plan_steps)

    data = {k: v[i_win[0], None] if len(v) > 0 else v for k, v in data.items()}
    return a[i_win[0]], data


def collect_groundtruth_data(model: MultiscaleDynamicsModelMK2,
                             history: Dict[str, torch.Tensor],
                             env: gym.Env,
                             n_steps: int,
                             predefined_actions: Sequence = ()):
    # prepare actions and first step's data (note: only the initial observation contains real environment info)
    n_a_required = n_steps - 1 - len(predefined_actions)
    groundtruth_data = {'a': [], 'o': [], 'r': [], 'term': []}
    groundtruth_data['a'] += [0] + list(predefined_actions) + [env.action_space.sample() for _ in range(n_a_required)]
    groundtruth_data['o'].append(env.reset())
    groundtruth_data['r'].append(0.0)
    groundtruth_data['term'].append(0.0)

    # collect ground truth observations, rewards and terminal flags
    for a in groundtruth_data['a'][1:]:  # first action is placeholder by convention, don't execute it
        o, r, term, _ = env.step(a)
        groundtruth_data['o'].append(o)
        groundtruth_data['r'].append(r)
        groundtruth_data['term'].append(term)
        if term:
            raise RuntimeError('Environment terminated during warmup')

    # prepare groundtruth data
    o = torch.from_numpy(np.stack(groundtruth_data['o']))
    a = torch.tensor(groundtruth_data['a'])
    r = torch.tensor(groundtruth_data['r'])
    term = torch.tensor(groundtruth_data['term'])

    o, a, r, term = o.to(model.device), a.to(model.device), r.to(model.device), term.to(model.device)
    o, a, r, term = o.unsqueeze(0), a.unsqueeze(0), r.unsqueeze(0), term.unsqueeze(0)  # need batch dim for preparation
    o, a, r, term = prepare_data(o, a, r, term, env)

    history['prim_o'] = o
    history['prim_a'] = a
    history['prim_r'] = r
    history['prim_term'] = term

    return history


def init_prim_s(model: MultiscaleDynamicsModelMK2,
                history: Dict[str, torch.Tensor]):
    o_start_batch = history['prim_o']
    a_start_batch = history['prim_a']
    r_start_batch = history['prim_r']
    term_start_batch = history['prim_term']

    mem, prim_current = model.rollout_primitive(a=a_start_batch, o=o_start_batch, r=r_start_batch,
                                                term=term_start_batch, n_posterior_steps=-1, sample=False)
    mem = model.pack_mem(mem)

    # remove the rollout data from mem that is already present as groundtruth data
    mem['prim_o'] = []
    mem['prim_a'] = []
    mem['prim_r'] = []
    mem['prim_term'] = []

    history = update_history(history, mem)
    return history


def init_abstr_s(model: MultiscaleDynamicsModelMK2,
                 history: Dict[str, torch.Tensor],
                 planner_prim: CrossentropyPlanner,
                 n_warmup_abstr: int,
                 n_rollouts: int,
                 allow_prim_imagination: bool = False):
    history_length = history['prim_a'].shape[1]
    required_steps = model.abstract_step_size * n_warmup_abstr
    remaining_steps = required_steps - history_length

    assert 0 < history_length <= model.abstract_step_size * n_warmup_abstr, f'History length: {history_length}'
    assert remaining_steps >= 0

    if remaining_steps > 0:
        if not allow_prim_imagination:
            raise RuntimeError('Not enough warmup data for abstract model and data synthesis by primitive model is '
                               'disabled.')
        z_batch = history['prim_z'][:, -1].repeat(n_rollouts, 1)
        rnn_state_batch = unpack_rnn_state(history['prim_rnn_state'][:, -1].repeat(n_rollouts, 1, 1, 1))

        def _rollout_fn(_a: torch.Tensor):
            _a = to_onehot(_a, n_classes=model.d_action)
            _mem, _prim_final = model.rollout_primitive(a=_a, z=z_batch, rnn_state=rnn_state_batch, sample=False)
            _mem = model.pack_mem(_mem)

            _criterion = _mem['prim_r'].squeeze(-1)
            _discount = _mem['prim_term'].squeeze(-1)
            return _criterion, _discount, _mem

        a, a_dist, i_win, R_win, data = planner_prim.plan(rollout_fn=_rollout_fn,
                                                          n_rollouts=n_rollouts,
                                                          n_plan_steps=remaining_steps)
        # filter out winner rollout for every entry in the memory
        data = {k: v[i_win[0], None] if len(v) > 0 else v for k, v in data.items()}
        history = update_history(history, data)

    a_binned = bin_every_k_steps(history['prim_a'][:, :required_steps], model.abstract_step_size)
    abstr_a = torch.stack([model.abstract_action_model(a_binned[:, i_chunk], sample=False)
                           for i_chunk in range(a_binned.shape[1])], dim=1)
    target = model.abstr_pred_target
    prim_data = bin_every_k_steps(history[target][:, :required_steps], model.abstract_step_size)[:, :, -1]
    abstr_r = model.calc_abstr_r_ground_truth(history['prim_r'][:, :required_steps])
    abstr_term = model.calc_abstr_term_ground_truth(history['prim_term'][:, :required_steps])
    #abstr_r = bin_every_k_steps(history['prim_r'][:, :required_steps], model.abstract_step_size).mean(dim=2)
    #abstr_term = bin_every_k_steps(history['prim_term'][:, :required_steps], model.abstract_step_size).max(dim=2).values

    mem, abstr_final = model.rollout_abstract(a=abstr_a, prim_data=prim_data, r=abstr_r,
                                              term=abstr_term, sample=False, n_posterior_steps=-1)
    mem = model.pack_mem(mem)
    history = update_history(history, mem)
    return history


def plan_abstract(model: MultiscaleDynamicsModelMK2,
                  history: Dict[str, torch.Tensor],
                  planner_abstr: CrossentropyPlanner,
                  n_plan_steps: int,
                  n_rollouts: int):
    if n_plan_steps == 0:
        return history

    abstr_z_start = history['abstr_z'][:, -1].repeat(n_rollouts, 1)
    abstr_rnn_state_start = unpack_rnn_state(history['abstr_rnn_state'][:, -1].repeat(n_rollouts, 1, 1, 1))

    def _rollout_fn(_a: torch.Tensor):
        if model.abstract_action_model.distribution_type is None:
            _a = torch.clamp(_a, -0.99, 0.99)  # limit action range to allowed values
        elif model.abstract_action_model.distribution_type == 'categorical':
            _a = to_onehot(_a, model.abstract_model.d_action)
        _mem, _abstr_final = model.rollout_abstract(_a, z=abstr_z_start, rnn_state=abstr_rnn_state_start, sample=False,
                                                    n_posterior_steps=0)
        _mem = model.pack_mem(_mem)

        _criterion = _mem['abstr_r'].squeeze(-1)
        _discount = _mem['abstr_term'].squeeze(-1)
        return _criterion, _discount, _mem

    a, a_dist, i_win, R_win, data = planner_abstr.plan(rollout_fn=_rollout_fn,
                                                       n_rollouts=n_rollouts,
                                                       n_plan_steps=n_plan_steps)
    # update history with new rollouts
    data = {k: v[i_win[0], None] if len(v) > 0 else v for k, v in data.items()}
    history = update_history(history, data)
    return history


def plan_section(model: MultiscaleDynamicsModelMK2,
                 history: Dict[str, torch.Tensor],
                 planner_prim: CrossentropyPlanner,
                 i_section: int,
                 n_rollouts: int):
    n_steps_done = history['prim_a'].shape[1]
    assert n_steps_done % model.abstract_step_size == 0, ('Warmup step count should be evenly divisible by the section '
                                                          f'length, but they are {n_steps_done} and '
                                                          f'{model.abstract_step_size}')
    z_start = history['prim_z'][:, -1].repeat(n_rollouts, 1)
    rnn_state_start = unpack_rnn_state(history['prim_rnn_state'][:, -1].repeat(n_rollouts, 1, 1, 1))
    target = history['abstr_o'][:, i_section].repeat(n_rollouts, 1)
    r_goal = history['abstr_r'][:, i_section].repeat(n_rollouts, 1)

    def _rollout_fn(_a: torch.Tensor):
        _a = to_onehot(_a, n_classes=model.d_action)
        _mem, _prim_final = model.rollout_primitive(a=_a, z=z_start, rnn_state=rnn_state_start, sample=False)
        _mem = model.pack_mem(_mem)

        # TODO: test KL divergence between distributions
        _criterion = - torch.mean((_mem[model.abstr_pred_target][:, -1] - target) ** 2, dim=1, keepdim=True)
        _criterion -= torch.mean((_mem['prim_r'].sum(dim=1) - r_goal) ** 2, dim=1, keepdim=True)
        _discount = None
        return _criterion, _discount, _mem

    a, a_dist, i_win, R_win, data = planner_prim.plan(rollout_fn=_rollout_fn,
                                                      n_rollouts=n_rollouts,
                                                      n_plan_steps=model.abstract_step_size)
    # filter out winner rollout for every entry in the memory
    data = {k: v[i_win[0], None] if len(v) > 0 else v for k, v in data.items()}
    history = update_history(history, data)

    return history, model.abstract_step_size


def plan_section_flexible(model: MultiscaleDynamicsModelMK2,
                          history: Dict[str, torch.Tensor],
                          planner_prim: CrossentropyPlanner,
                          i_section: int,
                          n_rollouts: int):
    if i_section >= history['abstr_o'].shape[1]:
        return history, 0

    best_data = None
    best_criterion = np.inf
    for seq_len in range(model.abstract_step_size - 1, model.abstract_step_size + 2):
        available_actions = list(range(model.d_action))
        a_seq = np.stack(list(product(available_actions, repeat=seq_len)))
        a_seq = torch.tensor(a_seq, device=model.device)
        a_seq = to_onehot(a_seq, n_classes=model.d_action)
        n_sequences = a_seq.shape[0]

        z_start = history['prim_z'][:, -1].repeat(n_sequences, 1)
        rnn_state_start = unpack_rnn_state(history['prim_rnn_state'][:, -1].repeat(n_sequences, 1, 1, 1))
        s_goal = history['abstr_o'][:, i_section].repeat(n_sequences, 1)
        r_goal = history['abstr_r'][:, i_section].repeat(n_sequences, 1)

        mem, _prim_final = model.rollout_primitive(a=a_seq, z=z_start, rnn_state=rnn_state_start, sample=False)
        mem = model.pack_mem(mem)
        _criterion = torch.mean((mem[model.abstr_pred_target][:, -1] - s_goal) ** 2, dim=1, keepdim=True)
        _criterion += torch.mean((mem['prim_r'].sum(dim=1) - r_goal) ** 2, dim=1, keepdim=True)
        ranking = torch.sort(_criterion, dim=0, descending=False)
        if ranking.values[0] < best_criterion:
            best_criterion = ranking.values[0]
            i_win = ranking.indices[0]
            best_data = {k: v[i_win] if len(v) > 0 else v for k, v in mem.items()}

    final_sec_len = best_data['prim_a'].shape[1]

    history = update_history(history, best_data)
    return history, final_sec_len