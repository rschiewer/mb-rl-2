from typing import Union, Tuple, Sequence, Dict, TypeVar, List
import copy
import matplotlib.pyplot as plt
from itertools import product

import gym
import torch
import numpy as np

from mdm.models.multiscale_model_mk2 import MultiscaleDynamicsModelMK2
from mdm.models.building_blocks import RnnStateType
from mdm.planning.cem_planner import CrossentropyPlanner, TensorData, DistributionType
from mdm.utils.utils import SliceType, to_onehot, prepare_data, flatten_and_unsqueeze, sensitivity_analysis, trajectory_uncertainty
from mdm.utils.torch_tools import (extract_sub_distribution, bin_every_k_steps, pack_rnn_state, unpack_rnn_state,
                                   TensorIndex, to_tensors)


def update_history(history: Dict[str, torch.Tensor],
                   new_data: Dict[str, torch.Tensor]):
    for k, v in new_data.items():
        if len(v) > 0:
            if len(history[k]) > 0:
                history[k] += v
            else:
                history[k] = v
    return history


def select_batch_items(mem: Dict[str, Union[torch.Tensor, torch.distributions.Distribution]],
                       i: TensorIndex,
                       keepdim: bool = False):
    if keepdim and type(i) is int:
        i = slice(i, i+1)

    ret = {}
    for name, val in mem.items():
        ret[name] = []
        for timestep in val:
            if timestep is None:
                ret[name].append(None)
            elif isinstance(timestep, torch.Tensor):
                ret[name].append(timestep[i])
            elif isinstance(timestep, torch.distributions.Distribution):
                ret[name].append(extract_sub_distribution(timestep, i, keepdim=keepdim))
            elif 'rnn_state' in name:
                ret[name].append(unpack_rnn_state(pack_rnn_state(timestep)[i]))
            else:
                raise ValueError(f'Unknown memory content for key {name}: {timestep}')
    return ret


def plan_prim_free(model: MultiscaleDynamicsModelMK2,
                   planner_prim: object,
                   prim_rnn_state: RnnStateType,
                   prim_z: torch.Tensor,
                   n_plan_steps: int,
                   n_rollouts: int):
    rnn_state_start = unpack_rnn_state(pack_rnn_state(prim_rnn_state).repeat(n_rollouts, 1, 1, 1))
    z_start = prim_z.repeat(n_rollouts, 1)

    def _rollout_fn(_a: torch.Tensor):
        _a = to_onehot(_a, n_classes=model.primitive_model.d_action)
        _a = _a.swapaxes(0, 1)
        _mem, _prim_final = model.rollout_primitive(a=_a, z=z_start, rnn_state=rnn_state_start, sample=True)

        _criterion = torch.stack(_mem['prim_r'], dim=1).squeeze(-1)
        _discount = torch.stack(_mem['prim_term'], dim=1).squeeze(-1)
        return _criterion, _discount, _mem

    a, a_dist, i_win, R_win, data = planner_prim.plan(rollout_fn=_rollout_fn, n_rollouts=n_rollouts,
                                                      n_plan_steps=n_plan_steps)

    # select winner per per memory element per timestep
    data = select_batch_items(data, i_win[0], keepdim=True)
    return a[i_win[0]], data


def plan_prim_with_warmup(model: MultiscaleDynamicsModelMK2,
                          planner_prim: CrossentropyPlanner,
                          env_data: Dict[str, List[torch.Tensor]],
                          n_plan_steps: int,
                          n_rollouts: int):
    o_start_batch = torch.stack(env_data['prim_o']).repeat(1, n_rollouts, 1, 1)
    a_start_batch = torch.stack(env_data['prim_a']).repeat(1, n_rollouts, 1)
    r_start_batch = torch.stack(env_data['prim_r']).repeat(1, n_rollouts, 1)
    term_start_batch = torch.stack(env_data['prim_term']).repeat(1, n_rollouts, 1)
    n_posterior_steps = o_start_batch.shape[0]

    var_z_max = []
    var_z_std = []
    var_r_max = []
    var_r_std = []

    init_uncertainty = None
    def _rollout_fn(_a: torch.Tensor):
        nonlocal init_uncertainty
        _a = _a[0]  # remove redundant env dimension
        _a = to_onehot(_a, n_classes=model.primitive_model.d_action)
        _a = _a.swapaxes(0, 1)
        _a = torch.cat([a_start_batch, _a], dim=0)
        _mem, _prim_final = model.rollout_primitive(a=_a, o=o_start_batch, r=r_start_batch, term=term_start_batch,
                                                    n_posterior_steps=n_posterior_steps, sample=True)
        _var_z, _var_r = trajectory_uncertainty(_mem)
        var_z_max.append(_var_z.max())
        var_z_std.append(_var_z.std())
        var_r_max.append(_var_r.max())
        var_r_std.append(_var_r.std())

        if init_uncertainty is None:
            init_uncertainty = _var_z.mean(dim=1, keepdim=True)

        diff = (_var_z.mean(dim=1, keepdim=True) - init_uncertainty).mean(-1)
        _criterion = torch.stack(_mem['prim_r'], dim=1).squeeze(-1)
        #_criterion -= diff.swapaxes(0, 1)
        #_uncertainty = torch.stack([d.scale for d in _mem['prim_r_dist']], dim=1).squeeze(-1)
        #_criterion -= _uncertainty
        _discount = torch.stack(_mem['prim_term'], dim=1).squeeze(-1)

        _criterion = _criterion.unsqueeze(0)  # add "env" dimension
        _discount = _discount.unsqueeze(0)
        return _criterion, _discount, _mem

    a, a_dist, i_win, R_win, data = planner_prim.plan(rollout_fn=_rollout_fn, n_rollouts=n_rollouts,
                                                      n_plan_steps=n_plan_steps)

    """
    var_z_max = np.array(torch.stack(var_z_max).detach().cpu().numpy())
    var_z_std = np.array(torch.stack(var_z_std).detach().cpu().numpy())
    var_r_max = np.array(torch.stack(var_r_max).detach().cpu().numpy())
    var_r_std = np.array(torch.stack(var_r_std).detach().cpu().numpy())
    fig, ax = plt.subplots(1, 2, figsize=(16, 10))
    ax[0].fill_between(range(len(var_z_max)), var_z_max - var_z_std, var_z_max + var_z_std, alpha=0.9)
    ax[0].plot(var_z_max, label='z')
    ax[0].set(xlabel='planning iteration', ylabel='max batch z sigma')
    ax[0].set_title('z')
    ax[1].fill_between(range(len(var_r_max)), var_r_max - var_r_std, var_r_max + var_r_std, alpha=0.9)
    ax[1].plot(var_r_max, label='r')
    ax[1].set(xlabel='planning iteration', ylabel='max batch r sigma')
    ax[1].set_title('r')
    plt.legend()
    plt.show()
    """

    # select winner batch item per per memory timestep
    data = select_batch_items(data, i_win[0, 0], keepdim=True)
    # CAUTION: the actions from planner are without the already performed warmup acitons!
    a_win = planner_prim.get_winner_actions(a, a_dist, i_win, resample=True)
    a_win = a_win[0]  # remove redundant "env" dimension
    return a_win, data


def plan_abstr_with_warmup(model: MultiscaleDynamicsModelMK2,
                           planner_abstr: CrossentropyPlanner,
                           prim_data: Dict[str, List[torch.Tensor]],
                           n_plan_steps: int,
                           n_rollouts: int):
    o_start_batch = torch.stack(prim_data[model.abstr_pred_target][::model.abstract_step_size]).repeat(1, n_rollouts, 1, 1)
    a_start_batch = model.calc_abstr_a(torch.stack(prim_data['prim_a']).repeat(1, n_rollouts, 1))
    r_start_batch = model.calc_abstr_r_ground_truth(torch.stack(prim_data['prim_r']).repeat(1, n_rollouts, 1))
    term_start_batch = model.calc_abstr_term_ground_truth(torch.stack(prim_data['prim_term']).repeat(1, n_rollouts, 1))
    n_posterior_steps = o_start_batch.shape[0]

    def _rollout_fn(_a: torch.Tensor):
        if model.abstract_action_model.model_type in ('det_tanh', 'prob_normal'):
            _a = torch.clamp(_a, -0.99, 0.99)  # limit action range to allowed values
        elif model.abstract_action_model.model_type in ('prob_categorical', 'det_mapping'):
            _a = to_onehot(_a, model.abstract_model.d_action)
        _a = _a.swapaxes(0, 1)
        _a = torch.cat([a_start_batch, _a], dim=0)
        _mem, _abstr_final = model.rollout_abstract(prim_data=o_start_batch, a=_a, r=r_start_batch,
                                                    term=term_start_batch, n_posterior_steps=0, sample=True)
        _mem = model.pack_mem(_mem)

        _criterion = torch.stack(_mem['abstr_r'], dim=1).squeeze(-1)
        #_criterion -= diff.swapaxes(0, 1)
        #_uncertainty = torch.stack([d.scale for d in _mem['prim_r_dist']], dim=1).squeeze(-1)
        #_criterion -= _uncertainty
        _discount = torch.stack(_mem['abstr_term'], dim=1).squeeze(-1)
        return _criterion, _discount, _mem

    a, a_dist, i_win, R_win, data = planner_abstr.plan(rollout_fn=_rollout_fn, n_rollouts=n_rollouts,
                                                       n_plan_steps=n_plan_steps)

    data = select_batch_items(data, i_win[0, 0], keepdim=True)
    a_win = planner_abstr.get_winner_actions(a, a_dist, i_win, resample=True)
    a_win = a_win[0]  # remove redundant "env" dimension
    return a_win, data


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
            _a = _a.swapaxes(0, 1)
            _mem, _prim_final = model.rollout_primitive(a=_a, z=z_start, rnn_state=rnn_state_start, sample=True)

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

    data_best = select_batch_items(data_best, i_win_best[0], keepdim=True)
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
        if model.abstract_action_model.model_type in ('det_tanh', 'prob_normal'):
            _a = torch.clamp(_a, -0.99, 0.99)  # limit action range to allowed values
        elif model.abstract_action_model.model_type in ('prob_categorical', 'det_mapping'):
            _a = to_onehot(_a, model.abstract_model.d_action)
        _mem, _abstr_final = model.rollout_abstract(a=_a, z=z_start, rnn_state=rnn_state_start,
                                                    n_posterior_steps=0, sample=True)
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
    groundtruth_data = {'a': [], 'o': [], 'r': [], 'term': [], 'trunc': []}
    groundtruth_data['a'] += [0] + list(predefined_actions) + [env.action_space.sample() for _ in range(n_a_required)]
    groundtruth_data['o'].append(env.reset()[0])
    groundtruth_data['r'].append(0.0)
    groundtruth_data['term'].append(0.0)
    groundtruth_data['trunc'].append(0.0)

    # collect ground truth observations, rewards and terminal flags
    for a in groundtruth_data['a'][1:]:  # first action is placeholder by convention, don't execute it
        o, r, term, trunc, _ = env.step(a)
        groundtruth_data['o'].append(o)
        groundtruth_data['r'].append(r)
        groundtruth_data['term'].append(term)
        groundtruth_data['trunc'].append(trunc)

        if term or trunc:
            raise RuntimeError('Environment terminated during warmup')

    # batch and convert groundtruth data
    o = torch.from_numpy(np.stack(groundtruth_data['o']))
    a = torch.tensor(groundtruth_data['a'])
    r = torch.tensor(groundtruth_data['r'])
    term = torch.tensor(groundtruth_data['term'])
    trunc = torch.tensor(groundtruth_data['trunc'])
    mask = torch.zeros_like(r)
    o, a, r, term, trunc, mask = [x.to(model.device) for x in (o, a, r, term, trunc, mask)]
    o, a, r, term, trunc, mask = [x.unsqueeze(0) for x in (o, a, r, term, trunc, mask)]
    o, a, r, term, trunc, mask = prepare_data(o, a, r, term, trunc, mask, env)

    # use lists instead of tensor objects to match new convention
    # this is ugly but works
    o = list(o.unbind(0))
    a = list(a.unbind(0))
    r = list(r.unbind(0))
    term = list(term.unbind(0))

    history['prim_o'] = o
    history['prim_a'] = a
    history['prim_r'] = r
    history['prim_term'] = term

    return history


def init_prim_s(model: MultiscaleDynamicsModelMK2,
                history: Dict[str, torch.Tensor]):
    o_start_batch = torch.stack(history['prim_o'])
    a_start_batch = torch.stack(history['prim_a'])
    r_start_batch = torch.stack(history['prim_r'])
    term_start_batch = torch.stack(history['prim_term'])

    mem, prim_current = model.rollout_primitive(a=a_start_batch, o=o_start_batch, r=r_start_batch,
                                                term=term_start_batch, n_posterior_steps=-1, sample=True)

    o_start_batch = o_start_batch.squeeze().argmax(-1)
    r_start_batch = r_start_batch.squeeze()
    term_start_batch = term_start_batch.squeeze()
    pred_o = torch.stack(mem['prim_o']).squeeze().argmax(-1)
    pred_r = torch.stack(mem['prim_r']).squeeze()
    pred_term = torch.stack(mem['prim_term']).squeeze()

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
    history_length = len(history['prim_a'])
    required_steps = model.abstract_step_size * n_warmup_abstr
    remaining_steps = required_steps - history_length

    assert 0 < history_length <= model.abstract_step_size * n_warmup_abstr, f'History length: {history_length}'
    assert remaining_steps >= 0

    if remaining_steps > 0:
        if not allow_prim_imagination:
            raise RuntimeError('Not enough warmup data for abstract model and data synthesis by primitive model is '
                               'disabled.')
        z_batch = history['prim_z'][-1].repeat(n_rollouts, 1)
        rnn_state_batch = unpack_rnn_state(pack_rnn_state(history['prim_rnn_state'][-1]).repeat(n_rollouts, 1, 1, 1))

        def _rollout_fn(_a: torch.Tensor):
            _a = to_onehot(_a, n_classes=model.d_action)
            _a = _a.swapaxes(0, 1)
            _mem, _prim_final = model.rollout_primitive(a=_a, z=z_batch, rnn_state=rnn_state_batch, sample=True)

            _criterion = torch.stack(_mem['prim_r'], dim=1).squeeze(-1)
            _discount = torch.stack(_mem['prim_term'], dim=1).squeeze(-1)
            return _criterion, _discount, _mem

        a, a_dist, i_win, R_win, data = planner_prim.plan(rollout_fn=_rollout_fn,
                                                          n_rollouts=n_rollouts,
                                                          n_plan_steps=remaining_steps)
        # filter out winner rollout for every entry in the memory
        data = select_batch_items(data, i_win[0], keepdim=True)
        history = update_history(history, data)

    a_binned = bin_every_k_steps(torch.stack(history['prim_a'][:required_steps]), model.abstract_step_size)
    abstr_a = [model.abstract_action_model(a_binned[i_chunk].swapaxes(0, 1), sample=False)
               for i_chunk in range(len(a_binned))]
    abstr_a = torch.stack(abstr_a)
    target = model.abstr_pred_target
    prim_data = bin_every_k_steps(torch.stack(history[target][:required_steps]), model.abstract_step_size)[:, -1]
    abstr_r = model.calc_abstr_r_ground_truth(torch.stack(history['prim_r'][:required_steps]))
    abstr_term = model.calc_abstr_term_ground_truth(torch.stack(history['prim_term'][:required_steps]))

    mem, abstr_final = model.rollout_abstract(a=abstr_a, prim_data=prim_data, r=abstr_r,
                                              term=abstr_term, sample=False, n_posterior_steps=-1)
    history = update_history(history, mem)
    return history


def plan_abstract(model: MultiscaleDynamicsModelMK2,
                  history: Dict[str, torch.Tensor],
                  planner_abstr: CrossentropyPlanner,
                  n_plan_steps: int,
                  n_rollouts: int):
    if n_plan_steps == 0:
        return history

    abstr_z_start = history['abstr_z'][-1].repeat(n_rollouts, 1)
    abstr_rnn_state_start = unpack_rnn_state(pack_rnn_state(history['abstr_rnn_state'][-1]).repeat(n_rollouts, 1, 1, 1))

    def _rollout_fn(_a: torch.Tensor):
        if model.abstract_action_model.model_type in ('det_tanh', 'prob_normal'):
            _a = torch.clamp(_a, -0.99, 0.99)  # limit action range to allowed values
        elif model.abstract_action_model.model_type in ('prob_categorical', 'det_mapping'):
            _a = to_onehot(_a, model.abstract_model.d_action)
        _a = _a.swapaxes(0, 1)
        _mem, _abstr_final = model.rollout_abstract(_a, z=abstr_z_start, rnn_state=abstr_rnn_state_start, sample=True,
                                                    n_posterior_steps=0)
        _criterion = torch.stack(_mem['abstr_r'], dim=1).squeeze(-1)
        _discount = torch.stack(_mem['abstr_term'], dim=1).squeeze(-1)
        return _criterion, _discount, _mem

    a, a_dist, i_win, R_win, data = planner_abstr.plan(rollout_fn=_rollout_fn,
                                                       n_rollouts=n_rollouts,
                                                       n_plan_steps=n_plan_steps)
    # update history with new rollouts
    data = select_batch_items(data, i_win[0], keepdim=True)
    history = update_history(history, data)
    return history


def plan_section(model: MultiscaleDynamicsModelMK2,
                 history: Dict[str, torch.Tensor],
                 planner_prim: CrossentropyPlanner,
                 i_section: int,
                 n_rollouts: int):
    n_steps_done = len(history['prim_a'])
    assert n_steps_done % model.abstract_step_size == 0, ('Warmup step count should be evenly divisible by the section '
                                                          f'length, but they are {n_steps_done} and '
                                                          f'{model.abstract_step_size}')
    i_t = i_section * model.abstract_step_size - 1
    z_start = history['prim_z'][i_t].repeat(n_rollouts, 1)
    rnn_state_start = unpack_rnn_state(pack_rnn_state(history['prim_rnn_state'][i_t]).repeat(n_rollouts, 1, 1, 1))
    target = history['abstr_o'][i_section].repeat(n_rollouts, *[1 for _ in model.abstract_model.o_shape])
    #target = torch.flatten(target, start_dim=1)
    target_dist = history['abstr_o_dist'][i_section]
    target_dist = target_dist.expand((n_rollouts, *target_dist.batch_shape[1:]))
    r_goal = history['abstr_r'][i_section].repeat(n_rollouts, 1)

    def _rollout_fn(_a: torch.Tensor):
        _a = to_onehot(_a, n_classes=model.primitive_model.d_action)
        _a = _a.swapaxes(0, 1)
        _mem, _prim_final = model.rollout_primitive(a=_a, z=z_start, rnn_state=rnn_state_start, sample=True)

        _targets_rollout = _mem[model.abstr_pred_target][-1]
        _criterion = - torch.flatten((target - _targets_rollout) ** 2, start_dim=1).sum(dim=1, keepdim=True)
        #_targets_rollout = torch.abs(_targets_rollout - 1e-5)
        #_targets_rollout /= _targets_rollout.sum(axis=-1, keepdims=True)

        #_criterion = target_dist.log_prob(_targets_rollout).sum(axis=-1, keepdims=True)
        _rollout_r = model.calc_abstr_r_ground_truth(torch.stack(_mem['prim_r']))[0]
        #_criterion -= torch.mean((r_goal - _rollout_r) ** 2, dim=0)
        #_rollout_goal = torch.flatten(_mem[model.abstr_pred_target][-1], start_dim=1)
        # TODO: test KL divergence between distributions
        #_criterion = - torch.mean((_rollout_goal - target) ** 2, dim=1, keepdim=True)
        #_criterion -= torch.mean((_rollout_r - r_goal) ** 2, dim=1, keepdim=True)
        _discount = None
        return _criterion, _discount, _mem

    a, a_dist, i_win, R_win, data = planner_prim.plan(rollout_fn=_rollout_fn,
                                                      n_rollouts=n_rollouts,
                                                      n_plan_steps=model.abstract_step_size)
    # filter out winner rollout for every entry in the memory
    data = select_batch_items(data, i_win[0], keepdim=True)
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