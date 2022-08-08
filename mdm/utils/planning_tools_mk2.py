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


def collect_groundtruth_data(model: MultiscaleDynamicsModelMK2,
                             history: Dict[str, torch.Tensor],
                             env: gym.Env,
                             n_warmup: int):
    # prepare actions and first step's data (note: only the initial observation contains real environment info)
    groundtruth_data = {'a': [], 'o': [], 'r': [], 'term': []}
    groundtruth_data['a'] += [0] + [env.action_space.sample() for _ in range(n_warmup - 1)]
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
                 n_warmup_abstr: int,
                 planner_prim: CrossentropyPlanner,
                 n_rollouts: int,
                 n_evolution_steps: int,
                 winning_perc: float,
                 discount: float,
                 a_noise_prim: float):
    history_length = history['prim_a'].shape[1]
    required_steps = model.abstract_step_size * n_warmup_abstr
    remaining_steps = required_steps - history_length

    assert 0 < history_length <= model.abstract_step_size * n_warmup_abstr, f'History length: {history_length}'
    assert remaining_steps >= 0

    if remaining_steps > 0:
        raise ValueError('Not supported yet')

    if remaining_steps > 0:
        s_batch = history['prim_s'][:, -1].repeat(n_rollouts, 1)
        rnn_state_batch = unpack_rnn_state(history['prim_rnn_state'][:, -1].repeat(n_rollouts, 1, 1, 1))

        def _rollout_fn(_a: torch.Tensor):
            _a = to_onehot(_a, n_classes=model.d_action)
            _mem, _prim_final = model.rollout_primitive(a=_a, s=s_batch, rnn_state=rnn_state_batch, sample=False)
            _mem = model.pack_mem(_mem)

            _criterion = _mem['prim_r'].squeeze(-1)
            _discount = _mem['prim_term'].squeeze(-1)
            return _criterion, _discount, _mem

        a, a_dist, i_win, R_win, data = planner_prim.plan(rollout_fn=_rollout_fn,
                                                          d_dist=history['prim_a'].shape[-1],
                                                          n_rollouts=n_rollouts,
                                                          n_plan_steps=remaining_steps,
                                                          n_evolution_steps=n_evolution_steps,
                                                          winning_perc=winning_perc,
                                                          discount=discount,
                                                          act_noise=a_noise_prim)
        # filter out winner rollout for every entry in the memory
        data = {k: v[i_win[0], None] if len(v) > 0 else v for k, v in data.items()}
        history = update_history(history, data)

    a_binned = bin_every_k_steps(history['prim_a'][:, :required_steps], model.abstract_step_size)
    abstr_a = torch.stack([model.abstract_action_model(a_binned[:, i_chunk], sample=False)
                           for i_chunk in range(a_binned.shape[1])], dim=1)
    abstr_o = bin_every_k_steps(history['prim_s'][:, :required_steps], model.abstract_step_size)[:, :, -1]
    abstr_r = bin_every_k_steps(history['prim_r'][:, :required_steps], model.abstract_step_size).sum(dim=2)
    abstr_term = bin_every_k_steps(history['prim_term'][:, :required_steps], model.abstract_step_size).max(dim=2).values

    mem, abstr_final = model.rollout_abstract(a=abstr_a, primitive_history=abstr_o, r=abstr_r, term=abstr_term,
                                              sample=False, n_posterior_steps=n_warmup_abstr)
    mem = model.pack_mem(mem)
    history = update_history(history, mem)
    return history


def init_model(model: MultiscaleDynamicsModelMK2,
               env: gym.Env,
               n_warmup_prim: int,
               n_warmup_abstr: int,
               planner: CrossentropyPlanner,
               n_rollouts: int,
               n_evolution_steps: int,
               winning_perc: float,
               discount: float,
               a_noise_prim: float):
    # prepare actions and first step, only the initial observation contains real environment info
    groundtruth_data = {'a': [], 'o': [], 'r': [], 'term': []}
    groundtruth_data['a'] += [0] + [env.action_space.sample() for _ in range(n_warmup_prim)]
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
    a_start = torch.tensor(groundtruth_data['a']).to(model.device)
    o_start = torch.from_numpy(np.stack(groundtruth_data['o'])).to(model.device)
    r_start = torch.tensor(groundtruth_data['r']).to(model.device)
    term_start = torch.tensor(groundtruth_data['term']).to(model.device)

    # do primitive model rollout with groundtruth data posterior to produce h and s
    a_start_batch = broadcast_to_batch(a_start, 1)
    o_start_batch = broadcast_to_batch(o_start, 1)
    r_start_batch = broadcast_to_batch(r_start, 1)
    term_start_batch = broadcast_to_batch(term_start, 1)
    o_start_batch, a_start_batch, r_start_batch, term_start_batch = prepare_data(o_start_batch, a_start_batch,
                                                                                 r_start_batch, term_start_batch, env)

    mem, prim_current = model.rollout_primitive(a=a_start_batch, o=o_start_batch, r=r_start_batch,
                                                term=term_start_batch,
                                                n_posterior_steps=n_warmup_prim, sample=False)
    mem = model.pack_mem(mem)
    trajectory = model.gen_mem()
    trajectory['prim_a'] = a_start_batch
    trajectory['prim_o'] = o_start_batch
    trajectory['prim_r'] = r_start_batch
    trajectory['prim_term'] = term_start_batch
    trajectory['prim_s_post'] = mem['prim_s_post']
    trajectory['prim_rnn_state'] = mem['prim_rnn_state']

    # produce remaining primitive model steps for abstract model warmup
    remaining_steps = model.abstract_step_size * n_warmup_abstr - n_warmup_prim - 1  # initial step counts as well
    if remaining_steps > 0:
        s_batch = broadcast_to_batch(prim_current['s'][0], n_rollouts)
        rnn_state_batch = unpack_rnn_state(broadcast_to_batch(pack_rnn_state(prim_current['rnn_state'])[0], n_rollouts))

        def _rollout_fn(_a: torch.Tensor):
            _a = to_onehot(_a, n_classes=model.d_action)
            _mem, _prim_final = model.rollout_primitive(a=_a, s=s_batch, rnn_state=rnn_state_batch, sample=False)
            _mem = model.pack_mem(_mem)

            _criterion = _mem['prim_r'].squeeze(-1)
            _discount = _mem['prim_term'].squeeze(-1)
            return _criterion, _discount, _mem

        plan = planner.plan(rollout_fn=_rollout_fn,
                            d_dist=env.action_space.n,
                            n_rollouts=n_rollouts,
                            n_plan_steps=remaining_steps,
                            n_evolution_steps=n_evolution_steps,
                            winning_perc=winning_perc,
                            discount=discount,
                            act_noise=a_noise_prim)

        # collect best possible trajectory
        a_rollout, a_rollout_dist, i_winners, R_winners, rollout_data = plan
        i_best = i_winners[0]
        a_rollout_onehot = to_onehot(a_rollout[i_best, None], env.action_space.n)
        trajectory['a'] = torch.concat([trajectory['prim_a'], a_rollout_onehot], dim=1)
        trajectory['prim_o'] = torch.concat([trajectory['prim_o'], rollout_data['prim_o'][i_best, None]], dim=1)
        trajectory['prim_r'] = torch.concat([trajectory['prim_r'], rollout_data['prim_r'][i_best, None]], dim=1)
        trajectory['prim_term'] = torch.concat([trajectory['prim_term'], rollout_data['prim_term'][i_best, None]],
                                               dim=1)
        trajectory['prim_s'] = torch.concat([trajectory['prim_s'], rollout_data['prim_s'][i_best, None]], dim=1)
        # TODO: pack all rnn states im some format so that selecting batch and time step is easy
        rnn_states = []
        trajectory['prim_rnn_state'] = unpack_rnn_state(pack_rnn_state(rollout_data['prim_rnn_state'])[i_best, None])

    # TODO: 1) collect data from primitive model to trajectory memory 2) use model.forward() to init abstract model
    # 1) DONE

    abstr_r = bin_every_k_steps(trajectory['prim_r'], model.abstract_step_size, padding_val=0).sum(dim=2)
    abstr_term = bin_every_k_steps(trajectory['prim_term'], model.abstract_step_size).max(dim=2).values
    abstr_o = bin_every_k_steps(trajectory['prim_s'], model.abstract_step_size, padding_val=0)[:, :,
              -1]  # (batch, bin, timestep_in_bin, ...)
    prim_a_onehot = to_onehot(trajectory['a'], env.action_space.n)
    abstr_a = model.abstract_action_model(bin_every_k_steps(prim_a_onehot, model.abstract_step_size, padding_val=0))

    mem, abstr_final = model.rollout_abstract(a=abstr_a, primitive_history=abstr_o, r=abstr_r, term=abstr_term,
                                              sample=False)

    abstr_a = abstr_a


def init_s_prim(model: MultiscaleDynamicsModelMK2,
                env: gym.Env,
                n_step: int,
                trajectory_history: dict):
    a_mem = [0] + [env.action_space.sample() for _ in range(n_step)]
    o_mem, r_mem, term_mem = [env.reset()], [0.0], [0.0]

    for a in a_mem[1:]:  # first action is zero placeholder by convention, don't execute it
        o, r, term, _ = env.step(a)
        o_mem.append(o)
        r_mem.append(r)
        term_mem.append(term)

    # transform initial samples to GPU tensors and add batch dimension
    a_mem = torch.tensor(a_mem).unsqueeze(0).to(model.device)
    o_mem = torch.from_numpy(np.stack(o_mem)).unsqueeze(0).to(model.device)
    r_mem = torch.tensor(r_mem).unsqueeze(0).to(model.device)
    term_mem = torch.tensor(term_mem).unsqueeze(0).to(model.device)
    o_mem, a_mem, r_mem, term_mem = prepare_data(o_mem, a_mem, r_mem, term_mem, env)

    mem, prim_current = model.rollout_primitive(a_mem, o_mem, r_mem, term_mem, sample=False)

    s = get_item_from_batch(prim_current['s'], 0, keep_dim=False)
    rnn_state = get_rnn_state_from_batch(prim_current['rnn_state'], model.primitive_model.rnn_type, 0, keep_dim=False)

    # TODO: init abstract s here as well

    return {'s': s, 'rnn_state': rnn_state}


def init_s_abstr(model: MultiscaleDynamicsModelMK2,
                 planner: CrossentropyPlanner,
                 env: gym.Env,
                 s_start: torch.tensor,
                 rnn_state_start: torch.tensor,
                 n_rollouts: int,
                 n_evolution_steps: int,
                 winning_perc: float,
                 discount: float,
                 act_noise: float):
    s_start = broadcast_to_batch(s_start, n_rollouts)
    rnn_state_start = broadcast_rnn_state_to_batch(rnn_state_start, model.primitive_model.rnn_type, n_rollouts)

    def _rollout_init_fn(a_: torch.Tensor):
        a_ = to_onehot(a_, n_classes=model.d_action)
        mem_, prim_final_ = model.rollout_primitive(a=a_, o=None, r=None, term=None, s=s_start,
                                                    rnn_state=rnn_state_start, ctx_high_level=None, mem=None,
                                                    sample=False)
        mem_ = model.pack_mem(mem_)
        return mem_['prim_r'].squeeze(-1), mem_['prim_term'].squeeze(-1), mem_

    # get best subtrajectory
    a, act_dist, i_winners, R_winners, rollout_data = planner.plan(rollout_fn=_rollout_init_fn,
                                                                   d_dist=env.action_space.n,
                                                                   n_rollouts=n_rollouts,
                                                                   n_plan_steps=model.abstract_step_size,
                                                                   n_evolution_steps=n_evolution_steps,
                                                                   winning_perc=winning_perc,
                                                                   discount=discount,
                                                                   act_noise=act_noise)
    i_bst = i_winners[0]

    best_prim_rnn_state = get_rnn_state_from_batch(rollout_data['prim_rnn_state'][-1], model.primitive_model.rnn_type,
                                                   i_bst)
    best_prim_s = get_item_from_batch(rollout_data['prim_s'], i_bst)[:, -1]
    best_ctx = add_time_dim(model.fuse_state(best_prim_s, best_prim_rnn_state))

    best_prim_a = get_item_from_batch(a, i_bst)
    best_prim_a = to_onehot(best_prim_a, model.d_action)
    best_prim_o = get_item_from_batch(rollout_data['prim_o'], i_bst)[:, -1]
    best_prim_r = get_item_from_batch(rollout_data['prim_r'], i_bst)[:, -1]
    best_prim_term = get_item_from_batch(rollout_data['prim_term'], i_bst)[:, -1]

    abstr_a = add_time_dim(model.abstract_action_model(best_prim_a))
    abstr_r = add_time_dim(best_prim_r.sum(dim=1, keepdim=True))
    abstr_term = add_time_dim(best_prim_term.max(dim=1, keepdim=True).values)
    mem, abstr_current = model.rollout_abstract(a=abstr_a, primitive_history=best_ctx, r=abstr_r, term=abstr_term,
                                                s=None, rnn_state=None, mem=None, sample=False)
    mem = model.pack_mem(mem)

    # remove batch dimensions before returning
    prim_a = best_prim_a[0]
    prim_o = best_prim_o[0]
    prim_r = best_prim_r[0]
    prim_term = best_prim_term[0]
    prim_rnn_state = get_rnn_state_from_batch(best_prim_rnn_state, model.primitive_model.rnn_type, 0, keep_dim=False)
    prim_s = best_prim_s[0]
    abstr_s = abstr_current['s'][0]
    abstr_rnn_state = get_rnn_state_from_batch(abstr_current['rnn_state'], model.abstract_model.rnn_type, 0,
                                               keep_dim=False)
    abstr_o = abstr_current['o'][0]
    abstr_r = abstr_current['r'][0]
    abstr_term = abstr_current['term'][0]

    return {'prim_o': prim_o,
            'prim_a': prim_a,
            'prim_r': prim_r,
            'prim_term': prim_term,
            'prim_s': prim_s,
            'prim_rnn_state': prim_rnn_state,
            'abstr_s': abstr_s,
            'abstr_rnn_state': abstr_rnn_state,
            'abstr_o': abstr_o,
            'abstr_r': abstr_r,
            'abstr_term': abstr_term}


def plan_abstract(model: MultiscaleDynamicsModelMK2,
                  history: Dict[str, torch.Tensor],
                  planner_abstr: CrossentropyPlanner,
                  n_plan_steps: int,
                  n_rollouts: int,
                  n_evolution_steps: int,
                  winning_perc: float,
                  discount: float,
                  a_noise_abstr: float):
    if n_plan_steps == 0:
        return history

    abstr_s_start = history['abstr_s'][:, -1].repeat(n_rollouts, 1)
    abstr_rnn_state_start = unpack_rnn_state(history['abstr_rnn_state'][:, -1].repeat(n_rollouts, 1, 1, 1))

    def _rollout_fn(_a: torch.Tensor):
        #_a = to_onehot(_a, model.abstract_model.d_action)
        _a = torch.clamp(_a, -0.99, 0.99)  # limit action range to allowed values
        _mem, _abstr_final = model.rollout_abstract(_a, s=abstr_s_start, rnn_state=abstr_rnn_state_start, sample=False,
                                                    n_posterior_steps=0)
        _mem = model.pack_mem(_mem)

        _criterion = _mem['abstr_r'].squeeze(-1)
        _discount = _mem['abstr_term'].squeeze(-1)
        return _criterion, _discount, _mem

    a, a_dist, i_win, R_win, data = planner_abstr.plan(rollout_fn=_rollout_fn,
                                                       d_dist=model.abstract_model.d_action,
                                                       n_rollouts=n_rollouts,
                                                       n_plan_steps=n_plan_steps,
                                                       n_evolution_steps=n_evolution_steps,
                                                       winning_perc=winning_perc,
                                                       discount=discount,
                                                       act_noise=a_noise_abstr)
    # update history with new rollouts
    data = {k: v[i_win[0], None] if len(v) > 0 else v for k, v in data.items()}
    history = update_history(history, data)
    return history


def plan_section(model: MultiscaleDynamicsModelMK2,
                 history: Dict[str, torch.Tensor],
                 planner_prim: CrossentropyPlanner,
                 n_rollouts: int,
                 n_evolution_steps: int,
                 winning_perc: float,
                 discount: float,
                 a_noise_prim: float):
    n_steps_done = history['prim_a'].shape[1]
    assert n_steps_done % model.abstract_step_size == 0, ('Warmup step count should be evenly divisible by the section '
                                                          f'length, but they are {n_steps_done} and '
                                                          f'{model.abstract_step_size}')
    i_section = n_steps_done // model.abstract_step_size # + n_steps_done // model.abstract_step_size

    s_start = history['prim_s'][:, -1].repeat(n_rollouts, 1)
    rnn_state_start = unpack_rnn_state(history['prim_rnn_state'][:, -1].repeat(n_rollouts, 1, 1, 1))
    s_goal = history['abstr_o'][:, i_section].repeat(n_rollouts, 1)
    r_goal = history['abstr_r'][:, i_section].repeat(n_rollouts, 1)

    def _rollout_fn(_a: torch.Tensor):
        _a = to_onehot(_a, n_classes=model.d_action)
        _mem, _prim_final = model.rollout_primitive(a=_a, s=s_start, rnn_state=rnn_state_start, sample=False)
        _mem = model.pack_mem(_mem)

        _criterion = - torch.mean((_mem['prim_s'][:, 0] - s_goal) ** 2, dim=1, keepdim=True)
        _discount = None
        return _criterion, _discount, _mem

    a, a_dist, i_win, R_win, data = planner_prim.plan(rollout_fn=_rollout_fn,
                                                      d_dist=history['prim_a'].shape[-1],
                                                      n_rollouts=n_rollouts,
                                                      n_plan_steps=model.abstract_step_size,
                                                      n_evolution_steps=3,
                                                      winning_perc=winning_perc,
                                                      discount=discount,
                                                      act_noise=a_noise_prim)
    # filter out winner rollout for every entry in the memory
    data = {k: v[i_win[0], None] if len(v) > 0 else v for k, v in data.items()}
    history = update_history(history, data)

    return history


def plan_abstract_old(model: MultiscaleDynamicsModelMK2,
                      planner: CrossentropyPlanner,
                      abstr_s_start: torch.Tensor,
                      abstr_rnn_state_start: RnnStateType,
                      abstr_r_start: torch.Tensor,
                      abstr_term_start: torch.Tensor,
                      n_plan_steps: int,
                      n_rollouts: int,
                      n_evolution_steps: int,
                      winning_perc: float,
                      discount: float,
                      act_noise: float):
    abstr_s_start = broadcast_to_batch(abstr_s_start, n_rollouts)
    abstr_rnn_state_start = broadcast_rnn_state_to_batch(abstr_rnn_state_start, model.abstract_model.rnn_type,
                                                         n_rollouts)
    abstr_r_start = broadcast_to_batch(abstr_r_start, n_rollouts)
    abstr_r_start = add_time_dim(abstr_r_start)
    abstr_term_start = broadcast_to_batch(abstr_term_start, n_rollouts)
    abstr_term_start = add_time_dim(abstr_term_start)

    def _rollout_abstract_fn(a_: torch.Tensor):
        mem_, abstr_final_ = model.rollout_abstract(a=a_, r=abstr_r_start, term=abstr_term_start,
                                                    s=abstr_s_start, rnn_state=abstr_rnn_state_start,
                                                    mem=None, sample=False)
        mem_ = model.pack_mem(mem_)
        return mem_['abstr_r'].squeeze(-1), mem_['abstr_term'].squeeze(-1), mem_

    abstr_a, a_dist, i_winners, R_winners, rollout_data = planner.plan(rollout_fn=_rollout_abstract_fn,
                                                                       d_dist=model.d_abstract_action,
                                                                       n_rollouts=n_rollouts,
                                                                       n_plan_steps=n_plan_steps,
                                                                       n_evolution_steps=n_evolution_steps,
                                                                       winning_perc=winning_perc,
                                                                       discount=discount,
                                                                       act_noise=act_noise)
    i_bst = i_winners[0]
    best_abstr_a = abstr_a[i_bst]
    best_abstr_rnn_state = [get_rnn_state_from_batch(h, model.abstract_model.rnn_type, 0, keep_dim=False)
                            for h in rollout_data['abstr_rnn_state']]
    best_abstr_s = rollout_data['abstr_s'][i_bst]
    best_abstr_o = rollout_data['abstr_o'][i_bst]
    best_abstr_r = rollout_data['abstr_r'][i_bst]
    best_abstr_term = rollout_data['abstr_term'][i_bst]

    return {'abstr_o': best_abstr_o,
            'abstr_a': best_abstr_a,
            'abstr_r': best_abstr_r,
            'abstr_term': best_abstr_term,
            'abstr_s': best_abstr_s,
            'abstr_rnn_state': best_abstr_rnn_state}


def plan_section_old(model: MultiscaleDynamicsModelMK2,
                     planner: CrossentropyPlanner,
                     env: gym.Env,
                     init_prim_o: torch.Tensor,
                     init_prim_r: torch.Tensor,
                     init_prim_term: torch.Tensor,
                     prim_s: torch.Tensor,
                     prim_rnn_state: torch.Tensor,
                     subtraj_hist_target: torch.Tensor,
                     abstr_s: torch.Tensor,
                     abstr_rnn_state: RnnStateType,
                     abstr_r: torch.Tensor,
                     abstr_term: torch.Tensor,
                     abstr_s_next: torch.Tensor,
                     n_rollouts: int,
                     n_evolution_steps: int,
                     winning_perc: float,
                     act_noise: float):
    # Tile init data along batch dimension for rollouts
    # init_prim_o = add_time_dim(broadcast_to_batch(init_prim_o, n_rollouts))
    # init_prim_r = add_time_dim(broadcast_to_batch(init_prim_r, n_rollouts))
    # init_prim_term = add_time_dim(broadcast_to_batch(init_prim_term, n_rollouts))
    prim_s = broadcast_to_batch(prim_s, n_rollouts)
    prim_rnn_state = broadcast_rnn_state_to_batch(prim_rnn_state, model.primitive_model.rnn_type, n_rollouts)
    # abstr_s = broadcast_to_batch(abstr_s, n_rollouts)
    # abstr_rnn_state = broadcast_rnn_state_to_batch(abstr_rnn_state, model.abstract_model.rnn_type, n_rollouts)
    abstr_r = broadcast_to_batch(abstr_r, n_rollouts)
    abstr_term = broadcast_to_batch(abstr_term, n_rollouts)
    # abstr_s_next = broadcast_to_batch(abstr_s_next, n_rollouts)
    # ctx_high_level = model.fuse_state(abstr_s, abstr_rnn_state)
    subtraj_hist_target = broadcast_to_batch(subtraj_hist_target, n_rollouts)

    # subtraj_hist_target = model.context_projector(subtraj_hist_target)

    # The only changing part for every batch item will be the combination of prim_a
    def _rollout_detailed_fn(a_: torch.Tensor):
        a_ = flatten_and_unsqueeze(a_)
        a_ = to_onehot(a_, n_classes=model.d_action)
        mem_, prim_final_ = model.rollout_primitive(a=a_, o=None, r=None,
                                                    term=None, s=prim_s,
                                                    rnn_state=prim_rnn_state,
                                                    ctx_high_level=None, mem=None, sample=False)
        mem_ = model.pack_mem(mem_)
        r_total = mem_['prim_r'].sum(dim=1)
        term_total = mem_['prim_term'].max(dim=1).values

        # ctx = model.fuse_state(prim_final_['s'], prim_final_['rnn_state'])
        # ctx = model.context_projector(ctx)
        # criterion = -torch.mean(torch.abs(ctx - subtraj_hist_target) ** 2, dim=1)
        s_post = build_gaussian(mem_['prim_s_post'][:, -1])
        s_similarity = s_post.log_prob(subtraj_hist_target).sum(dim=-1)
        r_err = torch.abs(r_total - abstr_r).squeeze(-1)
        term_err = torch.nn.functional.binary_cross_entropy(term_total, abstr_term, reduction='none').squeeze(-1)
        criterion = s_similarity - r_err - term_err
        criterion = criterion.unsqueeze(-1)  # criterion needs a batch and a time dimension
        # abstr_a_ = add_time_dim(model.abstract_action_model(a_))
        # abstr_r_target_ = mem_['prim_r'].sum(dim=1, keepdim=True)
        # abstr_term_target_ = mem_['prim_term'].max(dim=1, keepdim=True).values
        # ctx_low_level_ = add_time_dim(model.fuse_state(prim_final_['s'], prim_final_['rnn_state']))
        # mem_, abstr_current_ = model.rollout_abstract(a=abstr_a_, init_r=abstr_r, init_term=abstr_term, init_s=abstr_s,
        #                                              init_rnn_state=abstr_rnn_state, o_target=ctx_low_level_,
        #                                              r_target=abstr_r_target_, term_target=abstr_term_target_,
        #                                              mem=None, use_posterior=True, sample=False)
        # criterion = - torch.sum(torch.abs(abstr_s_next - abstr_current_['s']) ** 2, dim=1)
        # mem_ = model.pack_mem(mem_)

        # CAUTION: criterion does not have the expected dimension of (n_rollouts, n_plan_steps) which will lead to
        # a small error after calculating the discounted return during planning
        return criterion, None, mem_

    # R_winners = [torch.tensor(-10000, device=abstr_s.device)]
    # best_prim_a = None
    # while R_winners[0] < -0.1:
    a, act_dist, i_winners, R_winners, rollout_data = planner.plan(rollout_fn=_rollout_detailed_fn,
                                                                   d_dist=env.action_space.n,
                                                                   n_rollouts=n_rollouts,
                                                                   n_plan_steps=model.abstract_step_size,
                                                                   n_evolution_steps=n_evolution_steps,
                                                                   winning_perc=0.2,
                                                                   discount=1,
                                                                   act_noise=act_noise)
    i_bst = i_winners[0]
    best_prim_a = get_item_from_batch(a, i_bst, keep_dim=False)
    best_prim_a = to_onehot(best_prim_a, model.d_action)
    best_prim_o = get_item_from_batch(rollout_data['prim_o'], i_bst, keep_dim=False)[-1]
    best_prim_r = get_item_from_batch(rollout_data['prim_r'], i_bst, keep_dim=False)[-1]
    best_prim_term = get_item_from_batch(rollout_data['prim_term'], i_bst, keep_dim=False)[-1]
    best_rnn_state = get_rnn_state_from_batch(rollout_data['prim_rnn_state'][-1], model.primitive_model.rnn_type,
                                              i_bst, keep_dim=False)
    best_s = get_item_from_batch(rollout_data['prim_s'], i_bst, keep_dim=False)[-1]

    return {'prim_o': best_prim_o,
            'prim_a': best_prim_a,
            'prim_r': best_prim_r,
            'prim_term': best_prim_term,
            'prim_s': best_s,
            'prim_rnn_state': best_rnn_state}


def plan_section_2(model: MultiscaleDynamicsModelMK2,
                   planner: CrossentropyPlanner,
                   env: gym.Env,
                   init_prim_o: torch.Tensor,
                   init_prim_r: torch.Tensor,
                   init_prim_term: torch.Tensor,
                   prim_s: torch.Tensor,
                   prim_rnn_state: torch.Tensor,
                   subtraj_hist_target: torch.Tensor,
                   abstr_s: torch.Tensor,
                   abstr_rnn_state: RnnStateType,
                   abstr_r: torch.Tensor,
                   abstr_term: torch.Tensor,
                   abstr_s_next: torch.Tensor,
                   n_rollouts: int,
                   n_evolution_steps: int,
                   winning_perc: float,
                   act_noise: float):
    available_actions = list(range(env.action_space.n))
    a_sequences = list(product(available_actions, repeat=model.abstract_step_size))
    a_sequences = torch.tensor(a_sequences).to(model.device)
    a_sequences = to_onehot(a_sequences, model.d_action)

    prim_s = broadcast_to_batch(prim_s, a_sequences.shape[0])
    prim_rnn_state = broadcast_rnn_state_to_batch(prim_rnn_state, model.primitive_model.rnn_type, a_sequences.shape[0])
    abstr_r = broadcast_to_batch(abstr_r, a_sequences.shape[0])
    abstr_term = broadcast_to_batch(abstr_term, a_sequences.shape[0])
    subtraj_hist_target = broadcast_to_batch(subtraj_hist_target, a_sequences.shape[0])

    rollout_data, prim_final_ = model.rollout_primitive(a=a_sequences, s=prim_s, rnn_state=prim_rnn_state, sample=False)
    rollout_data = model.pack_mem(rollout_data)
    r_total = rollout_data['prim_r'].sum(dim=1)
    term_total = rollout_data['prim_term'].max(dim=1).values
    s_similarity = rollout_data['prim_s_post'][-1].log_prob(subtraj_hist_target).sum(dim=-1)
    # s_similarity = torch.abs(rollout_data['prim_s_prior'][-1].loc - subtraj_hist_target).sum(dim=-1)
    r_err = torch.abs(r_total - abstr_r).squeeze(-1)
    term_err = torch.nn.functional.binary_cross_entropy(term_total, abstr_term, reduction='none').squeeze(-1)
    criterion = s_similarity - 0.1 * r_err - 0.1 * term_err
    disc_ret_sorted = torch.sort(criterion, dim=0, descending=True)
    i_winners, R_winners = disc_ret_sorted.indices, disc_ret_sorted.values

    i_bst = i_winners[0]
    best_prim_a = get_item_from_batch(a_sequences, i_bst, keep_dim=False)
    # best_prim_a = to_onehot(best_prim_a, model.d_action)
    best_prim_o = get_item_from_batch(rollout_data['prim_o'], i_bst, keep_dim=False)[-1]
    best_prim_r = get_item_from_batch(rollout_data['prim_r'], i_bst, keep_dim=False)[-1]
    best_prim_term = get_item_from_batch(rollout_data['prim_term'], i_bst, keep_dim=False)[-1]
    best_rnn_state = get_rnn_state_from_batch(rollout_data['prim_rnn_state'][-1], model.primitive_model.rnn_type,
                                              i_bst, keep_dim=False)
    best_s = get_item_from_batch(rollout_data['prim_s'], i_bst, keep_dim=False)[-1]

    return {'prim_o': best_prim_o,
            'prim_a': best_prim_a,
            'prim_r': best_prim_r,
            'prim_term': best_prim_term,
            'prim_s': best_s,
            'prim_rnn_state': best_rnn_state}


def get_item_from_batch(data_batch: torch.Tensor,
                        idx: int,
                        keep_dim: bool = True):
    """
    Get item *idx* from *data_batch* and still maintain a batch dimension of 1 if keep_dim is True, which is the
    default. This is exactly *data_batch[idx]* if keep_dim is False.

    :param data_batch: data to select entry from
    :param idx: index of entry
    :param keep_dim: whether or not to keep the batch dimension of 1
    :return: data_batch[idx].unsqueeze(0)
    """
    if keep_dim:
        return data_batch[idx].unsqueeze(0)
    else:
        return data_batch[idx]


def broadcast_to_batch(tens: torch.Tensor,
                       d_batch: int):
    # assert tens.ndim <= 2
    repeat_per_dim = [d_batch] + [1 for _ in range(tens.ndim)]
    return torch.tile(tens.unsqueeze(0), dims=repeat_per_dim)


def get_rnn_state_from_batch(rnn_states: RnnStateType,
                             rnn_type: str,
                             idx: int,
                             keep_dim: bool = True):
    # rnn states are organized h=(layer, batch, data)
    if rnn_type == 'lstm':
        if keep_dim:
            # unsqueeze for batch dim
            return rnn_states[0][:, idx].unsqueeze(1), rnn_states[1][:, idx].unsqueeze(1)
        else:
            return rnn_states[0][:, idx], rnn_states[1][:, idx]
    elif rnn_type == 'gru':
        if keep_dim:
            ret_item = rnn_states[:, idx].unsqueeze(1)
        else:
            ret_item = rnn_states[:, idx]
    else:
        raise ValueError(f'Unknown rnn type: {rnn_type}')
    return ret_item


def broadcast_rnn_state_to_batch(rnn_state: RnnStateType,
                                 rnn_type: str,
                                 d_batch: int):
    if rnn_type == 'lstm':
        ret_item = (torch.tile(rnn_state[0].unsqueeze(1), dims=(1, d_batch, 1)),
                    torch.tile(rnn_state[1].unsqueeze(1), dims=(1, d_batch, 1)))
    elif rnn_type == 'gru':
        ret_item = torch.tile(rnn_state.unsqueeze(1), dims=(1, d_batch, 1))
    else:
        raise ValueError(f'Unknown rnn type: {rnn_type}')
    return ret_item
