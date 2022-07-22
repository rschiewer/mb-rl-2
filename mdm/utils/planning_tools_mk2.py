from typing import Union, Tuple, Sequence, Dict
from itertools import product


import gym
import torch
import numpy as np

from mdm.models.multiscale_model_mk2 import MultiscaleDynamicsModelMK2
from mdm.models.building_blocks import RnnStateType
from mdm.planning.cem_planner import CrossentropyPlanner, TensorData
from mdm.utils.utils import normalize_obs, to_onehot, prepare_data, flatten_and_unsqueeze, sensitivity_analysis
from mdm.utils.torch_tools import add_time_dim, add_data_dim


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


def plan_section(model: MultiscaleDynamicsModelMK2,
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
    #init_prim_o = add_time_dim(broadcast_to_batch(init_prim_o, n_rollouts))
    #init_prim_r = add_time_dim(broadcast_to_batch(init_prim_r, n_rollouts))
    #init_prim_term = add_time_dim(broadcast_to_batch(init_prim_term, n_rollouts))
    prim_s = broadcast_to_batch(prim_s, n_rollouts)
    prim_rnn_state = broadcast_rnn_state_to_batch(prim_rnn_state, model.primitive_model.rnn_type, n_rollouts)
    #abstr_s = broadcast_to_batch(abstr_s, n_rollouts)
    #abstr_rnn_state = broadcast_rnn_state_to_batch(abstr_rnn_state, model.abstract_model.rnn_type, n_rollouts)
    abstr_r = broadcast_to_batch(abstr_r, n_rollouts)
    abstr_term = broadcast_to_batch(abstr_term, n_rollouts)
    #abstr_s_next = broadcast_to_batch(abstr_s_next, n_rollouts)
    #ctx_high_level = model.fuse_state(abstr_s, abstr_rnn_state)
    subtraj_hist_target = broadcast_to_batch(subtraj_hist_target, n_rollouts)
    #subtraj_hist_target = model.context_projector(subtraj_hist_target)

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

        #ctx = model.fuse_state(prim_final_['s'], prim_final_['rnn_state'])
        #ctx = model.context_projector(ctx)
        #criterion = -torch.mean(torch.abs(ctx - subtraj_hist_target) ** 2, dim=1)
        s_similarity = mem_['prim_s_post'][-1].log_prob(subtraj_hist_target).sum(dim=-1)
        r_err = torch.abs(r_total - abstr_r).squeeze(-1)
        term_err = torch.nn.functional.binary_cross_entropy(term_total, abstr_term, reduction='none').squeeze(-1)
        criterion = s_similarity - r_err - term_err
        #abstr_a_ = add_time_dim(model.abstract_action_model(a_))
        #abstr_r_target_ = mem_['prim_r'].sum(dim=1, keepdim=True)
        #abstr_term_target_ = mem_['prim_term'].max(dim=1, keepdim=True).values
        #ctx_low_level_ = add_time_dim(model.fuse_state(prim_final_['s'], prim_final_['rnn_state']))
        #mem_, abstr_current_ = model.rollout_abstract(a=abstr_a_, init_r=abstr_r, init_term=abstr_term, init_s=abstr_s,
        #                                              init_rnn_state=abstr_rnn_state, o_target=ctx_low_level_,
        #                                              r_target=abstr_r_target_, term_target=abstr_term_target_,
        #                                              mem=None, use_posterior=True, sample=False)
        #criterion = - torch.sum(torch.abs(abstr_s_next - abstr_current_['s']) ** 2, dim=1)
        #mem_ = model.pack_mem(mem_)

        # CAUTION: criterion does not have the expected dimension of (n_rollouts, n_plan_steps) which will lead to
        # a small error after calculating the discounted return during planning
        return criterion, None, mem_

    #R_winners = [torch.tensor(-10000, device=abstr_s.device)]
    #best_prim_a = None
    #while R_winners[0] < -0.1:
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
    #s_similarity = torch.abs(rollout_data['prim_s_prior'][-1].loc - subtraj_hist_target).sum(dim=-1)
    r_err = torch.abs(r_total - abstr_r).squeeze(-1)
    term_err = torch.nn.functional.binary_cross_entropy(term_total, abstr_term, reduction='none').squeeze(-1)
    criterion = s_similarity - 3 * r_err - 0.1 * term_err
    disc_ret_sorted = torch.sort(criterion, dim=0, descending=True)
    i_winners, R_winners = disc_ret_sorted.indices, disc_ret_sorted.values

    i_bst = i_winners[0]
    best_prim_a = get_item_from_batch(a_sequences, i_bst, keep_dim=False)
    #best_prim_a = to_onehot(best_prim_a, model.d_action)
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
    #assert tens.ndim <= 2
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