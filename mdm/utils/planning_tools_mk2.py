from typing import Union, Tuple, Sequence, Dict

import gym
import torch

from mdm.models.multiscale_model_mk2 import MultiscaleDynamicsModelMK2
from mdm.models.building_blocks import RnnStateType
from mdm.planning.cem_planner import CrossentropyPlanner, TensorData
from mdm.utils.utils import normalize_obs, to_onehot
from mdm.utils.torch_tools import add_time_dim


def init_s_abstr(model: MultiscaleDynamicsModelMK2,
                 planner: CrossentropyPlanner,
                 env: gym.Env,
                 o_start: torch.Tensor,
                 n_rollouts: int,
                 n_evolution_steps: int,
                 winning_perc: float,
                 discount: float,
                 act_noise: float):
    # preprocess starting state
    o_start = normalize_obs(o_start, env)
    o_start = broadcast_to_batch(o_start, n_rollouts)
    if o_start.ndim == 2:
        o_start = add_time_dim(o_start)

    def _rollout_init_fn(a_: torch.Tensor):
        a_ = to_onehot(a_, n_classes=model.d_action)
        mem_, prim_final_ = model.rollout_primitive(a=a_, init_o=o_start, init_r=None, init_term=None,
                                                    init_s=None, init_rnn_state=None,
                                                    ctx_high_level=None, o_target=None, r_target=None,
                                                    term_target=None, mem=None, use_posterior=False, sample=False)
        mem_ = model.pack_mem(mem_)
        mem_['rnn_state'] = prim_final_['rnn_state']
        return mem_['prim_r'].squeeze(-1), None, mem_

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

    best_h = get_rnn_state_from_batch(rollout_data['rnn_state'], model.primitive_model.rnn_type, i_bst)
    best_s = get_item_from_batch(rollout_data['prim_s'], i_bst)[:, -1]
    best_ctx = add_time_dim(model.fuse_state(best_s, best_h))

    best_prim_a = get_item_from_batch(a, i_bst)
    best_prim_a = to_onehot(best_prim_a, model.d_action)
    best_prim_r = get_item_from_batch(rollout_data['prim_r'], i_bst)
    best_prim_term = get_item_from_batch(rollout_data['prim_term'], i_bst)

    a_abstr = add_time_dim(model.abstract_action_model(best_prim_a))
    r_target = best_prim_r.sum(dim=1, keepdim=True)
    term_target = best_prim_term.max(dim=1, keepdim=True).values
    mem, abstr_current = model.rollout_abstract(a=a_abstr, init_r=None, init_term=None, init_s=None,
                                                init_rnn_state=None,
                                                ctx_low_level=best_ctx, r_target=r_target,
                                                term_target=term_target, mem=None, use_posterior=True, sample=False)
    mem = model.pack_mem(mem)

    # remove batch dimensions before returning
    prim_a = best_prim_a[0]
    abstr_s = abstr_current['s'][0]
    abstr_rnn_state = get_rnn_state_from_batch(abstr_current['rnn_state'], model.abstract_model.rnn_type, 0,
                                               keep_dim=False)
    abstr_r = abstr_current['r'][0]
    abstr_term = abstr_current['term'][0]

    return {'prim_a': prim_a,
            'abstr_s': abstr_s,
            'abstr_rnn_state': abstr_rnn_state,
            'abstr_r': r_target[0, 0],  # use predictions from primitive model
            'abstr_term': term_target[0, 0]}
            #'abstr_r': abstr_r,
            #'abstr_term': abstr_term}


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
        mem_, abstr_final_ = model.rollout_abstract(a=a_, init_r=abstr_r_start, init_term=abstr_term_start,
                                                    init_s=abstr_s_start, init_rnn_state=abstr_rnn_state_start,
                                                    ctx_low_level=None, r_target=None, term_target=None, mem=None,
                                                    use_posterior=False, sample=False)
        mem_ = model.pack_mem(mem_)
        return mem_['abstr_r'].squeeze(-1), None, mem_

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
    best_abstr_r = rollout_data['abstr_r'][i_bst]
    best_abstr_term = rollout_data['abstr_term'][i_bst]

    return {'abstr_s': best_abstr_s,
            'abstr_rnn_state': best_abstr_rnn_state,
            'abstr_a': best_abstr_a,
            'abstr_r': best_abstr_r,
            'abstr_term': best_abstr_term}


def plan_section(model: MultiscaleDynamicsModelMK2,
                 planner: CrossentropyPlanner,
                 env: gym.Env,
                 abstr_s: torch.Tensor,
                 abstr_rnn_state: RnnStateType,
                 abstr_r: torch.Tensor,
                 abstr_term: torch.Tensor,
                 abstr_s_next: torch.Tensor,
                 n_rollouts: int,
                 n_evolution_steps: int,
                 winning_perc: float,
                 act_noise: float):
    abstr_s = broadcast_to_batch(abstr_s, n_rollouts)
    abstr_rnn_state = broadcast_rnn_state_to_batch(abstr_rnn_state, model.abstract_model.rnn_type, n_rollouts)
    abstr_r = add_time_dim(broadcast_to_batch(abstr_r, n_rollouts))
    abstr_term = add_time_dim(broadcast_to_batch(abstr_term, n_rollouts))
    abstr_s_next = broadcast_to_batch(abstr_s_next, n_rollouts)
    ctx_high_level = model.fuse_state(abstr_s, abstr_rnn_state)

    def _rollout_detailed_fn(a_: torch.Tensor):
        a_ = to_onehot(a_, n_classes=model.d_action)
        mem_, prim_final_ = model.rollout_primitive(a=a_, init_o=None, init_r=None, init_term=None,
                                                    init_s=None, init_rnn_state=None,
                                                    ctx_high_level=ctx_high_level, o_target=None, r_target=None,
                                                    term_target=None, mem=None, use_posterior=False, sample=False)
        mem_ = model.pack_mem(mem_)
        abstr_a_ = add_time_dim(model.abstract_action_model(a_))
        abstr_r_target_ = mem_['prim_r'].sum(dim=1, keepdim=True)
        abstr_term_target_ = mem_['prim_term'].max(dim=1, keepdim=True).values
        ctx_low_level_ = add_time_dim(model.fuse_state(prim_final_['s'], prim_final_['rnn_state']))
        mem_, abstr_current_ = model.rollout_abstract(a=abstr_a_, init_r=abstr_r, init_term=abstr_term, init_s=abstr_s,
                                                      init_rnn_state=abstr_rnn_state, ctx_low_level=ctx_low_level_,
                                                      r_target=abstr_r_target_, term_target=abstr_term_target_,
                                                      mem=None, use_posterior=True, sample=False)
        overlap = - torch.sum(torch.abs(abstr_s_next - abstr_current_['s']) ** 2, dim=1)
        mem_ = model.pack_mem(mem_)
        return overlap, None, mem_

    #R_winners = [torch.tensor(-10000, device=abstr_s.device)]
    #best_a = None
    #while R_winners[0] < -0.1:
    actions, act_dist, i_winners, R_winners, rollout_data = planner.plan(rollout_fn=_rollout_detailed_fn,
                                                              d_dist=env.action_space.n,
                                                              n_rollouts=n_rollouts,
                                                              n_plan_steps=model.abstract_step_size,
                                                              n_evolution_steps=n_evolution_steps,
                                                              winning_perc=winning_perc,
                                                              discount=1,
                                                              act_noise=act_noise)
    i_best = i_winners[0]
    best_a = torch.nn.functional.one_hot(actions[i_best], num_classes=model.d_action).float()

    return {'prim_a': best_a}


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
    assert tens.ndim <= 2
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