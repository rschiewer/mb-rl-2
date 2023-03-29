from typing import Dict, Union, Optional, List

import torch

from mdm.models.dynamics_model import DynamicsModel
from mdm.models.hierarchical_rssm import HierarchicalRSSM
from mdm.planning.cem_planner import CrossentropyPlanner
from mdm.utils.torch_tools import extract_sub_distribution, unpack_rnn_state, pack_rnn_state


def select_winners(mem: Dict[str, Union[torch.Tensor, torch.distributions.Distribution]],
                   i_win: torch.Tensor,
                   n_envs: int,
                   n_rollouts: int):
    assert i_win.ndim == 1

    offsets = torch.tensor([i * n_rollouts for i in range(n_envs)], dtype=torch.long, device=i_win.device)
    i_win_offs = i_win + offsets

    for k, v in mem.items():
        for t, x_t in enumerate(v):
            if isinstance(x_t, torch.Tensor):
                v[t] = x_t[i_win_offs]
            elif isinstance(x_t, torch.distributions.Distribution):
                v[t] = extract_sub_distribution(x_t, i_win_offs)
            elif 'rnn_state' in k:
                v[t] = unpack_rnn_state(pack_rnn_state(x_t)[i_win_offs])


def plan(model: HierarchicalRSSM,
         level: int,
         planner: CrossentropyPlanner,
         n_plan_steps: int,
         n_rollouts: int,
         n_warmup: int,
         env_data: Optional[Dict[str, torch.Tensor]] = None,
         model_state: Optional[Dict[str, torch.Tensor]] = None,
         goal_data: Optional[Dict[str, torch.Tensor]] = None):
    assert env_data is not None or model_state is not None, 'need at least warmup data or a model state'

    # repeat the starting data n_rollouts times per environment, so use repeat_interleave instead of repeat
    if env_data is None:
        assert model_state is not None
        assert n_warmup == 0
        n_envs = model_state['z'].shape[0]
        o_start_batch = torch.zeros(0, 0, 0, device=model.device)  # zero time, batch and data dim
        a_start_batch = torch.zeros(0, n_envs * n_rollouts, model.rssm_modules[level].d_a, device=model.device)
        r_start_batch = torch.zeros(0, 0, 0, device=model.device)
        term_start_batch = torch.zeros(0, 0, 0, device=model.device)
        z_rep = torch.repeat_interleave(model_state['z'], n_rollouts, dim=0)  # no time dimension here, so batch_dim=0
        rnn_rep = unpack_rnn_state(torch.repeat_interleave(pack_rnn_state(model_state['rnn_state']), n_rollouts, dim=0))
        model_state = {'z': z_rep, 'rnn_state': rnn_rep}
    else:
        assert model_state is None
        n_envs = env_data['o'].shape[1]
        o_start_batch = torch.repeat_interleave(env_data['o'], n_rollouts, dim=1)
        a_start_batch = torch.repeat_interleave(env_data['a'], n_rollouts, dim=1)
        r_start_batch = torch.repeat_interleave(env_data['r'], n_rollouts, dim=1)
        term_start_batch = torch.repeat_interleave(env_data['terminal'], n_rollouts, dim=1)

    if goal_data is None:
        def _calc_criterion(_mem):
            _criterion = torch.stack(_mem['r']).squeeze(-1).swapaxes(0, 1)
            #_exploration_bonus = torch.stack([x.scale for x in _mem['o_dist']]).mean(dim=2).swapaxes(0, 1)
            #_criterion += _exploration_bonus
            # _criterion -= torch.stack([d.scale / 2 for d in _mem['r_dist']]).squeeze(-1).swapaxes(0, 1)
            _discount = torch.stack(_mem['terminal']).squeeze(-1).swapaxes(0, 1)
            # _discount = torch.where(_discount > 0.75, 1.0, 0.0)
            _mem['valid_steps'] = list(torch.ones_like(_criterion).squeeze(-1).unbind(1))
            return _criterion, _discount
    else:
        # TODO: goal_data should never have a time dimension, so this should be simpler
        goal_data = {
            k: torch.repeat_interleave(v, n_rollouts, dim=0)
            if v.ndim <= 2  # no time dim, so first is batch
            else torch.repeat_interleave(v, n_rollouts, dim=1)
            for k, v in goal_data.items()
        }

        def _calc_criterion(_mem):
            _criterion = torch.zeros(n_envs * n_rollouts, 1, device=model.device)
            _discount = torch.ones_like(_criterion)
            # hack: support planning lengths shorter than n_plan_steps
            #_criterion = _criterion.reshape(n_envs, n_rollouts, 1)
            first_third = n_rollouts // 3
            second_third = first_third * 2
            third_third = n_rollouts
            for k, v in goal_data.items():
                #_target = goal_data[k].reshape(n_envs, n_rollouts, goal_data[k].shape[-1])
                #_first_goal = _mem[k][-3].reshape(n_envs, n_rollouts, *_mem[k][-3].shape[1:])
                #_second_goal = _mem[k][-2].reshape(n_envs, n_rollouts, *_mem[k][-2].shape[1:])
                #_third_goal = _mem[k][-1].reshape(n_envs, n_rollouts, *_mem[k][-1].shape[1:])
                #_criterion[:, 0:first_third] -= torch.mean((_first_goal[:, 0:first_third] - _target[:, 0:first_third]) ** 2, dim=-1, keepdim=True)
                #_criterion[:, first_third:second_third] -= torch.mean((_second_goal[:, first_third:second_third] - _target[:, first_third:second_third]) ** 2, dim=-1, keepdim=True)
                #_criterion[:, second_third:third_third] -= torch.mean((_third_goal[:, second_third:third_third] - _target[:, second_third:third_third]) ** 2, dim=-1, keepdim=True)
                _target = goal_data[k].reshape(n_envs * n_rollouts, *goal_data[k].shape[1:])
                _criterion -= torch.mean((_mem[k][-1] - goal_data[k]) ** 2, dim=-1, keepdim=True)
            #_a_counted = torch.zeros(n_envs, n_rollouts, n_plan_steps + 1, device=model.device)
            #_a_counted[:, 0:first_third, :] = 1
            #_a_counted[:, first_third:second_third, :-1] = 1
            #_a_counted[:, second_third:third_third, :-2] = 1
            #_a_counted = _a_counted.reshape(n_envs * n_rollouts, n_plan_steps + 1)
            #_a_counted = torch.ones(n_envs * n_rollouts, n_plan_steps, device=model.device)
            #_mem['valid_steps'] = list(_a_counted.unbind(1))

            _criterion = _criterion.reshape(n_envs * n_rollouts, 1)
            return _criterion, _discount

    def _rollout_fn(_a: torch.Tensor):
        # fold 'env' dimension into batch dimension for rollout
        _a = _a.reshape(_a.shape[0] * _a.shape[1], *_a.shape[2:])
        _a = _a.swapaxes(0, 1)  # swap batch and time dim since model is time-first but planner is batch-first
        _a = torch.cat([a_start_batch, _a], dim=0)
        _mem, _ = model(a=_a, o=o_start_batch, r=r_start_batch, terminal=term_start_batch,
                        n_warmup=n_warmup, level=level, start_state=model_state, sample_state=True, sample_output=True)
        _criterion, _discount = _calc_criterion(_mem)

        _criterion = _criterion.reshape(n_envs, n_rollouts, _criterion.shape[-1])
        _discount = _discount.reshape(n_envs, n_rollouts, _discount.shape[-1])
        return _criterion, _discount, _mem

    # hack: support planning lengths one step longer than n_plan_steps
    a, a_dist, i_win, R_win, data = planner.plan(rollout_fn=_rollout_fn, n_rollouts=n_rollouts,
                                                 n_plan_steps=n_plan_steps, n_envs=n_envs)

    # select winner batch item per per memory timestep
    select_winners(data, i_win[:, 0], n_envs, n_rollouts)
    # CAUTION: the actions from planner are without the already performed warmup acitons!
    a_win = planner.get_winner_actions(a, a_dist, i_win, resample=False)
    return a_win, R_win[:, 0], data


def run_model(model: DynamicsModel,
              level: int,
              env_data: Dict[str, torch.Tensor],
              discount: float = 1.0):
    mem, _ = model(**env_data, n_warmup=-1, level=level, sample_state=True, sample_output=True)
    step_rewards = torch.stack(mem['r'])
    disc_mat = torch.cumprod(torch.full_like(step_rewards, discount), dim=0)
    disc_mat = torch.roll(disc_mat, 1, dims=0)
    disc_mat[0, :, :] = 1
    R_win = torch.sum(step_rewards * disc_mat, dim=0).squeeze()
    a_win = env_data['a']
    #mem['valid_steps'] = list(torch.ones_like(step_rewards).squeeze(-1).unbind(0))
    return a_win, R_win, mem


def plan_hierarchical(model: HierarchicalRSSM,
                      env: Dict[str, torch.Tensor],
                      planners: List[CrossentropyPlanner],
                      n_plan_steps: int,
                      n_rollouts: List[int],
                      n_warmup: List[int]):
    n_groundtruth_steps, n_envs = env['o'].shape[:2]
    del env['truncated']
    del env['mask']

    min_init_steps = []
    for n_wu, stride in zip(n_warmup, model.strides):
        assert n_wu >= 1, 'every level needs at least one warmup step'
        min_init_steps.append(n_wu * stride)
    min_init_steps.append(n_warmup[-1])  # last hierarchy doesn't need to satisfy any requirements of above hierarchy
    assert n_groundtruth_steps >= min_init_steps.pop(0), 'not enough groundtruth data for lowest level'

    # min_init_steps now contains for every level the information how much model steps have to be taken in order to
    # satisfy the timestep requirements of the above hierarchy.

    # climb hierarchy and collect warmup data
    init_data = []
    inp_lvl = env
    for level in range(model.levels):
        filters = model.upwards_filters[level]
        filtered_inp_level = {k: filters[k](inp_lvl[k]) for k in inp_lvl}
        n_steps_available = filtered_inp_level['o'].shape[0]

        plan_steps_lvl = min_init_steps[level] - n_steps_available
        if plan_steps_lvl > 0:
            a_win, return_win, data = plan(model=model, level=level, planner=planners[level],
                                           n_plan_steps=plan_steps_lvl,
                                           n_rollouts=n_rollouts[level], n_warmup=n_warmup[level],
                                           env_data=filtered_inp_level)
        else:
            a_win, return_win, data = run_model(model=model, level=level, env_data=filtered_inp_level,
                                                discount=planners[level].discount)
        link = model.links[level]
        inp_lvl = {'o': torch.stack(data[link]), 'a': filtered_inp_level['a'], 'r': torch.stack(data['r']),
                   'terminal': torch.stack(data['terminal'])}
        # data['a'] = list(filtered_inp_level['a'].unbind(0))  # record actions as well; make list to match other buffers
        init_data.append(data)

    # plan top level to maximize reward
    i_top_lvl = model.levels - 1
    top_lvl_input_data = {k: torch.stack(init_data[i_top_lvl][k]) for k in ('o', 'a', 'r', 'terminal')}
    a_win_top, return_win_top, top_lvl_data = plan(model=model, level=i_top_lvl, planner=planners[i_top_lvl],
                                                   n_plan_steps=n_plan_steps, n_rollouts=n_rollouts[i_top_lvl],
                                                   n_warmup=n_warmup[i_top_lvl], env_data=top_lvl_input_data)

    # now plan top to bottom levels to maximize target similarity
    planning_data = [{} for _ in range(model.levels)]
    best_actions = [[] for _ in range(model.levels)]
    best_returns = [None for _ in range(model.levels)]

    # top level for all memories is already done, so fill it in
    planning_data[-1] = top_lvl_data
    best_actions[-1] = a_win_top  # 0 is env/batch dimension, 1 is time dimension, 2 is action dimension
    best_returns[-1] = return_win_top

    for level in reversed(range(model.levels - 1)):
        above_level = level + 1
        i_chunk_start = len(init_data[above_level]['o'])  # omit the warm up steps
        i_chunk_end = len(planning_data[above_level]['o'])
        n_plan_steps_level = model.strides[above_level]  # plan chunk by chunk
        a_win_level = []
        return_win_level = []

        # add init data portion of the trajectories
        for k, v in init_data[level].items():
            tmp = planning_data[level].get(k, [])
            tmp.extend(v)
            planning_data[level][k] = tmp

        # iterate through the chunks and do planning
        # note: just take the last model state to be independent of actual chunk sizes
        model_state = {'rnn_state': init_data[level]['rnn_state'][-1], 'z': init_data[level]['z'][-1]}
        for i_chunk in range(i_chunk_start, i_chunk_end):
            goal_data = {model.links[level]: planning_data[above_level]['o'][i_chunk]}  # TODO: check validity of goals
            a_win, return_win, data = plan(model=model, level=level, planner=planners[level],
                                           n_plan_steps=n_plan_steps_level, n_rollouts=n_rollouts[level],
                                           n_warmup=0, model_state=model_state,
                                           goal_data=goal_data)
            model_state = {'rnn_state': data['rnn_state'][-1], 'z': data['z'][-1]}
            #last_valid = torch.stack(data['valid_steps']).sum(dim=0).to(torch.long) - 1
            #rnn_state = torch.stack([pack_rnn_state(x) for x in data['rnn_state']])
            #rnn_state = rnn_state.swapaxes(0, 1)[torch.arange(n_envs), last_valid]
            #z = torch.stack(data['z']).swapaxes(0, 1)[torch.arange(n_envs), last_valid]
            #model_state['rnn_state'] = unpack_rnn_state(rnn_state)
            #model_state['z'] = z

            # bookkeeping
            a_win_level.extend(list(a_win.unbind(1)))  # 0 is env dimension, 1 is time dimension, 2 is action dimension
            return_win_level.append(return_win)
            for k, v in data.items():
                tmp = planning_data[level].get(k, [])
                tmp.extend(v)
                planning_data[level][k] = tmp
        best_actions[level] = torch.stack(a_win_level, dim=1)
        best_returns[level] = torch.stack(return_win_level).sum(dim=0)

    # filter only valid time steps, super inefficient but I don't know any better solution right now
    #valid_data = [{} for _ in range(model.levels)]
    #for i_level, raw_level_data in enumerate(planning_data):
    #    valid = torch.stack(raw_level_data['valid_steps']).squeeze().to(torch.bool)
    #    longest = valid.sum(dim=0).max()
    #    for k, v in raw_level_data.items():
    #        if isinstance(v[0], torch.Tensor):
    #            v = torch.stack(v)
    #            valid_data[i_level][k] = torch.zeros(longest, *v.shape[1:])
    #            for i_env in range(n_envs):
    #                valid_env_data = v[:, i_env][valid[:, i_env].nonzero(as_tuple=True)]
    #                valid_data[i_level][k][:len(valid_env_data), i_env] = valid_env_data

    best_lvl_0_actions = best_actions[0]
    #best_lvl_0_actions = valid_data[0]['a'][n_groundtruth_steps:].swapaxes(0, 1)
    return best_lvl_0_actions, best_returns, planning_data
