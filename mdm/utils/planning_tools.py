import warnings

import gym
import torch

from mdm.models.multiscale_model import MultiscaleDynamicsModel
from mdm.planning.cem_planner import CrossentropyPlanner
from mdm.utils.utils import normalize_obs, to_onehot




def init_macro_s(model: MultiscaleDynamicsModel,
                 planner: CrossentropyPlanner,
                 env: gym.Env,
                 s_start: torch.Tensor,
                 n_rollouts: int,
                 n_evolution_steps: int,
                 winning_perc: float,
                 discount: float,
                 act_noise: float):
    warnings.warn('This function is no longer maintained and shouldn\'t be used', DeprecationWarning)
    # preprocess starting state
    s_start = normalize_obs(s_start, env)
    s_start = torch.tile(s_start, dims=(n_rollouts, 1))  # copy same starting observation along batch
    s_start = s_start.unsqueeze(1)  # add time dimension of 1

    def _rollout_init_fn(start_states: torch.Tensor, actions: torch.Tensor):
        actions = to_onehot(actions, n_classes=model.d_action)
        predictions_ss = model.rollout_single_step(start_states, actions)
        return predictions_ss['r'].squeeze(), predictions_ss['term'].squeeze(), predictions_ss

    actions, act_dist, i_winners, rollout_data = planner.plan(rollout_fn=_rollout_init_fn,
                                                              init_data=s_start,
                                                              d_dist=env.action_space.n,
                                                              n_plan_steps=model.macro_step_size,
                                                              n_evolution_steps=n_evolution_steps,
                                                              winning_perc=winning_perc,
                                                              discount=discount,
                                                              act_noise=act_noise)
    i_best = i_winners[0]
    best_h = rollout_data['h'][0][:, i_best].unsqueeze(1), rollout_data['h'][1][:, i_best].unsqueeze(1)
    best_a = torch.nn.functional.one_hot(actions[i_best], num_classes=model.d_action).float()
    best_s = rollout_data['s'][i_best]
    zero_macro_s = torch.zeros(1, model.d_macro_state, device=model.device)
    zero_macro_a = torch.zeros(1, model.d_macro_action, device=model.device)
    pred = model.macro_next_posterior(zero_macro_s, zero_macro_a, best_h)
    pred.update({'ss': best_s, 'as': best_a})

    return pred


def plan_abstract(model: MultiscaleDynamicsModel,
                  planner: CrossentropyPlanner,
                  macro_s_start: torch.Tensor,
                  macro_s_start_dist: torch.distributions.Normal,
                  n_plan_steps: int,
                  n_rollouts: int,
                  n_evolution_steps: int,
                  winning_perc: float,
                  discount: float,
                  act_noise: float):
    warnings.warn('This function is no longer maintained and shouldn\'t be used', DeprecationWarning)
    def _rollout_abstract_fn(macro_start_state: torch.Tensor, macro_actions: torch.Tensor):
        #macro_actions = torch.nn.functional.one_hot(macro_actions, num_classes=mdl.d_macro_action)
        predictions = model.rollout_abstract(macro_start_state, macro_actions)
        return predictions['macro_r'].squeeze(), None, predictions

    # macro_s_next_post = extract_sub_distribution(macro_s_start_dist, 0)  # remove time dim
    # macro_s_batch = macro_s_start_dist.sample(sample_shape=(pln_d_batch,))  # sample some possible start states
    macro_s_batch = torch.tile(macro_s_start, dims=(n_rollouts, 1))  # time dim required
    macro_s_batch = macro_s_batch.unsqueeze(1)  # add time dimension of 1
    macro_actions, act_dist, i_winners, rollout_data = planner.plan(rollout_fn=_rollout_abstract_fn,
                                                                    init_data=macro_s_batch,
                                                                    d_dist=model.d_macro_action,
                                                                    n_plan_steps=n_plan_steps,
                                                                    n_evolution_steps=n_evolution_steps,
                                                                    winning_perc=winning_perc,
                                                                    discount=discount,
                                                                    act_noise=act_noise)
    i_top_cand = i_winners[0]
    # best_macro_as = torch.nn.functional.one_hot(macro_actions[i_top_cand], num_classes=mdl.d_macro_action).float()
    best_macro_as = macro_actions[i_top_cand]
    # extract best performer for each time step
    # best_macro_ss = [extract_sub_distribution(d, i_top_cand) for d in rollout_data['macro_s']]
    # best_macro_rs = [extract_sub_distribution(d, i_top_cand) for d in rollout_data['macro_r']]
    best_macro_ss = rollout_data['macro_s'][i_top_cand]
    best_macro_rs = rollout_data['macro_r'][i_top_cand]
    best_macro_terms = rollout_data['macro_term'][i_top_cand]

    return {'macro_ss': best_macro_ss,
            'macro_as': best_macro_as,
            'macro_rs': best_macro_rs,
            'macro_terms': best_macro_terms}


def plan_section(model: MultiscaleDynamicsModel,
                 planner: CrossentropyPlanner,
                 env: gym.Env,
                 macro_s: torch.Tensor,
                 macro_a: torch.Tensor,
                 macro_s_next: torch.Tensor,
                 n_rollouts: int,
                 n_evolution_steps: int,
                 winning_perc: float,
                 discount: float,
                 act_noise: float):
    warnings.warn('This function is no longer maintained and shouldn\'t be used', DeprecationWarning)

    s_batch = torch.zeros(n_rollouts, 1, model.d_state, device=model.device)
    macro_s_batch = torch.tile(macro_s, dims=(n_rollouts, 1))
    macro_a_batch = torch.tile(macro_a, dims=(n_rollouts, 1))
    macro_s_next_batch = torch.tile(macro_s_next, dims=(n_rollouts, 1))
    macro_a_batch_post = torch.zeros_like(macro_a_batch)

    # use closure to bind macro_x arguments inside the function to the above defined ones
    def _rollout_detailed_fn(start_states: torch.Tensor, actions: torch.Tensor):
        actions = to_onehot(actions, n_classes=model.d_action)
        pred_prim = model.rollout_single_step(start_states, actions, macro_s_batch, macro_a_batch)
        pred_abstr = model.macro_next_posterior(macro_s_batch, macro_a_batch_post, pred_prim['h'])
        #overlap = torch.distributions.kl_divergence(macro_s_next_post_dist,
        #                                            macro_s_next.expand((pln_d_batch, mdl.d_macro_state)))
        #overlap = - overlap.abs().sum(dim=1, keepdim=True)
        overlap = pred_abstr['macro_s_next_post'].log_prob(macro_s_next_batch).sum(dim=1)
        #overlap = -torch.sum(torch.abs(macro_s_next_post - macro_s_next_batch), dim=1)
        return overlap, None, pred_prim

    actions, act_dist, i_winners, rollout_data = planner.plan(rollout_fn=_rollout_detailed_fn,
                                                              init_data=s_batch,
                                                              d_dist=env.action_space.n,
                                                              n_plan_steps=model.macro_step_size,
                                                              n_evolution_steps=n_evolution_steps,
                                                              winning_perc=winning_perc,
                                                              discount=discount,
                                                              act_noise=act_noise)
    i_best = i_winners[0]
    best_a = torch.nn.functional.one_hot(actions[i_best], num_classes=model.d_action).float()
    best_s = rollout_data['s'][i_best]

    return {'ss': best_s, 'as': best_a}
