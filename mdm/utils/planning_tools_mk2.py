import gym
import torch
import numpy as np

from mdm.models.multiscale_model_mk2 import MultiscaleDynamicsModelMK2
from mdm.planning.cem_planner import CrossentropyPlanner
from mdm.utils.utils import normalize_obs, to_onehot


def init_abstr_s(model: MultiscaleDynamicsModelMK2,
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
    o_start = torch.tile(o_start, dims=(n_rollouts, 1))  # copy same starting observation along batch
    o_start = o_start.unsqueeze(1)  # add time dimension of 1

    def _rollout_init_fn(o_start_: torch.Tensor, a_: torch.Tensor):
        a_ = to_onehot(a_, n_classes=model.d_action)
        pred = model.rollout_primitive(o_start_, a_, sample=False)
        return pred['prim_r'].squeeze(), pred['prim_term'].squeeze(), pred

    actions, act_dist, i_winners, rollout_data = planner.plan(rollout_fn=_rollout_init_fn,
                                                              start_states=o_start,
                                                              d_dist=env.action_space.n,
                                                              n_plan_steps=model.abstract_step_size,
                                                              n_evolution_steps=n_evolution_steps,
                                                              winning_perc=winning_perc,
                                                              discount=discount,
                                                              act_noise=act_noise)
    i_best = i_winners[0]
    best_h = rollout_data['prim_h'][0][:, i_best].unsqueeze(1), rollout_data['prim_h'][1][:, i_best].unsqueeze(1)
    best_a = torch.nn.functional.one_hot(actions[i_best], num_classes=model.d_action).float()
    best_o = rollout_data['prim_o'][i_best]
    empty_abstr_s = torch.zeros(1, 1, model.d_abstract_state, device=model.device)
    abstr_a = model.abstract_action_model(best_a.unsqueeze(0)).unsqueeze(0)
    pred = model.rollout_abstract(empty_abstr_s, abstr_a, [best_h], sample=False)
    pred.update({'prim_o': best_o, 'prim_a': best_a})

    return {'abstr_s': pred['abstr_s'].squeeze(0),  # remove batch dimension
            'abstr_s_post': pred['abstr_s_post'],
            'prim_o': best_o,
            'prim_a': best_a}


def plan_abstract(model: MultiscaleDynamicsModelMK2,
                  planner: CrossentropyPlanner,
                  abstr_s_start: torch.Tensor,
                  abstr_s_start_dist: torch.distributions.Normal,
                  n_plan_steps: int,
                  n_rollouts: int,
                  n_evolution_steps: int,
                  winning_perc: float,
                  discount: float,
                  act_noise: float):

    def _rollout_abstract_fn(abstr_s_start_: torch.Tensor, abstr_a_: torch.Tensor):
        #abstr_a = torch.nn.functional.one_hot(abstr_a, num_classes=mdl.d_macro_action)
        predictions = model.rollout_abstract(abstr_s_start_, abstr_a_, sample=False)
        return predictions['abstr_r'].squeeze(), None, predictions

    # macro_s_next_post = extract_sub_distribution(macro_s_start_dist, 0)  # remove time dim
    # abstr_s_start_batch = macro_s_start_dist.sample(sample_shape=(pln_d_batch,))  # sample some possible start states
    abstr_s_start_batch = torch.tile(abstr_s_start, dims=(n_rollouts, 1))  # time dim required
    abstr_s_start_batch = abstr_s_start_batch.unsqueeze(1)  # add time dimension of 1
    abstr_a, a_dist, i_winners, rollout_data = planner.plan(rollout_fn=_rollout_abstract_fn,
                                                                          start_states=abstr_s_start_batch,
                                                                          d_dist=model.d_abstract_action,
                                                                          n_plan_steps=n_plan_steps,
                                                                          n_evolution_steps=n_evolution_steps,
                                                                          winning_perc=winning_perc,
                                                                          discount=discount,
                                                                          act_noise=act_noise)
    i_top_cand = i_winners[0]
    # best_abstr_a = torch.nn.functional.one_hot(abstr_a[i_top_cand], num_classes=mdl.d_macro_action).float()
    best_abstr_a = abstr_a[i_top_cand]
    # extract best performer for each time step
    # best_abstr_s = [extract_sub_distribution(d, i_top_cand) for d in rollout_data['macro_s']]
    # best_abstr_r = [extract_sub_distribution(d, i_top_cand) for d in rollout_data['macro_r']]
    best_abstr_s = rollout_data['abstr_s'][i_top_cand]
    best_abstr_r = rollout_data['abstr_r'][i_top_cand]
    best_macro_terms = rollout_data['abstr_term'][i_top_cand]

    return {'abstr_s': best_abstr_s,
            'abstr_a': best_abstr_a,
            'abstr_r': best_abstr_r,
            'abstr_term': best_macro_terms}


def plan_section(model: MultiscaleDynamicsModelMK2,
                 planner: CrossentropyPlanner,
                 env: gym.Env,
                 abstr_s: torch.Tensor,
                 abstr_s_next: torch.Tensor,
                 n_rollouts: int,
                 n_evolution_steps: int,
                 winning_perc: float,
                 discount: float,
                 act_noise: float):
    o_batch = torch.zeros(n_rollouts, 1, model.d_observation, device=model.device)
    abstr_s_batch = torch.tile(abstr_s, dims=(n_rollouts, 1))
    abstr_s_batch_next = torch.tile(abstr_s_next, dims=(n_rollouts, 1))

    # use closure to bind macro_x arguments inside the function to the above defined ones
    def _rollout_detailed_fn(o_start_: torch.Tensor, a_: torch.Tensor):
        a_ = to_onehot(a_, n_classes=model.d_action)
        pred_prim = model.rollout_primitive(o_start_, a_, abstr_s_batch)
        abstr_a_batch_post = model.abstract_action_model(a_)
        pred_abstr = model.rollout_abstract(abstr_s_batch.unsqueeze(1), abstr_a_batch_post.unsqueeze(1),
                                            [pred_prim['prim_h']], sample=False)
        #overlap = torch.distributions.kl_divergence(macro_s_next_post_dist,
        #                                            macro_s_next.expand((pln_d_batch, mdl.d_macro_state)))
        #overlap = - overlap.abs().sum(dim=1, keepdim=True)
        #overlap = pred_abstr['abstr_s_post'][0].log_prob(abstr_s_batch_next).sum(dim=1)
        overlap = -torch.sum(torch.abs(abstr_s_batch_next - pred_abstr['abstr_s'][:, 0]), dim=1)
        return overlap, None, pred_prim

    actions, act_dist, i_winners, rollout_data = planner.plan(rollout_fn=_rollout_detailed_fn,
                                                              start_states=o_batch,
                                                              d_dist=env.action_space.n,
                                                              n_plan_steps=model.abstract_step_size,
                                                              n_evolution_steps=n_evolution_steps,
                                                              winning_perc=winning_perc,
                                                              discount=discount,
                                                              act_noise=act_noise)
    i_best = i_winners[0]
    best_a = torch.nn.functional.one_hot(actions[i_best], num_classes=model.d_action).float()
    best_o = rollout_data['prim_o'][i_best]

    return {'prim_o': best_o, 'prim_a': best_a}
