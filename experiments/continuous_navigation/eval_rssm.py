import argparse

import torch
import gym
import numpy as np

from mdm.planning.planning_tools import plan_hierarchical
from mdm.utils.utils import here, load_yaml, random_walk_success_rate
# from mdm.utils.planning_tools_mk2 import *
from mdm.planning.cem_planner import CrossentropyPlanner, DistributionType
from mdm.logging.neptune_logger import NeptuneLogger
from mdm.logging.not_logger import NotLogger
from mdm.logging.logger import Scope
from mdm.utils.torch_tools import to_tensors
from mdm.utils.utils import prepare_data
from mdm.utils.gym_wrappers import CacheLastStepEnv
from mdm.training.gym_driver import collect_data, CacheLastStepVecEnv
from mdm.policies.random_policy import RandomPolicy
from mdm.policies.predefined_policy import PredefinedPolicy

"""
def select_batch_items(mem: Dict[str, Union[torch.Tensor, torch.distributions.Distribution]],
                       i: TensorIndex,
                       keepdim: bool = False):
    if keepdim and type(i) is not slice:
        if isinstance(i, torch.Tensor):
            i = i.detach().cpu().numpy().item()
        i = slice(i, i + 1)

    ret = {}
    for name, val in mem.items():
        ret[name] = []
        for timestep in val:
            if timestep is None:
                ret[name].append(None)
            elif isinstance(timestep, torch.Tensor):
                ret[name].append(timestep[i])
            elif isinstance(timestep, torch.distributions.Distribution):
                ret[name].append(extract_sub_distribution(timestep, i))  # keepdim is handled by calling function
            elif 'rnn_state' in name:
                ret[name].append(unpack_rnn_state(pack_rnn_state(timestep)[i]))
            else:
                raise ValueError(f'Unknown memory content for key {name}: {timestep}')
    return ret


def plan_prim_with_warmup(model: DynamicsModel,
                          planner_prim: CrossentropyPlanner,
                          env_data: Dict[str, torch.Tensor],
                          n_plan_steps: int,
                          n_rollouts: int,
                          n_warmup: int):
    o_start_batch = env_data['o'].repeat(1, n_rollouts, 1)
    a_start_batch = env_data['a'].repeat(1, n_rollouts, 1)
    r_start_batch = env_data['r'].repeat(1, n_rollouts, 1)
    term_start_batch = env_data['terminal'].repeat(1, n_rollouts, 1)

    def _rollout_fn(_a: torch.Tensor):
        _a = _a[0]  # remove redundant env dimension
        _a = _a.swapaxes(0, 1)
        _a = torch.cat([a_start_batch, _a], dim=0)
        _mem, _ = model(a=_a, o=o_start_batch, r=r_start_batch, terminal=term_start_batch,
                        n_warmup=n_warmup, sample_state=True, sample_output=True)
        _criterion = torch.stack(_mem['r']).squeeze(-1).swapaxes(0, 1)
        #_criterion -= torch.stack([d.scale / 2 for d in _mem['r_dist']]).squeeze(-1).swapaxes(0, 1)
        _discount = torch.stack(_mem['terminal']).squeeze(-1).swapaxes(0, 1)

        _criterion = _criterion.unsqueeze(0)  # add "env" dimension
        _discount = _discount.unsqueeze(0)
        return _criterion, _discount, _mem

    a, a_dist, i_win, R_win, data = planner_prim.plan(rollout_fn=_rollout_fn, n_rollouts=n_rollouts,
                                                      n_plan_steps=n_plan_steps)

    # select winner batch item per per memory timestep
    data = select_batch_items(data, i_win[0, 0], keepdim=True)
    # CAUTION: the actions from planner are without the already performed warmup acitons!
    a_win = planner_prim.get_winner_actions(a, a_dist, i_win, resample=False)
    a_win = a_win[0]  # remove redundant "env" dimension
    return a_win, data
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Provide neptune run_id for loading the correct model')
    parser.add_argument('id', type=str, nargs=1)
    parser.add_argument('-render', action='store_true')
    parser.add_argument('-log', action='store_true')
    args = parser.parse_args()

    cfg = load_yaml(here() / 'cfg_rssm_train.yaml')
    neptune_cfg = load_yaml(here() / cfg['neptune_cfg'])
    planning_cfg = load_yaml(here() / 'cfg_rssm_plan.yaml')

    if len(args.id) == 0:
        model_path = f'{cfg["final_model_path"]}-1.ptmdl'
    else:
        model_path = f'{cfg["final_model_path"]}_{args.id[0]}.ptmdl'

    map_version = 'VeryEasy'
    env = gym.make(f'gym_nav2d:nav2d{map_version}-v0')
    env = CacheLastStepEnv(env)
    model = torch.load(here() / model_path).to('cuda')
    model.eval()  # deactivate dropout in RNN

    # new planning
    def make_env_fn():
        return gym.make(f'gym_nav2d:nav2d{map_version}-v0')
    n_eval_envs = cfg['eval']['eval_envs']
    eval_env = gym.vector.AsyncVectorEnv([make_env_fn] * n_eval_envs)
    eval_env = CacheLastStepVecEnv(eval_env)
    planners = [CrossentropyPlanner(DistributionType.NORMAL, d_dist=m.d_a, device=model.device, debug_env=None,
                                    **pln_cfg, a_min=-1.0, a_max=1.0)
                for pln_cfg, m in zip(planning_cfg['planners'], model.rssm_modules)]

    if args.log:
        logger = NeptuneLogger(neptune_cfg['PROJECT_NAME'], api_token=neptune_cfg['NEPTUNE_API_TOKEN'],
                               run_id=args.id[0])
    else:
        logger = NotLogger()
    logger.start_session()

    logger.log(planning_cfg, Scope.HYPERPARAMETERS() / 'plan/flat')

    l0_steps = planning_cfg['n_plan_steps'] * np.prod(model.strides)
    success, r_ep, l_ep = random_walk_success_rate(env, l0_steps, planning_cfg['n_rollouts'][0])
    print(f'Random walk statistics with planning parameters:')
    print(f'Steps taken in environment: {l0_steps}')
    print(f'Expected initial success rate: {success}')
    print(f'Expected initial average return: {r_ep.mean()}')
    print(f'Expected initial average episode length: {l_ep.mean()}')

    eval_env.reset()
    warmup_data_trajectories = collect_data(eval_env, planning_cfg['n_warmup'][0], RandomPolicy(eval_env))
    warmup_data = prepare_data(to_tensors(warmup_data_trajectories, model.device))
    a_win, R_win, _ = plan_hierarchical(model, warmup_data, planners,
                                        planning_cfg['n_plan_steps'],
                                        planning_cfg['n_rollouts'],
                                        planning_cfg['n_warmup'])
    collect_policy = PredefinedPolicy(eval_env, a_win.detach().cpu().numpy().swapaxes(0, 1))
    collected_data_trajectories = collect_data(eval_env, collect_policy.max_timestep, collect_policy)
    mem = [{k: np.concatenate([wu[k], col[k]]) for k in wu}
           for wu, col in zip(warmup_data_trajectories, collected_data_trajectories)]

    ep_len = 0
    success = 0
    avg_return = 0
    for ep in mem:
        avg_return += np.stack(ep['r']).sum()
        if ep['terminal'].sum() == 1:
            ep_len += len(ep['terminal'])
            success += 1
        elif ep['terminal'].sum() > 1:
            raise RuntimeError('More than one terminal flag, there is something wrong!')
        else:
            ep_len += len(ep['terminal'])
    success /= n_eval_envs
    ep_len /= n_eval_envs
    avg_return /= n_eval_envs

    print(f'evaluation runs: {n_eval_envs}')
    print(f'success: {success}')
    print(f'average episode length: {ep_len}')
    print(f'average return: {avg_return}')

    logger.log({'ep_len': ep_len, 'success': success, 'avg_return': avg_return},
               Scope.TEST() / 'planning')
    logger.stop_session()

    """
    succeeded = 0
    n_steps = []
    for i_ep in tqdm(range(planning_cfg['n_episodes'])):
        planner_prim = CrossentropyPlanner(DistributionType.NORMAL, d_dist=2,
                                           device=mdl.device, debug_env=None, **planning_cfg['pln_prim'],
                                           a_min=-1.0, a_max=1.0)

        env.reset()
        init_data = collect_data(env, planning_cfg['n_warmup_prim'], RandomPolicy(env))
        init_data = prepare_data(**to_tensors(init_data, mdl.device))
        a, i_win = plan_prim_with_warmup(mdl, planner_prim, init_data, planning_cfg['n_plan_steps_prim'],
                                         planning_cfg['n_rollouts'], planning_cfg['n_warmup_prim'])

        action_iter = iter(a.detach().cpu().numpy())

        i_step = planning_cfg['n_warmup_prim']
        done = False
        while not done:
            if args.render:
                env.render()
                time.sleep(0.1)
            i_step += 1
            try:
                a = next(action_iter)
                s_, r, terminal, truncated, info = env.step(a)

                if terminal:
                    succeeded += 1

                #agent_pos = s_[:2]
                #goal_pos = s_[2:4]
                #plt.clf()
                #plt.scatter(agent_pos[0], agent_pos[1], c='red')
                #plt.scatter(goal_pos[0], goal_pos[1], c='green')
                #plt.xlim((-1, 1))
                #plt.ylim((-1, 1))
                #plt.ion()
                #plt.pause(0.001)
                #plt.show()

                done = terminal or truncated
            except StopIteration:
                break

        n_steps.append(i_step)

    # final debug output
    print(succeeded / planning_cfg['n_episodes'])
    print(n_steps, f' mean: {np.mean(n_steps)}')

    logger.log({'success_rate': succeeded / planning_cfg['n_episodes'],
                'n_steps': n_steps},
               Scope.TEST() / 'plan/flat')
    logger.stop_session()
    """
