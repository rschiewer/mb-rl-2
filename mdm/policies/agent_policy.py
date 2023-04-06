from typing import Dict, Union

import torch
import numpy as np

from mdm.policies.actor_critic_agent import ActorCriticAgent
from mdm.policies.policy import Policy
from mdm.models.hierarchical_rssm import HierarchicalRSSM
from mdm.utils.torch_tools import to_tensors
from mdm.utils.utils import prepare_data
from mdm.utils.gym_wrappers import CacheLastStepEnv, CacheLastStepVecEnv


class AgentPolicy(Policy):

    def __init__(self,
                 agent: ActorCriticAgent):
        self.agent = agent

    def __call__(self, env):
        last_o = torch.from_numpy(env.last_o).to(self.agent.device)
        a_dist, a, v = self.agent(last_o)
        return a.detach().cpu().numpy()


class LatentAgentPolicy(Policy):

    def __init__(self,
                 agent: ActorCriticAgent,
                 model: HierarchicalRSSM,
                 init_data: Dict[str, torch.Tensor] = None,
                 use_ema_modules: bool = False):
        assert np.prod(agent.d_o) == model.rssm_modules[agent.level].d_z

        self.agent = agent
        self.model = model
        if init_data is not None:
            mem, env_state = model(o=init_data['o'], a=init_data['a'], r=init_data['r'], terminal=init_data['terminal'],
                                   level=agent.level, use_ema_modules=use_ema_modules)
            self._current_env_state = env_state
        else:
            self._current_env_state = None
        self._use_ema_modules = use_ema_modules

    def __call__(self, env: Union[CacheLastStepEnv, CacheLastStepVecEnv]):
        device = self.agent.device

        if env.current_step == 0:
            if isinstance(env, CacheLastStepVecEnv):
                d_batch = env.last_o.shape[0]
            else:
                d_batch = 1
            self._current_env_state = self.model.rssm_modules[self.agent.level].init_state(d_batch, device)

        # get data and add time dim
        o = torch.from_numpy(env.last_o).unsqueeze(0).to(device=device, dtype=torch.float32)
        a = torch.from_numpy(env.last_a).unsqueeze(0).to(device=device, dtype=torch.float32)
        r = torch.from_numpy(env.last_r).unsqueeze(0).to(device=device, dtype=torch.float32)
        terminal = torch.from_numpy(env.last_term).unsqueeze(0).to(device=device, dtype=torch.float32)
        truncated = torch.from_numpy(env.last_trunc).unsqueeze(0).to(device=device, dtype=torch.float32)
        if isinstance(env, CacheLastStepEnv):  # add batch dim if unbatched env
            o, a, r, terminal, truncated = [x.unsqueeze(1) for x in (o, a, r, terminal, truncated)]

        env_data = {'o': o, 'a': a, 'r': r, 'terminal': terminal, 'truncated': truncated, 'mask': torch.empty_like(r)}
        env_data = prepare_data(env_data)
        mem, self._current_env_state = self.model(o=env_data['o'], a=env_data['a'], r=env_data['r'],
                                                  terminal=env_data['terminal'], start_state=self._current_env_state,
                                                  level=self.agent.level, use_ema_modules=self._use_ema_modules)
        a_dist, a, v = self.agent(mem[self.agent.observation_key][-1])
        return a.detach().cpu().numpy()
