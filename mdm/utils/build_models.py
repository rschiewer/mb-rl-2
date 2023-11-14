import gymnasium
import torch

from mdm.models.building_blocks import RSSMCell
from mdm.policies.actor_critic_agent import ActorCriticAgent


def build_model_opt(model: torch.nn.Module, cfg: dict):
    optim_type = cfg['optim'].pop('type')
    params = model.parameters()

    if optim_type == 'adam':
        opt_model = torch.optim.Adam(params, **cfg['optim'])
    elif optim_type == 'adamW':
        opt_model = torch.optim.AdamW(params, **cfg['optim'])
    elif optim_type == 'sgd':
        opt_model = torch.optim.SGD(params, **cfg['optim'])
    else:
        raise ValueError(f'Unknown optimizer type: {optim_type}')

    return opt_model


def build_rssms(cfg: dict):
    for i_module, module_args in enumerate(cfg['mdm']['rssm_modules']):
        for k, v in module_args.items():  # generate encoders and decoder objects for current RSSM
            if isinstance(v, dict) and 'class' in v:
                cls_name = v.pop('class')
                instance = globals()[cls_name](**v)
                module_args[k] = instance
        cfg['mdm']['rssm_modules'][i_module] = RSSMCell(**module_args)  # generate RSSM
    for i_filter, filter_args in enumerate(cfg['mdm']['upwards_filters']):  # generate filter objects
        for k, v in filter_args.items():
            cls_name = v.pop('class')
            instance = globals()[cls_name](**v)
            filter_args[k] = instance
    return cfg


def build_agents(cfg: dict,
                 env: gym.Env,
                 device: torch.device):
    if not isinstance(env.action_space, gym.spaces.Box):
        raise ValueError('Only enviornments with continuous action space are supported')

    def gen_agent_fn(level: int, goal_seeking: bool, cfg) -> (
            ActorCriticAgent, torch.optim.Optimizer, torch.optim.Optimizer):
        agent = ActorCriticAgent(level=level, observation_key=None, goal_seeking=goal_seeking, **cfg)
        agent = agent.to(device)
        actor_optimizer = torch.optim.Adam(agent.actor_net.parameters(), lr=cfg['lr_actor'])
        critic_optimizer = torch.optim.Adam(agent.critic_net.parameters(), lr=cfg['lr_critic'])
        return agent, actor_optimizer, critic_optimizer

    r_max_agents = []
    goal_seeking_agents = []
    for agent_lvl in range(len(cfg['mdm']['rssm_modules'])):
        cfg_r_max = cfg['agents']['r_max'][agent_lvl]
        r_max_agents.append(gen_agent_fn(agent_lvl, False, cfg_r_max))

        if agent_lvl < len(cfg['mdm']['rssm_modules']) - 1:
            cfg_goal_seeking = cfg['agents']['goal_seeking'][agent_lvl]
            goal_seeking_agents.append(gen_agent_fn(agent_lvl, True, cfg_goal_seeking))
    # goal_seeking_agents.append(None)  # no homing agent needed on last level

    return r_max_agents, goal_seeking_agents
