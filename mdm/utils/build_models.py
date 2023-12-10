import gymnasium as gym
import torch

from mdm.models.building_blocks import RSSMCell
from mdm.policies.actor_critic_agent import ActorCriticAgent
from mdm.models.building_blocks import *


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
        # make one optimizer for all parameters not actor or critic net related
        exclude_params = list(agent.actor_net.parameters()) + list(agent.critic_net.parameters())
        remaining_params = []
        for p in agent.parameters():  # parameters need to be explicitly compared with 'is'
            found = False
            for p_other in exclude_params:
                if torch.equal(p, p_other):
                    found = True
                    break
            if not found:
                remaining_params.append(p)
        other_params_optimizer = torch.optim.Adam(remaining_params, lr=cfg['lr_other'])
        return agent, {'actor_optimizer': actor_optimizer, 'critic_optimizer': critic_optimizer,
                       'other_optimizer': other_params_optimizer}

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


def cfg_infer_missing_values(cfg: dict,
                             env: gym.Env):
    # infer missing config values for RSSMs
    for i_module, module_args in enumerate(cfg['mdm']['rssm_modules']):
        # calculate z sample size
        if module_args['d_s_embedding'] is None:
            if module_args['latent_dist'] == 'normal':
                d_state = module_args['d_z'] + module_args['d_h']
            elif module_args['latent_dist'] == 'categorical':
                d_state = module_args['d_z'] * module_args['n_latent_categories'] + module_args['d_h']
            else:
                raise ValueError(f'Unknown latent distribution')
            module_args['d_s_embedding'] = d_state
        else:
            d_state = module_args['d_s_embedding']

        # calculate state embedding size if necessary and store it explicitly in config for later

        # calculate observation dimension for RSSM encoders/decoders
        if i_module == 0:
            module_args['d_a'] = env.action_space.shape[0]
            s_o = env.observation_space.shape
        else:
            if cfg['mdm']['links'][i_module - 1] == 'z':
                s_o = cfg['mdm']['rssm_modules'][i_module - 1]['d_z']
            elif cfg['mdm']['links'][i_module - 1] == 'h':
                s_o = cfg['mdm']['rssm_modules'][i_module - 1]['d_h']
            elif cfg['mdm']['links'][i_module - 1] == 's_embedding':
                s_o = cfg['mdm']['rssm_modules'][i_module - 1]['d_s_embedding']
            elif cfg['mdm']['links'][i_module - 1] == 'o':
                s_o = cfg['mdm']['rssm_modules'][i_module - 1]['o_decoder']['s_x_orig']
            else:
                raise ValueError(f'Unknown link key: {cfg["mdm"]["links"][i_module - 1]}')

        module_args['o_encoder']['s_x_orig'] = s_o
        module_args['o_decoder']['s_x_orig'] = s_o
        module_args['o_decoder']['d_x_encoded'] = d_state
        module_args['r_decoder']['s_x_orig'] = 1
        module_args['r_decoder']['d_x_encoded'] = d_state
        module_args['term_decoder']['s_x_orig'] = 1
        module_args['term_decoder']['d_x_encoded'] = d_state
        module_args['name'] = f'rssm_level_{i_module}'

    # just for logging
    cfg['hierarchy_levels'] = len(cfg['mdm']['rssm_modules'])

    # agents
    for agent_lvl in range(len(cfg['mdm']['rssm_modules'])):
        try:
            cfg['agents']
        except KeyError:
            print('No agent configuration found, config values for agents won\'t be inferred')
            continue

        if cfg['mdm']['rssm_modules'][agent_lvl]['latent_dist'] == 'normal':
            d_z = cfg['mdm']['rssm_modules'][agent_lvl]['d_z']
        elif cfg['mdm']['rssm_modules'][agent_lvl]['latent_dist'] == 'categorical':
            d_z = cfg['mdm']['rssm_modules'][agent_lvl]['d_z'] * cfg['mdm']['rssm_modules'][agent_lvl][
                'n_latent_categories']
        else:
            d_z = cfg['mdm']['rssm_modules'][agent_lvl]['d_z']
        d_h = cfg['mdm']['rssm_modules'][agent_lvl]['d_h']

        cfg_r_max = cfg['agents']['r_max'][agent_lvl]
        #if agent_lvl == 0:
        #    cfg_r_max['min_a'] = tuple(env.action_space.low)
        #    cfg_r_max['max_a'] = tuple(env.action_space.high)
        cfg_r_max['d_a'] = cfg['mdm']['rssm_modules'][agent_lvl]['d_a']
        if cfg_r_max['observation_type'] == 'z':
            cfg_r_max['d_o'] = d_z
        elif cfg_r_max['observation_type'] == 'h':
            cfg_r_max['d_o'] = d_h
        elif cfg_r_max['observation_type'] == 's_embedding':
            cfg_r_max['d_o'] = cfg['mdm']['rssm_modules'][agent_lvl]['d_s_embedding']

        if agent_lvl < len(cfg['mdm']['rssm_modules']) - 1:
            cfg_goal_seeking = cfg['agents']['goal_seeking'][agent_lvl]
            #if agent_lvl == 0:
            #    cfg_goal_seeking['min_a'] = tuple(env.action_space.low)
            #    cfg_goal_seeking['max_a'] = tuple(env.action_space.high)
            cfg_goal_seeking['d_a'] = cfg['mdm']['rssm_modules'][agent_lvl]['d_a']
            if cfg_goal_seeking['observation_type'] == 'z':
                cfg_goal_seeking['d_o'] = d_z
            elif cfg_goal_seeking['observation_type'] == 'h':
                cfg_goal_seeking['d_o'] = d_h
            elif cfg_goal_seeking['observation_type'] == 's_embedding':
                cfg_goal_seeking['d_o'] = cfg['mdm']['rssm_modules'][agent_lvl]['d_s_embedding']

    def _check_complete(name, entry):
        if isinstance(entry, dict):
            for _k, _v in entry.items():
                _check_complete(_k, _v)
        elif isinstance(entry, (list, tuple)):
            for x in entry:
                _check_complete(name, x)
        elif entry == '<infer>':
            raise ValueError(f'Found config value that should\'ve been inferred from other values but hasn\'t: {name}')

    for k, v in cfg.items():
        _check_complete(k, v)

    return cfg
