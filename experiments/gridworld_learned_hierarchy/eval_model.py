import torch

from mdm.gridworld.gridworld import Gridworld
from mdm.training.gym_driver import GymEpisodeDriver
from mdm.planning.cem_planner import CrossentropyPlanner, DistributionType
from mdm.utils.utils import here

if __name__ == '__main__':
    env = Gridworld.from_cleartext(here() / '../../mdm/gridworld/8x8_v0.mapdata')
    multiscale_model = torch.load(here() / 'model.ptmdl')
    planner = CrossentropyPlanner(DistributionType.CATEGORICAL)

    pln_n_planning_steps = 20

    def collect_policy(observation):
        return env.action_space.sample()

    collect_driver = GymEpisodeDriver(env, collect_policy)
