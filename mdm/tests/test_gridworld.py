from pathlib import Path
import unittest
from time import sleep

import numpy as np

from mdm.gridworld.gridworld import Gridworld, CellType


class GridworldTest(unittest.TestCase):

    def setUp(self) -> None:
        self.world = Gridworld(5, 5, 100, 1.0, 0.0, 0.99)
        self.agent_start_pos = (2, 2)
        self.reward_pos = (0, 4)
        self.wall_pos = (3, 1)

        self.world.add_object(self.agent_start_pos, CellType.AGENT)
        self.world.add_object(self.reward_pos, CellType.REWARD)
        self.world.add_object(self.wall_pos, CellType.WALL)

    def test_from_cleartext(self):
        world = Gridworld.from_cleartext(Path(__file__).parent / 'testmap.mapdata')

        self.assertEqual(world.grid_h, 8)
        self.assertEqual(world.grid_w, 8)
        self.assertEqual(world.grid[0, 0], CellType.WALL.value)
        self.assertEqual(world.grid[-1, -1], CellType.WALL.value)
        self.assertEqual(world.grid[3, 1], CellType.WALL.value)
        self.assertEqual(world.grid[0, -1], CellType.REWARD.value)
        self.assertEqual(world.grid[-2, 1], CellType.AGENT.value)
        self.assertEqual(world.grid[2, 2], CellType.FREE.value)

    def test_reset(self):
        world = self.world

        # move away from start position
        for action in [0, 0, 1]:
            world.step(action)
        self.assertEqual(world.grid[self.agent_start_pos], CellType.FREE)
        self.assertEqual(world.grid[self.reward_pos], CellType.REWARD)
        self.assertEqual(world.grid[self.wall_pos], CellType.WALL)

        s = world.reset()
        self.assertTrue((s == (2, 2)).all())
        self.assertEqual(world.grid[self.agent_start_pos], CellType.AGENT)
        self.assertEqual(world.grid[self.reward_pos], CellType.REWARD)
        self.assertEqual(world.grid[self.wall_pos], CellType.WALL)
        self.assertTrue(world.observation_space.contains(s))

    def test_step_basic_moving(self):
        world = self.world

        for action in [0, 1, 2, 3]:
            agent_pos = np.array(self.agent_start_pos) + world.move_offset[action]
            world.step(action)
            self.assertEqual(world.grid[tuple(agent_pos)], CellType.AGENT)
            world.reset()

    def test_step_map_boundary_collision(self):
        world = self.world

        for action in [0, 1, 2, 3]:
            agent_pos = np.array(self.agent_start_pos)
            for _ in range(2):
                world.step(action)
                agent_pos += world.move_offset[action]
            world.step(action)  # don't update agent_pos after this one since we're bumping into the map border
            self.assertEqual(world.grid[tuple(agent_pos)], CellType.AGENT)
            world.reset()

    def test_step_wall_collision(self):
        world = self.world

        world.step(3)
        self.assertEqual(world.grid[2, 1], CellType.AGENT)
        world.step(2)
        self.assertEqual(world.grid[2, 1], CellType.AGENT)
        world.step(3)
        self.assertEqual(world.grid[2, 0], CellType.AGENT)
        world.step(2)
        self.assertEqual(world.grid[3, 0], CellType.AGENT)
        world.step(1)
        self.assertEqual(world.grid[3, 0], CellType.AGENT)
        world.step(2)
        self.assertEqual(world.grid[4, 0], CellType.AGENT)
        world.step(1)
        self.assertEqual(world.grid[4, 1], CellType.AGENT)
        world.step(0)
        self.assertEqual(world.grid[4, 1], CellType.AGENT)
        world.step(1)
        self.assertEqual(world.grid[4, 2], CellType.AGENT)
        world.step(0)
        self.assertEqual(world.grid[3, 2], CellType.AGENT)
        world.step(3)
        self.assertEqual(world.grid[3, 2], CellType.AGENT)
        world.step(0)
        self.assertEqual(world.grid[self.agent_start_pos], CellType.AGENT)

    def test_step_return_values(self):
        world = self.world

        actions = [0, 0, 0, 1, 1]
        rewards = [0, 0, 0, 0, 1.0]
        dones = [False, False, False, False, True]

        s = world.observation_space.sample()

        for a, r_target, done_target in zip(actions, rewards, dones):
            s, r, done, info = world.step(a)
            self.assertTrue((s == world._get_agent_cell()).all())
            self.assertEqual(r, r_target)
            self.assertEqual(done, done_target)
            self.assertTrue(world.observation_space.contains(s))

    @unittest.skip
    def test_render(self):
        world = self.world

        world.reset()
        for t in range(10):
            world.render()
            o, r, done, info = world.step(world.action_space.sample())
            if done:
                break
            sleep(1)

if __name__ == '__main__':
    unittest.main()
