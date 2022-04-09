from pathlib import Path
import unittest
from time import sleep
from copy import deepcopy

import numpy as np

from mdm.gridworld.gridworld import Gridworld, CellType
from mdm.utils.utils import here


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
            self.assertTrue((s == world.find_cell_type(CellType.AGENT)).all())
            self.assertEqual(r, r_target)
            self.assertEqual(done, done_target)
            self.assertTrue(world.observation_space.contains(s))

    def test_step_reward(self):
        world = self.world
        world.step_reward = -0.01

        actions = [0, 0, 0, 1, 1]
        rewards = [-0.01, -0.01, -0.01, -0.01, 0.99]
        dones = [False, False, False, False, True]

        s = world.observation_space.sample()

        for a, r_target, done_target in zip(actions, rewards, dones):
            s, r, done, info = world.step(a)
            self.assertTrue((s == world.find_cell_type(CellType.AGENT)).all())
            self.assertEqual(r, r_target)
            self.assertEqual(done, done_target)
            self.assertTrue(world.observation_space.contains(s))

    def test_fully_random_start_pos(self):
        world = Gridworld.from_cleartext(here() / 'testmap_no_start_pos.mapdata')
        for i in range(50):
            world.reset()
            agent_pos = world.find_cell_type(CellType.AGENT)
            self.assertEqual(agent_pos.shape, (2,))
            self.assertEqual(agent_pos.size, 2)  # only one agent after reset
            self.assertTrue((agent_pos < (world.grid_h, world.grid_w)).all())  # check out of bounds
            self.assertTrue((agent_pos >= 0).all())
            self.assertEqual(world.grid[0, 0], CellType.WALL)  # check that no existing entities were overwritten
            self.assertEqual(world.grid[3, 1], CellType.WALL)
            self.assertEqual(world.grid[6, 2], CellType.WALL)
            self.assertEqual(world.grid[7, 7], CellType.WALL)
            self.assertEqual(world.grid[0, 7], CellType.REWARD)

    def test_multi_start_pos_set(self):
        world = Gridworld.from_cleartext(here() / 'testmap_multi_start_pos.mapdata')
        for i in range(50):
            world.reset()
            agent_pos = world.find_cell_type(CellType.AGENT)
            self.assertEqual(agent_pos.shape, (2,))
            self.assertEqual(agent_pos.size, 2)  # only one agent after reset
            self.assertTrue((agent_pos < (world.grid_h, world.grid_w)).all())  # check out of bounds
            self.assertTrue((agent_pos >= 0).all())
            self.assertEqual(world.grid[0, 0], CellType.WALL)  # check that no existing entities were overwritten
            self.assertEqual(world.grid[3, 1], CellType.WALL)
            self.assertEqual(world.grid[6, 2], CellType.WALL)
            self.assertEqual(world.grid[7, 7], CellType.WALL)
            self.assertEqual(world.grid[0, 7], CellType.REWARD)
            self.assertTrue((world.grid[(0, 3, 3, 6), (1, 2, 3, 0)] == CellType.AGENT).any())

    def test_enact_sequence(self):
        world = Gridworld.from_cleartext(here() / 'testmap_multi_start_pos.mapdata')

        s_mem, a_mem, r_mem, terminal_mem = [], [], [], []

        s_mem.append(world.reset())
        done = False
        while not done:
            a = world.action_space.sample()
            s_next, r, done, info = world.step(a)

            s_mem.append(s_next)
            a_mem.append(a)
            r_mem.append(r)
            terminal_mem.append(done)

        # correct sequence
        world.enact_sequence(s_mem, a_mem, r_mem, terminal_mem, False, 0)

        # wrong sequences
        for i in range(len(s_mem)):
            s_mem_cp = deepcopy(s_mem)
            s_mem_cp[i] = s_mem_cp[i] + 1
            with self.assertRaises(RuntimeError):
                world.enact_sequence(s_mem_cp, a_mem, r_mem, terminal_mem, False, 0)
        # randomly changing action sequences could end up in the trajectory not changing at all
        # so this is not tested here
        for i in range(len(r_mem)):
            r_mem_cp = deepcopy(r_mem)
            r_mem_cp[i] = r_mem_cp[i] + 1
            with self.assertRaises(RuntimeError):
                world.enact_sequence(s_mem, a_mem, r_mem_cp, terminal_mem, False, 0)
        for i in range(len(terminal_mem)):
            terminal_mem_cp = deepcopy(terminal_mem)
            terminal_mem_cp[i] = not terminal_mem_cp[i]
            with self.assertRaises(RuntimeError):
                world.enact_sequence(s_mem, a_mem, r_mem, terminal_mem_cp, False, 0)


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
