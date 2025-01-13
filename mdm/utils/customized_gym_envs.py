import gymnasium as gym
import minigrid
from minigrid.envs import FourRoomsEnv
from minigrid.core.world_object import Goal
from minigrid.core.grid import Grid

C = 'c'
G = 'g'
R = 'r'

static_u_maze_map_small = [[1, 1, 1, 1, 1],
                           [1, G, 0, 0, 1],
                           [1, 1, 1, R, 1],
                           [1, R, 0, 0, 1],
                           [1, 1, 1, 1, 1]]

static_u_maze_map_small_far = [[1, 1, 1, 1, 1],
                               [1, G, 0, 0, 1],
                               [1, 1, 1, 0, 1],
                               [1, R, 0, 0, 1],
                               [1, 1, 1, 1, 1]]

static_u_maze_map_small_1 = [[1, 1, 1, 1, 1],
                             [1, G, R, R, 1],
                             [1, 1, 1, R, 1],
                             [1, R, R, R, 1],
                             [1, 1, 1, 1, 1]]

static_u_maze_map_small_2 = [[1, 1, 1, 1, 1],
                             [1, R, R, R, 1],
                             [1, 1, 1, R, 1],
                             [1, G, R, R, 1],
                             [1, 1, 1, 1, 1]]

empty_room = [[1, 1, 1, 1, 1, 1, 1, 1],
              [1, C, C, C, C, C, C, 1],
              [1, C, C, C, C, C, C, 1],
              [1, C, C, C, C, C, C, 1],
              [1, C, C, C, C, C, C, 1],
              [1, C, C, C, C, C, C, 1],
              [1, C, C, C, C, C, C, 1],
              [1, 1, 1, 1, 1, 1, 1, 1]]

medium_maze_map = [[1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [1, C, C, C, 1, C, C, C, 1],
                   [1, C, C, C, 1, C, C, C, 1],
                   [1, C, C, 1, 1, C, C, C, 1],
                   [1, C, C, C, C, C, C, C, 1],
                   [1, C, C, C, C, C, C, C, 1],
                   [1, C, C, C, 1, 1, C, C, 1],
                   [1, C, C, C, 1, C, C, C, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1]]

medium_maze_map_2 = [[1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 0, 0, 0, 1, 0, 0, 0, 1],
                     [1, 0, 0, R, 1, 0, 0, 0, 1],
                     [1, 0, 0, 1, 1, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 1, 1, 0, 0, 1],
                     [1, 0, 0, 0, 1, G, 0, 0, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1]]

gym.register(
    id=f"PointMaze_UMaze_static-v3",
    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
    kwargs={"maze_map": static_u_maze_map_small, 'continuing_task': False, 'reset_target': False},
    max_episode_steps=300,
)

gym.register(
    id=f"PointMaze_UMaze_static_far-v3",
    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
    kwargs={"maze_map": static_u_maze_map_small_far, 'continuing_task': False, 'reset_target': False},
    max_episode_steps=300,
)

gym.register(
    id=f"PointMaze_UMaze_static_1-v3",
    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
    kwargs={"maze_map": static_u_maze_map_small_1, 'continuing_task': False, 'reset_target': False},
    max_episode_steps=300,
)

gym.register(
    id=f"PointMaze_UMaze_static_2-v3",
    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
    kwargs={"maze_map": static_u_maze_map_small_2, 'continuing_task': False, 'reset_target': False},
    max_episode_steps=300,
)

gym.register(
    id=f"PointMaze_empty_6x6-v0",
    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
    kwargs={"maze_map": empty_room, 'continuing_task': False, 'reset_target': False},
    max_episode_steps=300,
)

gym.register(
    id=f"PointMaze_SplitMaze_dense-v0",
    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
    kwargs={"maze_map": medium_maze_map, 'continuing_task': False, 'reset_target': False},
    max_episode_steps=300,
)

gym.register(
    id=f"PointMaze_SplitMaze_sparse-v0",
    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
    kwargs={"maze_map": medium_maze_map, 'continuing_task': False, 'reset_target': False},
    max_episode_steps=300,
)

gym.register(
    id="MiniGrid-FixedLayoutFourRooms-v0",
    entry_point="mdm.utils.customized_gym_envs:FixedLayoutFourRooms",
    kwargs={'goal_pos': (2, 2), 'door_pos': [(9, 2), (9, 14), (4, 9), (16, 9)]},
    max_episode_steps=100
)


class FixedLayoutFourRooms(FourRoomsEnv):

    def __init__(self, agent_pos=None, goal_pos=None, door_pos=None, max_steps=100, **kwargs):
        super().__init__(agent_pos, goal_pos, max_steps, **kwargs)
        self.door_pos = door_pos

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):
            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    pos = (xR, self._rand_int(yT + 1, yB))
                    if self.door_pos is None:
                        self.grid.set(*pos, None)

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    pos = (self._rand_int(xL + 1, xR), yB)
                    if self.door_pos is None:
                        self.grid.set(*pos, None)

        if self.door_pos is not None:
            for pos in self.door_pos:
                self.grid.set(pos[0], pos[1], None)

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            # assuming random start direction
            self.agent_dir = self._rand_int(0, 4)
        else:
            self.place_agent()

        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
            self.place_obj(Goal())
