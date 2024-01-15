import gymnasium as gym

static_u_maze_map_small = [[1, 1, 1, 1, 1],
                           [1, 'g', 0, 0, 1],
                           [1, 1, 1, 'r', 1],
                           [1, 'r', 0, 0, 1],
                           [1, 1, 1, 1, 1]]

static_u_maze_map_small_far = [[1, 1, 1, 1, 1],
                               [1, 'g', 0, 0, 1],
                               [1, 1, 1, 0, 1],
                               [1, 'r', 0, 0, 1],
                               [1, 1, 1, 1, 1]]

static_u_maze_map_small_1 = [[1, 1, 1, 1, 1],
                             [1, 'g', 'r', 'r', 1],
                             [1, 1, 1, 'r', 1],
                             [1, 'r', 'r', 'r', 1],
                             [1, 1, 1, 1, 1]]

static_u_maze_map_small_2 = [[1, 1, 1, 1, 1],
                             [1, 'r', 'r', 'r', 1],
                             [1, 1, 1, 'r', 1],
                             [1, 'g', 'r', 'r', 1],
                             [1, 1, 1, 1, 1]]

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
