import gymnasium as gym

static_u_maze_map_small = [[1, 1, 1, 1, 1],
                           [1, 'g', 0, 0, 1],
                           [1, 1, 1, 0, 1],
                           [1, 'r', 0, 0, 1],
                           [1, 1, 1, 1, 1]]

gym.register(
    id=f"PointMaze_UMaze_static-v3",
    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
    kwargs={"maze_map": static_u_maze_map_small, 'continuing_task': False, 'reset_target': False},
    max_episode_steps=300,
)
