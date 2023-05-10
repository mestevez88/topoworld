import argparse
from os import PathLike
from typing import Union, Tuple

import pickle

import gymnasium as gym
import miniworld

from topoworld.utils.argparse import add_miniworld_arguments, add_maze_selection_arguments, add_maze_argument


def n_edges_grid(lx, lz):
    return (lx - 1) * lz + (lz - 1) * lx


def max_pi(lx, lz):
    v = lx * lz
    e_grid = n_edges_grid(lx, lz)
    e_spanning_tree = v - 1
    return e_grid - e_spanning_tree


def load_maze(mazes_path: Union[PathLike, str], pi: int, env_idx: int, check_square_grid: bool = True):

    mazes = pickle.load(open(mazes_path, "rb"))

    if check_square_grid:
        assert mazes["info"]["lx"] == mazes["info"]["lz"], "Only square mazes are supported"

    adjacency = mazes["mazes"][pi][env_idx]

    return adjacency, mazes["info"]


def set_up_miniworld_topomaze_env(env_name):
    parser = argparse.ArgumentParser()
    add_miniworld_arguments(parser)
    add_maze_argument(parser)
    add_maze_selection_arguments(parser)

    args = parser.parse_args()
    view_mode = "top" if args.top_view else "agent"

    adjacency, maze_info = load_maze(args.maze_path, args.pi, args.env_idx)

    print(f"Load maze: {maze_info}")

    env = gym.make(env_name, adjacency=adjacency, chinese_postman=False, reset_seed=2056,
                   num_rows=maze_info["lx"], num_cols=maze_info["lz"], view=view_mode, render_mode="human")

    miniworld_version = miniworld.__version__

    print(f"Miniworld v{miniworld_version}, Env: {env_name}")

    return env, adjacency, maze_info, args
