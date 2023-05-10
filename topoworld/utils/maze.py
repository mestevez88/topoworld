import argparse
import itertools
from os import PathLike
from typing import Union, Tuple

import pickle

import gymnasium as gym
import miniworld
import numpy

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


def cell_maze_from_adjacency(
        adjacency,
        n,
        random_goal=False,
        random_start=False
):

    max_cells = n * 2 + 1

    cell_array = numpy.ones(shape=(max_cells, max_cells), dtype="int32")

    # Dig the vertex cells
    for i in range(1, max_cells, 2):
        for j in range(1, max_cells, 2):
            cell_array[i, j] = 0

    if random_start:
        # Randomize a start, around the bottom-left
        s_x = numpy.random.randint(0, (max_cells - 1) // 2 - 1) * 2 + 1
        s_y = numpy.random.randint(0, (max_cells - 1) // 2 - 1) * 2 + 1
        start = (s_x, s_y)
    else:
        s_x = 1
        s_y = 1
        start = (s_x, s_y)

    if random_goal:
        # Randomize a goal (not quite efficiently)
        goal = (max_cells - 2, max_cells - 2)
        minimum_goal_dist = 0.45 * max_cells  # (at least that far)
        for e_x in range(1, max_cells, 2):
            for e_y in range(1, max_cells, 2):
                e_x = numpy.random.randint(0, (max_cells - 1) // 2 - 1) * 2 + 1
                e_y = numpy.random.randint(0, (max_cells - 1) // 2 - 1) * 2 + 1
                d_s_e = numpy.sqrt((e_x - s_x) ** 2 + (e_y - s_y) ** 2)
                if d_s_e > minimum_goal_dist:
                    goal = (e_x, e_y)
                    break
    else:
        goal = (max_cells-2, max_cells-2)

    for i, j in itertools.product(range(n*n), range(n*n)):
        if adjacency[i, j] and i < j:
            if j-i == 1:
                cell_array[(i % n) * 2 + 2, (i // n) * 2 + 1] = 0
            else:
                cell_array[(i % n) * 2 + 1, (i // n) * 2 + 2] = 0

    return cell_array
