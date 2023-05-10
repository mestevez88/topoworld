import argparse


def add_maze_argument(parser: argparse.ArgumentParser):
    parser.add_argument('maze_path', help="path to maze dictionary file")


def add_experiment_path_argument(parser: argparse.ArgumentParser):
    parser.add_argument('exp_path', help="path to experiment folder")


def add_maze_selection_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('pi', type=int, help="select environment with this fundamental group (pi_1")
    parser.add_argument('env_idx', type=int, help="select environment with this env index")


def add_algo_argument(parser: argparse.ArgumentParser):
    parser.add_argument('--algo', default="PPO", type=str, help="name of the algorithm")


def add_miniworld_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--domain-rand", action="store_true", help="enable domain randomization"
    )
    parser.add_argument(
        "--no-time-limit", action="store_true", help="ignore time step limits"
    )
    parser.add_argument(
        "--top_view",
        action="store_true",
        help="show the top view instead of the agent view",
    )
