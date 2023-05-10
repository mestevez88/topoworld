import os

import numpy as np
import yaml


def get_maze_adjacency(experiment_path):
    return np.load(os.path.join(experiment_path, "maze_adjacency.npz"))['arr_0']


def get_experiment_info(experiment_path):
    with open(os.path.join(experiment_path, 'info.yml'), "r") as f:
        info = yaml.load(f, yaml.Loader)

    return info


def write_adjacency(experiment_path, adjacency):
    with open(os.path.join(experiment_path, "maze_adjacency.npz"), "wb") as f:
        np.savez(f, adjacency)


def write_info(experiment_path, info):
    with open(os.path.join(experiment_path, "info.yml"), "w") as f:
        f.write(yaml.dump(info))
