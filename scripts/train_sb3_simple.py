import argparse
import datetime
import os
import random
import time
from os import PathLike
from typing import Union

import gymnasium as gym
import miniworld
import numpy as np
import yaml
from torch import nn

from stable_baselines3 import A2C, SAC, PPO
from stable_baselines3.common.callbacks import EvalCallback
from sb3_contrib import RecurrentPPO

from topoworld.utils.argparse import add_miniworld_arguments, add_maze_argument, add_maze_selection_arguments
from topoworld.utils.experiment import write_adjacency, write_info
from topoworld.utils.maze import load_maze, set_up_miniworld_topomaze_env
from stable_baselines3.common.logger import configure as sb3_configure_logger


def play_visual(viz_env, trained_model=None, save_trajectory_path: Union[PathLike, str] = None):
    obs = viz_env.reset()
    done = False
    sum_reward = 0
    while not done:
        viz_env.render()
        if trained_model:
            time.sleep(0.2)
            action, _ = trained_model.predict(obs, deterministic=True)
        else:
            action = None
        obs, reward, done, _ = viz_env.step(action)
        sum_reward += reward

    print("Episode is over! You got %.1f score." % sum_reward)

    if save_trajectory_path:
        viz_env.save_trajectory(save_trajectory_path)


def train_sb3():

    env_name = "MiniWorld-TopoMaze-v0"
    env, adjacency, maze_info, args = set_up_miniworld_topomaze_env(env_name)
    algo_name = "PPO"

    with open(os.path.join("config", "sb3_params", f'{algo_name}_params.yml'), "r") as f:
        params = yaml.load(f, yaml.Loader)[env_name]
        if params.get("policy_kwargs"):
            params["policy_kwargs"] = exec(params["policy_kwargs"])

    print(f'Training {algo_name} with params {params}')

    model = PPO(env=env, **params)

    exp_key = f"{random.getrandbits(32):X}"

    experiment_path = os.path.join("results", env_name, f"sb3_{algo_name}", f"exp_{exp_key}")
    os.makedirs(experiment_path, exist_ok=True)

    eval_callback = EvalCallback(env, log_path=experiment_path, eval_freq=1000, deterministic=True,
                                 render=False, n_eval_episodes=3, warn=False, verbose=False)

    os.makedirs(experiment_path, exist_ok=True)

    write_adjacency(experiment_path, adjacency)

    write_info(experiment_path, {
            "algo": algo_name,
            "env_type": env_name,
            "pi": args.pi,
            "lx": maze_info["lx"],
            "lz": maze_info["lz"],
            "experiment": exp_key,
            "algo_impl": "sb3",
            "training_start_time": datetime.datetime.now()
    })

    new_logger = sb3_configure_logger(experiment_path, ["csv", "tensorboard"])
    model.set_logger(new_logger)

    model.learn(
        total_timesteps=100_000,
        log_interval=1000,
        progress_bar=True,
        callback=[eval_callback]  # , state_trajectory_callback
    )

    model.save(os.path.join(experiment_path, "model.zip"))

    return model


if __name__ == '__main__':
    train_sb3()
