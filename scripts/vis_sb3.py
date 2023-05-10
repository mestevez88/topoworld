import argparse
import os
import time

import gymnasium as gym
from stable_baselines3 import PPO

from topoworld.utils.argparse import add_experiment_path_argument, add_miniworld_arguments
from topoworld.utils.experiment import get_experiment_info, get_maze_adjacency

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_experiment_path_argument(parser)
    add_miniworld_arguments(parser)
    args = parser.parse_args()

    experiment_path = args.exp_path
    info = get_experiment_info(experiment_path)
    adjacency = get_maze_adjacency(experiment_path)

    view_mode = "top" if args.top_view else "agent"

    env = gym.make(info["env_type"], adjacency=adjacency, chinese_postman=False, reset_seed=2056,
                   num_rows=info["lx"], num_cols=info["lz"], view=view_mode, render_mode="human")

    model = PPO.load(os.path.join(experiment_path, "model"), print_system_info=True)

    obs, _ = env.reset()
    for i in range(3000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        env.render()
        time.sleep(0.2)

        if done:
            break
