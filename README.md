# TopoWorld
TopoWorld: Topological game environments for benchmarking Reinforcement Learning

# Setup
Set up python environment for this project (anaconda, venv, etc.) and install requirements:

```shell
pip install -e .
```
please let me know if requirements are missing.

# Usage

## Mazes
First, we have to generate some mazes 
(the script also comes with a help page, so just pass the `-h` or `--help flag to see how to customize):
```shell
python -m topoworld.utils.maze_sampler --lx 5 --lz 5
```
It will generate a maze dictionary file at `mazes/mazes_<lx>x<lz>_<maze_dict_key>.p`.

You can play a specific maze by running 
`python scripts/manual_control.py mazes/mazes_<lx>x<lz>_<maze_dict_key>.p <pi> <env_idx> --top_view`
(the `--top_view` flag being optional). Make sure to replace `mazes_<lx>x<lz>_<maze_dict_key>.p` with the 
directory you've just generated

This will for example look like:
```shell
python scripts/manual_control.py mazes/mazes_10x10_71C2204D.p 40 0 --top_view
```

## Training
Finally, you can train PPO on a fixed navigation task, e.g.:
```shell
python scripts/train_sb3_simple.py mazes/mazes_4x4_527F7AFC.p 5 0
```

This will drop all the logs, training info and the trained model at
```shell
/results/Miniworld_TopoMaze-v0/sb3_PP0/exp_<experiment_key>/
```

## Visualization
You have multiple options to visualize the results. 
- You can watch the agent solve the navigation problem with 
(make sure to replace `exp_<exp_key> with the actual folder name):
```shell
python scripts/vis_sb3.py results/MiniWorld-TopoMaze-v0/sb3_PPO/exp_496EDB38
```
- You can inspect the learning curves for PPO with tensorboard:
```shell
tensorboard --logdir results/MiniWorld-TopoMaze-v0/sb3_PPO/exp_496EDB38/
```
- If you want to do more in-depth data analysis of the training run, take a look 
into the `notebooks` directory for inspiration