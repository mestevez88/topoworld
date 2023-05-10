import gymnasium as gym

from topoworld.topomaze import TopoMaze, TopoMazeMeta

gym.register(
    id="MiniWorld-TopoMaze-v0",
    entry_point="topoworld.topomaze:TopoMaze",
)

gym.register(
    id="MiniWorld-TopoMazeMeta-v0",
    entry_point="topoworld.topomaze:TopoMazeMeta",
)
