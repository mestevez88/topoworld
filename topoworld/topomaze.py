import itertools

from gymnasium.core import ObsType
from gymnasium.spaces import Dict
from miniworld.entity import Box
from miniworld.params import DEFAULT_PARAMS
from miniworld.envs.maze import Maze

from enum import Enum
from typing import Union, Optional, List, Any, Tuple
from typing import Dict as DictType

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import itertools
import networkx as nx


default_params = DEFAULT_PARAMS.no_random()
default_params.set("forward_step", 0.7)
default_params.set("turn_step", 25)


def n_edges_grid(lx, lz):
    return (lx - 1) * lz + (lz - 1) * lx


def max_pi(lx, lz):
    v = lx * lz
    e_grid = n_edges_grid(lx, lz)
    e_spanning_tree = v - 1
    return e_grid - e_spanning_tree


class GridAction(Enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3


class TopogridTabular(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, a_env, lx, lz, enable_render=False, sparse_reward=True, max_reward=10, action_noise=0.2):
        super(TopogridTabular, self).__init__()

        self.lx = lx
        self.lz = lz

        self.max_reward = max_reward

        self.n_grid_actions = 4

        self.action_space = spaces.Discrete(self.n_grid_actions)
        self.n_states = self.lx * self.lz
        self.observation_space = spaces.Discrete(self.n_states)

        self.a_env = a_env

        self.start_state = 0
        self.goal_state = self.n_states - 1

        self.state = self.start_state

        if sparse_reward:
            self.reward_func = self.sparse_reward_func
        else:
            self.reward_func = self.shaped_reward_func

        self.distance_to_goal = nx.shortest_path_length(nx.from_numpy_array(self.a_env), target=self.goal_state)

        self.action_certainty = 1 - min(0.5, action_noise)

    def next_state(self, action: Union[GridAction, int]) -> int:

        # take a random action 1-action_certainty fraction of the time
        if np.random.rand() > self.action_certainty:
            action = self.action_space.sample()

        if action is GridAction.LEFT and (self.state % self.lz != 0) and self.a_env[self.state, self.state - 1]:
            self.state = self.state - 1

        if action is GridAction.DOWN and (self.state <= (self.lx * self.lz - self.lz - 1)) and self.a_env[self.state, self.state + self.lz]:
            self.state = self.state + self.lz

        if action is GridAction.RIGHT and (self.state % self.lz < (self.lz - 1)) and self.a_env[self.state, self.state + 1]:
            self.state = self.state + 1

        if action is GridAction.UP and (self.state >= self.lz) and self.a_env[self.state, self.state - self.lz]:
            self.state = self.state - self.lz

        return self.state

    def sparse_reward_func(self, state):
        if state == self.goal_state:
            return self.max_reward
        else:
            return -1

    def shaped_reward_func(self, state):
        return -self.distance_to_goal[state]/self.action_certainty

    def step(self, action: Union[int, List[int]]):
        state = self.next_state(GridAction(action))

        if state == self.goal_state:
            done = True
        else:
            done = False

        reward = self.reward_func(state)

        info = {"state": state, "reward": reward, "done": done}
        return state, reward, done, info

    def reset(self):
        self.state = self.start_state
        return self.state

    def render(self, mode='human'):
        pass

    def close(self):
        pass


class TopoMaze(Maze):
    def __init__(
            self,
            adjacency,
            num_rows=5,
            num_cols=5,
            max_episode_steps=300,
            params=default_params,
            domain_rand=False,
            chinese_postman=False,
            reset_seed=None,
            **kwargs,
    ):
        self.adjacency = adjacency
        self.reset_seed = reset_seed
        super(TopoMaze, self).__init__(
            num_rows=num_rows,
            num_cols=num_cols,
            max_episode_steps=max_episode_steps,
            params=params,
            domain_rand=domain_rand,
            **kwargs,
        )

        self.chinese_postman = chinese_postman
        if self.chinese_postman:
            self.agent_room = self.is_in_room()
            self.traversed_edges = set()

    def add_rooms(self, room1, room2):
        edge = frozenset({room1, room2})
        if edge not in self.traversed_edges:
            self.traversed_edges.add(edge)
            return True
        return False

    def is_in_room(self):
        for room in self.rooms:
            if room.min_x < self.agent.pos[0] < room.max_x \
                    and room.min_z < self.agent.pos[2] < room.max_z \
                    and room.max_x - room.min_x > 0.25 \
                    and room.max_z - room.min_z > 0.25:
                return room
        return self.agent_room

    def chinese_postman_reward(self):
        tmp_agent_room = self.is_in_room()
        if self.agent_room is not tmp_agent_room:
            if self.add_rooms(self.agent_room, tmp_agent_room):
                reward = 1
            else:
                reward = -1
            self.agent_room = tmp_agent_room
        else:
            reward = 0

        return reward

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            termination = True

        if self.chinese_postman:
            reward += self.chinese_postman_reward()

        return obs, reward, termination, truncation, info

    def _gen_world(self):
        rows = []

        # For each row
        for j in range(self.num_rows):
            row = []

            # For each column
            for i in range(self.num_cols):
                min_x = i * (self.room_size + self.gap_size)
                max_x = min_x + self.room_size

                min_z = j * (self.room_size + self.gap_size)
                max_z = min_z + self.room_size

                room = self.add_rect_room(
                    min_x=min_x,
                    max_x=max_x,
                    min_z=min_z,
                    max_z=max_z,
                    wall_tex="brick_wall",
                    # floor_tex='asphalt'
                )
                row.append(room)

            rows.append(row)

        neighbors = [(0, 1), (0, -1), (-1, 0), (1, 0)]

        for i, j in itertools.product(range(self.num_rows), range(self.num_cols)):
            room = rows[j][i]
            for dj, di in neighbors:
                ni = i + di
                nj = j + dj

                if nj < 0 or nj >= self.num_rows:
                    continue

                if ni < 0 or ni >= self.num_cols:
                    continue

                ns = nj*self.num_cols + ni
                s = j*self.num_cols + i

                if not self.adjacency[s, ns]:
                    continue

                if ns <= s:
                    continue

                neighbor = rows[nj][ni]

                if di == 0:
                    self.connect_rooms(
                        room, neighbor, min_x=room.min_x, max_x=room.max_x
                    )
                elif dj == 0:
                    self.connect_rooms(
                        room, neighbor, min_z=room.min_z, max_z=room.max_z
                    )

        self.box = self.place_entity(Box(color="red"))

        self.place_agent()

    def reset(
        self,
        *,
        seed: Union[int, None] = None,
        options: Union[DictType[str, Any], None] = None,
    ) -> Tuple[ObsType, DictType[str, Any]]:
        if self.reset_seed:
            return super().reset(seed=self.reset_seed)
        else:
            return super().reset(seed=seed)


class TopoMazeMeta(gym.Env):
    def __init__(
            self,
            adjacency,
            num_rows=5,
            num_cols=5,
            max_episode_steps=3000,
            params=default_params,
            domain_rand=False,
            chinese_postman=True,
            n_inner_episodes=10,
            **kwargs
    ):
        self.inner_maze = TopoMaze(
            adjacency,
            num_rows=num_rows,
            num_cols=num_cols,
            max_episode_steps=max_episode_steps,
            params=params,
            domain_rand=domain_rand,
            chinese_postman=chinese_postman,
            **kwargs
        )
        self.inner_reward_space = gym.spaces.Box(low=-1.0, high=2.0, shape=(1,))
        self.inner_done_space = gym.spaces.Discrete(2)
        self.observation_space = Dict({
            "prev_observation": self.inner_maze.observation_space,
            "prev_action": self.inner_maze.action_space,
            "prev_reward": self.inner_reward_space,
            "prev_done": self.inner_done_space,
            "observation": self.inner_maze.observation_space
        })
        self.action_space = self.inner_maze.action_space

        self.prev_observation = np.zeros_like(self.inner_maze.observation_space.sample())
        self.prev_action = np.zeros_like(self.inner_maze.action_space.sample())
        self.prev_reward = np.zeros_like(self.inner_reward_space.sample())
        self.prev_done = np.zeros_like(self.inner_done_space.sample())

        self.n_inner_episodes = n_inner_episodes
        self.inner_reset_seed = self.new_reset_seed()
        self.termination_counter = 0

        self.reset()

    @staticmethod
    def new_reset_seed():
        return np.random.randint(0, 2500)

    def reset(
        self,
        *,
        seed: Union[int, None] = None,
        options: Union[DictType[str, Any], None] = None,
    ) -> Tuple[ObsType, DictType[str, Any]]:
        super().reset(seed=seed)
        self.prev_observation = np.zeros_like(self.inner_maze.observation_space.sample())
        self.prev_action = np.zeros_like(self.inner_maze.action_space.sample())
        self.prev_reward = np.zeros_like(self.inner_reward_space.sample())
        self.prev_done = np.zeros_like(self.inner_done_space.sample())
        obs, _ = self.inner_maze.reset(seed=self.inner_reset_seed)
        return {
            "prev_observation": self.prev_observation,
            "prev_action": self.prev_action,
            "prev_reward": self.prev_reward,
            "prev_done": self.prev_done,
            "observation": obs
        }, {}

    def step(self, action):
        obs, reward, termination, truncation, _ = self.inner_maze.step(action)
        meta_obs = {
            "prev_observation": self.prev_observation,
            "prev_action": self.prev_action,
            "prev_reward": self.prev_reward,
            "prev_done": self.prev_done,
            "observation": obs
        }
        self.prev_action = action
        self.prev_observation = obs
        self.prev_reward = reward
        self.prev_done = np.array([int(termination)])

        if termination:
            self.termination_counter += 1
            self.inner_maze.reset(seed=self.inner_reset_seed)

        if self.termination_counter == (self.n_inner_episodes - 1):
            done = True

        else:
            done = False

        return meta_obs, reward, done, truncation, {}

    def render(self):
        self.inner_maze.render()
