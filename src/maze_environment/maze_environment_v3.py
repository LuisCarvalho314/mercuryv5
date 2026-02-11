import logging
import random
from typing import Any, Optional

import gymnasium as gym
import numpy as np

from src.maze_environment.agent import Agent
from src.maze_environment.plots import MazeEnvironmentVisualization
from src.utils.setup_logging import setup_logging


class MazeEnvironment(gym.Env):
    def __init__(
        self,
        level: Any,
        plotting: bool = False,
        agent_sensors: Optional[dict] = None,
        colour_tile_levels: bool = False,
        seed: int = 0,
    ):
        setup_logging()
        self.logger = logging.getLogger(self.__class__.__name__)

        if colour_tile_levels:
            self.maze, self.initial_agent_position = self._read_colour_level(level)
        else:
            self.maze, self.initial_agent_position = self._read_level(level)

        self.agent_position = self.initial_agent_position
        self.agent = Agent(initial_position=self.agent_position, sensors=agent_sensors, maze=self.maze)

        self.action_keys = list(self.agent.action_dict.keys())
        self.rng = random.Random(seed)

        self.plotting = plotting
        self.visualization = MazeEnvironmentVisualization(self) if plotting else None
        if self.visualization is not None:
            self.visualization.update_loop(self)

    @staticmethod
    def _read_level(level: Any) -> tuple[np.ndarray, tuple[int, int]]:
        maze = np.zeros((len(level), len(level[0])), dtype=np.uint8)
        initial_agent_position = (0, 0)

        for row_index, row in enumerate(level):
            for col_index, element in enumerate(row):
                if element == "X":
                    maze[row_index, col_index] = 1
                elif element == "P":
                    initial_agent_position = (row_index, col_index)

        return maze, initial_agent_position

    @staticmethod
    def _read_colour_level(level: Any) -> tuple[np.ndarray, tuple[int, int]]:
        maze = np.array(level)
        initial_agent_position = (1, 1)
        return maze, initial_agent_position

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if seed is not None:
            self.rng = random.Random(seed)

        self.agent_position = self.initial_agent_position
        self.agent.position = self.initial_agent_position
        self.agent.make_observation(self.maze)

        return self.agent.observation, {}

    def step(self, action: str):
        self.agent_position, self.agent.action, collision = self.agent.take_action(self.maze, action=action)
        self.agent.make_observation(self.maze)

        self.logger.debug("STEP")

        if self.visualization is not None:
            self.visualization.update_loop(self)

        return self.agent.observation, self.agent.action, bool(collision)

    def random_action(self) -> str:
        return self.rng.choice(self.action_keys)

    def random_policy(self, previous_action: str, collision: bool, rand_prob: float) -> str:
        do_random = (self.rng.random() < rand_prob) or collision
        return self.random_action() if do_random else previous_action
