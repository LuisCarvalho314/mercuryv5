import logging
import random
import numpy as np
import gymnasium as gym

from src.utils.setup_logging import setup_logging
from src.maze_environment.LEVEL import *
from src.maze_environment.colour_tile_levels import *
from src.maze_environment.agent import Agent
from src.maze_environment.plots import MazeEnvironmentVisualization
import time

class MazeEnvironment(gym.Env):
    def __init__(self, level, plotting=False, agent_sensors=None,
                 colour_tile_levels=False):
        setup_logging()
        self.logger = logging.getLogger(self.__class__.__name__)
        if colour_tile_levels:
            self.maze, self.initial_agent_position = self.read_colour_level(level)
        else:
            self.maze, self.initial_agent_position = self.read_level(level)
        self.agent_position = self.initial_agent_position
        self.agent = Agent(initial_position=self.agent_position,
                           sensors=agent_sensors, maze = self.maze)
        self.plotting = plotting
        if plotting:
            self.visualization = MazeEnvironmentVisualization(self)
            self.visualization.update_loop(self)


    def read_level(self, level):
        maze = np.zeros((len(level), len(level[0])), dtype=np.uint8)
        initial_agent_position = (0,0)
        for i, row in enumerate(level):
            for j, element in enumerate(row):
                if element == "X":
                    maze[i, j] = 1
                if element == "P":
                    initial_agent_position = (i, j)
        return maze, initial_agent_position

    def read_colour_level(self, level):
        maze = np.array(level)
        initial_agent_position = (1,1)
        return maze, initial_agent_position


    def reset(self):
        self.logger.debug("RESET")
        self.agent_position = self.initial_agent_position

    def _get_obs(self):
        return {"agent": self.agent.make_observation(self.maze)}

    def step(self, action):
        # print("-"*50)
        # print(f"POS | {self.agent.position}")
        # print(f"ACT | {action}")
        # print(f"OBS | {self.agent.observation}")
        self.agent_position, self.agent.action, collision = (
            self.agent.take_action(
            self.maze,
            action=action))
        self._get_obs()
        self.logger.debug("STEP")
        if self.plotting:
            self.visualization.update_loop(self)
        return self.agent.observation, self.agent.action, bool(collision)

    def random_action(self):
        return random.choice(list(self.agent.action_dict.keys()))

if __name__ == '__main__':
    level = levels[13]
    agent_sensors = {"sensor": "cardinal distance", "range" : None}
    env = MazeEnvironment(level, plotting=True, agent_sensors=agent_sensors)
    env.reset()
    for i in range(100):
        action = random.choice(list(env.agent.action_dict.keys()))
        obs = env.step(action)
        print(obs)
        time.sleep(.001)


