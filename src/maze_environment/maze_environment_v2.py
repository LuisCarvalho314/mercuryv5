import random

import numpy as np
from LEVEL import *
from agent import Agent
from plots import MazeEnvironmentVisualization
import time

class Maze:
    def __init__(self, level):
        self.agent_position = None
        self.maze = self.read_level(level)

    def read_level(self, level):
        maze = np.zeros((len(level), len(level[0])), dtype=np.uint8)
        for i, row in enumerate(level):
            for j, element in enumerate(row):
                if element == "X":
                    maze[i, j] = 1
                if element == "P":
                    self.agent_position = (i, j)
        return maze


class MazeEnvironment(Maze):
    def __init__(self, level, plotting=False):
        super().__init__(level)
        self.agent = Agent(initial_position=self.agent_position, sensors={"sensor": "cardinal distance", "range": None})
        if plotting:
            self.visualization = MazeEnvironmentVisualization(self)
            self.visualization.update_loop(self)


    def update_loop(self, action):
        self.agent_position, _ = self.agent.take_action(self.maze, action=action)
        print(self.agent.make_observation(maze=self.maze))
        self.visualization.update_loop(self)



if __name__ == '__main__':
    level = levels[5]
    env = MazeEnvironment(level, plotting=True)
    actions = ["east, west, north, south"]
    for i in range(1000):
        action = random.choice(list(env.agent.action_dict.keys()))
        print(action)
        env.update_loop(action)
        print(env.agent.position)