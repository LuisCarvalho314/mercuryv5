import logging
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import colors

from utils.setup_logging import setup_logging

logger = logging.getLogger(__name__)
class MazeEnvironmentVisualization:
    def __init__(self, env):
        setup_logging()
        plt.style.use("dark_background")
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        colormap = colors.ListedColormap(["xkcd:slate grey", "xkcd:marine blue"])
        self.ax.imshow(env.maze, cmap=colormap,origin="upper")
        self.agent_sc = self.ax.scatter(env.agent.position[0], env.agent.position[1], s=100, c="xkcd:bright red",
                                        )

        self.ax.set_xticks(np.arange(0, env.maze.shape[1], step=1))
        self.ax.set_yticks(np.arange(0, env.maze.shape[0], step=1))
        self.ax.set_xlim(0, env.maze.shape[1] - .5)
        self.ax.set_ylim(0,env.maze.shape[0] - .5)

        # Minor ticks
        self.ax.set_xticks(np.arange(-.5, env.maze.shape[1] - 1, 1), minor=True)
        self.ax.set_yticks(np.arange(-.5, env.maze.shape[0] - 1, 1), minor=True)

        # Gridlines based on minor ticks
        self.ax.grid(which='minor', color='xkcd:light grey', linestyle='-', linewidth=2)

        # Remove minor ticks
        self.ax.tick_params(which='minor', bottom=False, left=False)
        self.ax.invert_yaxis()

    def update_loop(self, env):
        time.sleep(0.1)
        self.agent_sc.set_offsets((env.agent.position[1], env.agent.position[0]))
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        logger.debug(f"OBS | {env.agent.observation} POS |"
                      f" {env.agent.position}")
        pass
