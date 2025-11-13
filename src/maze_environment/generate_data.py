from maze_environment_v3 import MazeEnvironment
from colour_tile_levels import *
from LEVEL import levels
import random
import numpy as np
import datetime
import os

random.seed(0)

level = 17
agent_sensors = {"sensor": "cardinal distance", "range": 2}
# agent_sensors = {"sensor": "cartesian"}

if agent_sensors["sensor"] == "cardinal distance":
    directory = (f"../../datasets/level{level}/{agent_sensors["sensor"]}/"
             f"range_{agent_sensors['range']}/")
else:
    directory = f"../../datasets/level{level}/{agent_sensors["sensor"]}/"

if not os.path.exists(directory):
    os.makedirs(directory)

file_name = f"{directory}{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

env = MazeEnvironment(level=levels[level], plotting=False,
                      agent_sensors=agent_sensors)

observations = []
actions = []
collisions = []

for i in np.arange(20000):
    action = env.random_action()
    observation, action, collision = env.step(action)
    observations.append(observation)
    actions.append(action)
    collisions.append(collision)

observations = np.array(observations)
actions = np.array(actions)
collisions = np.array(collisions)

np.savetxt(
    f"{file_name}_observations.csv",
    observations,
    delimiter=",",
    fmt='%i'
)
np.savetxt(
    f"{file_name}_actions.csv",
    actions,
    delimiter=",",
    fmt='%i'
)

np.savetxt(
    f"{file_name}_collisions.csv",
    collisions,
    delimiter=",",
    fmt='%i'
)


