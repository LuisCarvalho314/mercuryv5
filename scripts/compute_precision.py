import numpy as np

from utils import *
from utils.metrics import compute_precision
import matplotlib.pyplot as plt

level = 17

sensory_states = np.loadtxt(f"sensory{level}.csv", delimiter=",")
latent_states = np.loadtxt(f"latent{level}.csv", delimiter=",")
ground_truth = np.loadtxt(f"ground_truth{level}.csv", delimiter=",")


sensory_precisions = []
for i in range(1,len(sensory_states)):
        min_i = max(0, i-500)
        sensory_states_so_far = sensory_states[min_i:i]
        ground_truth_so_far = ground_truth[min_i:i]
        precision = compute_precision(sensory_states_so_far, ground_truth_so_far)
        sensory_precisions.append(precision)

latent_precisions = []
for i in range(1,len(latent_states)):
        min_i = max(0, i-500)
        latent_states_so_far = latent_states[min_i:i]
        ground_truth_so_far = ground_truth[min_i:i]
        precision = compute_precision(latent_states_so_far, ground_truth_so_far)
        latent_precisions.append(precision)

fig, ax = plt.subplots()
ax.plot(range(1,len(sensory_states)),sensory_precisions, label="sensory")
ax.plot(range(1,len(latent_states)),latent_precisions, label="latent")
ax.set_xlabel("Iterations")
ax.set_ylabel("Precision")
ax.set_title(f"Precision Corridor Length {level} Sensor Range 1")
ax.legend()
plt.savefig(f"L{level}_range_1_precision.png")
plt.show()
