# scripts/run_sensory_numpy.py
from __future__ import annotations

from pathlib import Path
import pprint
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from data_helper.csv_loader import CSVConfig, load_level_csv, iter_sequence
from mercury.action_map.adapter import ActionMap
from mercury.sensory.params import SensoryParams
from mercury.sensory.state import sensory_step, init_state, sensory_step_frozen
from mercury.memory.state import init_mem, update_memory, add_memory, MemoryState
from mercury.latent.state import LatentState, latent_step, init_latent_state, \
    _remove_aliased_connections
from mercury.latent.params import LatentParams

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"

# ----- data -----
level = 17
# data_cfg = CSVConfig(root=str(DATASETS_DIR), level=13, coords="floor")
data_cfg = CSVConfig(root=str(DATASETS_DIR), level=level,coords="cartesian")

obs, act, col = load_level_csv(data_cfg)
data_dim = int(obs.shape[1])
action_dim = int(act.shape[1]) if act.ndim == 2 else 1

# ----- models -----
state = init_state(data_dim)
cfg = SensoryParams(activation_threshold=0.95, sensory_weighting=1)
am = ActionMap.random(n_codebook=4, dim=action_dim, lr=0.5, sigma=0.0, key=0)
states = []
# ----- loop -----
for t, (observation, action, collision) in enumerate(iter_sequence(obs, act,
                                                                 col)):
    action = np.atleast_1d(action).astype(np.float32)          # (A,)
    action_bmu, action_vector = am.step(action)                              #
    # update
    # AM, get BMU
    state = sensory_step(observation.astype(np.float32),       # update sensory graph
                         int(action_bmu),
                         state,
                         cfg,
                         am)
    states.append(state.prev_bmu)

np.savetxt(f"ground_truth{level}.csv", states, delimiter=",")