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
level = 16
# data_cfg = CSVConfig(root=str(DATASETS_DIR), level=1, coords="floor")
data_cfg = CSVConfig(root=str(DATASETS_DIR), level=level)

obs, act, col = load_level_csv(data_cfg)
data_dim = int(obs.shape[1])
action_dim = int(act.shape[1]) if act.ndim == 2 else 1

# ----- models -----
state = init_state(data_dim)
cfg = SensoryParams(activation_threshold=0.95)
am = ActionMap.random(n_codebook=4, dim=action_dim, lr=0.5, sigma=0.0, key=0)
# ----- loop -----
sensory_states = []

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
    sensory_states.append(state.prev_bmu)


# ---- Models ----
mem_length = 100
mem = init_mem(state.gs.n, mem_length)
mem_vec = []
action_mem = []
state_mem = []
latent = init_latent_state(mem)
latent_cfg = LatentParams()

prev_bmu = None
latent_states = []
for t, (observation, action, collision) in enumerate(iter_sequence(obs, act,
                                                                   col)):
    action = np.atleast_1d(action).astype(np.float32)          # (A,)
    action_bmu, action_vector = am.step(action)
    state = sensory_step_frozen(observation.astype(np.float32),       # update
                          # sensory graph
                         int(action_bmu),
                         state,
                         cfg,
                         am, )
    S = state.gs.n
    if mem is None or mem.gs.n != S * mem_length:
        mem = init_mem(state.gs.n, length=mem_length)

    if prev_bmu is None or not collision:
        s_act = np.asarray(state.gs.node_features["activation"],
                           dtype=np.float32)
        mem = update_memory(mem)
        mem = add_memory(mem, s_act)
        mem_vec.append(s_act)
        action_mem.append(action_bmu)
        latent, bmu, state_mem = latent_step(mem, mem_vec, latent, action_bmu,
                                  latent_cfg, am,
                                  action_mem, state_mem)
    latent_states.append(int(latent.prev_bmu))

    prev_bmu = state.prev_bmu
    # if t % 5000 == 0:
    #     print(t)
    #     g = latent.g
    #     for i in range(g.n):
    #         removed = range(g.n)
    #         g = _remove_aliased_connections(g, i, removed)
#     #     latent.g = g
np.savetxt(f"sensory{level}.csv", sensory_states, delimiter=",")
np.savetxt(f"latent{level}.csv", latent_states, delimiter=",")