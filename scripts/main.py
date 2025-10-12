# scripts/run_sensory_numpy.py
from __future__ import annotations

from pathlib import Path
import pprint
import numpy as np

from data_helper.csv_loader import CSVConfig, load_level_csv, iter_sequence
from mercury.action_map.adapter import ActionMap
from mercury.sensory.params import SensoryParams
from mercury.sensory.state import sensory_step, init_state

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"

# ----- data -----
data_cfg = CSVConfig(root=str(DATASETS_DIR))
obs, act = load_level_csv(data_cfg)
data_dim = int(obs.shape[1])
action_dim = int(act.shape[1]) if act.ndim == 2 else 1

# ----- models -----
state = init_state(data_dim)
cfg = SensoryParams(activation_threshold=0.95)
am = ActionMap.random(n_codebook=4, dim=action_dim, lr=0.5, sigma=0.0, key=0)

# ----- loop -----
for t, (observation, action) in enumerate(iter_sequence(obs, act), start=1):
    action = np.atleast_1d(action).astype(np.float32)          # (A,)
    action_bmu, action_vector = am.step(action)                              #
    # update
    # AM, get BMU
    state = sensory_step(observation.astype(np.float32),       # update sensory graph
                         int(action_bmu),
                         state,
                         cfg,
                         am)
    if state.step_idx % 1000 == 0:
        pprint.pprint(state.gs.adj)
        pprint.pprint(state.mapping)
        pprint.pprint(state.gs.node_features["activation"])
