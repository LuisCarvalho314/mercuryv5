# scripts/run_sensory_numpy_live.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from data_helper.csv_loader import CSVConfig, load_level_csv, iter_sequence
from mercury.action_map.adapter import ActionMap
from mercury.sensory.params import SensoryParams
from mercury.sensory.state import sensory_step, init_state
from mercury.graph.plot_graph import to_networkx  # reuses your converter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"

def _edge_attr(G, key_primary: str, key_fallback: str, default: float = 0.0):
    vals = []
    for u, v in G.edges:
        d = G[u][v]
        if key_primary in d:
            vals.append(float(d[key_primary]))
        elif key_fallback in d:
            vals.append(float(d[key_fallback]))
        else:
            vals.append(default)
    return np.asarray(vals, dtype=float)

def main(max_steps: int = 100) -> None:
    # ----- data -----
    cfg_csv = CSVConfig(root=str(DATASETS_DIR))
    obs, act = load_level_csv(cfg_csv)
    data_dim = int(obs.shape[1])
    action_dim = int(act.shape[1]) if act.ndim == 2 else 1

    # ----- models -----
    state = init_state(data_dim)
    cfg = SensoryParams(activation_threshold=0.95)
    am = ActionMap.random(n_codebook=4, dim=action_dim, lr=0.5, sigma=0.0, key=0)

    # ----- single figure -----
    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)

    for t, (observation, action) in enumerate(iter_sequence(obs, act), start=1):
        if t > max_steps:
            break

        action = np.atleast_1d(action).astype(np.float32)
        action_bmu, _ = am.step(action)

        state = sensory_step(
            observation.astype(np.float32),
            int(action_bmu),
            state,
            cfg,
            am,
        )

        # ---- draw current graph on the same axes ----
        ax.clear()
        G = to_networkx(state.gs)

        # layout each frame; seed ensures stable-ish layout
        pos = nx.spring_layout(G, seed=0)

        # node colors from "activation" (default 0)
        node_color = [float(G.nodes[i].get("activation", 0.0)) for i in G.nodes]

        # edge widths from "weight" scaled
        widths = np.array([float(G[u][v].get("weight", 1.0)) for u, v in G.edges], dtype=float)
        if widths.size:
            wmin, wptp = widths.min(), np.ptp(widths)
            widths = 1.0 + 2.0 * (widths - wmin) / (wptp + 1e-9)

        # edge colors from "age" or "age[0]"
        edge_color = _edge_attr(G, "age", "age[0]", default=0.0)

        nx.draw_networkx_nodes(G, pos, node_color=node_color, cmap="viridis", node_size=400, ax=ax)
        nx.draw_networkx_edges(G, pos, width=widths.tolist() if widths.size else 1.0,
                               edge_color=edge_color, edge_cmap=plt.cm.viridis,
                               arrows=True, ax=ax)
        # labels optional; keep off for speed
        # nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

        ax.set_title(f"Sensory graph at step {state.step_idx}")
        ax.axis("off")
        fig.tight_layout()
        plt.pause(0.5)  # single window, 0.5s delay

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main(max_steps=100)
