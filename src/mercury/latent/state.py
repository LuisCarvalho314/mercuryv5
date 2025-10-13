# latent/state.py

from dataclasses import dataclass, replace
from typing import Any, Tuple

import numpy as np

from ..action_map.adapter import ActionMap
from mercury.graph.core import Graph, Array
from mercury.graph.maintenance import age_maintenance
from .params import LatentParams
from ..memory.state import MemoryState
from ..sensory.state import SensoryState

Array = np.ndarray


@dataclass
class LatentState:
    g: Graph
    mapping: Array
    prev_bmu: int | None = None
    step_idx: int = 0



def _register_features(g: Graph, mem_len: int) -> Graph:
    """
    Register required features.

    Node features
    -------------
    activation  : (n,)
    mem_adj     : (n, mem_len)

    Edge features
    -------------
    age         : (n, n, 1)
    action      : (n, n, 1)
    """

    g.register_node_feature("activation", 1, init_value=0)
    g.register_node_feature("mem_adj", mem_len, init_value=0)
    g.register_edge_feature("age", 1, dtype=np.int32, init_value=0)
    g.register_edge_feature("action", 1, dtype=np.int32, init_value=0)
    return g


def build_initial_graph(ms: MemoryState) -> Graph:
    g = Graph(directed=True)
    _register_features(g, ms.length)
    for _ in range(ms.sensory_n_nodes):
        g = _add_node(g)



def _add_node(g: Graph) -> Graph:
    g.add_node(
        node_feat={
            "activation": np.array(0.0, np.float32),
            "mem_adj": np.array(0.0, np.float32),
        },
        edge_defaults_for_new_node={"age": np.array([0], np.int32),
                                    "action": np.array([0], np.int32)},
    )
    return g