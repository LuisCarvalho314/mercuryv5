import numpy as np

from mercury.graph.core import Graph
from mercury.latent.params import LatentParams
from mercury.latent.state import LatentState, _register_features, compute_bmu
from mercury.memory.state import add_memory, init_mem


def test_compute_bmu_uses_configured_lambda_trace() -> None:
    memory_state = add_memory(init_mem(1, length=2), np.array([1.0], dtype=np.float32))

    graph = Graph(directed=True)
    _register_features(graph, mem_len=memory_state.gs.n, n_actions=1)
    for _ in range(2):
        graph.add_node()
    graph.node_features["mem_adj"] = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    no_penalty = compute_bmu(
        state=LatentState(
            g=graph,
            mapping=np.array([0, 1]),
            prev_activations=np.array([100.0, 0.0], dtype=np.float32),
        ),
        memory_state=memory_state,
        action_bmu=0,
        cfg=LatentParams(
            allow_self_loops=False,
            trace_decay=0.99,
            lambda_trace=0.0,
            weight_memory=1.0,
            weight_undirected=0.0,
            weight_base=0.0,
            weight_action=0.0,
        ),
    )
    with_penalty = compute_bmu(
        state=LatentState(
            g=graph,
            mapping=np.array([0, 1]),
            prev_activations=np.array([100.0, 0.0], dtype=np.float32),
        ),
        memory_state=memory_state,
        action_bmu=0,
        cfg=LatentParams(
            allow_self_loops=False,
            trace_decay=0.99,
            lambda_trace=0.05,
            weight_memory=1.0,
            weight_undirected=0.0,
            weight_base=0.0,
            weight_action=0.0,
        ),
    )

    assert no_penalty == 0
    assert with_penalty == 1
