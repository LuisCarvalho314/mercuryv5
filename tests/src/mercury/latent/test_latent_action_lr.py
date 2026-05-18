import numpy as np
import pytest

import mercury.latent.state as latent_state_mod
from mercury.graph.core import Graph
from mercury.latent.params import LatentParams
from mercury.latent.state import _register_features, _update_ambiguity_counter, _update_edges_with_actions


def test_update_edges_with_actions_uses_passed_beta(monkeypatch):
    g = Graph(directed=True)
    _register_features(g, mem_len=4, n_actions=3)
    for _ in range(2):
        g.add_node(
            node_feat={
                "activation": np.array(0.0, np.float32),
                "mem_adj": np.zeros(4, np.float32),
                "ambiguity_score": np.zeros(3, np.float32),
            },
            edge_defaults_for_new_node={
                "age": np.array([0], np.int32),
                "action": np.array([0], np.int32),
            },
        )

    class AMStub:
        def __init__(self):
            self.state = type("S", (), {})()
            self.state.codebook = np.eye(3, dtype=np.float32)

        @property
        def params(self):
            return type("P", (), {"dim": 3})()

    recorded: dict[str, float] = {}

    def fake_update_edge(g_in, u, v, prior_row, observed_row, action_bmu, beta, gaussian_shape, **kwargs):
        recorded["beta"] = float(beta)
        return g_in

    monkeypatch.setattr(latent_state_mod, "_update_edge", fake_update_edge)

    out = _update_edges_with_actions(
        g=g,
        prev_bmu=0,
        bmu=1,
        action_bmu=2,
        action_map=AMStub(),
        gaussian_shape=1.0,
        beta=0.37,
    )

    assert out is g
    assert recorded["beta"] == pytest.approx(0.37)


def test_ambiguity_counter_action_mode_updates_selected_action_only():
    g = Graph(directed=True)
    _register_features(g, mem_len=4, n_actions=3)
    g.add_node()
    g.node_features["ambiguity_score"][0, 1] = 5.0
    g.node_features["ambiguity_score_node"][0] = 7.0

    score = _update_ambiguity_counter(
        g,
        prev_bmu=0,
        action_bmu=1,
        preds_size=2,
        cfg=LatentParams(ambiguity_decay=0.5, ambiguity_counter_mode="action"),
    )

    assert score == pytest.approx(3.5)
    assert g.node_features["ambiguity_score"][0, 1] == pytest.approx(3.5)
    assert g.node_features["ambiguity_score"][0, 0] == pytest.approx(0.0)
    assert g.node_features["ambiguity_score_node"][0] == pytest.approx(7.0)


def test_ambiguity_counter_node_mode_updates_node_counter_only():
    g = Graph(directed=True)
    _register_features(g, mem_len=4, n_actions=3)
    g.add_node()
    g.node_features["ambiguity_score"][0, 1] = 5.0
    g.node_features["ambiguity_score_node"][0] = 7.0

    score = _update_ambiguity_counter(
        g,
        prev_bmu=0,
        action_bmu=1,
        preds_size=2,
        cfg=LatentParams(ambiguity_decay=0.5, ambiguity_counter_mode="node"),
    )

    assert score == pytest.approx(4.5)
    assert g.node_features["ambiguity_score_node"][0] == pytest.approx(4.5)
    assert g.node_features["ambiguity_score"][0, 1] == pytest.approx(5.0)
