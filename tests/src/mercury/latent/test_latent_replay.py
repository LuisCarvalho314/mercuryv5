import numpy as np
import pytest

import mercury.latent.state as latent_state_mod
from mercury.graph.core import Graph
from mercury.latent.params import LatentParams
from mercury.latent.state import (
    LatentState,
    ReplayWindow,
    _build_replay_window,
    _reconstruct_replay_transitions,
    _refine_ambiguous_replay_contexts,
    _register_features,
    _resolve_prev_with_memory_replay,
    _set_activation,
    _update_edges_with_actions,
    latent_step,
)


class _StubGS:
    def __init__(self, n: int):
        self.n = n


class StubMSFull:
    def __init__(self, activ_mat: np.ndarray):
        assert activ_mat.ndim == 2
        sensory_count, length = activ_mat.shape
        self.length = length
        self.sensory_n_nodes = sensory_count
        self.gs = _StubGS(n=sensory_count * length)
        self.activations = activ_mat.reshape(sensory_count * length).astype(np.float32)


class StubActionMap:
    def __init__(self, dim: int, n_codes: int):
        self.state = type("S", (), {})()
        self.state.codebook = np.eye(n_codes, dtype=np.float32)
        self._params = type("P", (), {"dim": dim})()

    @property
    def params(self):
        return self._params


def _make_graph_with_nodes(mem_len: int, n_lat: int, n_actions: int = 1) -> Graph:
    g = Graph(directed=True)
    _register_features(g, mem_len, n_actions=n_actions)
    for _ in range(n_lat):
        g.add_node(
            node_feat={
                "activation": np.array(0.0, dtype=np.float32),
                "mem_adj": np.zeros(mem_len, dtype=np.float32),
                "first_incoming_at": np.array(-1, dtype=np.int32),
                "first_outgoing_at": np.array(-1, dtype=np.int32),
            },
            edge_defaults_for_new_node={
                "age": np.array([0], dtype=np.int32),
                "action": np.array([0], dtype=np.int32),
            },
        )
    return g


def _cfg(**overrides) -> LatentParams:
    values = {
        "gaussian_shape": 2,
        "action_lr": 0.37,
        "memory_replay": True,
        "memory_disambiguation": True,
        "ambiguity_threshold": 10,
        "max_age": 9999,
    }
    values.update(overrides)
    return LatentParams(**values)


def test_build_replay_window_retains_observations_and_actions():
    ms = StubMSFull(np.zeros((1, 3), dtype=np.float32))
    mem_vec = [np.array([i], dtype=np.float32) for i in range(8)]
    action_mem = [4, 5, 6, 7]

    window = _build_replay_window(ms, mem_vec, action_mem)

    assert window.length == 3
    np.testing.assert_allclose(window.observations.reshape(6, 1), np.arange(2, 8).reshape(6, 1))
    np.testing.assert_array_equal(window.actions, np.array([5, 6, 7], dtype=np.int32))


def test_build_replay_window_rejects_short_history():
    ms = StubMSFull(np.zeros((1, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="Replay requires 6 observations"):
        _build_replay_window(ms, [np.array([1]), np.array([2])], [0, 1, 2])

    with pytest.raises(ValueError, match="Replay requires 3 actions"):
        _build_replay_window(
            ms,
            [np.array([i], dtype=np.float32) for i in range(6)],
            [0, 1],
        )


def test_refine_ambiguous_replay_contexts_advances_and_counts_clones(monkeypatch):
    ms = StubMSFull(np.zeros((1, 3), dtype=np.float32))
    window = ReplayWindow(
        length=3,
        observations=np.array([[0], [1], [2], [3], [4], [5]], dtype=np.float32),
        actions=np.array([10, 11, 12], dtype=np.int32),
    )
    replay_state = LatentState(
        g=_make_graph_with_nodes(mem_len=3, n_lat=2),
        mapping=np.arange(2, dtype=np.int32),
    )
    calls = {"infer": [], "add": []}
    inferred = iter([0, 1, 0])

    monkeypatch.setattr(latent_state_mod, "_initialise_replay_memory", lambda *_: "mem0")

    def fake_infer(replay_state, replay_mem, action_bmu, cfg, *, use_temporal_memory):
        calls["infer"].append((replay_mem, action_bmu, use_temporal_memory))
        return next(inferred)

    def fake_update_memory(replay_mem):
        return f"{replay_mem}|shift"

    def fake_add_memory(replay_mem, observation):
        calls["add"].append(int(observation[0]))
        return f"{replay_mem}|obs{int(observation[0])}"

    def fake_clone(g, replay_mem, deficient_bmu, preds, *, created_step):
        calls["add"].append(("clone", replay_mem, deficient_bmu, created_step))
        return g, deficient_bmu + 10, True

    monkeypatch.setattr(latent_state_mod, "_infer_replay_bmu", fake_infer)
    monkeypatch.setattr(latent_state_mod, "update_memory", fake_update_memory)
    monkeypatch.setattr(latent_state_mod, "add_memory", fake_add_memory)
    monkeypatch.setattr(latent_state_mod, "add_temporal_edge_at_first_gap_clone", fake_clone)

    clone_count = _refine_ambiguous_replay_contexts(
        replay_state=replay_state,
        ms=ms,
        window=window,
        deficient_bmu=0,
        preds=np.array([1, 2], dtype=np.int32),
        cfg=_cfg(memory_disambiguation=False),
        created_step=42,
    )

    assert clone_count == 2
    assert calls["infer"] == [
        ("mem0", 10, False),
        ("mem0|shift|obs3", 11, False),
        ("mem0|shift|obs3|shift|obs4", 12, False),
    ]
    assert calls["add"] == [
        ("clone", "mem0", 0, 42),
        3,
        4,
        ("clone", "mem0|shift|obs3|shift|obs4", 0, 42),
        5,
    ]


def test_reconstruct_replay_transitions_trains_historical_edges_only(monkeypatch):
    ms = StubMSFull(np.zeros((1, 3), dtype=np.float32))
    window = ReplayWindow(
        length=3,
        observations=np.array([[0], [1], [2], [3], [4], [5]], dtype=np.float32),
        actions=np.array([20, 21, 22], dtype=np.int32),
    )
    replay_state = LatentState(
        g=_make_graph_with_nodes(mem_len=3, n_lat=4),
        mapping=np.arange(4, dtype=np.int32),
    )
    inferred = iter([10, 11, 12, 13])
    edges = []

    monkeypatch.setattr(latent_state_mod, "_initialise_replay_memory", lambda *_: "mem0")
    monkeypatch.setattr(latent_state_mod, "update_memory", lambda replay_mem: f"{replay_mem}|shift")
    monkeypatch.setattr(
        latent_state_mod,
        "add_memory",
        lambda replay_mem, observation: f"{replay_mem}|obs{int(observation[0])}",
    )
    monkeypatch.setattr(
        latent_state_mod,
        "_infer_replay_bmu",
        lambda *args, **kwargs: next(inferred),
    )

    def fake_update_edges(g, **kwargs):
        edges.append(
            (
                kwargs["prev_bmu"],
                kwargs["bmu"],
                kwargs["action_bmu"],
                kwargs["gaussian_shape"],
                kwargs["beta"],
                kwargs["created_step"],
            )
        )
        return g

    monkeypatch.setattr(latent_state_mod, "_update_edges_with_actions", fake_update_edges)

    resolved_prev, resolved_bmu = _reconstruct_replay_transitions(
        replay_state=replay_state,
        ms=ms,
        window=window,
        action_map=StubActionMap(dim=3, n_codes=30),
        cfg=_cfg(gaussian_shape=5, action_lr=0.25),
        created_step=99,
    )

    assert (resolved_prev, resolved_bmu) == (12, 13)
    assert edges == [
        (10, 11, 20, 5, 0.25, 99),
        (11, 12, 21, 5, 0.25, 99),
    ]


def test_resolve_prev_with_memory_replay_early_returns_before_window_ready():
    ms = StubMSFull(np.zeros((1, 3), dtype=np.float32))
    g = _make_graph_with_nodes(mem_len=3, n_lat=2)
    state = LatentState(g=g, mapping=np.arange(2), prev_bmu=0, step_idx=5)

    g_out, resolved_prev, resolved_bmu, clone_count = _resolve_prev_with_memory_replay(
        state=state,
        ms=ms,
        mem_vec=[],
        prev_bmu=0,
        bmu=1,
        action_mem=[],
        action_map=StubActionMap(dim=3, n_codes=4),
        cfg=_cfg(),
        preds=np.array([1], dtype=np.int32),
    )

    assert g_out is not g
    assert resolved_prev == 0
    assert resolved_bmu == 1
    assert clone_count == 0


def test_resolve_prev_refines_without_reconstructing_or_reassigning(monkeypatch):
    ms = StubMSFull(np.zeros((1, 2), dtype=np.float32))
    state = LatentState(
        g=_make_graph_with_nodes(mem_len=2, n_lat=3),
        mapping=np.arange(3),
        prev_bmu=1,
        step_idx=4,
    )
    calls = {"refine": 0, "reconstruct": 0}

    def fake_refine(*args, **kwargs):
        calls["refine"] += 1
        return 1

    def fail_reconstruct(*args, **kwargs):
        calls["reconstruct"] += 1
        raise AssertionError("transition reconstruction must be disabled")

    monkeypatch.setattr(latent_state_mod, "_refine_ambiguous_replay_contexts", fake_refine)
    monkeypatch.setattr(latent_state_mod, "_reconstruct_replay_transitions", fail_reconstruct)

    _, resolved_prev, resolved_bmu, clone_count = _resolve_prev_with_memory_replay(
        state=state,
        ms=ms,
        mem_vec=[np.array([i], dtype=np.float32) for i in range(4)],
        prev_bmu=1,
        bmu=2,
        action_mem=[0, 1],
        action_map=StubActionMap(dim=2, n_codes=2),
        cfg=_cfg(transition_reconstruction_enabled=False),
        preds=np.array([0, 2], dtype=np.int32),
    )

    assert calls == {"refine": 1, "reconstruct": 0}
    assert (resolved_prev, resolved_bmu, clone_count) == (1, 2, 1)
    assert state.replay_inference_count == 2
    assert state.replay_context_count == 0
    assert state.reconstructed_transition_count == 0


def test_resolve_prev_local_repair_reconstructs_without_refining(monkeypatch):
    ms = StubMSFull(np.zeros((1, 2), dtype=np.float32))
    state = LatentState(
        g=_make_graph_with_nodes(mem_len=2, n_lat=3),
        mapping=np.arange(3),
        prev_bmu=1,
        step_idx=4,
    )

    monkeypatch.setattr(
        latent_state_mod,
        "_refine_ambiguous_replay_contexts",
        lambda *args, **kwargs: pytest.fail("context refinement must be disabled"),
    )
    monkeypatch.setattr(
        latent_state_mod,
        "_reconstruct_replay_transitions",
        lambda *args, **kwargs: (0, 2),
    )

    _, resolved_prev, resolved_bmu, clone_count = _resolve_prev_with_memory_replay(
        state=state,
        ms=ms,
        mem_vec=[np.array([i], dtype=np.float32) for i in range(4)],
        prev_bmu=1,
        bmu=2,
        action_mem=[0, 1],
        action_map=StubActionMap(dim=2, n_codes=2),
        cfg=_cfg(ambiguity_refinement_enabled=False),
        preds=np.array([0, 2], dtype=np.int32),
    )

    assert (resolved_prev, resolved_bmu, clone_count) == (0, 2, 0)
    assert state.replay_inference_count == 3
    assert state.replay_context_count == 3
    assert state.reconstructed_transition_count == 1


def test_update_edges_with_actions_uses_passed_beta(monkeypatch):
    g = _make_graph_with_nodes(mem_len=3, n_lat=2)
    action_map = StubActionMap(dim=3, n_codes=3)
    recorded: dict[str, float] = {}

    def fake_update_edge(g_in, u, v, prior_row, observed_row, action_bmu, beta, gaussian_shape, created_step=0):
        recorded["beta"] = float(beta)
        recorded["created_step"] = int(created_step)
        return g_in

    monkeypatch.setattr(latent_state_mod, "_update_edge", fake_update_edge)

    out = _update_edges_with_actions(
        g=g,
        prev_bmu=0,
        bmu=1,
        action_bmu=2,
        action_map=action_map,
        gaussian_shape=1.5,
        beta=0.37,
        created_step=8,
    )

    assert out is g
    assert recorded == {"beta": pytest.approx(0.37), "created_step": 8}


def test_update_edge_records_first_transition_support_once():
    g = _make_graph_with_nodes(mem_len=3, n_lat=2)
    zeros = np.zeros((2,), dtype=np.float32)
    observed = np.array([1.0, 0.0], dtype=np.float32)

    latent_state_mod._update_edge(
        g,
        0,
        1,
        zeros,
        observed,
        action_bmu=0,
        beta=0.5,
        gaussian_shape=2,
        created_step=8,
    )
    latent_state_mod._update_edge(
        g,
        0,
        1,
        observed,
        observed,
        action_bmu=0,
        beta=0.5,
        gaussian_shape=2,
        created_step=12,
    )

    assert int(g.node_features["first_outgoing_at"][0]) == 8
    assert int(g.node_features["first_incoming_at"][1]) == 8


def test_latent_step_uses_replay_boundary_and_trains_live_edge(monkeypatch):
    g = _make_graph_with_nodes(mem_len=4, n_lat=3, n_actions=2)
    state = LatentState(g=g, mapping=np.arange(g.n), prev_bmu=1, step_idx=7)
    ms = StubMSFull(np.array([[1, 0], [0, 1]], dtype=np.float32))
    action_map = StubActionMap(dim=2, n_codes=2)
    state_mem = [1]
    calls = {"resolve": [], "edges": []}

    monkeypatch.setattr(latent_state_mod, "compute_bmu", lambda *_args, **_kwargs: 2)
    monkeypatch.setattr(latent_state_mod, "_predict", lambda *_args, **_kwargs: np.array([0, 2], dtype=np.int32))
    monkeypatch.setattr(latent_state_mod, "_remove_aliased_connections", lambda g, *_args, **_kwargs: g)
    monkeypatch.setattr(
        latent_state_mod,
        "age_maintenance",
        lambda prev_bmu, bmu, g, cfg, exclude, return_stats: (
            g,
            np.arange(g.n, dtype=np.int32),
            type("Stats", (), {"edge_prune_count": 0, "node_prune_count": 0})(),
        ),
    )

    def fake_resolve_prev(**kwargs):
        calls["resolve"].append((kwargs["prev_bmu"], kwargs["bmu"]))
        return kwargs["state"].g, 0, 2, 1

    def fake_update_edges(g, **kwargs):
        calls["edges"].append(
            (
                kwargs["prev_bmu"],
                kwargs["bmu"],
                kwargs["action_bmu"],
                kwargs["beta"],
                kwargs["created_step"],
            )
        )
        return g

    monkeypatch.setattr(latent_state_mod, "_resolve_prev_with_memory_replay", fake_resolve_prev)
    monkeypatch.setattr(latent_state_mod, "_update_edges_with_actions", fake_update_edges)

    state_out, bmu_now, state_mem_out = latent_step(
        ms=ms,
        mem_vec=[np.array([0, 1], dtype=np.float32)] * 8,
        state=state,
        action_bmu=1,
        cfg=_cfg(ambiguity_threshold=-1, action_lr=0.5),
        action_map=action_map,
        action_mem=[0, 1, 1],
        state_mem=state_mem,
    )

    assert bmu_now == 2
    assert state_out.prev_bmu == 2
    assert state_out.step_idx == 8
    assert state_out.clone_created_count == 1
    assert state_out.ambiguity_response_count == 1
    assert state_mem_out[-2:] == [0, 2]
    assert calls["resolve"] == [(1, 2)]
    assert calls["edges"] == [(0, 2, 1, 0.5, 7)]


def test_set_activation_sets_single_hot_in_place():
    g = _make_graph_with_nodes(mem_len=6, n_lat=3)
    g2 = _set_activation(g, bmu=1)

    assert g2 is g
    vec = g.node_features["activation"]
    assert vec.dtype == np.float32
    assert vec.shape == (3,)
    assert np.count_nonzero(vec) == 1
    assert vec[1] == 1.0
    assert vec[0] == 0.0 and vec[2] == 0.0
