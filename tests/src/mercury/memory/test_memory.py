# tests/test_memory_network.py
import numpy as np
import pytest

from mercury.memory.state import (
    init_mem,
    add_memory,
    update_memory,
    mem_id,
    activations_at_t,
    MemoryState,
    memory_view_at_global_timestep,  # new
)


# --- minimal sensory stub ---
class _DummyGS:
    def __init__(self, n: int):
        self.n = n

class _DummySensory:
    def __init__(self, n_sensory: int):
        self.gs = _DummyGS(n_sensory)

def _has_edge(u, v, A):  # u->v
    return bool(A[u, v] != 0)


def test_init_state_bank_and_edges():
    S, L = 3, 4
    ss = _DummySensory(S)
    ms: MemoryState = init_mem(ss, length=L)

    assert ms.gs.n == S * L
    A = ms.gs.adj
    assert int(np.count_nonzero(A)) == S * (L - 1)

    for s in range(S):
        base = s * L
        for t in range(1, L):
            assert _has_edge(base + t, base + (t - 1), A)
        for t in range(L):
            nz = set(np.nonzero(A[base + t])[0].tolist())
            expect = {base + (t - 1)} if t > 0 else set()
            assert nz == expect


def test_add_memory_updates_only_last_nodes_and_validates_length():
    S, L = 3, 5
    ss = _DummySensory(S)
    ms: MemoryState = init_mem(ss, length=L)
    n = ms.gs.n
    A = ms.gs.adj

    # seed baseline activations to verify non-last entries stay the same
    base = np.arange(n, dtype=A.dtype) * 0.1
    ms.gs.set_node_feat("activation", base.copy())
    ms0 = MemoryState(ms.gs, L, n)

    # memory length must be S = n // L
    mem = np.array([10.0, 20.0, 30.0], dtype=A.dtype)  # shape (S,)
    ms1 = add_memory(ms0, mem)

    last_idx = np.arange(S, dtype=np.int32) * L + (L - 1)
    act1 = ms1.gs.node_features["activation"]

    # last positions set to mem
    assert np.allclose(act1[last_idx], mem)
    # all other positions unchanged
    mask = np.ones((n,), dtype=bool)
    mask[last_idx] = False
    assert np.allclose(act1[mask], base[mask])

    # wrong length raises
    with pytest.raises(ValueError):
        add_memory(ms0, np.ones((S - 1,), dtype=A.dtype))


def test_update_memory_matches_matmul_and_moves_back_one():
    S, L = 2, 3
    ss = _DummySensory(S)
    ms: MemoryState = init_mem(ss, length=L)
    A = ms.gs.adj

    # write only into last nodes (length S)
    mem = np.array([1.0, 2.0], dtype=A.dtype)
    ms = add_memory(ms, mem)

    # one-step propagation equals act @ A
    act0 = ms.gs.node_features["activation"]
    expected = act0 @ A
    ms1 = update_memory(ms)
    assert np.allclose(ms1.gs.node_features["activation"], expected)

    # explicit check: from t=L-1 to t=L-2
    for s in range(S):
        assert ms1.gs.node_features["activation"][mem_id(s, L - 2, L)] == mem[s]


def test_mem_id_linear_mapping():
    S, L = 4, 3
    for s in range(S):
        for t in range(L):
            assert mem_id(s, t, L) == s * L + t


def test_update_memory_matches_matmul_and_moves_back_one1():
    S, L = 2, 3
    ss = _DummySensory(S)
    ms: MemoryState = init_mem(ss, length=L)
    A = ms.gs.adj

    # write only into last nodes (length S)
    mem = np.array([1.0, 2.0], dtype=A.dtype)
    ms = add_memory(ms, mem)

    # one-step propagation equals act @ A
    act0 = ms.gs.node_features["activation"]
    expected = act0 @ A
    ms1 = update_memory(ms)
    assert np.allclose(ms1.gs.node_features["activation"], expected)

    # explicit check: from t=L-1 to t=L-2 across all strips
    assert np.array_equal(activations_at_t(ms1, 1), mem)

@pytest.mark.parametrize("k", [0, 1, 3])
def test_memory_view_at_global_timestep_alignment_and_padding(k):
    # set up deterministic memory state
    S, L = 3, 4
    ss = _DummySensory(S)
    ms: MemoryState = init_mem(ss, length=L)

    # assign activation pattern:
    # strip s, timestep t -> 10*s + t
    # so:
    # strip 0: [0, 1, 2, 3]
    # strip 1: [10,11,12,13]
    # strip 2: [20,21,22,23]
    mem_matrix = np.zeros((S, L), dtype=np.float32)
    for s in range(S):
        for t in range(L):
            mem_matrix[s, t] = 10 * s + t
    ms.gs.set_node_feat("activation", mem_matrix.reshape(S * L))

    # call function under test
    view_state = memory_view_at_global_timestep(ms, k)

    # return type should be MemoryState
    assert isinstance(view_state, MemoryState)

    # metadata should be preserved
    assert view_state.length == L
    assert view_state.sensory_n_nodes == S
    assert view_state.gs is ms.gs  # same graph object by design

    # read back activation from returned state
    act_flat = view_state.gs.node_features["activation"]
    assert act_flat.shape == (S * L,)
    act_view = act_flat.reshape(S, L)

    # build expected matrix:
    # column 0 in expected = original column k
    # column 1 in expected = original column k+1
    # columns past L-1 are zero
    expected = np.zeros((S, L), dtype=np.float32)
    remaining_cols = L - k
    if remaining_cols > 0:
        expected[:, :remaining_cols] = mem_matrix[:, k:]

    # numeric check
    assert np.allclose(act_view, expected), (
        f"unexpected reindexed view for k={k}\n"
        f"got:\n{act_view}\n"
        f"expected:\n{expected}"
    )

