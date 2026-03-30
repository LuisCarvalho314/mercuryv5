# tests/test_som_numpy.py
import numpy as np

from mercury.action_map.som import (
    SOMParams, SOMState,
    som_init, som_init_zeros,
    som_predict, som_predict_batch,
    som_update_one, som_epoch,
)

Array = np.ndarray


def test_init_shapes_and_dtypes():
    p = SOMParams(n_codebook=8, dim=4, lr=0.3, sigma=0.0)
    s0 = som_init_zeros(p)
    assert s0.codebook.shape == (8, 4)
    assert s0.codebook.dtype == np.float32
    assert isinstance(s0.bmu, (int, np.integer))

    rng = np.random.default_rng(0)
    s1 = som_init(p, rng)
    assert s1.codebook.shape == (8, 4)
    assert (s1.codebook != 0).any()


def test_predict_single_and_batch_consistency():
    p = SOMParams(n_codebook=5, dim=3, lr=0.1, sigma=0.0)
    # put distinct rows to make BMU deterministic
    codebook = np.array(
        [[0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0],
         [1.0, 1.0, 0.0]], dtype=np.float32
    )
    s = SOMState(codebook=codebook, bmu=0)

    x = np.array([0.9, 0.05, 0.0], dtype=np.float32)
    bmu_single = som_predict(p, s, x)
    assert int(bmu_single) == 1

    X = np.stack([x, np.array([0.0, 0.9, 0.1], np.float32)], axis=0)
    bmu_batch = som_predict_batch(p, s, X)
    assert bmu_batch.shape == (2,)
    assert int(bmu_batch[0]) == 1
    assert int(bmu_batch[1]) == 2


def test_update_moves_winner_toward_input():
    p = SOMParams(n_codebook=3, dim=2, lr=0.5, sigma=0.0)
    rng = np.random.default_rng(0)
    s = som_init(p, rng)

    x = np.array([1.0, -1.0], dtype=np.float32)# tests/test_som_numpy.py
import numpy as np

from mercury.action_map.som import (
    SOMParams, SOMState,
    som_init, som_init_zeros,
    som_predict, som_predict_batch,
    som_update_one, som_epoch,
)

Array = np.ndarray


def test_init_shapes_and_dtypes():
    p = SOMParams(n_codebook=8, dim=4, lr=0.3, sigma=0.0)
    s0 = som_init_zeros(p)
    assert s0.codebook.shape == (8, 4)
    assert s0.codebook.dtype == np.float32
    assert isinstance(s0.bmu, (int, np.integer))

    key = np.random.default_rng(0)
    s1 = som_init(p, key)
    assert s1.codebook.shape == (8, 4)
    assert (s1.codebook != 0).any()


def test_predict_single_and_batch_consistency():
    p = SOMParams(n_codebook=5, dim=3, lr=0.1, sigma=0.0)
    codebook = np.array(
        [[0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0],
         [1.0, 1.0, 0.0]], dtype=np.float32
    )
    s = SOMState(codebook=codebook, bmu=0)

    x = np.array([0.9, 0.05, 0.0], dtype=np.float32)
    bmu_single = som_predict(p, s, x)
    assert int(bmu_single) == 1

    X = np.stack([x, np.array([0.0, 0.9, 0.1], np.float32)], axis=0)
    bmu_batch = som_predict_batch(p, s, X)
    assert bmu_batch.shape == (2,)
    assert int(bmu_batch[0]) == 1
    assert int(bmu_batch[1]) == 2


def test_update_moves_winner_toward_input():
    p = SOMParams(n_codebook=3, dim=2, lr=0.5, sigma=0.0)
    key = np.random.default_rng(0)
    s = som_init(p, key)

    x = np.array([1.0, -1.0], dtype=np.float32)
    pre_bmu = som_predict(p, s, x)
    pre_w = s.codebook[pre_bmu].copy()

    s2, post_bmu = som_update_one(p, s, x)
    assert int(pre_bmu) == int(post_bmu)

    post_w = s2.codebook[post_bmu]
    pre_d2 = float(np.sum((x - pre_w) ** 2))
    post_d2 = float(np.sum((x - post_w) ** 2))
    assert post_d2 < pre_d2


def test_sigma_neighborhood_updates_multiple_units():
    p = SOMParams(n_codebook=5, dim=1, lr=1.0, sigma=1.0)
    s = som_init_zeros(p)
    x = np.array([2.0], dtype=np.float32)

    s2, bmu = som_update_one(p, s, x)
    assert float(s2.codebook[bmu, 0]) > 0.0
    left = max(0, int(bmu) - 1)
    right = min(p.n_codebook - 1, int(bmu) + 1)
    neighborhood_sum = float(np.sum(s2.codebook[left:right+1, 0]))
    assert neighborhood_sum > float(s2.codebook[bmu, 0])


def test_epoch_scans_batch_and_reduces_loss():
    p = SOMParams(n_codebook=4, dim=2, lr=0.3, sigma=0.0)
    key = np.random.default_rng(0)
    s = som_init(p, key)
    X = np.array([[1.0, 1.0],
                  [0.9, 1.1],
                  [1.1, 0.9],
                  [1.0, 0.95]], dtype=np.float32)

    def avg_min_d2(state):
        return float(np.mean([np.min(np.sum((state.codebook - x) ** 2, axis=1)) for x in X]))

    pre = avg_min_d2(s)
    s2 = som_epoch(p, s, X)
    post = avg_min_d2(s2)
    assert post < pre

    pre_bmu = som_predict(p, s, x)
    pre_w = s.codebook[pre_bmu].copy()

    s2, post_bmu = som_update_one(p, s, x)
    assert int(pre_bmu) == int(post_bmu)

    post_w = s2.codebook[post_bmu]
    pre_d2 = float(np.sum((x - pre_w) ** 2))
    post_d2 = float(np.sum((x - post_w) ** 2))
    assert post_d2 < pre_d2


def test_sigma_neighborhood_updates_multiple_units():
    p = SOMParams(n_codebook=5, dim=1, lr=1.0, sigma=1.0)  # wide neighborhood
    s = som_init_zeros(p)
    x = np.array([2.0], dtype=np.float32)

    s2, bmu = som_update_one(p, s, x)
    # winner positive
    assert float(s2.codebook[bmu, 0]) > 0.0
    # at least one neighbor also updated
    left = max(0, int(bmu) - 1)
    right = min(p.n_codebook - 1, int(bmu) + 1)
    neighborhood_sum = float(np.sum(s2.codebook[left:right+1, 0]))
    assert neighborhood_sum > float(s2.codebook[bmu, 0])  # neighbors contributed


def test_epoch_scans_batch_and_reduces_loss():
    p = SOMParams(n_codebook=4, dim=2, lr=0.3, sigma=0.0)
    rng = np.random.default_rng(0)
    s = som_init(p, rng)
    # synthetic cluster around (1,1)
    X = np.array([[1.0, 1.0],
                  [0.9, 1.1],
                  [1.1, 0.9],
                  [1.0, 0.95]], dtype=np.float32)

    def avg_min_d2(state):
        # min squared distance from each x to any prototype
        d2 = []
        for x in X:
            d2.append(np.min(np.sum((state.codebook - x) ** 2, axis=1)))
        return float(np.mean(d2))

    pre = avg_min_d2(s)
    s2 = som_epoch(p, s, X)
    post = avg_min_d2(s2)
    assert post < pre
