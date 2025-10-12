# tests/test_adapter.py
import numpy as np
import pytest

from mercury.action_map.adapter import ActionMap
from mercury.action_map.som import SOMParams, som_predict, som_predict_batch

Array = np.ndarray


def test_adapter_random_and_step_and_predict_single():
    key = np.random.default_rng(42)
    am = ActionMap.random(n_codebook=6, dim=3, lr=0.4, sigma=0.0, key=key)

    x = np.array([0.2, -0.1, 0.3], dtype=np.float32)

    pre_idx = som_predict(am.params, am.state, x)
    idx, proto = am.step(x)
    assert isinstance(idx, (int, np.integer))
    assert proto.shape == (3,)
    assert idx == int(pre_idx)


def test_adapter_batch_predict_matches_functional():
    key = np.random.default_rng(0)
    am = ActionMap.random(n_codebook=5, dim=2, lr=0.2, sigma=0.0, key=key)

    X = np.array([[0.0, 0.0],
                  [1.0, 0.0],
                  [0.0, 1.0]], dtype=np.float32)

    idx_func = som_predict_batch(am.params, am.state, X)
    idx_adapt = am.predict(actions=X)
    assert np.array_equal(idx_func, idx_adapt)


def test_adapter_predict_argument_validation():
    key = np.random.default_rng(1)
    am = ActionMap.random(n_codebook=3, dim=2, key=key)

    x = np.array([0.0, 0.0], dtype=np.float32)
    X = np.stack([x, x], axis=0)

    _ = am.predict(action=x)
    _ = am.predict(actions=X)
    with pytest.raises(ValueError):
        am.predict()
    with pytest.raises(ValueError):
        am.predict(action=x, actions=X)
