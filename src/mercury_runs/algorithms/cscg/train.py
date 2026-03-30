from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np


def train_cscg_model(
    model: Any,
    *,
    algo: str,
    obs_idx: np.ndarray,
    act_idx: np.ndarray,
    n_iter: int,
    term_early: bool,
    iteration_callback=None,
) -> tuple[dict[str, np.ndarray], list[str]]:
    if iteration_callback is not None:
        return train_cscg_model_with_callbacks(
            model,
            algo=algo,
            obs_idx=obs_idx,
            act_idx=act_idx,
            n_iter=n_iter,
            term_early=term_early,
            iteration_callback=iteration_callback,
        )
    if algo == "em":
        em_convergence = np.asarray(
            model.learn_em_T(obs_idx, act_idx, n_iter=n_iter, term_early=term_early),
            dtype=np.float32,
        )
        return {"em": em_convergence}, ["em"]
    if algo == "viterbi":
        em_convergence = np.asarray(
            model.learn_em_T(obs_idx, act_idx, n_iter=n_iter, term_early=term_early),
            dtype=np.float32,
        )
        model.pseudocount = 0.0
        viterbi_convergence = np.asarray(model.learn_viterbi_T(obs_idx, act_idx, n_iter=n_iter), dtype=np.float32)
        return {"em": em_convergence, "viterbi": viterbi_convergence}, ["em", "viterbi"]
    raise ValueError(f"Unsupported CSCG train_algorithm: {algo}")


def train_cscg_model_with_callbacks(
    model: Any,
    *,
    algo: str,
    obs_idx: np.ndarray,
    act_idx: np.ndarray,
    n_iter: int,
    term_early: bool,
    iteration_callback,
) -> tuple[dict[str, np.ndarray], list[str]]:
    previous = -np.inf
    em_convergence: list[float] = []
    for iteration in range(1, n_iter + 1):
        value = float(np.asarray(model.learn_em_T(obs_idx, act_idx, n_iter=1, term_early=False), dtype=np.float32)[-1])
        em_convergence.append(value)
        iteration_callback(stage="em", iteration=iteration, model=model, step=int(obs_idx.shape[0]) * iteration)
        if value <= previous and term_early:
            break
        previous = value
    if algo == "em":
        return {"em": np.asarray(em_convergence, dtype=np.float32)}, ["em"]
    if algo != "viterbi":
        raise ValueError(f"Unsupported CSCG train_algorithm: {algo}")
    model.pseudocount = 0.0
    previous = -np.inf
    viterbi_convergence: list[float] = []
    for iteration in range(1, n_iter + 1):
        value = float(np.asarray(model.learn_viterbi_T(obs_idx, act_idx, n_iter=1), dtype=np.float32)[-1])
        viterbi_convergence.append(value)
        iteration_callback(stage="viterbi", iteration=iteration, model=model, step=int(obs_idx.shape[0]) * iteration)
        if value <= previous:
            break
        previous = value
    return {
        "em": np.asarray(em_convergence, dtype=np.float32),
        "viterbi": np.asarray(viterbi_convergence, dtype=np.float32),
    }, ["em", "viterbi"]


def objective_scores(model: Any, algo: str, obs_idx: np.ndarray, act_idx: np.ndarray) -> tuple[str, np.ndarray]:
    if algo == "em":
        score_name = "forward_bits_per_symbol"
        scores = model.bps(obs_idx, act_idx)
    elif algo == "viterbi":
        score_name = "viterbi_bits_per_symbol"
        scores = model.bpsV(obs_idx, act_idx)
    else:
        raise ValueError(f"Unsupported CSCG train_algorithm: {algo}")
    return score_name, np.asarray(scores, dtype=np.float64)


def run_cscg_online_em(
    model: Any,
    *,
    chmm_mod: Any,
    batch_iterator_factory: Callable[[], Iterable[dict[str, Any]]],
    map_observations: Callable[[np.ndarray], np.ndarray],
    map_actions: Callable[[np.ndarray], np.ndarray],
    online_lambda: float,
    max_batches: int,
    term_early: bool,
    iteration_callback=None,
) -> np.ndarray:
    running_counts = np.zeros_like(model.C, dtype=np.float32)
    em_convergence: list[float] = []
    previous = -np.inf
    processed_batches = 0
    processed_steps = 0

    for batch in batch_iterator_factory():
        if processed_batches >= max_batches:
            break
        filtered_steps = int(batch["filtered_steps"])
        if filtered_steps == 0:
            continue
        obs_idx = map_observations(np.asarray(batch["observations"]))
        act_idx = map_actions(np.asarray(batch["actions"]))
        if obs_idx.shape[0] < 2 or act_idx.shape[0] < 2:
            processed_steps += filtered_steps
            continue

        log2_lik, mess_fwd = chmm_mod.forward(
            model.T.transpose(0, 2, 1),
            model.Pi_x,
            model.n_clones,
            obs_idx,
            act_idx,
            store_messages=True,
        )
        mess_bwd = chmm_mod.backward(model.T, model.n_clones, obs_idx, act_idx)
        batch_counts = np.zeros_like(model.C, dtype=np.float32)
        chmm_mod.updateC(batch_counts, model.T, model.n_clones, mess_fwd, mess_bwd, obs_idx, act_idx)

        if np.isclose(online_lambda, 1.0):
            running_counts += batch_counts
        else:
            running_counts *= float(online_lambda)
            running_counts += (1.0 - float(online_lambda)) * batch_counts

        model.C = running_counts.astype(model.dtype, copy=True)
        model.update_T()
        processed_batches += 1
        processed_steps += filtered_steps
        value = float((-np.asarray(log2_lik, dtype=np.float32)).mean())
        em_convergence.append(value)
        if iteration_callback is not None:
            iteration_callback(stage="em", iteration=processed_batches, model=model, step=processed_steps)
        if term_early and value <= previous:
            break
        previous = value

    return np.asarray(em_convergence, dtype=np.float32)
