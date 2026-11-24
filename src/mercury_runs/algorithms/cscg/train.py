from __future__ import annotations

import copy
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from .prepare import align_post_step_actions_to_cscg


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
    cscg_actions = align_post_step_actions_to_cscg(act_idx)
    if iteration_callback is not None:
        return train_cscg_model_with_callbacks(
            model,
            algo=algo,
            obs_idx=obs_idx,
            act_idx=cscg_actions,
            n_iter=n_iter,
            term_early=term_early,
            iteration_callback=iteration_callback,
            actions_aligned=True,
        )
    if algo == "em":
        em_convergence = np.asarray(
            model.learn_em_T(obs_idx, cscg_actions, n_iter=n_iter, term_early=term_early),
            dtype=np.float32,
        )
        return {"em": em_convergence}, ["em"]
    if algo == "viterbi":
        em_convergence = np.asarray(
            model.learn_em_T(obs_idx, cscg_actions, n_iter=n_iter, term_early=term_early),
            dtype=np.float32,
        )
        model.pseudocount = 0.0
        viterbi_convergence = np.asarray(model.learn_viterbi_T(obs_idx, cscg_actions, n_iter=n_iter), dtype=np.float32)
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
    actions_aligned: bool = False,
) -> tuple[dict[str, np.ndarray], list[str]]:
    cscg_actions = np.asarray(act_idx) if actions_aligned else align_post_step_actions_to_cscg(act_idx)
    previous = -np.inf
    em_convergence: list[float] = []
    for iteration in range(1, n_iter + 1):
        value = float(np.asarray(model.learn_em_T(obs_idx, cscg_actions, n_iter=1, term_early=False), dtype=np.float32)[-1])
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
        value = float(np.asarray(model.learn_viterbi_T(obs_idx, cscg_actions, n_iter=1), dtype=np.float32)[-1])
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
    cscg_actions = align_post_step_actions_to_cscg(act_idx)
    if algo == "em":
        score_name = "forward_bits_per_symbol"
        scores = model.bps(obs_idx, cscg_actions)
    elif algo == "viterbi":
        score_name = "viterbi_bits_per_symbol"
        scores = model.bpsV(obs_idx, cscg_actions)
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
    iterations_per_batch: int = 1,
    max_steps: int | None = None,
    initial_counts: np.ndarray | None = None,
    repeat_batches: bool = False,
    iteration_callback=None,
) -> np.ndarray:
    if iterations_per_batch < 1:
        raise ValueError("iterations_per_batch must be at least 1.")
    running_counts = (
        np.asarray(initial_counts, dtype=np.float32).copy()
        if initial_counts is not None
        else np.zeros_like(model.C, dtype=np.float32)
    )
    if running_counts.shape != model.C.shape:
        raise ValueError(f"initial_counts shape {running_counts.shape} does not match model.C shape {model.C.shape}.")
    em_convergence: list[float] = []
    previous = -np.inf
    processed_batches = 0
    processed_steps = 0
    step_limit = None if max_steps is None else max(1, int(max_steps))

    while processed_batches < int(max_batches) and (step_limit is None or processed_steps < step_limit):
        processed_batches_at_pass_start = processed_batches
        for batch in batch_iterator_factory():
            if processed_batches >= max_batches:
                break
            if step_limit is not None and processed_steps >= step_limit:
                break
            filtered_steps = int(batch["filtered_steps"])
            if filtered_steps == 0:
                continue
            obs_idx = map_observations(np.asarray(batch["observations"]))
            act_idx = align_post_step_actions_to_cscg(map_actions(np.asarray(batch["actions"])))
            if obs_idx.shape[0] < 2 or act_idx.shape[0] < 2:
                processed_steps += filtered_steps
                continue

            batch_model = model if iterations_per_batch == 1 else copy.deepcopy(model)
            for local_iteration in range(iterations_per_batch):
                log2_lik, mess_fwd = chmm_mod.forward(
                    batch_model.T.transpose(0, 2, 1),
                    batch_model.Pi_x,
                    batch_model.n_clones,
                    obs_idx,
                    act_idx,
                    store_messages=True,
                )
                mess_bwd = chmm_mod.backward(batch_model.T, batch_model.n_clones, obs_idx, act_idx)
                batch_counts = np.zeros_like(model.C, dtype=np.float32)
                chmm_mod.updateC(batch_counts, batch_model.T, batch_model.n_clones, mess_fwd, mess_bwd, obs_idx, act_idx)
                if local_iteration + 1 < iterations_per_batch:
                    batch_model.C = batch_counts.astype(batch_model.dtype, copy=True)
                    batch_model.update_T()

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
            if step_limit is not None and processed_steps >= step_limit:
                return np.asarray(em_convergence, dtype=np.float32)
            if term_early and value <= previous:
                return np.asarray(em_convergence, dtype=np.float32)
            previous = value
        if not repeat_batches or processed_batches == processed_batches_at_pass_start:
            break

    return np.asarray(em_convergence, dtype=np.float32)


def viterbi_refined_eval_model_npz(
    model: Any,
    *,
    obs_idx: np.ndarray,
    act_idx: np.ndarray,
    n_clones: np.ndarray,
    n_iter: int,
    pseudocount: float = 0.0,
) -> dict[str, np.ndarray]:
    eval_model = copy.deepcopy(model)
    eval_model.pseudocount = float(pseudocount)
    cscg_actions = align_post_step_actions_to_cscg(act_idx)
    eval_model.learn_viterbi_T(
        np.asarray(obs_idx, dtype=np.int64),
        np.asarray(cscg_actions, dtype=np.int64),
        n_iter=int(n_iter),
    )
    return {
        "T": np.asarray(eval_model.T).copy(),
        "n_clones": np.asarray(n_clones, dtype=np.int64).copy(),
    }
