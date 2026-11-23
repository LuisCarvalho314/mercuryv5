"""Exploratory transCSSR fit on existing canonical evaluation walks.

Run from the repository root with transCSSR installed in an isolated environment;
see README.md. This script only reads evaluation caches and writes pilot results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score
from transcssr import infer_transducer

from src.mercury_runs.application.evaluation_service import _load_walk_cache
from src.mercury_runs.infrastructure.paper_precision import (
    action_ids_from_action_vectors,
    compute_fixed_map_precision_details,
    exact_cartesian_reference_positions,
    validate_evaluation_walk_action_alignment,
)


ROOT = Path(__file__).resolve().parents[3]
CACHE_ROOT = ROOT / "results/current/studies/hybrid_cscg_full/evaluation_caches/canonical-v4-8922ca1c45c19cc7"
RESULT_ROOT = ROOT / "results/current/transcssr_pilot"
REVISION = "23a53ba21101cf26bdfcb83570d60bb6147d07ba"
L_VALUES = (3, 5, 10)
LENGTHS = (2_000, 5_000, 10_000)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _tokens(rows: np.ndarray, vocabulary: list[tuple[int, ...]]) -> list[int]:
    index = {row: i for i, row in enumerate(vocabulary)}
    return [index[tuple(int(x) for x in row)] for row in rows]


def _vocabulary(rows: list[np.ndarray]) -> list[tuple[int, ...]]:
    return sorted({tuple(int(x) for x in row) for arr in rows for row in arr})


def _load_level(level: int) -> tuple[list[dict], dict]:
    filename = f"level={level}_sensor=cardinal distance_range=1_rand=0.3_eval-seed=0.npz"
    cache_path = CACHE_ROOT / "walks" / filename
    raw = _load_walk_cache(cache_path.parent, cache_path.name)
    reference = exact_cartesian_reference_positions(level)
    walks = []
    diagnostics = []
    for i in (0, 1, 2):
        obs = np.asarray(raw["filtered_obs_per_walk"][i], dtype=np.int64)
        act = np.asarray(raw["filtered_actions_per_walk"][i], dtype=np.int64)
        gt = np.asarray(raw["gt_exact_per_walk"][i], dtype=np.int64)
        coll = np.asarray(raw["collisions_per_walk"][i], dtype=bool)
        cached_action = np.asarray(raw["gt_action_per_walk"][i], dtype=np.int64)
        assert obs.shape == act.shape == (10_000, 4)
        assert gt.shape == coll.shape == cached_action.shape == (10_000,)
        assert np.array_equal(action_ids_from_action_vectors(act), cached_action)
        assert np.all((gt >= 0) & (gt < len(reference)))
        positions = reference[gt]
        checked = validate_evaluation_walk_action_alignment(positions=positions, actions=act)
        moved = np.any(positions[1:] != positions[:-1], axis=1)
        assert np.array_equal(moved, ~coll[1:]), "collision flags must agree with motion"
        assert not np.any(np.abs(np.diff(positions, axis=0)).sum(axis=1) > 1)
        walks.append({"obs": obs, "act": act, "gt": gt, "coll": coll})
        diagnostics.append({"walk": i, "steps": len(gt), "collisions": int(coll.sum()),
                            "checked_nonself_actions": checked})
    action_vocab = _vocabulary([w["act"] for w in walks])
    obs_vocab = _vocabulary([w["obs"] for w in walks])
    assert len(action_vocab) == 4
    for w in walks:
        w["x"] = _tokens(w["act"], action_vocab)
        w["cardinal"] = _tokens(w["obs"], obs_vocab)
        w["cartesian"] = [int(x) for x in w["gt"]]
    transitions: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for w in walks:
        for t in range(1, len(w["gt"])):
            source = int(w["gt"][t - 1])
            action = w["x"][t]
            outcome = (int(w["gt"][t]), w["cardinal"][t])
            transitions.setdefault((source, action), set()).add(outcome)
    assert len(transitions) == len(reference) * len(action_vocab)
    assert all(len(outcomes) == 1 for outcomes in transitions.values())
    blocks = [tuple(range(len(reference)))]
    for _ in range(len(reference)):
        owner = {state: i for i, block in enumerate(blocks) for state in block}
        groups: dict[tuple, list[int]] = {}
        for state in range(len(reference)):
            signature = tuple(
                (next(iter(transitions[(state, action)]))[1],
                 owner[next(iter(transitions[(state, action)]))[0]])
                for action in range(len(action_vocab))
            )
            groups.setdefault(signature, []).append(state)
        new_blocks = sorted(tuple(group) for group in groups.values())
        if new_blocks == blocks:
            break
        blocks = new_blocks
    metadata = {
        "cache": str(cache_path.relative_to(ROOT)),
        "cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        "protocol": json.loads((CACHE_ROOT / "protocol.json").read_text()),
        "walks": diagnostics,
        "action_vocabulary": [list(x) for x in action_vocab],
        "cardinal_vocabulary": [list(x) for x in obs_vocab],
        "cartesian_vocabulary": list(range(len(reference))),
        "heldout_only_cardinal_tokens": sorted(set(walks[1]["cardinal"] + walks[2]["cardinal"]) - set(walks[0]["cardinal"])),
        "heldout_only_cartesian_tokens": sorted(set(walks[1]["cartesian"] + walks[2]["cartesian"]) - set(walks[0]["cartesian"])),
        "observed_state_action_pairs": len(transitions),
        "deterministic_predictive_equivalence_classes": [list(block) for block in blocks],
    }
    return walks, metadata


def _relationship(states: np.ndarray, targets: np.ndarray, name: str) -> dict:
    valid = states >= 0
    details = compute_fixed_map_precision_details(
        inferred_states=states[valid], ground_truth_states=targets[valid],
        metric_name=f"{name}_precision",
    )
    details["ari"] = float(adjusted_rand_score(targets[valid], states[valid])) if valid.any() else None
    return details


def _evaluate(states: np.ndarray, walk: dict, model_state_count: int) -> dict:
    # Filter state t predicts Y_t from action X_t and history through t-1.
    # Cached ground truth t is the position after that action, so compare with t-1.
    aligned = states[1:]
    previous_position = walk["gt"][:-1]
    previous_observation = np.asarray(walk["cardinal"][:-1], dtype=np.int64)
    current_output = np.asarray(walk["cardinal"][1:], dtype=np.int64)
    valid = aligned >= 0
    physical = _relationship(aligned, previous_position, "physical")
    counts = np.asarray(physical["cooccurrence_matrix"], dtype=np.int64)
    return {
        "steps": len(states), "scored_steps": len(aligned),
        "synchronized_steps": int(valid.sum()),
        "synchronized_fraction": float(valid.mean()),
        "unsynchronized_steps": int((~valid).sum()),
        "model_state_count": model_state_count,
        "visited_state_count": int(np.unique(aligned[valid]).size),
        "physical_previous": physical,
        "observation_previous": _relationship(aligned, previous_observation, "observation"),
        "observation_current": _relationship(aligned, current_output, "observation"),
        "physical_states_per_inferred_state": [int((row > 0).sum()) for row in counts],
        "inferred_states_per_physical_state": [int((col > 0).sum()) for col in counts.T],
    }


def run(levels: tuple[int, ...], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    summary = {"upstream_revision": REVISION, "alpha": 0.001, "test": "chi2",
               "stats_mode": "correct", "levels": {}, "runs": [], "comparisons": []}
    for level in levels:
        walks, metadata = _load_level(level)
        summary["levels"][str(level)] = metadata
        _write_json(output / f"level={level}_inputs.json", metadata)
        for sensor in ("cardinal", "cartesian"):
            alphabet = sorted({token for w in walks for token in w[sensor]})
            for length in LENGTHS:
                for L_max in L_VALUES:
                    key = f"level={level}_output={sensor}_train={length}_L={L_max}"
                    record = {"key": key, "level": level, "output": sensor,
                              "train_length": length, "L_max": L_max}
                    started = time.monotonic()
                    try:
                        model = infer_transducer(
                            walks[0]["x"][:length], walks[0][sensor][:length],
                            L_max=L_max, alpha=0.001, test="chi2", stats_mode="correct",
                            input_alphabet=list(range(4)), output_alphabet=alphabet,
                        )
                        record["model_state_count"] = model.n_states
                        record["history_count"] = len(model.epsilon)
                        model.to_json(output / "models" / f"{key}.json")
                        record["heldout"] = {}
                        for walk_index in (1, 2):
                            result = model.filter(walks[walk_index]["x"], walks[walk_index][sensor])
                            states = np.asarray(result.states, dtype=np.int64)
                            assert states.shape == (10_000,)
                            np.savez_compressed(output / "models" / f"{key}_walk={walk_index}.npz", states=states)
                            record["heldout"][str(walk_index)] = _evaluate(states, walks[walk_index], model.n_states)
                        record["status"] = "ok"
                    except Exception:
                        record["status"] = "failed"
                        record["error"] = traceback.format_exc()
                    record["elapsed_seconds"] = round(time.monotonic() - started, 3)
                    summary["runs"].append(record)
                    _write_json(output / "summary.json", summary)
                    print(key, record["status"], record.get("model_state_count"), record["elapsed_seconds"], flush=True)

            for L_max in L_VALUES:
                for left, right in ((2_000, 5_000), (2_000, 10_000), (5_000, 10_000)):
                    left_key = f"level={level}_output={sensor}_train={left}_L={L_max}"
                    right_key = f"level={level}_output={sensor}_train={right}_L={L_max}"
                    for walk_index in (1, 2):
                        paths = [output / "models" / f"{key}_walk={walk_index}.npz" for key in (left_key, right_key)]
                        if not all(p.exists() for p in paths):
                            continue
                        a, b = [np.load(p)["states"][1:] for p in paths]
                        both = (a >= 0) & (b >= 0)
                        summary["comparisons"].append({
                            "level": level, "output": sensor, "L_max": L_max,
                            "train_lengths": [left, right], "walk": walk_index,
                            "both_synchronized_steps": int(both.sum()),
                            "both_synchronized_fraction": float(both.mean()),
                            "ari": float(adjusted_rand_score(a[both], b[both])) if both.any() else None,
                        })
            _write_json(output / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, nargs="+", default=[20, 13], choices=[13, 20])
    parser.add_argument("--output", type=Path, default=RESULT_ROOT)
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.preflight_only:
        summary_path = arguments.output / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
        for level in arguments.levels:
            _, metadata = _load_level(level)
            _write_json(arguments.output / f"level={level}_inputs.json", metadata)
            if summary is not None:
                summary["levels"][str(level)] = metadata
            print(f"level={level}: inputs checked", flush=True)
        if summary is not None:
            _write_json(summary_path, summary)
    else:
        run(tuple(arguments.levels), arguments.output)
