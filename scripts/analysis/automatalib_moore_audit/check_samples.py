"""Check 2,000-step training datasets and 10,000-step walks against the oracle audit.

Coverage is reported per trajectory. An incomplete trajectory is still checked
against the certified machine but cannot independently certify its full table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from scripts.analysis.automatalib_moore_audit.run import (
    HERE,
    ROOT,
    _compile_java_runner,
    _machine_tsv,
    _parse_java,
    _reference_positions,
)


LEVELS = (13, 20, 21, 22)
OBS_COLUMNS = [f"observation_{i}" for i in range(4)]
ACTION_COLUMNS = [f"action_{direction}" for direction in ("north", "east", "south", "west")]


def _oracle_arrays(machine: dict) -> tuple[np.ndarray, np.ndarray]:
    outputs = np.asarray(machine["outputs"], dtype=np.int64)
    transitions = np.full((machine["physical_state_count"], 4), -1, dtype=np.int64)
    for source, action, target in machine["transitions"]:
        transitions[source, action] = target
    if np.any(transitions < 0):
        raise ValueError("Stored oracle table is incomplete")
    return outputs, transitions


def _action_ids(vectors: np.ndarray) -> np.ndarray:
    if vectors.ndim != 2 or vectors.shape[1] != 4:
        raise ValueError("Expected four action columns")
    if not np.all((vectors == 0) | (vectors == 1)) or not np.all(vectors.sum(axis=1) == 1):
        raise ValueError("Action vectors are not one-hot")
    return vectors.argmax(axis=1).astype(np.int64)


def _coverage(states: np.ndarray, sources: np.ndarray, actions: np.ndarray, count: int) -> dict:
    state_set = set(int(x) for x in np.unique(states))
    pair_set = {(int(s), int(a)) for s, a in zip(sources, actions)}
    expected_states = set(range(count))
    expected_pairs = {(s, a) for s in range(count) for a in range(4)}
    return {
        "observed_state_count": len(state_set),
        "observed_state_action_pair_count": len(pair_set),
        "complete": state_set == expected_states and pair_set == expected_pairs,
        "missing_states": sorted(expected_states - state_set),
        "missing_state_action_pairs": [list(pair) for pair in sorted(expected_pairs - pair_set)],
    }


def _sample_machine(
    states: np.ndarray, observations: np.ndarray, sources: np.ndarray,
    actions: np.ndarray, targets: np.ndarray, count: int,
) -> dict | None:
    """Build a full table solely from the trajectory when its coverage permits."""
    coverage = _coverage(states, sources, actions, count)
    if not coverage["complete"]:
        return None
    outputs: dict[int, tuple[int, ...]] = {}
    transitions: dict[tuple[int, int], int] = {}
    for state, observation in zip(states, observations):
        key = int(state)
        output = tuple(int(x) for x in observation)
        if key in outputs and outputs[key] != output:
            raise ValueError(f"Conflicting sampled output at state {key}")
        outputs[key] = output
    for source, action, target in zip(sources, actions, targets):
        key = (int(source), int(action))
        value = int(target)
        if key in transitions and transitions[key] != value:
            raise ValueError(f"Conflicting sampled transition at {key}")
        transitions[key] = value
    return {
        "physical_state_count": count,
        "outputs": [list(outputs[state]) for state in range(count)],
        "transitions": [[state, action, transitions[(state, action)]]
                        for state in range(count) for action in range(4)],
    }


def check_dataset_arrays(
    *, initial_state: int, observations: np.ndarray, vectors: np.ndarray,
    collisions: np.ndarray, positions: list[tuple[int, int]], oracle: dict,
) -> dict:
    expected_obs, expected_transitions = _oracle_arrays(oracle)
    actions = _action_ids(vectors)
    if observations.shape != (len(actions), 4) or collisions.shape != (len(actions),):
        raise ValueError("Dataset rows are misaligned")
    if not 0 <= initial_state < len(positions):
        raise ValueError("Initial physical position is not open")
    # Replay independently from the maze layout, using the privileged start
    # position. The parquet file itself contains no physical-state column.
    position_index = {position: state for state, position in enumerate(positions)}
    geometry = np.empty_like(expected_transitions)
    deltas = ((-1, 0), (0, 1), (1, 0), (0, -1))
    for source, (row, col) in enumerate(positions):
        for action, (dr, dc) in enumerate(deltas):
            geometry[source, action] = position_index.get((row + dr, col + dc), source)
    sources = np.empty(len(actions), dtype=np.int64)
    targets = np.empty(len(actions), dtype=np.int64)
    state = initial_state
    for step, action in enumerate(actions):
        sources[step] = state
        state = int(geometry[state, action])
        targets[step] = state
    if not np.array_equal(targets, expected_transitions[sources, actions]):
        raise ValueError("Dataset transition disagrees with certified oracle")
    if not np.array_equal(np.asarray(collisions, dtype=bool), sources == targets):
        raise ValueError("Dataset collision flags disagree with physical replay")
    if not np.array_equal(np.asarray(observations, dtype=np.int64), expected_obs[targets]):
        raise ValueError("Dataset observations disagree with physical replay")
    coverage = _coverage(targets, sources, actions, len(positions))
    coverage["_sample_machine"] = _sample_machine(
        targets, observations, sources, actions, targets, len(positions)
    )
    return coverage


def check_dataset(metadata_path: Path, oracle: dict) -> dict:
    metadata = json.loads(metadata_path.read_text())
    parquet_path = metadata_path.parent / metadata["artifact"]["parquet"]
    frame = pl.read_parquet(parquet_path)
    if frame.height != 2000 or not np.array_equal(frame["iteration"].to_numpy(), np.arange(2000)):
        raise ValueError("Expected 2,000 consecutive dataset rows")
    positions = _reference_positions(int(metadata["maze_id"]))
    start = tuple(int(x) for x in metadata["initial_agent_position"])
    if start not in positions:
        raise ValueError("Invalid metadata initial_agent_position")
    coverage = check_dataset_arrays(
        initial_state=positions.index(start),
        observations=frame.select(OBS_COLUMNS).to_numpy(),
        vectors=frame.select(ACTION_COLUMNS).to_numpy(),
        collisions=frame["collision"].to_numpy(),
        positions=positions,
        oracle=oracle,
    )
    return {
        "source": str(parquet_path.relative_to(ROOT)),
        "seed": int(metadata["random_seed"]),
        "random_prob": float(metadata["random_prob"]),
        **coverage,
        "consistent_with_oracle": True,
    }


def check_evaluation_walk(cached, walk: int, oracle: dict) -> dict:
    prefix = f"walk_{walk:05d}"
    if not bool(cached[f"{prefix}_valid"]):
        raise ValueError("Selected walk is invalid")
    observations = np.asarray(cached[f"{prefix}_obs"])
    vectors = np.asarray(cached[f"{prefix}_act"])
    states = np.asarray(cached[f"{prefix}_gt_exact"], dtype=np.int64)
    cached_actions = np.asarray(cached[f"{prefix}_gt_action"], dtype=np.int64)
    collisions = np.asarray(cached[f"{prefix}_collisions"], dtype=bool)
    if not (
        observations.shape == vectors.shape == (10_000, 4)
        and states.shape == cached_actions.shape == collisions.shape == (10_000,)
    ):
        raise ValueError("Expected 10,000 aligned evaluation rows")
    actions = _action_ids(vectors)
    if not np.array_equal(actions, cached_actions):
        raise ValueError("Cached action IDs disagree with vectors")
    expected_obs, expected_transitions = _oracle_arrays(oracle)
    if not np.all((states >= 0) & (states < len(expected_obs))):
        raise ValueError("Evaluation walk has an invalid physical state")
    if not np.array_equal(observations, expected_obs[states]):
        raise ValueError("Evaluation observations disagree with certified oracle")
    sources = states[:-1]
    pair_actions = actions[1:]
    targets = states[1:]
    if not np.array_equal(targets, expected_transitions[sources, pair_actions]):
        raise ValueError("Evaluation transitions disagree with certified oracle")
    if not np.array_equal(collisions[1:], sources == targets):
        raise ValueError("Evaluation collision flags disagree with physical states")
    coverage = _coverage(states, sources, pair_actions, len(expected_obs))
    coverage["_sample_machine"] = _sample_machine(
        states, observations, sources, pair_actions, targets, len(expected_obs)
    )
    return {"walk": walk, **coverage, "consistent_with_oracle": True}


def _dataset_paths() -> list[Path]:
    selected = []
    for path in (ROOT / "datasets").rglob("*steps2000_*.metadata.json"):
        metadata = json.loads(path.read_text())
        sensor = metadata.get("agent_config", {})
        if (
            metadata.get("maze_id") in LEVELS
            and metadata.get("num_steps") == 2000
            and sensor.get("sensor") == "cardinal distance"
            and sensor.get("range") == 1
            and not metadata.get("valid_trajectories_only", False)
        ):
            selected.append(path)
    return sorted(selected)


def run(output_dir: Path = HERE, maven: Path = Path("mvn")) -> dict:
    certified = json.loads((HERE / "RESULTS.json").read_text())
    oracles = {machine["level"]: machine for machine in certified["levels"]}
    datasets = []
    for path in _dataset_paths():
        metadata = json.loads(path.read_text())
        level = int(metadata["maze_id"])
        try:
            result = check_dataset(path, oracles[level])
            result.update({"level": level, "status": "ok"})
        except Exception as exc:
            result = {"level": level, "source": str(path.relative_to(ROOT)),
                      "status": "failed", "error": str(exc)}
        datasets.append(result)
    evaluation = []
    for level in LEVELS:
        path = ROOT / oracles[level]["cache"]
        with np.load(path, allow_pickle=False) as cached:
            for walk in range(int(cached["n_walks"])):
                try:
                    result = check_evaluation_walk(cached, walk, oracles[level])
                    result.update({"level": level, "status": "ok"})
                except Exception as exc:
                    result = {"level": level, "walk": walk, "status": "failed", "error": str(exc)}
                evaluation.append(result)
    distinct_machines = {}
    for row in datasets + evaluation:
        if row["status"] != "ok":
            row["equivalence_status"] = "unavailable_failed_consistency_check"
            continue
        machine = row.pop("_sample_machine")
        if machine is None:
            row["equivalence_status"] = "unavailable_incomplete_coverage"
            row["minimized_moore_state_count"] = None
            row["equivalence_classes"] = None
            row["all_classes_singleton"] = None
            row["separating_sequences"] = None
            continue
        oracle = oracles[row["level"]]
        if machine != {key: oracle[key] for key in
                       ("physical_state_count", "outputs", "transitions")}:
            raise ValueError("Complete sampled machine disagrees with the certified oracle")
        encoded = json.dumps(machine, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        distinct_machines[digest] = machine
        row["sample_machine_sha256"] = digest
        row["equivalence_status"] = "pending_automatalib"
    results_by_hash = {}
    with tempfile.TemporaryDirectory(prefix="automatalib-samples-") as temporary:
        temp = Path(temporary)
        classpath = _compile_java_runner(maven, temp)
        for digest, machine in sorted(distinct_machines.items()):
            path = temp / f"{digest}.tsv"
            _machine_tsv(machine, path)
            completed = subprocess.run(
                ["java", "-cp", classpath, "MooreAudit", str(path)],
                check=True, capture_output=True, text=True,
            )
            results_by_hash[digest] = _parse_java(completed.stdout, machine)
    for row in datasets + evaluation:
        digest = row.get("sample_machine_sha256")
        if digest is None:
            continue
        result = results_by_hash[digest]
        oracle = oracles[row["level"]]
        for key in ("minimized_moore_state_count", "equivalence_classes",
                    "all_classes_singleton", "separating_sequences",
                    "characterizing_set", "characterizing_set_pair_coverage"):
            if result[key] != oracle[key]:
                raise ValueError(f"Sample AutomataLib result disagrees with oracle at level {row['level']}")
        # Keep per-sample output focused; global W and all-pair coverage live in
        # the benchmark-level RESULTS.json, not repeated hundreds of times.
        for key in ("minimized_moore_state_count", "equivalence_classes",
                    "all_classes_singleton", "separating_sequences",
                    "perceptually_aliased_pair_count"):
            row[key] = result[key]
        row["equivalence_status"] = "certified_by_automatalib"
    output = {"dataset_scope": "2,000-step cardinal-distance range-1 all-step datasets on levels 13, 20, 21, 22",
              "evaluation_scope": "all 10,000-step walks in the certified canonical cache",
              "automatalib_version": "13.0.0",
              "distinct_complete_tables_analyzed": len(distinct_machines),
              "datasets": datasets, "evaluation_walks": evaluation}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "SAMPLE_CHECKS.json").write_text(json.dumps(output, indent=2) + "\n")
    (output_dir / "SAMPLE_CHECKS.md").write_text(_report(output))
    return output


def _report(result: dict) -> str:
    dataset_rows = result["datasets"]
    evaluation_rows = result["evaluation_walks"]
    complete_datasets = sum(row.get("complete", False) for row in dataset_rows)
    complete_walks = sum(row.get("complete", False) for row in evaluation_rows)
    lines = ["# Sample-level Moore equivalence and coverage", "",
             "Each 2,000-step parquet is replayed from its privileged metadata start position. "
             "Each 10,000-step evaluation walk is checked against its cached physical states. "
             "Observations, actions, transitions, and collisions must match the certified Moore machine.",
             "", "| Level | 2,000-step datasets consistent | Datasets certified | "
             "10,000-step walks consistent | Walks certified | Moore classes |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for level in LEVELS:
        ds = [row for row in result["datasets"] if row["level"] == level]
        ev = [row for row in result["evaluation_walks"] if row["level"] == level]
        lines.append(
            f"| {level} | {sum(row['status'] == 'ok' for row in ds)}/{len(ds)} | "
            f"{sum(row.get('equivalence_status') == 'certified_by_automatalib' for row in ds)}/{len(ds)} | "
            f"{sum(row['status'] == 'ok' for row in ev)}/{len(ev)} | "
            f"{sum(row.get('equivalence_status') == 'certified_by_automatalib' for row in ev)}/{len(ev)} | "
            f"{next(row['minimized_moore_state_count'] for row in ev if row.get('equivalence_status') == 'certified_by_automatalib')} |"
        )
    lines += [
              "",
              f"The {complete_datasets} complete datasets and {complete_walks} complete "
              "evaluation walks each supply a full sampled Moore table. The tables were "
              f"deduplicated into {result['distinct_complete_tables_analyzed']} distinct machines, "
              "then minimized with AutomataLib. Every resulting class is a singleton. "
              "Per-sample classes and separating words are in `SAMPLE_CHECKS.json`.",
              "",
              "A sample with missing state–action pairs cannot by itself establish the "
              "minimized machine. Its observed rows still agree with the complete "
              "oracle machine; its equivalence fields are null. Full per-sample coverage "
              "and failure details are in `SAMPLE_CHECKS.json`.", "",
              "The 2,000-step parquet files do not contain physical-state paths. "
              "Their paths are reconstructed from privileged initial positions and "
              "the benchmark maze layout; this is an oracle check, not a learner evaluation.",
              ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maven", type=Path, default=Path("mvn"))
    parser.add_argument("--output-dir", type=Path, default=HERE)
    arguments = parser.parse_args()
    run(arguments.output_dir, arguments.maven)
