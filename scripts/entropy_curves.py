from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Hashable, Iterable, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SRC_ROOT):
    import_root_string = str(import_root)
    if import_root_string not in sys.path:
        sys.path.insert(0, import_root_string)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import polars as pl

from src.maze_environment.LEVEL import levels
from src.maze_environment.agent import Agent
from src.maze_environment.generate_data import generate_data


State = Hashable
Action = Hashable
Observation = Hashable

LEVEL_INDEX = 20
LEVEL_INDICES = (13, 20, 21)

SENSOR = "cardinal distance"
SENSOR_RANGE = 1
MAX_HISTORY_ORDER = 8
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "outputs" / "entropy_curves"
DATASETS_ROOT = PROJECT_ROOT / "datasets"
EMPIRICAL_SEEDS = tuple(range(10))
EMPIRICAL_RANDOM_PROBABILITIES = tuple(round(i / 10, 1) for i in range(11))
EMPIRICAL_VALID_ONLY_OPTIONS = (False, True)
EMPIRICAL_NUM_STEPS = 20_000
EMPIRICAL_RANDOM_START = True
SENSORY_DATASET_KEY = "cardinal distance:range=1"
TRUE_STATE_DATASET_KEY = "cartesian"
SUMMARY_QUANTILE_PERCENTAGES = tuple(range(0, 101))
DISPLAY_QUANTILE_PERCENTAGES = tuple(range(0, 101, 10))
SUMMARY_QUANTILE_FRACTIONS = tuple(
    percentage / 100.0 for percentage in SUMMARY_QUANTILE_PERCENTAGES
)
DISPLAY_QUANTILE_FRACTIONS = tuple(
    percentage / 100.0 for percentage in DISPLAY_QUANTILE_PERCENTAGES
)
NEAR_ZERO_ENTROPY_THRESHOLD = 1e-2


def shannon_entropy(
    probabilities: Mapping[Hashable, float],
    log_base: float = 2.0,
) -> float:
    """Compute Shannon entropy of a discrete distribution."""
    entropy = 0.0
    log_denominator = math.log(log_base)
    for probability in probabilities.values():
        if probability > 0.0:
            entropy -= probability * (math.log(probability) / log_denominator)
    return entropy


def normalise_distribution(
    weights: Mapping[Hashable, float],
) -> dict[Hashable, float]:
    """Return a normalised copy of a discrete distribution."""
    total_weight = sum(weights.values())
    if total_weight <= 0.0:
        raise ValueError("Distribution must have positive total mass.")
    return {
        key: value / total_weight
        for key, value in weights.items()
        if value > 0.0
    }


def get_all_actions(
    transition_probabilities: Mapping[State, Mapping[Action, Mapping[State, float]]],
) -> list[Action]:
    """Collect the union of all actions present in the transition model."""
    all_actions: set[Action] = set()
    for action_mapping in transition_probabilities.values():
        all_actions.update(action_mapping.keys())
    if not all_actions:
        raise ValueError("No actions found in transition_probabilities.")
    return sorted(all_actions, key=str)


def _sensor_config(sensor: str, sensor_range: int | None) -> dict[str, Any]:
    config: dict[str, Any] = {"sensor": sensor}
    if sensor == "cardinal distance":
        config["range"] = sensor_range
    return config


def _read_level_grid(level: list[str]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    if not level:
        raise ValueError("Level must contain at least one row.")

    row_width = len(level[0])
    if row_width == 0:
        raise ValueError("Level rows must be non-empty.")
    if any(len(row) != row_width for row in level):
        raise ValueError("All level rows must have the same width.")

    maze = np.zeros((len(level), row_width), dtype=np.uint8)
    open_positions: list[tuple[int, int]] = []
    for row_index, row in enumerate(level):
        for col_index, value in enumerate(row):
            if value == "X":
                maze[row_index, col_index] = 1
            else:
                open_positions.append((row_index, col_index))

    if not open_positions:
        raise ValueError("Level does not contain any accessible positions.")

    return maze, open_positions


def _normalise_observation(observation: Any) -> Observation:
    if isinstance(observation, np.ndarray):
        return tuple(_normalise_observation(value) for value in observation.tolist())
    if isinstance(observation, list):
        return tuple(_normalise_observation(value) for value in observation)
    if isinstance(observation, tuple):
        return tuple(_normalise_observation(value) for value in observation)
    if isinstance(observation, np.generic):
        return observation.item()
    return observation


def _level_output_stem(*, level_index: int, sensor: str, sensor_range: int | None) -> str:
    sensor_stem = sensor.replace(" ", "_")
    range_stem = "unlimited" if sensor_range is None else f"range_{sensor_range}"
    return f"level_{int(level_index)}_{sensor_stem}_{range_stem}"


def _dataset_directory(
    *,
    datasets_root: str | Path,
    level_index: int,
    sensor: str,
    sensor_range: int | None = None,
) -> Path:
    directory = Path(datasets_root) / f"level={level_index}" / f"sensor={sensor}"
    if sensor == "cardinal distance":
        directory = directory / f"range={sensor_range}"
    return directory


def dataset_artifact_paths(
    *,
    datasets_root: str | Path,
    level_index: int,
    sensor_key: str,
    run_id: str,
) -> tuple[Path, Path]:
    if sensor_key.startswith("cardinal distance"):
        sensor = "cardinal distance"
        sensor_range = int(sensor_key.split("range=", maxsplit=1)[1])
    else:
        sensor = sensor_key
        sensor_range = None

    directory = _dataset_directory(
        datasets_root=datasets_root,
        level_index=level_index,
        sensor=sensor,
        sensor_range=sensor_range,
    )
    return directory / f"{run_id}.parquet", directory / f"{run_id}.metadata.json"


def generate_empirical_entropy_run_ids(
    *,
    level_index: int = LEVEL_INDEX,
    datasets_root: str | Path = DATASETS_ROOT,
    seeds: Iterable[int] = EMPIRICAL_SEEDS,
    random_probabilities: Iterable[float] = EMPIRICAL_RANDOM_PROBABILITIES,
    valid_only_options: Iterable[bool] = EMPIRICAL_VALID_ONLY_OPTIONS,
    num_steps: int = EMPIRICAL_NUM_STEPS,
    random_start: bool = EMPIRICAL_RANDOM_START,
    reuse_existing: bool = True,
) -> dict[tuple[float, bool], list[dict[str, Any]]]:
    runs_by_condition: dict[tuple[float, bool], list[dict[str, Any]]] = defaultdict(list)
    for rand_prob in random_probabilities:
        for valid_only in valid_only_options:
            condition_key = (float(rand_prob), bool(valid_only))
            for seed in seeds:
                run_ids = generate_data(
                    level=level_index,
                    seed=int(seed),
                    rand_prob=float(rand_prob),
                    num_steps=int(num_steps),
                    valid_trajectories_only=bool(valid_only),
                    random_start=bool(random_start),
                    output_root=datasets_root,
                    reuse_existing=bool(reuse_existing),
                )
                runs_by_condition[condition_key].append(
                    {
                        "seed": int(seed),
                        "rand_prob": float(rand_prob),
                        "valid_trajectories_only": bool(valid_only),
                        "run_ids": run_ids,
                    }
                )
    return runs_by_condition


def _observation_columns(frame: pl.DataFrame) -> list[str]:
    columns = [name for name in frame.columns if name.startswith("observation_")]
    return sorted(columns, key=lambda name: int(name.rsplit("_", maxsplit=1)[1]))


def _action_columns(frame: pl.DataFrame) -> list[str]:
    return [name for name in frame.columns if name.startswith("action_")]


def _rows_as_tuples(frame: pl.DataFrame, columns: list[str]) -> list[tuple[Any, ...]]:
    return [tuple(row) for row in frame.select(columns).iter_rows()]


def _load_metadata(metadata_path: Path) -> dict[str, Any]:
    import json

    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_paired_empirical_dataset(
    *,
    datasets_root: str | Path,
    level_index: int,
    run_ids: Mapping[str, str],
    sensory_key: str = SENSORY_DATASET_KEY,
    true_state_key: str = TRUE_STATE_DATASET_KEY,
) -> tuple[list[Observation], list[Action], list[State], dict[str, Any], dict[str, Any]]:
    sensory_path, sensory_metadata_path = dataset_artifact_paths(
        datasets_root=datasets_root,
        level_index=level_index,
        sensor_key=sensory_key,
        run_id=run_ids[sensory_key],
    )
    state_path, state_metadata_path = dataset_artifact_paths(
        datasets_root=datasets_root,
        level_index=level_index,
        sensor_key=true_state_key,
        run_id=run_ids[true_state_key],
    )

    sensory_frame = pl.read_parquet(sensory_path)
    state_frame = pl.read_parquet(state_path)
    if sensory_frame.height != state_frame.height:
        raise ValueError(
            f"Paired datasets have different row counts: {sensory_path} has "
            f"{sensory_frame.height}, {state_path} has {state_frame.height}."
        )

    action_columns = _action_columns(sensory_frame)
    if not action_columns:
        raise ValueError(f"Sensory dataset has no action columns: {sensory_path}")

    sensory_rows = _rows_as_tuples(sensory_frame, _observation_columns(sensory_frame))
    action_rows = _rows_as_tuples(sensory_frame, action_columns)
    state_rows = _rows_as_tuples(state_frame, _observation_columns(state_frame))
    return (
        [_normalise_observation(row) for row in sensory_rows],
        [tuple(int(value) for value in row) for row in action_rows],
        [tuple(int(value) for value in row) for row in state_rows],
        _load_metadata(sensory_metadata_path),
        _load_metadata(state_metadata_path),
    )


def compute_empirical_history_entropy_summaries(
    *,
    sensory_histories_by_seed: Iterable[tuple[list[Observation], list[State]]],
    max_history_length: int = MAX_HISTORY_ORDER,
    log_base: float = 2.0,
    weight_by_history_mass: bool = False,
) -> dict[str, list[float]]:
    quantile_keys = [f"q{percentage:02d}" for percentage in SUMMARY_QUANTILE_PERCENTAGES]
    summaries = {"mean": []}
    for quantile_key in quantile_keys:
        summaries[quantile_key] = []

    for history_order in range(max_history_length + 1):
        history_to_state_counts: dict[tuple[Observation, ...], dict[State, float]] = defaultdict(lambda: defaultdict(float))

        for sensory_rows, state_rows in sensory_histories_by_seed:
            if len(sensory_rows) != len(state_rows):
                raise ValueError("Sensory and true-state rows must be aligned before aggregation.")
            for row_index in range(history_order, len(sensory_rows)):
                history = tuple(sensory_rows[row_index - history_order : row_index + 1])
                state = state_rows[row_index]
                history_to_state_counts[history][state] += 1.0

        summary = compute_history_entropy_summary(
            history_to_state_counts,
            log_base=log_base,
            aliased_states=None,
            weight_by_history_mass=bool(weight_by_history_mass),
        )
        summaries["mean"].append(summary["mean"])
        for quantile_key in quantile_keys:
            summaries[quantile_key].append(summary[quantile_key])

    return summaries


def compute_empirical_sensorimotor_history_entropy_summaries(
    *,
    sensory_histories_by_seed: Iterable[tuple[list[Observation], list[Action], list[State]]],
    max_history_length: int = MAX_HISTORY_ORDER,
    log_base: float = 2.0,
    weight_by_history_mass: bool = False,
) -> dict[str, list[float]]:
    quantile_keys = [f"q{percentage:02d}" for percentage in SUMMARY_QUANTILE_PERCENTAGES]
    summaries = {"mean": []}
    for quantile_key in quantile_keys:
        summaries[quantile_key] = []

    for history_order in range(max_history_length + 1):
        history_to_state_counts: dict[
            tuple[tuple[Observation, ...], tuple[Action, ...]],
            dict[State, float],
        ] = defaultdict(lambda: defaultdict(float))

        for sensory_rows, action_rows, state_rows in sensory_histories_by_seed:
            if len(sensory_rows) != len(action_rows) or len(sensory_rows) != len(state_rows):
                raise ValueError("Sensory, action, and true-state rows must be aligned before aggregation.")
            for row_index in range(history_order, len(sensory_rows)):
                observation_history = tuple(sensory_rows[row_index - history_order : row_index + 1])
                action_history = tuple(action_rows[row_index - history_order : row_index + 1])
                state = state_rows[row_index]
                history_to_state_counts[(observation_history, action_history)][state] += 1.0

        summary = compute_history_entropy_summary(
            history_to_state_counts,
            log_base=log_base,
            aliased_states=None,
            weight_by_history_mass=bool(weight_by_history_mass),
        )
        summaries["mean"].append(summary["mean"])
        for quantile_key in quantile_keys:
            summaries[quantile_key].append(summary[quantile_key])

    return summaries


def compute_empirical_next_state_history_entropy_summaries(
    *,
    sensory_histories_by_seed: Iterable[tuple[list[Observation], list[State]]],
    max_history_length: int = MAX_HISTORY_ORDER,
    log_base: float = 2.0,
    weight_by_history_mass: bool = False,
) -> dict[str, list[float]]:
    quantile_keys = [f"q{percentage:02d}" for percentage in SUMMARY_QUANTILE_PERCENTAGES]
    summaries = {"mean": []}
    for quantile_key in quantile_keys:
        summaries[quantile_key] = []

    for history_order in range(max_history_length + 1):
        history_to_state_counts: dict[tuple[Observation, ...], dict[State, float]] = defaultdict(lambda: defaultdict(float))

        for sensory_rows, state_rows in sensory_histories_by_seed:
            if len(sensory_rows) != len(state_rows):
                raise ValueError("Sensory and true-state rows must be aligned before aggregation.")
            for row_index in range(history_order, len(sensory_rows) - 1):
                history = tuple(sensory_rows[row_index - history_order : row_index + 1])
                next_state = state_rows[row_index + 1]
                history_to_state_counts[history][next_state] += 1.0

        summary = compute_history_entropy_summary(
            history_to_state_counts,
            log_base=log_base,
            aliased_states=None,
            weight_by_history_mass=bool(weight_by_history_mass),
        )
        summaries["mean"].append(summary["mean"])
        for quantile_key in quantile_keys:
            summaries[quantile_key].append(summary[quantile_key])

    return summaries


def compute_empirical_next_state_sensorimotor_history_entropy_summaries(
    *,
    sensory_histories_by_seed: Iterable[tuple[list[Observation], list[Action], list[State]]],
    max_history_length: int = MAX_HISTORY_ORDER,
    log_base: float = 2.0,
    weight_by_history_mass: bool = False,
) -> dict[str, list[float]]:
    quantile_keys = [f"q{percentage:02d}" for percentage in SUMMARY_QUANTILE_PERCENTAGES]
    summaries = {"mean": []}
    for quantile_key in quantile_keys:
        summaries[quantile_key] = []

    for history_order in range(max_history_length + 1):
        history_to_state_counts: dict[
            tuple[tuple[Observation, ...], tuple[Action, ...]],
            dict[State, float],
        ] = defaultdict(lambda: defaultdict(float))

        for sensory_rows, action_rows, state_rows in sensory_histories_by_seed:
            if len(sensory_rows) != len(action_rows) or len(sensory_rows) != len(state_rows):
                raise ValueError("Sensory, action, and true-state rows must be aligned before aggregation.")
            for row_index in range(history_order, len(sensory_rows) - 1):
                observation_history = tuple(sensory_rows[row_index - history_order : row_index + 1])
                action_history = tuple(action_rows[row_index - history_order : row_index + 1])
                next_state = state_rows[row_index + 1]
                history_to_state_counts[(observation_history, action_history)][next_state] += 1.0

        summary = compute_history_entropy_summary(
            history_to_state_counts,
            log_base=log_base,
            aliased_states=None,
            weight_by_history_mass=bool(weight_by_history_mass),
        )
        summaries["mean"].append(summary["mean"])
        for quantile_key in quantile_keys:
            summaries[quantile_key].append(summary[quantile_key])

    return summaries


def compute_empirical_condition_outputs(
    *,
    runs: Iterable[Mapping[str, Any]],
    datasets_root: str | Path = DATASETS_ROOT,
    level_index: int = LEVEL_INDEX,
    max_history_length: int = MAX_HISTORY_ORDER,
    log_base: float = 2.0,
) -> tuple[dict[str, dict[str, list[float]]], dict[str, dict[str, list[float]]], dict[str, Any]]:
    paired_rows: list[tuple[list[Observation], list[State]]] = []
    sensorimotor_paired_rows: list[tuple[list[Observation], list[Action], list[State]]] = []
    metadata_totals: dict[str, Any] = {
        "seeds": set(),
        "saved_rows": 0,
        "attempted_rows": 0,
        "dropped_collision_rows": 0,
        "start_positions": set(),
    }

    for run in runs:
        run_ids = run["run_ids"]
        sensory_rows, action_rows, state_rows, sensory_metadata, _ = load_paired_empirical_dataset(
            datasets_root=datasets_root,
            level_index=level_index,
            run_ids=run_ids,
        )
        paired_rows.append((sensory_rows, state_rows))
        sensorimotor_paired_rows.append((sensory_rows, action_rows, state_rows))

        stats = sensory_metadata.get("generation_stats", {}) or {}
        metadata_totals["seeds"].add(int(sensory_metadata.get("random_seed", run.get("seed", 0))))
        metadata_totals["saved_rows"] += int(stats.get("saved_steps", len(sensory_rows)))
        metadata_totals["attempted_rows"] += int(stats.get("attempted_steps", len(sensory_rows)))
        metadata_totals["dropped_collision_rows"] += int(stats.get("dropped_collision_steps", 0))
        start_position = sensory_metadata.get("initial_agent_position")
        if start_position is not None:
            metadata_totals["start_positions"].add(tuple(int(value) for value in start_position))

    unweighted = {
        "observation_only": compute_empirical_history_entropy_summaries(
            sensory_histories_by_seed=paired_rows,
            max_history_length=max_history_length,
            log_base=log_base,
            weight_by_history_mass=False,
        ),
        "sensorimotor": compute_empirical_sensorimotor_history_entropy_summaries(
            sensory_histories_by_seed=sensorimotor_paired_rows,
            max_history_length=max_history_length,
            log_base=log_base,
            weight_by_history_mass=False,
        ),
    }
    weighted = {
        "observation_only": compute_empirical_history_entropy_summaries(
            sensory_histories_by_seed=paired_rows,
            max_history_length=max_history_length,
            log_base=log_base,
            weight_by_history_mass=True,
        ),
        "sensorimotor": compute_empirical_sensorimotor_history_entropy_summaries(
            sensory_histories_by_seed=sensorimotor_paired_rows,
            max_history_length=max_history_length,
            log_base=log_base,
            weight_by_history_mass=True,
        ),
    }

    printable_totals = {
        "num_seeds": len(metadata_totals["seeds"]),
        "saved_rows": int(metadata_totals["saved_rows"]),
        "attempted_rows": int(metadata_totals["attempted_rows"]),
        "dropped_collision_rows": int(metadata_totals["dropped_collision_rows"]),
        "start_positions": sorted(metadata_totals["start_positions"]),
    }
    return unweighted, weighted, printable_totals


def compute_empirical_next_state_condition_outputs(
    *,
    runs: Iterable[Mapping[str, Any]],
    datasets_root: str | Path = DATASETS_ROOT,
    level_index: int = LEVEL_INDEX,
    max_history_length: int = MAX_HISTORY_ORDER,
    log_base: float = 2.0,
) -> tuple[dict[str, dict[str, list[float]]], dict[str, dict[str, list[float]]], dict[str, Any]]:
    paired_rows: list[tuple[list[Observation], list[State]]] = []
    sensorimotor_paired_rows: list[tuple[list[Observation], list[Action], list[State]]] = []
    metadata_totals: dict[str, Any] = {
        "seeds": set(),
        "saved_rows": 0,
        "attempted_rows": 0,
        "dropped_collision_rows": 0,
        "start_positions": set(),
    }

    for run in runs:
        run_ids = run["run_ids"]
        sensory_rows, action_rows, state_rows, sensory_metadata, _ = load_paired_empirical_dataset(
            datasets_root=datasets_root,
            level_index=level_index,
            run_ids=run_ids,
        )
        paired_rows.append((sensory_rows, state_rows))
        sensorimotor_paired_rows.append((sensory_rows, action_rows, state_rows))

        stats = sensory_metadata.get("generation_stats", {}) or {}
        metadata_totals["seeds"].add(int(sensory_metadata.get("random_seed", run.get("seed", 0))))
        metadata_totals["saved_rows"] += int(stats.get("saved_steps", len(sensory_rows)))
        metadata_totals["attempted_rows"] += int(stats.get("attempted_steps", len(sensory_rows)))
        metadata_totals["dropped_collision_rows"] += int(stats.get("dropped_collision_steps", 0))
        start_position = sensory_metadata.get("initial_agent_position")
        if start_position is not None:
            metadata_totals["start_positions"].add(tuple(int(value) for value in start_position))

    unweighted = {
        "observation_only": compute_empirical_next_state_history_entropy_summaries(
            sensory_histories_by_seed=paired_rows,
            max_history_length=max_history_length,
            log_base=log_base,
            weight_by_history_mass=False,
        ),
        "sensorimotor": compute_empirical_next_state_sensorimotor_history_entropy_summaries(
            sensory_histories_by_seed=sensorimotor_paired_rows,
            max_history_length=max_history_length,
            log_base=log_base,
            weight_by_history_mass=False,
        ),
    }
    weighted = {
        "observation_only": compute_empirical_next_state_history_entropy_summaries(
            sensory_histories_by_seed=paired_rows,
            max_history_length=max_history_length,
            log_base=log_base,
            weight_by_history_mass=True,
        ),
        "sensorimotor": compute_empirical_next_state_sensorimotor_history_entropy_summaries(
            sensory_histories_by_seed=sensorimotor_paired_rows,
            max_history_length=max_history_length,
            log_base=log_base,
            weight_by_history_mass=True,
        ),
    }

    printable_totals = {
        "num_seeds": len(metadata_totals["seeds"]),
        "saved_rows": int(metadata_totals["saved_rows"]),
        "attempted_rows": int(metadata_totals["attempted_rows"]),
        "dropped_collision_rows": int(metadata_totals["dropped_collision_rows"]),
        "start_positions": sorted(metadata_totals["start_positions"]),
    }
    return unweighted, weighted, printable_totals


def print_empirical_metadata_totals(*, heading: str, metadata_totals: Mapping[str, Any]) -> None:
    print(heading)
    print(f"seeds_included | {metadata_totals['num_seeds']}")
    print(f"saved_rows | {metadata_totals['saved_rows']}")
    print(f"attempted_rows | {metadata_totals['attempted_rows']}")
    print(f"dropped_collision_rows | {metadata_totals['dropped_collision_rows']}")
    print(f"start_positions | {metadata_totals['start_positions']}")


def build_level_entropy_model(
    *,
    level_index: int,
    sensor: str = SENSOR,
    sensor_range: int | None = SENSOR_RANGE,
    collision_filtered: bool = False,
) -> tuple[
    dict[tuple[int, int], dict[str, dict[tuple[int, int], float]]],
    dict[tuple[int, int], Observation],
]:
    """
    Build exact deterministic transitions and observations from LEVEL.py.

    States are accessible grid positions. Actions and sensor readings use the
    same Agent implementation as data generation. If collision_filtered is true,
    actions that would collide with a wall are omitted from the transition model.
    """
    try:
        level = levels[int(level_index)]
    except IndexError as exc:
        raise ValueError(f"Unknown level index {level_index}.") from exc

    maze, open_positions = _read_level_grid(level)
    sensors = _sensor_config(sensor, sensor_range)
    transition_probabilities: dict[
        tuple[int, int],
        dict[str, dict[tuple[int, int], float]],
    ] = {}
    observation_by_state: dict[tuple[int, int], Observation] = {}

    for position in open_positions:
        observation_agent = Agent(initial_position=position, maze=maze, sensors=sensors)
        observation_by_state[position] = _normalise_observation(
            observation_agent.make_observation(maze)
        )

        action_agent = Agent(initial_position=position, maze=maze, sensors=sensors)
        transition_probabilities[position] = {}
        for action_name in action_agent.action_dict:
            action_agent.position = position
            _, _, collision = action_agent.take_action(maze, action=action_name)
            if collision_filtered and bool(collision):
                continue
            next_position = tuple(int(value) for value in action_agent.position)
            transition_probabilities[position][action_name] = {next_position: 1.0}

    return transition_probabilities, observation_by_state


def count_level_states(level_index: int) -> int:
    """Return the number of accessible grid positions in a level."""
    try:
        level = levels[int(level_index)]
    except IndexError as exc:
        raise ValueError(f"Unknown level index {level_index}.") from exc

    _, open_positions = _read_level_grid(level)
    return len(open_positions)


def state_count_entropy_normalizer(level_index: int) -> float:
    """Return log2(|S|), the denominator for state-count entropy normalization."""
    state_count = count_level_states(level_index)
    if state_count <= 1:
        raise ValueError("At least two states are required for entropy normalization.")
    return math.log2(state_count)


def bit_threshold_to_state_normalized_threshold(
    *,
    threshold_bits: float,
    level_index: int,
) -> float:
    """Convert a threshold in bits to bits/log2(|S|) normalized entropy units."""
    return float(threshold_bits) / state_count_entropy_normalizer(level_index)


def scale_summary_family(
    summary_family: Mapping[str, list[float]],
    scale: float,
) -> dict[str, list[float]]:
    """Return a copy of one summary family with all values multiplied by scale."""
    return {
        statistic_name: [float(value) * float(scale) for value in values]
        for statistic_name, values in summary_family.items()
    }


def scale_entropy_summaries(
    summaries: Mapping[str, Mapping[str, list[float]]],
    scale: float,
) -> dict[str, dict[str, list[float]]]:
    """Return a copy of observation/sensorimotor summaries with all entropy values scaled."""
    return {
        history_type: scale_summary_family(summary_family, scale)
        for history_type, summary_family in summaries.items()
    }


def compute_quantile(sorted_values: list[float], quantile: float) -> float:
    """
    Compute a quantile using linear interpolation.
    """
    if not sorted_values:
        raise ValueError("sorted_values must be non-empty.")
    if quantile <= 0.0:
        return sorted_values[0]
    if quantile >= 1.0:
        return sorted_values[-1]

    position = quantile * (len(sorted_values) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))

    if lower_index == upper_index:
        return sorted_values[lower_index]

    interpolation_weight = position - lower_index
    return (
        (1.0 - interpolation_weight) * sorted_values[lower_index]
        + interpolation_weight * sorted_values[upper_index]
    )


def compute_weighted_quantile(
    sorted_value_weights: list[tuple[float, float]],
    quantile: float,
) -> float:
    """
    Compute the empirical weighted quantile for non-negative sample weights.
    """
    if not sorted_value_weights:
        raise ValueError("sorted_value_weights must be non-empty.")

    total_weight = sum(weight for _, weight in sorted_value_weights)
    if total_weight <= 0.0:
        raise ValueError("sorted_value_weights must have positive total weight.")

    if quantile <= 0.0:
        return sorted_value_weights[0][0]
    if quantile >= 1.0:
        return sorted_value_weights[-1][0]

    threshold = quantile * total_weight
    cumulative_weight = 0.0
    for value, weight in sorted_value_weights:
        cumulative_weight += weight
        if cumulative_weight >= threshold:
            return value

    return sorted_value_weights[-1][0]


def validate_model(
    transition_probabilities: Mapping[State, Mapping[Action, Mapping[State, float]]],
    observation_by_state: Mapping[State, Observation],
) -> None:
    """Basic consistency checks for the transition and observation model."""
    states = set(transition_probabilities.keys())

    missing_observations = states.difference(observation_by_state.keys())
    if missing_observations:
        raise ValueError(
            f"Missing observations for states: {sorted(missing_observations, key=str)}"
        )

    for current_state, action_mapping in transition_probabilities.items():
        for action, next_state_distribution in action_mapping.items():
            if not next_state_distribution:
                continue

            distribution_total = sum(next_state_distribution.values())
            if not math.isclose(distribution_total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(
                    f"Transition probabilities for state={current_state!r}, "
                    f"action={action!r} sum to {distribution_total}, not 1.0."
                )

            unknown_next_states = set(next_state_distribution.keys()).difference(states)
            if unknown_next_states:
                raise ValueError(
                    f"Unknown next states from state={current_state!r}, action={action!r}: "
                    f"{sorted(unknown_next_states, key=str)}"
                )


def make_state_prior(
    states: Iterable[State],
    state_prior: Mapping[State, float] | None,
) -> dict[State, float]:
    """Create a normalised prior over states."""
    state_list = list(states)
    if not state_list:
        raise ValueError("State set must be non-empty.")

    if state_prior is None:
        uniform_probability = 1.0 / len(state_list)
        return {state: uniform_probability for state in state_list}

    prior_weights = {state: state_prior.get(state, 0.0) for state in state_list}
    return normalise_distribution(prior_weights)


def get_aliased_states(
    observation_by_state: Mapping[State, Observation],
) -> set[State]:
    """
    Return the set of states that are ambiguous under instantaneous observation.
    """
    states_by_observation: dict[Observation, list[State]] = defaultdict(list)
    for state, observation in observation_by_state.items():
        states_by_observation[observation].append(state)

    aliased_states: set[State] = set()
    for grouped_states in states_by_observation.values():
        if len(grouped_states) > 1:
            aliased_states.update(grouped_states)

    return aliased_states


def compute_history_entropy_summary(
    history_to_state_weights: Mapping[Hashable, Mapping[State, float]],
    log_base: float,
    aliased_states: set[State] | None = None,
    weight_by_history_mass: bool = False,
) -> dict[str, float]:
    """
    Compute mean and quantiles of posterior entropy across aliased-state-relevant histories.

    If aliased_states is provided, only histories whose posterior support contains
    at least one aliased state are included.

    By default each distinct history key contributes equally. If
    weight_by_history_mass is true, each history is weighted by its marginal
    probability mass, making the mean the usual conditional entropy.
    """
    entropy_values: list[float] = []
    weighted_entropy_values: list[tuple[float, float]] = []

    for state_weights in history_to_state_weights.values():
        history_total_mass = sum(state_weights.values())
        if history_total_mass <= 0.0:
            continue

        posterior_state_distribution = normalise_distribution(state_weights)

        if aliased_states is not None:
            if not any(state in aliased_states for state in posterior_state_distribution):
                continue

        entropy_value = shannon_entropy(
            posterior_state_distribution,
            log_base=log_base,
        )
        entropy_values.append(entropy_value)
        weighted_entropy_values.append((entropy_value, history_total_mass))

    if not entropy_values:
        raise ValueError("No reachable aliased-state histories with positive posterior mass.")

    sorted_entropy_values = sorted(entropy_values)
    sorted_weighted_entropy_values = sorted(weighted_entropy_values)

    if weight_by_history_mass:
        total_history_mass = sum(weight for _, weight in sorted_weighted_entropy_values)
        mean_value = (
            sum(value * weight for value, weight in sorted_weighted_entropy_values)
            / total_history_mass
        )
    else:
        mean_value = sum(sorted_entropy_values) / len(sorted_entropy_values)

    summary = {"mean": mean_value}
    for percentage in SUMMARY_QUANTILE_PERCENTAGES:
        quantile = percentage / 100.0
        if weight_by_history_mass:
            summary[f"q{percentage:02d}"] = compute_weighted_quantile(
                sorted_weighted_entropy_values,
                quantile,
            )
        else:
            summary[f"q{percentage:02d}"] = compute_quantile(
                sorted_entropy_values,
                quantile,
            )

    return summary



def posterior_is_singleton_state(
    state_weights: Mapping[State, float],
    target_state: State,
    absolute_tolerance: float = 1e-12,
) -> bool:
    """
    Return True if the posterior induced by state_weights assigns all mass to target_state.
    """
    total_mass = sum(state_weights.values())
    if total_mass <= 0.0:
        return False

    target_mass = state_weights.get(target_state, 0.0)
    return abs(target_mass - total_mass) <= absolute_tolerance


def compute_history_conditioned_state_entropy_summaries(
    transition_probabilities: Mapping[State, Mapping[Action, Mapping[State, float]]],
    observation_by_state: Mapping[State, Observation],
    max_history_length: int,
    state_prior: Mapping[State, float] | None = None,
    log_base: float = 2.0,
    weight_by_history_mass: bool = False,
    restrict_to_aliased_histories: bool = True,
) -> dict[str, dict[str, list[float]]]:
    """
    Compute posterior-entropy summaries over reachable histories for:

        H(S_t | O_{t-k:t})
        H(S_t | O_{t-k:t}, A_{t-1-k:t-1})

    for history order k = 0, ..., max_history_length.
    Under this indexing convention, k = 0 sensorimotor histories condition on
    the current observation o_t and the action a_{t-1} that produced it.

    By default, summaries are restricted to histories whose posterior support
    contains at least one aliased state.
    """
    if max_history_length < 0:
        raise ValueError("max_history_length must be non-negative.")

    validate_model(transition_probabilities, observation_by_state)

    states = list(transition_probabilities.keys())
    initial_state_prior = make_state_prior(states, state_prior)
    aliased_states = get_aliased_states(observation_by_state)
    summary_aliased_states = aliased_states if restrict_to_aliased_histories else None

    quantile_keys = [
        f"q{percentage:02d}" for percentage in SUMMARY_QUANTILE_PERCENTAGES
    ]

    observation_summaries = {"mean": []}
    sensorimotor_summaries = {"mean": []}

    for quantile_key in quantile_keys:
        observation_summaries[quantile_key] = []
        sensorimotor_summaries[quantile_key] = []

    summary_keys = ["mean"] + [
        f"q{percentage:02d}" for percentage in SUMMARY_QUANTILE_PERCENTAGES
    ]

    # Keys are ((observation history), (action history)).
    # With this convention, k = 0 is ((o_t,), (a_{t-1},)).
    history_to_state_weights: dict[
        tuple[tuple[Observation, ...], tuple[Action, ...]],
        dict[State, float],
    ] = defaultdict(lambda: defaultdict(float))

    # Expand all actions under a uniform action policy.
    for previous_state, previous_state_probability in initial_state_prior.items():
        if previous_state_probability <= 0.0:
            continue

        action_mapping = transition_probabilities[previous_state]
        if not action_mapping:
            continue

        action_probability = 1.0 / len(action_mapping)
        for action, next_state_distribution in action_mapping.items():
            for next_state, transition_probability in next_state_distribution.items():
                if transition_probability <= 0.0:
                    continue

                joint_mass = (
                    previous_state_probability
                    * action_probability
                    * transition_probability
                )
                next_observation = observation_by_state[next_state]
                history_key = ((next_observation,), (action,))
                history_to_state_weights[history_key][next_state] += joint_mass

    for history_order in range(0, max_history_length + 1):
        # Aggregate sensorimotor histories down to observation-only histories.
        state_weights_by_observation_history: dict[
            tuple[Observation, ...], dict[State, float]
        ] = defaultdict(lambda: defaultdict(float))

        for history_key, state_weights in history_to_state_weights.items():
            observation_history, _ = history_key
            for state, mass in state_weights.items():
                state_weights_by_observation_history[observation_history][state] += mass

        observation_summary = compute_history_entropy_summary(
            state_weights_by_observation_history,
            log_base=log_base,
            aliased_states=summary_aliased_states,
            weight_by_history_mass=weight_by_history_mass,
        )
        sensorimotor_summary = compute_history_entropy_summary(
            history_to_state_weights,
            log_base=log_base,
            aliased_states=summary_aliased_states,
            weight_by_history_mass=weight_by_history_mass,
        )

        for statistic_name in summary_keys:
            observation_summaries[statistic_name].append(observation_summary[statistic_name])
            sensorimotor_summaries[statistic_name].append(sensorimotor_summary[statistic_name])

        if history_order == max_history_length:
            break

        next_history_to_state_weights: dict[
            tuple[tuple[Observation, ...], tuple[Action, ...]],
            dict[State, float],
        ] = defaultdict(lambda: defaultdict(float))

        for history_key, state_weights in history_to_state_weights.items():
            observation_history, action_history = history_key

            for current_state, current_mass in state_weights.items():
                if current_mass <= 0.0:
                    continue

                action_mapping = transition_probabilities[current_state]
                if not action_mapping:
                    continue

                action_probability = 1.0 / len(action_mapping)
                for action, next_state_distribution in action_mapping.items():
                    for next_state, transition_probability in next_state_distribution.items():
                        if transition_probability <= 0.0:
                            continue

                        next_mass = (
                            current_mass
                            * action_probability
                            * transition_probability
                        )
                        next_observation = observation_by_state[next_state]
                        next_history_key = (
                            observation_history + (next_observation,),
                            action_history + (action,),
                        )
                        next_history_to_state_weights[next_history_key][next_state] += next_mass

        history_to_state_weights = next_history_to_state_weights

    return {
        "observation_only": observation_summaries,
        "sensorimotor": sensorimotor_summaries,
    }


def compute_history_conditioned_next_state_entropy_summaries(
    transition_probabilities: Mapping[State, Mapping[Action, Mapping[State, float]]],
    observation_by_state: Mapping[State, Observation],
    max_history_length: int,
    state_prior: Mapping[State, float] | None = None,
    log_base: float = 2.0,
    weight_by_history_mass: bool = False,
    restrict_to_aliased_histories: bool = True,
) -> dict[str, dict[str, list[float]]]:
    """
    Compute posterior-entropy summaries over reachable histories for:

        H(S_{t+1} | O_{t-k:t})
        H(S_{t+1} | O_{t-k:t}, A_{t-k:t})

    for history order k = 0, ..., max_history_length.
    """
    if max_history_length < 0:
        raise ValueError("max_history_length must be non-negative.")

    validate_model(transition_probabilities, observation_by_state)

    states = list(transition_probabilities.keys())
    initial_state_prior = make_state_prior(states, state_prior)
    aliased_states = get_aliased_states(observation_by_state)
    summary_aliased_states = aliased_states if restrict_to_aliased_histories else None

    observation_summaries = {"mean": []}
    sensorimotor_summaries = {"mean": []}
    for percentage in SUMMARY_QUANTILE_PERCENTAGES:
        quantile_key = f"q{percentage:02d}"
        observation_summaries[quantile_key] = []
        sensorimotor_summaries[quantile_key] = []

    summary_keys = ["mean"] + [
        f"q{percentage:02d}" for percentage in SUMMARY_QUANTILE_PERCENTAGES
    ]

    history_to_next_state_weights: dict[
        tuple[tuple[Observation, ...], tuple[Action, ...]],
        dict[State, float],
    ] = defaultdict(lambda: defaultdict(float))

    for current_state, current_state_probability in initial_state_prior.items():
        if current_state_probability <= 0.0:
            continue

        action_mapping = transition_probabilities[current_state]
        if not action_mapping:
            continue

        current_observation = observation_by_state[current_state]
        action_probability = 1.0 / len(action_mapping)
        for action, next_state_distribution in action_mapping.items():
            for next_state, transition_probability in next_state_distribution.items():
                if transition_probability <= 0.0:
                    continue

                joint_mass = (
                    current_state_probability
                    * action_probability
                    * transition_probability
                )
                history_key = ((current_observation,), (action,))
                history_to_next_state_weights[history_key][next_state] += joint_mass

    for history_order in range(0, max_history_length + 1):
        state_weights_by_observation_history: dict[
            tuple[Observation, ...], dict[State, float]
        ] = defaultdict(lambda: defaultdict(float))

        for history_key, state_weights in history_to_next_state_weights.items():
            observation_history, _ = history_key
            for state, mass in state_weights.items():
                state_weights_by_observation_history[observation_history][state] += mass

        observation_summary = compute_history_entropy_summary(
            state_weights_by_observation_history,
            log_base=log_base,
            aliased_states=summary_aliased_states,
            weight_by_history_mass=weight_by_history_mass,
        )
        sensorimotor_summary = compute_history_entropy_summary(
            history_to_next_state_weights,
            log_base=log_base,
            aliased_states=summary_aliased_states,
            weight_by_history_mass=weight_by_history_mass,
        )

        for statistic_name in summary_keys:
            observation_summaries[statistic_name].append(observation_summary[statistic_name])
            sensorimotor_summaries[statistic_name].append(sensorimotor_summary[statistic_name])

        if history_order == max_history_length:
            break

        next_history_to_next_state_weights: dict[
            tuple[tuple[Observation, ...], tuple[Action, ...]],
            dict[State, float],
        ] = defaultdict(lambda: defaultdict(float))

        for history_key, state_weights in history_to_next_state_weights.items():
            observation_history, action_history = history_key

            for current_state, current_mass in state_weights.items():
                if current_mass <= 0.0:
                    continue

                action_mapping = transition_probabilities[current_state]
                if not action_mapping:
                    continue

                current_observation = observation_by_state[current_state]
                action_probability = 1.0 / len(action_mapping)
                for action, next_state_distribution in action_mapping.items():
                    for next_state, transition_probability in next_state_distribution.items():
                        if transition_probability <= 0.0:
                            continue

                        next_mass = (
                            current_mass
                            * action_probability
                            * transition_probability
                        )
                        next_history_key = (
                            observation_history + (current_observation,),
                            action_history + (action,),
                        )
                        next_history_to_next_state_weights[next_history_key][next_state] += next_mass

        history_to_next_state_weights = next_history_to_next_state_weights

    return {
        "observation_only": observation_summaries,
        "sensorimotor": sensorimotor_summaries,
    }


def compute_alias_state_disambiguation_curves(
    transition_probabilities: Mapping[State, Mapping[Action, Mapping[State, float]]],
    observation_by_state: Mapping[State, Observation],
    max_history_length: int,
    state_prior: Mapping[State, float] | None = None,
) -> dict[str, dict[State, list[bool]]]:
    """
    For each aliased state s, compute whether there exists a reachable history of
    order k that uniquely identifies s.

    For k = 0, sensorimotor histories condition on o_t and a_{t-1}.
    """
    if max_history_length < 0:
        raise ValueError("max_history_length must be non-negative.")

    validate_model(transition_probabilities, observation_by_state)

    states = list(transition_probabilities.keys())
    initial_state_prior = make_state_prior(states, state_prior)
    aliased_states = get_aliased_states(observation_by_state)

    observation_flags_by_state = {
        state: [] for state in sorted(aliased_states, key=str)
    }
    sensorimotor_flags_by_state = {
        state: [] for state in sorted(aliased_states, key=str)
    }

    history_to_state_weights: dict[
        tuple[tuple[Observation, ...], tuple[Action, ...]],
        dict[State, float],
    ] = defaultdict(lambda: defaultdict(float))

    for previous_state, previous_state_probability in initial_state_prior.items():
        if previous_state_probability <= 0.0:
            continue

        action_mapping = transition_probabilities[previous_state]
        if not action_mapping:
            continue

        action_probability = 1.0 / len(action_mapping)
        for action, next_state_distribution in action_mapping.items():
            for next_state, transition_probability in next_state_distribution.items():
                if transition_probability <= 0.0:
                    continue

                joint_mass = (
                    previous_state_probability
                    * action_probability
                    * transition_probability
                )
                next_observation = observation_by_state[next_state]
                history_key = ((next_observation,), (action,))
                history_to_state_weights[history_key][next_state] += joint_mass

    for history_order in range(0, max_history_length + 1):
        state_weights_by_observation_history: dict[
            tuple[Observation, ...], dict[State, float]
        ] = defaultdict(lambda: defaultdict(float))

        for history_key, state_weights in history_to_state_weights.items():
            observation_history, _ = history_key
            for state, mass in state_weights.items():
                state_weights_by_observation_history[observation_history][state] += mass

        for state in observation_flags_by_state:
            observation_flags_by_state[state].append(
                any(
                    posterior_is_singleton_state(state_weights, state)
                    for state_weights in state_weights_by_observation_history.values()
                )
            )
            sensorimotor_flags_by_state[state].append(
                any(
                    posterior_is_singleton_state(state_weights, state)
                    for state_weights in history_to_state_weights.values()
                )
            )

        if history_order == max_history_length:
            break

        next_history_to_state_weights: dict[
            tuple[tuple[Observation, ...], tuple[Action, ...]],
            dict[State, float],
        ] = defaultdict(lambda: defaultdict(float))

        for history_key, state_weights in history_to_state_weights.items():
            observation_history, action_history = history_key

            for current_state, current_mass in state_weights.items():
                if current_mass <= 0.0:
                    continue

                action_mapping = transition_probabilities[current_state]
                if not action_mapping:
                    continue

                action_probability = 1.0 / len(action_mapping)
                for action, next_state_distribution in action_mapping.items():
                    for next_state, transition_probability in next_state_distribution.items():
                        if transition_probability <= 0.0:
                            continue

                        next_mass = (
                            current_mass
                            * action_probability
                            * transition_probability
                        )
                        next_observation = observation_by_state[next_state]
                        next_history_key = (
                            observation_history + (next_observation,),
                            action_history + (action,),
                        )
                        next_history_to_state_weights[next_history_key][next_state] += next_mass

        history_to_state_weights = next_history_to_state_weights

    return {
        "observation_only": observation_flags_by_state,
        "sensorimotor": sensorimotor_flags_by_state,
    }


def find_environment_aliasing_order(
    disambiguation_flags_by_state: Mapping[State, list[bool]],
) -> int | None:
    """
    Return the smallest history order k such that every aliased state has at
    least one disambiguating history of that order.
    """
    if not disambiguation_flags_by_state:
        return None

    max_curve_length = max(len(flags) for flags in disambiguation_flags_by_state.values())

    for flag_index in range(max_curve_length):
        if all(
            flag_index < len(flags) and flags[flag_index]
            for flags in disambiguation_flags_by_state.values()
        ):
            return flag_index

    return None

# def plot_entropy_quantile_family(
#     summary_family: Mapping[str, list[float]],
#     title: str,
#     output_path: str | Path | None = None,
#     disambiguation_order: int | None = None,
# ) -> None:
#     """
#     Plot posterior-entropy quantile curves at 10% increments, with the mean highlighted.
#     """
#     history_lengths = list(range(len(summary_family["mean"])))
#     quantile_keys = [f"q{percentage:02d}" for percentage in range(0, 101, 10)]
#
#     plt.figure(figsize=(8, 5))
#
#     for quantile_key in quantile_keys:
#         percentage = int(quantile_key[1:])
#         label = f"{percentage}%"
#         line_width = 2.5 if percentage in (0, 50, 100) else 1.25
#         alpha = 0.9 if percentage in (0, 50, 100) else 0.5
#         line_style = "-" if percentage in (0, 50, 100) else "--"
#
#         plt.plot(
#             history_lengths,
#             summary_family[quantile_key],
#             linestyle=line_style,
#             linewidth=line_width,
#             alpha=alpha,
#             label=label,
#         )
#
#     plt.plot(
#         history_lengths,
#         summary_family["mean"],
#         linewidth=3.0,
#         label="Mean",
#     )
#
#     if disambiguation_order is not None:
#         plt.axvline(
#             disambiguation_order,
#             linestyle="-.",
#             linewidth=1.5,
#             alpha=0.8,
#             label=fr"$k^\star={disambiguation_order}$",
#         )
#
#     plt.xlabel("History order $k$")
#     plt.ylabel("Posterior entropy (bits)")
#     plt.title(title)
#     plt.xticks(history_lengths)
#     plt.ylim(bottom=0.0)
#     plt.grid(True, alpha=0.3)
#     plt.legend(ncol=2, fontsize=9)
#     plt.tight_layout()
#
#     if output_path is not None:
#         output_path = Path(output_path)
#         output_path.parent.mkdir(parents=True, exist_ok=True)
#         plt.savefig(output_path, dpi=300, bbox_inches="tight")
#
#     plt.show()

def plot_entropy_quantile_family(
    summary_family: Mapping[str, list[float]],
    title: str,
    output_path: str | Path | None = None,
    disambiguation_order: int | None = None,
    entropy_unit_label: str = "bits",
) -> None:
    """
    Plot posterior-entropy quantile bands at 10% increments, with the mean highlighted.

    The region between adjacent quantiles is filled to show how the distribution
    over reachable aliased-state-relevant histories evolves with history order.
    """
    history_orders = list(range(len(summary_family["mean"])))
    quantile_percentages = list(DISPLAY_QUANTILE_PERCENTAGES)
    quantile_keys = [f"q{percentage:02d}" for percentage in quantile_percentages]

    plt.figure(figsize=(8, 5))

    # Filled bands between adjacent quantiles.
    for lower_percentage, upper_percentage in zip(
        quantile_percentages[:-1],
        quantile_percentages[1:],
    ):
        lower_key = f"q{lower_percentage:02d}"
        upper_key = f"q{upper_percentage:02d}"

        # Make central bands slightly more visible than outer bands.
        band_midpoint = 0.5 * (lower_percentage + upper_percentage)
        if 40 <= band_midpoint <= 60:
            band_alpha = 0.30
        elif 20 <= band_midpoint <= 80:
            band_alpha = 0.22
        else:
            band_alpha = 0.14

        plt.fill_between(
            history_orders,
            summary_family[lower_key],
            summary_family[upper_key],
            alpha=band_alpha,
            label=f"{lower_percentage / 100.0:.1f}-{upper_percentage / 100.0:.1f}",
        )

    # Reference quantile lines.
    reference_quantiles = [0, 50, 100]
    for percentage in reference_quantiles:
        quantile_key = f"q{percentage:02d}"
        plt.plot(
            history_orders,
            summary_family[quantile_key],
            linewidth=2.0,
            linestyle="-" if percentage != 50 else "--",
            alpha=0.9,
            label=f"{percentage / 100.0:.1f}",
        )

    # Mean line.
    plt.plot(
        history_orders,
        summary_family["mean"],
        linewidth=3.0,
        label="Mean",
    )

    if disambiguation_order is not None:
        plt.axvline(
            disambiguation_order,
            linestyle="-.",
            linewidth=1.5,
            alpha=0.8,
            label=fr"$k^\star={disambiguation_order}$",
        )

    plt.xlabel("History order $k$")
    plt.ylabel(f"Posterior entropy ({entropy_unit_label})")
    plt.title(title)
    plt.xticks(history_orders)
    plt.ylim(bottom=0.0)
    plt.grid(True, alpha=0.3)

    # Keep legend compact.
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_entropy_quantile_surface(
    summary_family: Mapping[str, list[float]],
    title: str,
    output_path: str | Path | None = None,
    disambiguation_order: int | None = None,
    entropy_unit_label: str = "bits",
) -> None:
    """
    Plot posterior entropy as a history-by-history-fraction colour surface.

    Fraction of histories and history order are the spatial axes. Entropy is
    represented by colour, not by a third spatial axis.
    """
    history_orders = np.asarray(range(len(summary_family["mean"])), dtype=float)
    quantile_percentages = np.asarray(SUMMARY_QUANTILE_PERCENTAGES, dtype=float)
    quantile_fractions = np.asarray(SUMMARY_QUANTILE_FRACTIONS, dtype=float)
    entropy_surface = np.asarray(
        [
            summary_family[f"q{int(percentage):02d}"]
            for percentage in quantile_percentages
        ],
        dtype=float,
    )

    if entropy_surface.shape != (len(quantile_percentages), len(history_orders)):
        raise ValueError("Quantile entropy summaries must form a percentile-by-history grid.")

    figure, axis = plt.subplots(figsize=(8, 5))
    mesh = axis.pcolormesh(
        history_orders,
        quantile_fractions,
        entropy_surface,
        shading="nearest",
        cmap="viridis",
    )

    if disambiguation_order is not None:
        axis.axvline(
            disambiguation_order,
            color="white",
            linestyle="-.",
            linewidth=1.5,
            alpha=0.9,
        )

    axis.set_xlabel("History order $k$")
    axis.set_ylabel("Fraction of histories")
    axis.set_title(title)
    axis.set_xticks(history_orders)
    axis.set_yticks(DISPLAY_QUANTILE_FRACTIONS)
    axis.set_ylim(quantile_fractions[0], quantile_fractions[-1])

    colourbar = figure.colorbar(mesh, ax=axis)
    colourbar.set_label(f"Posterior entropy ({entropy_unit_label})")

    figure.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
    else:
        plt.show()


def entropy_heatmap_grid(
    summary_family: Mapping[str, list[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return history coordinates, history-fraction coordinates, and entropy grid.
    """
    history_orders = np.asarray(range(len(summary_family["mean"])), dtype=float)
    quantile_percentages = np.asarray(SUMMARY_QUANTILE_PERCENTAGES, dtype=float)
    quantile_fractions = np.asarray(SUMMARY_QUANTILE_FRACTIONS, dtype=float)
    entropy_surface = np.asarray(
        [
            summary_family[f"q{int(percentage):02d}"]
            for percentage in quantile_percentages
        ],
        dtype=float,
    )

    if entropy_surface.shape != (len(quantile_fractions), len(history_orders)):
        raise ValueError("Quantile entropy summaries must form a fraction-by-history grid.")

    return history_orders, quantile_fractions, entropy_surface


def integrate_entropy_heatmap(
    summary_family: Mapping[str, list[float]],
) -> float:
    """
    Integrate the entropy heatmap over history order and fraction of histories.
    """
    history_orders, quantile_fractions, entropy_surface = entropy_heatmap_grid(
        summary_family
    )
    integral_by_history = np.trapezoid(entropy_surface, x=quantile_fractions, axis=0)
    return float(np.trapezoid(integral_by_history, x=history_orders))


def integrate_entropy_heatmap_area_below_threshold(
    summary_family: Mapping[str, list[float]],
    threshold: float = NEAR_ZERO_ENTROPY_THRESHOLD,
) -> float:
    """
    Integrate the heatmap domain area where entropy is below a threshold.
    """
    history_orders, quantile_fractions, entropy_surface = entropy_heatmap_grid(
        summary_family
    )
    threshold_mask = (entropy_surface < float(threshold)).astype(float)
    area_by_history = np.trapezoid(threshold_mask, x=quantile_fractions, axis=0)
    return float(np.trapezoid(area_by_history, x=history_orders))


def plot_entropy_summary_family(
    summary_family: Mapping[str, list[float]],
    title: str,
    output_path: str | Path | None = None,
    disambiguation_order: int | None = None,
) -> None:
    """Plot min/mean/max entropy summaries with an interquartile band."""
    history_orders = list(range(len(summary_family["mean"])))

    min_values = summary_family["min"]
    q25_values = summary_family["q25"]
    mean_values = summary_family["mean"]
    q75_values = summary_family["q75"]
    max_values = summary_family["max"]

    plt.figure(figsize=(7, 4.5))

    plt.fill_between(
        history_orders,
        q25_values,
        q75_values,
        alpha=0.2,
        label="25–75%",
    )
    plt.plot(history_orders, mean_values, linestyle="-", linewidth=2.0, label="Mean")
    plt.plot(history_orders, min_values, linestyle="--", label="Min")
    plt.plot(history_orders, max_values, linestyle=":", linewidth=2.0, label="Max")

    if disambiguation_order is not None:
        plt.axvline(
            disambiguation_order,
            linestyle="-.",
            linewidth=1.5,
            alpha=0.8,
            label=fr"$k^\star={disambiguation_order}$",
        )

    plt.xlabel("History order $k$")
    plt.ylabel("Posterior entropy (bits)")
    plt.title(title)
    plt.xticks(history_orders)
    plt.ylim(bottom=0.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_observation_and_sensorimotor_entropy_summaries(
    summaries: Mapping[str, Mapping[str, list[float]]],
    output_directory: str | Path | None = None,
    filename_stem: str = "entropy_curve",
    observation_order: int | None = None,
    sensorimotor_order: int | None = None,
    title_prefix: str = "",
    target_state_label: str = r"S_t",
    sensorimotor_action_label: str = r"A_{t-1-k:t-1}",
    entropy_unit_label: str = "bits",
) -> None:
    """
    Plot separate figures for observation-only and sensorimotor quantile summaries.
    """
    output_directory_path = Path(output_directory) if output_directory is not None else None

    observation_output_path = (
        output_directory_path / f"{filename_stem}_observation_only.png"
        if output_directory_path is not None
        else None
    )
    sensorimotor_output_path = (
        output_directory_path / f"{filename_stem}_sensorimotor.png"
        if output_directory_path is not None
        else None
    )
    observation_surface_output_path = (
        output_directory_path / f"{filename_stem}_observation_only_surface.png"
        if output_directory_path is not None
        else None
    )
    sensorimotor_surface_output_path = (
        output_directory_path / f"{filename_stem}_sensorimotor_surface.png"
        if output_directory_path is not None
        else None
    )

    # plot_entropy_quantile_family(
    #     summary_family=summaries["observation_only"],
    #     title=title_prefix + r"Observation-only histories: $H(S_t \mid O_{t-k:t})$",
    #     output_path=observation_output_path,
    #     disambiguation_order=observation_order,
    # )

    # plot_entropy_quantile_family(
    #     summary_family=summaries["sensorimotor"],
    #     title=title_prefix + r"Sensorimotor histories: $H(S_t \mid O_{t-k:t}, A_{t-1-k:t-1})$",
    #     output_path=sensorimotor_output_path,
    #     disambiguation_order=sensorimotor_order,
    # )

    plot_entropy_quantile_surface(
        summary_family=summaries["observation_only"],
        title=title_prefix + rf"Observation-only histories: $H({target_state_label} \mid O_{{t-k:t}})$",
        output_path=observation_surface_output_path,
        disambiguation_order=observation_order,
        entropy_unit_label=entropy_unit_label,
    )

    plot_entropy_quantile_surface(
        summary_family=summaries["sensorimotor"],
        title=title_prefix + rf"Sensorimotor histories: $H({target_state_label} \mid O_{{t-k:t}}, {sensorimotor_action_label})$",
        output_path=sensorimotor_surface_output_path,
        disambiguation_order=sensorimotor_order,
        entropy_unit_label=entropy_unit_label,
    )


def print_summary_table(
    *,
    heading: str,
    summaries: Mapping[str, Mapping[str, list[float]]],
) -> None:
    print(heading)
    for history_order in range(len(summaries["observation_only"]["mean"])):
        observation_values = [
            f"{summaries['observation_only'][f'q{percentage:02d}'][history_order]:.4f}"
            for percentage in DISPLAY_QUANTILE_PERCENTAGES
        ]
        sensorimotor_values = [
            f"{summaries['sensorimotor'][f'q{percentage:02d}'][history_order]:.4f}"
            for percentage in DISPLAY_QUANTILE_PERCENTAGES
        ]

        print(
            f"{history_order} | "
            + " | ".join(observation_values)
            + f" | {summaries['observation_only']['mean'][history_order]:.4f} | "
            + " | ".join(sensorimotor_values)
            + f" | {summaries['sensorimotor']['mean'][history_order]:.4f}"
        )


def print_heatmap_integrals(
    *,
    heading: str,
    summaries: Mapping[str, Mapping[str, list[float]]],
) -> None:
    print(heading)
    print(
        "observation_only | "
        f"{integrate_entropy_heatmap(summaries['observation_only']):.6f}"
    )
    print(
        "sensorimotor | "
        f"{integrate_entropy_heatmap(summaries['sensorimotor']):.6f}"
    )


def build_empirical_integral_records(
    *,
    rand_prob: float,
    valid_trajectories_only: bool,
    unweighted_summaries: Mapping[str, Mapping[str, list[float]]],
    weighted_summaries: Mapping[str, Mapping[str, list[float]]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for summary_mode, summaries in (
        ("structural_unweighted", unweighted_summaries),
        ("probability_weighted", weighted_summaries),
    ):
        for history_type in ("observation_only", "sensorimotor"):
            records.append(
                {
                    "rand_prob": float(rand_prob),
                    "valid_trajectories_only": bool(valid_trajectories_only),
                    "summary_mode": summary_mode,
                    "history_type": history_type,
                    "integral": integrate_entropy_heatmap(summaries[history_type]),
                }
            )
    return records


def build_empirical_near_zero_area_records(
    *,
    rand_prob: float,
    valid_trajectories_only: bool,
    unweighted_summaries: Mapping[str, Mapping[str, list[float]]],
    weighted_summaries: Mapping[str, Mapping[str, list[float]]],
    threshold: float = NEAR_ZERO_ENTROPY_THRESHOLD,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for summary_mode, summaries in (
        ("structural_unweighted", unweighted_summaries),
        ("probability_weighted", weighted_summaries),
    ):
        for history_type in ("observation_only", "sensorimotor"):
            records.append(
                {
                    "rand_prob": float(rand_prob),
                    "valid_trajectories_only": bool(valid_trajectories_only),
                    "summary_mode": summary_mode,
                    "history_type": history_type,
                    "area": integrate_entropy_heatmap_area_below_threshold(
                        summaries[history_type],
                        threshold=threshold,
                    ),
                }
            )
    return records


def plot_empirical_integral_records(
    records: Iterable[Mapping[str, Any]],
    *,
    output_path: str | Path,
    title: str,
    y_label: str = "Entropy heatmap integral",
) -> None:
    records_by_line: dict[tuple[bool, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            bool(record["valid_trajectories_only"]),
            str(record["summary_mode"]),
            str(record["history_type"]),
        )
        records_by_line[key].append(record)

    if not records_by_line:
        raise ValueError("At least one integral record is required.")

    colour_by_primary_condition = {
        (False, "structural_unweighted"): "tab:blue",
        (False, "probability_weighted"): "tab:orange",
        (True, "structural_unweighted"): "tab:green",
        (True, "probability_weighted"): "tab:red",
    }
    marker_by_history_type = {
        "observation_only": "o",
        "sensorimotor": "s",
    }
    linestyle_by_history_type = {
        "observation_only": "-",
        "sensorimotor": "--",
    }

    figure, axis = plt.subplots(figsize=(8.5, 5.25))

    for (valid_only, summary_mode, history_type), line_records in sorted(records_by_line.items()):
        if summary_mode == "structural_unweighted":
            # Structural-unweighted integral lines are intentionally hidden from this plot.
            continue
        sorted_records = sorted(line_records, key=lambda item: float(item["rand_prob"]))
        validity_label = "validonly" if valid_only else "allsteps"
        summary_label = (
            "structural"
            if summary_mode == "structural_unweighted"
            else "probability"
        )
        history_label = "obs" if history_type == "observation_only" else "sensorimotor"
        axis.plot(
            [float(record["rand_prob"]) for record in sorted_records],
            [float(record["integral"]) for record in sorted_records],
            color=colour_by_primary_condition[(valid_only, summary_mode)],
            marker=marker_by_history_type[history_type],
            linestyle=linestyle_by_history_type[history_type],
            linewidth=2.0,
            markersize=4.5,
            label=f"{validity_label} {summary_label} {history_label}",
        )

    axis.set_xlabel("Random action probability $p$")
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.set_xticks([round(i / 10, 1) for i in range(11)])
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(bottom=0.0)
    axis.grid(True, alpha=0.3)
    axis.legend(ncol=2, fontsize=8.5)
    figure.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_empirical_integral_records_for_summary_mode(
    records: Iterable[Mapping[str, Any]],
    *,
    output_path: str | Path,
    title: str,
    summary_mode: str,
    summary_label: str,
    metric_key: str = "integral",
    y_label: str = "Entropy heatmap integral",
) -> None:
    records_by_line: dict[tuple[bool, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if str(record["summary_mode"]) != summary_mode:
            continue
        key = (
            bool(record["valid_trajectories_only"]),
            str(record["history_type"]),
        )
        records_by_line[key].append(record)

    if not records_by_line:
        raise ValueError(f"At least one {summary_label} integral record is required.")

    colour_by_validity = {
        False: "tab:blue",
        True: "tab:green",
    }
    marker_by_history_type = {
        "observation_only": "o",
        "sensorimotor": "s",
    }
    linestyle_by_history_type = {
        "observation_only": "-",
        "sensorimotor": "--",
    }

    figure, axis = plt.subplots(figsize=(8.5, 5.25))

    for (valid_only, history_type), line_records in sorted(records_by_line.items()):
        sorted_records = sorted(line_records, key=lambda item: float(item["rand_prob"]))
        validity_label = "validonly" if valid_only else "allsteps"
        history_label = "obs" if history_type == "observation_only" else "sensorimotor"
        axis.plot(
            [float(record["rand_prob"]) for record in sorted_records],
            [float(record[metric_key]) for record in sorted_records],
            color=colour_by_validity[valid_only],
            marker=marker_by_history_type[history_type],
            linestyle=linestyle_by_history_type[history_type],
            linewidth=2.0,
            markersize=4.5,
            label=f"{validity_label} {summary_label} {history_label}",
        )

    axis.set_xlabel("Random action probability $p$")
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.set_xticks([round(i / 10, 1) for i in range(11)])
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(bottom=0.0)
    axis.grid(True, alpha=0.3)
    axis.legend(ncol=2, fontsize=8.5)
    figure.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_empirical_structural_integral_records(
    records: Iterable[Mapping[str, Any]],
    *,
    output_path: str | Path,
    title: str,
    y_label: str = "Entropy heatmap integral",
) -> None:
    _plot_empirical_integral_records_for_summary_mode(
        records,
        output_path=output_path,
        title=title,
        summary_mode="structural_unweighted",
        summary_label="structural",
        y_label=y_label,
    )


def plot_empirical_near_zero_area_records(
    records: Iterable[Mapping[str, Any]],
    *,
    output_path: str | Path,
    title: str,
    y_label: str = "Near-zero entropy area",
) -> None:
    _plot_empirical_integral_records_for_summary_mode(
        records,
        output_path=output_path,
        title=title,
        summary_mode="probability_weighted",
        summary_label="probability",
        metric_key="area",
        y_label=y_label,
    )


def plot_empirical_structural_near_zero_area_records(
    records: Iterable[Mapping[str, Any]],
    *,
    output_path: str | Path,
    title: str,
    y_label: str = "Near-zero entropy area",
) -> None:
    _plot_empirical_integral_records_for_summary_mode(
        records,
        output_path=output_path,
        title=title,
        summary_mode="structural_unweighted",
        summary_label="structural",
        metric_key="area",
        y_label=y_label,
    )


def _integral_records_by_level_for_summary_mode(
    records: Iterable[Mapping[str, Any]],
    *,
    summary_mode: str,
) -> dict[tuple[int, bool, str], list[Mapping[str, Any]]]:
    records_by_line: dict[tuple[int, bool, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if str(record["summary_mode"]) != summary_mode:
            continue
        key = (
            int(record["level_index"]),
            bool(record["valid_trajectories_only"]),
            str(record["history_type"]),
        )
        records_by_line[key].append(record)
    return records_by_line


def _probability_weighted_integral_records_by_level(
    records: Iterable[Mapping[str, Any]],
) -> dict[tuple[int, bool, str], list[Mapping[str, Any]]]:
    return _integral_records_by_level_for_summary_mode(
        records,
        summary_mode="probability_weighted",
    )


def _structural_unweighted_integral_records_by_level(
    records: Iterable[Mapping[str, Any]],
) -> dict[tuple[int, bool, str], list[Mapping[str, Any]]]:
    return _integral_records_by_level_for_summary_mode(
        records,
        summary_mode="structural_unweighted",
    )


def _plot_empirical_integral_records_by_level_for_summary_mode(
    records: Iterable[Mapping[str, Any]],
    *,
    output_path: str | Path,
    title: str,
    summary_mode: str,
    summary_label: str,
    metric_key: str = "integral",
    y_label: str = "Entropy heatmap integral",
) -> None:
    records_by_line = _integral_records_by_level_for_summary_mode(
        records,
        summary_mode=summary_mode,
    )
    if not records_by_line:
        raise ValueError(f"At least one {summary_label} integral record is required.")

    levels_in_plot = sorted({level_index for level_index, _, _ in records_by_line})
    colour_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not colour_cycle:
        colour_cycle = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    colour_by_level = {
        level_index: colour_cycle[index % len(colour_cycle)]
        for index, level_index in enumerate(levels_in_plot)
    }
    marker_by_validity = {
        False: "o",
        True: "s",
    }
    linestyle_by_history_type = {
        "observation_only": "-",
        "sensorimotor": "--",
    }

    figure, axis = plt.subplots(figsize=(9.25, 5.5))

    for (level_index, valid_only, history_type), line_records in sorted(records_by_line.items()):
        sorted_records = sorted(line_records, key=lambda item: float(item["rand_prob"]))
        validity_label = "validonly" if valid_only else "allsteps"
        history_label = "obs" if history_type == "observation_only" else "sensorimotor"
        axis.plot(
            [float(record["rand_prob"]) for record in sorted_records],
            [float(record[metric_key]) for record in sorted_records],
            color=colour_by_level[level_index],
            marker=marker_by_validity[valid_only],
            linestyle=linestyle_by_history_type[history_type],
            linewidth=2.0,
            markersize=4.5,
            label=f"level {level_index} {validity_label} {history_label}",
        )

    axis.set_xlabel("Random action probability $p$")
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.set_xticks([round(i / 10, 1) for i in range(11)])
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(bottom=0.0)
    axis.grid(True, alpha=0.3)
    axis.legend(ncol=2, fontsize=8.0)
    figure.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_empirical_integral_records_by_level(
    records: Iterable[Mapping[str, Any]],
    *,
    output_path: str | Path,
    title: str,
    y_label: str = "Entropy heatmap integral",
) -> None:
    _plot_empirical_integral_records_by_level_for_summary_mode(
        records,
        output_path=output_path,
        title=title,
        summary_mode="probability_weighted",
        summary_label="probability-weighted",
        y_label=y_label,
    )


def plot_empirical_structural_integral_records_by_level(
    records: Iterable[Mapping[str, Any]],
    *,
    output_path: str | Path,
    title: str,
    y_label: str = "Entropy heatmap integral",
) -> None:
    _plot_empirical_integral_records_by_level_for_summary_mode(
        records,
        output_path=output_path,
        title=title,
        summary_mode="structural_unweighted",
        summary_label="structural-unweighted",
        y_label=y_label,
    )


def plot_empirical_near_zero_area_records_by_level(
    records: Iterable[Mapping[str, Any]],
    *,
    output_path: str | Path,
    title: str,
    y_label: str = "Near-zero entropy area",
) -> None:
    _plot_empirical_integral_records_by_level_for_summary_mode(
        records,
        output_path=output_path,
        title=title,
        summary_mode="probability_weighted",
        summary_label="probability-weighted",
        metric_key="area",
        y_label=y_label,
    )


def plot_empirical_structural_near_zero_area_records_by_level(
    records: Iterable[Mapping[str, Any]],
    *,
    output_path: str | Path,
    title: str,
    y_label: str = "Near-zero entropy area",
) -> None:
    _plot_empirical_integral_records_by_level_for_summary_mode(
        records,
        output_path=output_path,
        title=title,
        summary_mode="structural_unweighted",
        summary_label="structural-unweighted",
        metric_key="area",
        y_label=y_label,
    )


def compute_entropy_outputs(
    *,
    transition_probabilities: Mapping[State, Mapping[Action, Mapping[State, float]]],
    observation_by_state: Mapping[State, Observation],
    log_base: float = 2.0,
) -> tuple[
    dict[str, dict[str, list[float]]],
    dict[str, dict[str, list[float]]],
    dict[str, dict[State, list[bool]]],
    int | None,
    int | None,
]:
    unweighted_summaries = compute_history_conditioned_state_entropy_summaries(
        transition_probabilities=transition_probabilities,
        observation_by_state=observation_by_state,
        max_history_length=MAX_HISTORY_ORDER,
        log_base=log_base,
        weight_by_history_mass=False,
        restrict_to_aliased_histories=True,
    )
    weighted_summaries = compute_history_conditioned_state_entropy_summaries(
        transition_probabilities=transition_probabilities,
        observation_by_state=observation_by_state,
        max_history_length=MAX_HISTORY_ORDER,
        log_base=log_base,
        weight_by_history_mass=True,
        restrict_to_aliased_histories=True,
    )

    disambiguation_flags = compute_alias_state_disambiguation_curves(
        transition_probabilities=transition_probabilities,
        observation_by_state=observation_by_state,
        max_history_length=MAX_HISTORY_ORDER,
    )

    observation_order = find_environment_aliasing_order(
        disambiguation_flags["observation_only"]
    )
    sensorimotor_order = find_environment_aliasing_order(
        disambiguation_flags["sensorimotor"]
    )

    return (
        unweighted_summaries,
        weighted_summaries,
        disambiguation_flags,
        observation_order,
        sensorimotor_order,
    )


def compute_next_state_entropy_outputs(
    *,
    transition_probabilities: Mapping[State, Mapping[Action, Mapping[State, float]]],
    observation_by_state: Mapping[State, Observation],
    log_base: float = 2.0,
) -> tuple[
    dict[str, dict[str, list[float]]],
    dict[str, dict[str, list[float]]],
]:
    unweighted_summaries = compute_history_conditioned_next_state_entropy_summaries(
        transition_probabilities=transition_probabilities,
        observation_by_state=observation_by_state,
        max_history_length=MAX_HISTORY_ORDER,
        log_base=log_base,
        weight_by_history_mass=False,
        restrict_to_aliased_histories=True,
    )
    weighted_summaries = compute_history_conditioned_next_state_entropy_summaries(
        transition_probabilities=transition_probabilities,
        observation_by_state=observation_by_state,
        max_history_length=MAX_HISTORY_ORDER,
        log_base=log_base,
        weight_by_history_mass=True,
        restrict_to_aliased_histories=True,
    )

    return unweighted_summaries, weighted_summaries


def run_empirical_entropy_outputs_for_level(
    level_index: int,
    *,
    log_base: float = 2.0,
    output_stem_suffix: str = "",
    title_entropy_label: str = "entropy",
    entropy_unit_label: str = "bits",
    near_zero_threshold: float = NEAR_ZERO_ENTROPY_THRESHOLD,
    entropy_scale: float = 1.0,
) -> dict[str, list[dict[str, Any]]]:
    print()
    print(f"=== Empirical entropy outputs for level {level_index} ===")

    runs_by_condition = generate_empirical_entropy_run_ids(
        level_index=level_index,
        datasets_root=DATASETS_ROOT,
        seeds=EMPIRICAL_SEEDS,
        random_probabilities=EMPIRICAL_RANDOM_PROBABILITIES,
        valid_only_options=EMPIRICAL_VALID_ONLY_OPTIONS,
        num_steps=EMPIRICAL_NUM_STEPS,
        random_start=EMPIRICAL_RANDOM_START,
        reuse_existing=True,
    )

    output_stem = _level_output_stem(
        level_index=level_index,
        sensor=SENSOR,
        sensor_range=SENSOR_RANGE,
    )
    output_stem = f"{output_stem}{output_stem_suffix}"

    empirical_integral_records: list[dict[str, Any]] = []
    empirical_near_zero_area_records: list[dict[str, Any]] = []

    for (rand_prob, valid_only), runs in sorted(runs_by_condition.items()):
        validity_stem = "validonly" if valid_only else "allsteps"
        condition_stem = f"{output_stem}_empirical_p{rand_prob:g}_{validity_stem}_randomstart"
        condition_label = (
            f"Empirical p={rand_prob:g}, {validity_stem}, random_start: "
        )

        unweighted_summaries, weighted_summaries, metadata_totals = compute_empirical_condition_outputs(
            runs=runs,
            datasets_root=DATASETS_ROOT,
            level_index=level_index,
            max_history_length=MAX_HISTORY_ORDER,
            log_base=log_base,
        )
        if not math.isclose(float(entropy_scale), 1.0):
            unweighted_summaries = scale_entropy_summaries(
                unweighted_summaries,
                entropy_scale,
            )
            weighted_summaries = scale_entropy_summaries(
                weighted_summaries,
                entropy_scale,
            )
        condition_integral_records = build_empirical_integral_records(
            rand_prob=rand_prob,
            valid_trajectories_only=valid_only,
            unweighted_summaries=unweighted_summaries,
            weighted_summaries=weighted_summaries,
        )
        condition_near_zero_area_records = build_empirical_near_zero_area_records(
            rand_prob=rand_prob,
            valid_trajectories_only=valid_only,
            unweighted_summaries=unweighted_summaries,
            weighted_summaries=weighted_summaries,
            threshold=near_zero_threshold,
        )
        for record in condition_integral_records:
            empirical_integral_records.append(
                {
                    **record,
                    "level_index": int(level_index),
                }
            )
        for record in condition_near_zero_area_records:
            empirical_near_zero_area_records.append(
                {
                    **record,
                    "level_index": int(level_index),
                }
            )

        print()
        print_empirical_metadata_totals(
            heading=f"Empirical metadata totals for p={rand_prob:g}, {validity_stem}:",
            metadata_totals=metadata_totals,
        )

        print()
        print_summary_table(
            heading=f"Empirical {title_entropy_label} structural unweighted summaries for p={rand_prob:g}, {validity_stem}:",
            summaries=unweighted_summaries,
        )
        print_heatmap_integrals(
            heading=f"Empirical structural unweighted {title_entropy_label} heatmap integrals for p={rand_prob:g}, {validity_stem}:",
            summaries=unweighted_summaries,
        )

        print()
        print_summary_table(
            heading=f"Empirical {title_entropy_label} probability-weighted summaries for p={rand_prob:g}, {validity_stem}:",
            summaries=weighted_summaries,
        )
        print_heatmap_integrals(
            heading=f"Empirical probability-weighted {title_entropy_label} heatmap integrals for p={rand_prob:g}, {validity_stem}:",
            summaries=weighted_summaries,
        )

        plot_observation_and_sensorimotor_entropy_summaries(
            summaries=unweighted_summaries,
            output_directory=OUTPUT_DIRECTORY,
            filename_stem=f"{condition_stem}_structural_unweighted",
            title_prefix=condition_label + f"structural unweighted {title_entropy_label} sensory histories: ",
            entropy_unit_label=entropy_unit_label,
        )
        plot_observation_and_sensorimotor_entropy_summaries(
            summaries=weighted_summaries,
            output_directory=OUTPUT_DIRECTORY,
            filename_stem=f"{condition_stem}_probability_weighted",
            title_prefix=condition_label + f"probability-weighted {title_entropy_label} sensory histories: ",
            entropy_unit_label=entropy_unit_label,
        )

    plot_empirical_integral_records(
        empirical_integral_records,
        output_path=OUTPUT_DIRECTORY / f"{output_stem}_empirical_integrals_by_rand_prob.png",
        title=f"Empirical {title_entropy_label} heatmap integrals for {output_stem}",
        y_label=f"{title_entropy_label.capitalize()} heatmap integral",
    )
    plot_empirical_structural_integral_records(
        empirical_integral_records,
        output_path=OUTPUT_DIRECTORY / f"{output_stem}_empirical_structural_integrals_by_rand_prob.png",
        title=f"Empirical structural {title_entropy_label} heatmap integrals for {output_stem}",
        y_label=f"{title_entropy_label.capitalize()} heatmap integral",
    )
    plot_empirical_near_zero_area_records(
        empirical_near_zero_area_records,
        output_path=OUTPUT_DIRECTORY / f"{output_stem}_empirical_near_zero_area_by_rand_prob.png",
        title=f"Empirical near-zero {title_entropy_label} area for {output_stem}",
    )
    plot_empirical_structural_near_zero_area_records(
        empirical_near_zero_area_records,
        output_path=OUTPUT_DIRECTORY / f"{output_stem}_empirical_structural_near_zero_area_by_rand_prob.png",
        title=f"Empirical structural near-zero {title_entropy_label} area for {output_stem}",
    )

    return {
        "integral": empirical_integral_records,
        "near_zero_area": empirical_near_zero_area_records,
    }


def run_empirical_next_state_entropy_outputs_for_level(
    level_index: int,
    *,
    log_base: float = 2.0,
    output_stem_suffix: str = "",
    title_entropy_label: str = "entropy",
    entropy_unit_label: str = "bits",
    near_zero_threshold: float = NEAR_ZERO_ENTROPY_THRESHOLD,
    entropy_scale: float = 1.0,
) -> dict[str, list[dict[str, Any]]]:
    print()
    print(f"=== Empirical next-state entropy outputs for level {level_index} ===")

    runs_by_condition = generate_empirical_entropy_run_ids(
        level_index=level_index,
        datasets_root=DATASETS_ROOT,
        seeds=EMPIRICAL_SEEDS,
        random_probabilities=EMPIRICAL_RANDOM_PROBABILITIES,
        valid_only_options=EMPIRICAL_VALID_ONLY_OPTIONS,
        num_steps=EMPIRICAL_NUM_STEPS,
        random_start=EMPIRICAL_RANDOM_START,
        reuse_existing=True,
    )

    output_stem = _level_output_stem(
        level_index=level_index,
        sensor=SENSOR,
        sensor_range=SENSOR_RANGE,
    )
    output_stem = f"{output_stem}_next_state{output_stem_suffix}"

    empirical_integral_records: list[dict[str, Any]] = []
    empirical_near_zero_area_records: list[dict[str, Any]] = []

    for (rand_prob, valid_only), runs in sorted(runs_by_condition.items()):
        validity_stem = "validonly" if valid_only else "allsteps"
        condition_stem = f"{output_stem}_empirical_p{rand_prob:g}_{validity_stem}_randomstart"
        condition_label = (
            f"Empirical p={rand_prob:g}, {validity_stem}, random_start: "
        )

        unweighted_summaries, weighted_summaries, metadata_totals = compute_empirical_next_state_condition_outputs(
            runs=runs,
            datasets_root=DATASETS_ROOT,
            level_index=level_index,
            max_history_length=MAX_HISTORY_ORDER,
            log_base=log_base,
        )
        if not math.isclose(float(entropy_scale), 1.0):
            unweighted_summaries = scale_entropy_summaries(
                unweighted_summaries,
                entropy_scale,
            )
            weighted_summaries = scale_entropy_summaries(
                weighted_summaries,
                entropy_scale,
            )
        condition_integral_records = build_empirical_integral_records(
            rand_prob=rand_prob,
            valid_trajectories_only=valid_only,
            unweighted_summaries=unweighted_summaries,
            weighted_summaries=weighted_summaries,
        )
        condition_near_zero_area_records = build_empirical_near_zero_area_records(
            rand_prob=rand_prob,
            valid_trajectories_only=valid_only,
            unweighted_summaries=unweighted_summaries,
            weighted_summaries=weighted_summaries,
            threshold=near_zero_threshold,
        )
        for record in condition_integral_records:
            empirical_integral_records.append(
                {
                    **record,
                    "level_index": int(level_index),
                }
            )
        for record in condition_near_zero_area_records:
            empirical_near_zero_area_records.append(
                {
                    **record,
                    "level_index": int(level_index),
                }
            )

        print()
        print_empirical_metadata_totals(
            heading=f"Empirical next-state metadata totals for p={rand_prob:g}, {validity_stem}:",
            metadata_totals=metadata_totals,
        )

        print()
        print_summary_table(
            heading=f"Empirical next-state {title_entropy_label} structural unweighted summaries for p={rand_prob:g}, {validity_stem}:",
            summaries=unweighted_summaries,
        )
        print_heatmap_integrals(
            heading=f"Empirical next-state structural unweighted {title_entropy_label} heatmap integrals for p={rand_prob:g}, {validity_stem}:",
            summaries=unweighted_summaries,
        )

        print()
        print_summary_table(
            heading=f"Empirical next-state {title_entropy_label} probability-weighted summaries for p={rand_prob:g}, {validity_stem}:",
            summaries=weighted_summaries,
        )
        print_heatmap_integrals(
            heading=f"Empirical next-state probability-weighted {title_entropy_label} heatmap integrals for p={rand_prob:g}, {validity_stem}:",
            summaries=weighted_summaries,
        )

        plot_observation_and_sensorimotor_entropy_summaries(
            summaries=unweighted_summaries,
            output_directory=OUTPUT_DIRECTORY,
            filename_stem=f"{condition_stem}_structural_unweighted",
            title_prefix=condition_label + f"structural unweighted {title_entropy_label} sensory histories: ",
            target_state_label=r"S_{t+1}",
            sensorimotor_action_label=r"A_{t-k:t}",
            entropy_unit_label=entropy_unit_label,
        )
        plot_observation_and_sensorimotor_entropy_summaries(
            summaries=weighted_summaries,
            output_directory=OUTPUT_DIRECTORY,
            filename_stem=f"{condition_stem}_probability_weighted",
            title_prefix=condition_label + f"probability-weighted {title_entropy_label} sensory histories: ",
            target_state_label=r"S_{t+1}",
            sensorimotor_action_label=r"A_{t-k:t}",
            entropy_unit_label=entropy_unit_label,
        )

    plot_empirical_integral_records(
        empirical_integral_records,
        output_path=OUTPUT_DIRECTORY / f"{output_stem}_empirical_integrals_by_rand_prob.png",
        title=f"Empirical next-state {title_entropy_label} heatmap integrals for {output_stem}",
        y_label=f"{title_entropy_label.capitalize()} heatmap integral",
    )
    plot_empirical_structural_integral_records(
        empirical_integral_records,
        output_path=OUTPUT_DIRECTORY / f"{output_stem}_empirical_structural_integrals_by_rand_prob.png",
        title=f"Empirical next-state structural {title_entropy_label} heatmap integrals for {output_stem}",
        y_label=f"{title_entropy_label.capitalize()} heatmap integral",
    )
    plot_empirical_near_zero_area_records(
        empirical_near_zero_area_records,
        output_path=OUTPUT_DIRECTORY / f"{output_stem}_empirical_near_zero_area_by_rand_prob.png",
        title=f"Empirical next-state near-zero {title_entropy_label} area for {output_stem}",
    )
    plot_empirical_structural_near_zero_area_records(
        empirical_near_zero_area_records,
        output_path=OUTPUT_DIRECTORY / f"{output_stem}_empirical_structural_near_zero_area_by_rand_prob.png",
        title=f"Empirical next-state structural near-zero {title_entropy_label} area for {output_stem}",
    )

    return {
        "integral": empirical_integral_records,
        "near_zero_area": empirical_near_zero_area_records,
    }


def run_empirical_next_state_entropy_outputs_for_levels(
    level_indices: Iterable[int] = LEVEL_INDICES,
) -> dict[str, list[dict[str, Any]]]:
    empirical_integral_records: list[dict[str, Any]] = []
    empirical_near_zero_area_records: list[dict[str, Any]] = []
    for level_index in level_indices:
        output_records = run_empirical_next_state_entropy_outputs_for_level(level_index)
        empirical_integral_records.extend(output_records["integral"])
        empirical_near_zero_area_records.extend(output_records["near_zero_area"])

    plot_empirical_integral_records_by_level(
        empirical_integral_records,
        output_path=OUTPUT_DIRECTORY / "empirical_next_state_integrals_by_level_and_rand_prob.png",
        title="Empirical next-state entropy heatmap integrals by level",
    )
    plot_empirical_structural_integral_records_by_level(
        empirical_integral_records,
        output_path=OUTPUT_DIRECTORY / "empirical_next_state_structural_integrals_by_level_and_rand_prob.png",
        title="Empirical next-state structural entropy heatmap integrals by level",
    )
    plot_empirical_near_zero_area_records_by_level(
        empirical_near_zero_area_records,
        output_path=OUTPUT_DIRECTORY / "empirical_next_state_near_zero_area_by_level_and_rand_prob.png",
        title="Empirical next-state near-zero entropy area by level",
    )
    plot_empirical_structural_near_zero_area_records_by_level(
        empirical_near_zero_area_records,
        output_path=OUTPUT_DIRECTORY / "empirical_next_state_structural_near_zero_area_by_level_and_rand_prob.png",
        title="Empirical next-state structural near-zero entropy area by level",
    )
    return {
        "integral": empirical_integral_records,
        "near_zero_area": empirical_near_zero_area_records,
    }


def run_empirical_normalized_next_state_entropy_outputs_for_level(
    level_index: int,
) -> dict[str, list[dict[str, Any]]]:
    log_base = state_count_entropy_log_base(level_index)
    near_zero_threshold = bit_threshold_to_state_normalized_threshold(
        threshold_bits=NEAR_ZERO_ENTROPY_THRESHOLD,
        level_index=level_index,
    )
    return run_empirical_next_state_entropy_outputs_for_level(
        level_index,
        log_base=log_base,
        output_stem_suffix="_normalized_by_num_states",
        title_entropy_label="normalized entropy",
        entropy_unit_label=r"log base $|S|$",
        near_zero_threshold=near_zero_threshold,
    )


def run_empirical_normalized_next_state_entropy_outputs_for_levels(
    level_indices: Iterable[int] = LEVEL_INDICES,
) -> dict[str, list[dict[str, Any]]]:
    empirical_integral_records: list[dict[str, Any]] = []
    empirical_near_zero_area_records: list[dict[str, Any]] = []
    for level_index in level_indices:
        output_records = run_empirical_normalized_next_state_entropy_outputs_for_level(level_index)
        empirical_integral_records.extend(output_records["integral"])
        empirical_near_zero_area_records.extend(output_records["near_zero_area"])

    plot_empirical_integral_records_by_level(
        empirical_integral_records,
        output_path=OUTPUT_DIRECTORY / "empirical_next_state_normalized_by_num_states_integrals_by_level_and_rand_prob.png",
        title="Empirical next-state normalized entropy heatmap integrals by level",
        y_label="Normalized entropy heatmap integral",
    )
    plot_empirical_structural_integral_records_by_level(
        empirical_integral_records,
        output_path=OUTPUT_DIRECTORY / "empirical_next_state_normalized_by_num_states_structural_integrals_by_level_and_rand_prob.png",
        title="Empirical next-state normalized structural entropy heatmap integrals by level",
        y_label="Normalized entropy heatmap integral",
    )
    plot_empirical_near_zero_area_records_by_level(
        empirical_near_zero_area_records,
        output_path=OUTPUT_DIRECTORY / "empirical_next_state_normalized_by_num_states_near_zero_area_by_level_and_rand_prob.png",
        title="Empirical next-state normalized near-zero entropy area by level",
    )
    plot_empirical_structural_near_zero_area_records_by_level(
        empirical_near_zero_area_records,
        output_path=OUTPUT_DIRECTORY / "empirical_next_state_normalized_by_num_states_structural_near_zero_area_by_level_and_rand_prob.png",
        title="Empirical next-state normalized structural near-zero entropy area by level",
    )
    return {
        "integral": empirical_integral_records,
        "near_zero_area": empirical_near_zero_area_records,
    }


def run_empirical_entropy_outputs_for_levels(
    level_indices: Iterable[int] = LEVEL_INDICES,
) -> dict[str, list[dict[str, Any]]]:
    empirical_integral_records: list[dict[str, Any]] = []
    empirical_near_zero_area_records: list[dict[str, Any]] = []
    for level_index in level_indices:
        output_records = run_empirical_entropy_outputs_for_level(level_index)
        empirical_integral_records.extend(output_records["integral"])
        empirical_near_zero_area_records.extend(output_records["near_zero_area"])

    plot_empirical_integral_records_by_level(
        empirical_integral_records,
        output_path=OUTPUT_DIRECTORY / "empirical_integrals_by_level_and_rand_prob.png",
        title="Empirical entropy heatmap integrals by level",
    )
    plot_empirical_structural_integral_records_by_level(
        empirical_integral_records,
        output_path=OUTPUT_DIRECTORY / "empirical_structural_integrals_by_level_and_rand_prob.png",
        title="Empirical structural entropy heatmap integrals by level",
    )
    plot_empirical_near_zero_area_records_by_level(
        empirical_near_zero_area_records,
        output_path=OUTPUT_DIRECTORY / "empirical_near_zero_area_by_level_and_rand_prob.png",
        title="Empirical near-zero entropy area by level",
    )
    plot_empirical_structural_near_zero_area_records_by_level(
        empirical_near_zero_area_records,
        output_path=OUTPUT_DIRECTORY / "empirical_structural_near_zero_area_by_level_and_rand_prob.png",
        title="Empirical structural near-zero entropy area by level",
    )
    return {
        "integral": empirical_integral_records,
        "near_zero_area": empirical_near_zero_area_records,
    }


def run_empirical_normalized_entropy_outputs_for_level(
    level_index: int,
) -> dict[str, list[dict[str, Any]]]:
    log_base = state_count_entropy_log_base(level_index)
    near_zero_threshold = bit_threshold_to_state_normalized_threshold(
        threshold_bits=NEAR_ZERO_ENTROPY_THRESHOLD,
        level_index=level_index,
    )
    return run_empirical_entropy_outputs_for_level(
        level_index,
        log_base=log_base,
        output_stem_suffix="_normalized_by_num_states",
        title_entropy_label="normalized entropy",
        entropy_unit_label=r"log base $|S|$",
        near_zero_threshold=near_zero_threshold,
    )


def run_empirical_normalized_entropy_outputs_for_levels(
    level_indices: Iterable[int] = LEVEL_INDICES,
) -> dict[str, list[dict[str, Any]]]:
    empirical_integral_records: list[dict[str, Any]] = []
    empirical_near_zero_area_records: list[dict[str, Any]] = []
    for level_index in level_indices:
        output_records = run_empirical_normalized_entropy_outputs_for_level(level_index)
        empirical_integral_records.extend(output_records["integral"])
        empirical_near_zero_area_records.extend(output_records["near_zero_area"])

    plot_empirical_integral_records_by_level(
        empirical_integral_records,
        output_path=OUTPUT_DIRECTORY / "empirical_normalized_by_num_states_integrals_by_level_and_rand_prob.png",
        title="Empirical normalized entropy heatmap integrals by level",
        y_label="Normalized entropy heatmap integral",
    )
    plot_empirical_structural_integral_records_by_level(
        empirical_integral_records,
        output_path=OUTPUT_DIRECTORY / "empirical_normalized_by_num_states_structural_integrals_by_level_and_rand_prob.png",
        title="Empirical normalized structural entropy heatmap integrals by level",
        y_label="Normalized entropy heatmap integral",
    )
    plot_empirical_near_zero_area_records_by_level(
        empirical_near_zero_area_records,
        output_path=OUTPUT_DIRECTORY / "empirical_normalized_by_num_states_near_zero_area_by_level_and_rand_prob.png",
        title="Empirical normalized near-zero entropy area by level",
    )
    plot_empirical_structural_near_zero_area_records_by_level(
        empirical_near_zero_area_records,
        output_path=OUTPUT_DIRECTORY / "empirical_normalized_by_num_states_structural_near_zero_area_by_level_and_rand_prob.png",
        title="Empirical normalized structural near-zero entropy area by level",
    )
    return {
        "integral": empirical_integral_records,
        "near_zero_area": empirical_near_zero_area_records,
    }


if __name__ == "__main__":
    run_empirical_entropy_outputs_for_levels()
