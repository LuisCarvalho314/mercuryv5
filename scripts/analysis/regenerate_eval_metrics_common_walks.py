from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mercury_runs.algorithms.cscg import evaluate as cscg_evaluate
from mercury_runs.algorithms.cscg.prepare import import_cscg_module
from mercury_runs.infrastructure.paper_precision import (
    generate_random_start_walks,
    resolve_structure_metric_epsilons,
)


DEFAULT_STUDY_ROOT = (
    REPO_ROOT
    / "results"
    / "current"
    / "studies"
    / "mercury_cscg_hybrid_vs_cscg_batch_size_30_seed"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "current"
    / "notebook_plots"
    / "mercury_cscg_hybrid_vs_cscg_batch_size_30_seed_common_walk_eval"
)
SNAPSHOT_METHODS = ("cscg", "mercury_cscg_hybrid")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _run_config(run_root: Path) -> dict[str, Any]:
    payload = _read_json(run_root / "run_config.json")
    config = payload.get("cli_args") or payload.get("config") or payload
    config = dict(config)
    config["requested_run_id"] = payload.get("requested_run_id") or payload.get("run_id") or run_root.name
    return config


def _config_namespace(config: dict[str, Any], run_root: Path, *, method: str) -> SimpleNamespace:
    run_id = str(config.get("requested_run_id") or run_root.name)
    paper_rand_prob = config.get("paper_precision_rand_prob")
    if paper_rand_prob is None:
        paper_rand_prob = config.get("rand_prob", 0.3)
    paper_seed = config.get("paper_precision_seed")
    if paper_seed is None:
        paper_seed = config.get("seed", 0)
    seed = config.get("cscg_seed")
    if seed is None:
        seed = config.get("seed", 0)
    return SimpleNamespace(
        run_id=run_id,
        method=method,
        level=int(config["level"]),
        sensor=str(config.get("sensor", "cardinal distance")),
        sensor_range=config.get("sensor_range"),
        seed=int(seed),
        rand_prob=float(config.get("rand_prob", paper_rand_prob)),
        paper_precision_rand_prob=float(paper_rand_prob),
        paper_precision_seed=int(paper_seed),
        paper_precision_num_walks=int(config.get("paper_precision_num_walks", 100)),
        paper_precision_walk_length=int(config.get("paper_precision_walk_length", 10_000)),
        paper_precision_mode=str(config.get("paper_precision_mode", "per_iteration")),
        paper_precision_eval_interval=int(config.get("paper_precision_eval_interval", 1)),
        paper_precision_eval_checkpoints=config.get("paper_precision_eval_checkpoints"),
        paper_precision_num_points=config.get("paper_precision_num_points"),
        paper_precision_schedule=str(config.get("paper_precision_schedule", "linear")),
        valid_trajectories_only=bool(config.get("valid_trajectories_only", False)),
        pseudocount=float(config.get("cscg_pseudocount", config.get("pseudocount", 0.002)) or 0.002),
        training_mode=str(config.get("cscg_training_mode", config.get("training_mode", "online_em"))),
        train_algorithm=str(config.get("cscg_train_algorithm", config.get("train_algorithm", "em"))),
        n_iter=int(config.get("cscg_n_iter", config.get("n_iter", 0)) or 0),
        eval_viterbi=bool(config.get("cscg_eval_viterbi", config.get("eval_viterbi", False))),
        output_root=run_root / "bundles" / method,
        structure_metrics_enabled=bool(config.get("structure_metrics", True)),
        structure_metrics_epsilon=float(config.get("structure_metrics_epsilon", 1e-6)),
        structure_metrics_epsilons=config.get("structure_metrics_epsilons"),
        structure_metrics_ignore_self_loops=bool(config.get("structure_metrics_ignore_self_loops", True)),
        structure_metrics_scope=str(config.get("structure_metrics_scope", "exact_level")),
        structure_metrics_ground_truth_source=str(config.get("structure_metrics_ground_truth_source", "empirical_walks")),
    )


def _walk_key(config: SimpleNamespace) -> tuple[Any, ...]:
    return (
        int(config.level),
        str(config.sensor),
        None if config.sensor_range is None else int(config.sensor_range),
        float(config.paper_precision_rand_prob),
        int(config.paper_precision_num_walks),
        int(config.paper_precision_walk_length),
        int(config.paper_precision_seed),
        bool(config.valid_trajectories_only),
    )


def _walk_cache_id(key: tuple[Any, ...]) -> str:
    level, sensor, sensor_range, rand_prob, num_walks, walk_length, seed, valid_only = key
    range_label = "none" if sensor_range is None else str(sensor_range)
    return (
        f"level={level}|sensor={sensor}|range={range_label}|rand_prob={rand_prob}|"
        f"num_walks={num_walks}|walk_length={walk_length}|base_seed={seed}|valid_only={valid_only}"
    )


class WalkCache:
    def __init__(self) -> None:
        self._cache: dict[tuple[Any, ...], list[Any]] = {}

    def get(self, config: SimpleNamespace) -> list[Any]:
        key = _walk_key(config)
        if key not in self._cache:
            self._cache[key] = generate_random_start_walks(
                level=int(config.level),
                sensor=str(config.sensor),
                sensor_range=config.sensor_range,
                rand_prob=float(config.paper_precision_rand_prob),
                num_walks=int(config.paper_precision_num_walks),
                walk_length=int(config.paper_precision_walk_length),
                base_seed=int(config.paper_precision_seed),
                valid_trajectories_only=bool(config.valid_trajectories_only),
            )
        return self._cache[key]


@contextmanager
def _patched_cscg_walks(walks: list[Any]) -> Iterator[None]:
    original = cscg_evaluate.generate_random_start_walks
    cscg_evaluate.generate_random_start_walks = lambda **_kwargs: walks
    try:
        yield
    finally:
        cscg_evaluate.generate_random_start_walks = original


def _snapshot_dirs(run_root: Path, method: str, checkpoints: set[int] | None) -> list[Path]:
    root = run_root / "snapshots" / method / "paper_precision"
    if not root.exists():
        return []
    snapshots = sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith("step=")],
        key=lambda path: int(path.name.split("=", 1)[1]),
    )
    if checkpoints is not None:
        snapshots = [path for path in snapshots if int(path.name.split("=", 1)[1]) in checkpoints]
    return snapshots


def _load_snapshot_model(snapshot_dir: Path, method: str) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    if method == "cscg":
        model = np.load(snapshot_dir / "model.npz", allow_pickle=False)
        return (
            model,
            np.asarray(model["n_clones"], dtype=np.int64),
            np.asarray(model["obs_uniques"]),
            np.asarray(model["action_uniques"]),
        )
    model = np.load(snapshot_dir / "model.npz", allow_pickle=False)
    vocab = np.load(snapshot_dir / "observation_vocabulary.npz", allow_pickle=False)
    return (
        model,
        np.asarray(model["n_clones"], dtype=np.int64),
        np.asarray(vocab["obs_uniques"]),
        np.asarray(vocab["action_uniques"]),
    )


def _history_row(method: str, snapshot_meta: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    step = int(snapshot_meta.get("step", 0))
    observed_samples = int(snapshot_meta.get("observed_samples", step))
    row = {
        "stage": str(snapshot_meta.get("stage", "em")),
        "iteration": int(snapshot_meta.get("iteration", step)),
        "step": step,
        "observed_samples": observed_samples,
        **metrics,
    }
    if method == "mercury_cscg_hybrid":
        row.update(
            {
                "method": "mercury_cscg_hybrid_cscg",
                "series_label": "Hybrid CSCG",
                "phase": "cscg",
                "phase_step": step,
                "cumulative_step": observed_samples,
            }
        )
    return row


def _protocol(config: SimpleNamespace, method: str, history: list[dict[str, Any]]) -> dict[str, Any]:
    training_mode = str(config.training_mode).strip().lower()
    schedule_unit = "final_only" if not history else ("training_step" if training_mode == "online_em" else "em_iteration")
    inferred_series = "mercury_cscg_hybrid_state_id" if method == "mercury_cscg_hybrid" else "cscg_state_id"
    return {
        "num_walks": int(config.paper_precision_num_walks),
        "walk_length": int(config.paper_precision_walk_length),
        "rand_prob": float(config.paper_precision_rand_prob),
        "base_seed": int(config.paper_precision_seed),
        "inferred_state_series": inferred_series,
        "ground_truth_state_series": "cartesian_state_id",
        "state_decode_algorithm": "viterbi",
        "valid_trajectories_only": bool(config.valid_trajectories_only),
        "mode": str(config.paper_precision_mode),
        "eval_interval": int(config.paper_precision_eval_interval),
        "eval_checkpoints": config.paper_precision_eval_checkpoints,
        "num_points": config.paper_precision_num_points,
        "schedule_unit": schedule_unit,
        "x_axis_unit": "observed_samples" if training_mode == "online_em" else schedule_unit,
        "structure_metrics_epsilons": resolve_structure_metric_epsilons(config),
        "eval_viterbi_enabled": bool(config.eval_viterbi),
        "eval_viterbi_n_iter": int(config.n_iter) if bool(config.eval_viterbi) else None,
        "eval_viterbi_pseudocount": 0.0 if bool(config.eval_viterbi) else None,
        "common_walks": {
            "enabled": True,
            "scope": "shared across all evaluated checkpoints with the same key",
            "cache_key": _walk_cache_id(_walk_key(config)),
        },
    }


def regenerate_run_method(
    *,
    run_root: Path,
    method: str,
    output_root: Path,
    in_place: bool,
    checkpoints: set[int] | None,
    walk_cache: WalkCache,
    chmm_mod: Any,
) -> Path | None:
    snapshots = _snapshot_dirs(run_root, method, checkpoints)
    if not snapshots:
        return None
    raw_config = _run_config(run_root)
    config = _config_namespace(raw_config, run_root, method=method)
    walks = walk_cache.get(config)
    history: list[dict[str, Any]] = []
    with _patched_cscg_walks(walks):
        for snapshot_dir in snapshots:
            snapshot_meta = _read_json(snapshot_dir / "snapshot.json")
            model_npz, n_clones, obs_uniques, action_uniques = _load_snapshot_model(snapshot_dir, method)
            metrics = cscg_evaluate.compute_cscg_paper_precision_metrics_from_model(
                model_npz=model_npz,
                n_clones=n_clones,
                config=config,
                obs_uniques=obs_uniques,
                action_uniques=action_uniques,
                chmm_mod=chmm_mod,
            )
            history.append(_history_row(method, snapshot_meta, metrics))
    history = sorted(history, key=lambda row: int(row.get("observed_samples", row.get("step", 0))))
    final_metrics = {
        key: value
        for key, value in (history[-1].items() if history else [])
        if key
        not in {
            "stage",
            "iteration",
            "step",
            "observed_samples",
            "method",
            "series_label",
            "phase",
            "phase_step",
            "cumulative_step",
        }
    }
    payload = {
        "method": method,
        "schema": "paper_precision_v1",
        "regenerated_by": Path(__file__).name,
        "source_run_root": str(run_root),
        "protocol": _protocol(config, method, history),
        "metrics": final_metrics,
        "history": history,
    }
    if in_place:
        out_path = run_root / "metrics" / method / f"{run_root.name}_paper_precision.json"
    else:
        out_path = output_root / run_root.name / "metrics" / method / f"{run_root.name}_paper_precision.json"
    _write_json(out_path, payload)
    return out_path


def _iter_runs(study_root: Path, run_ids: set[str] | None, rand_prob: float | None) -> list[Path]:
    runs: list[Path] = []
    for run_root in sorted(path for path in study_root.iterdir() if path.is_dir()):
        if run_ids is not None and run_root.name not in run_ids:
            continue
        if not (run_root / "run_config.json").exists():
            continue
        config = _run_config(run_root)
        if rand_prob is not None and float(config.get("rand_prob", -1.0)) != float(rand_prob):
            continue
        runs.append(run_root)
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate paper-precision/eval metrics from stored snapshots using one shared random-walk "
            "set across checkpoints for each environment/rand-prob/seed key."
        )
    )
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--rand-prob", type=float, default=None)
    parser.add_argument("--run-id", action="append", default=None, help="Run id to process. Can be repeated.")
    parser.add_argument("--method", choices=SNAPSHOT_METHODS, action="append", default=None)
    parser.add_argument("--checkpoint", type=int, action="append", default=None, help="Restrict to checkpoint step. Can be repeated.")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--in-place", action="store_true", help="Overwrite metrics inside the study run directories.")
    args = parser.parse_args()

    run_ids = set(args.run_id) if args.run_id else None
    checkpoints = set(int(value) for value in args.checkpoint) if args.checkpoint else None
    methods = tuple(args.method) if args.method else SNAPSHOT_METHODS
    runs = _iter_runs(args.study_root, run_ids, args.rand_prob)
    if args.max_runs is not None:
        runs = runs[: int(args.max_runs)]
    args.output_root.mkdir(parents=True, exist_ok=True)
    chmm_mod = import_cscg_module(REPO_ROOT)
    walk_cache = WalkCache()
    written: list[str] = []
    skipped = 0
    for index, run_root in enumerate(runs, start=1):
        print(f"[{index}/{len(runs)}] {run_root.name}")
        for method in methods:
            out_path = regenerate_run_method(
                run_root=run_root,
                method=method,
                output_root=args.output_root,
                in_place=bool(args.in_place),
                checkpoints=checkpoints,
                walk_cache=walk_cache,
                chmm_mod=chmm_mod,
            )
            if out_path is None:
                skipped += 1
                continue
            print(f"  wrote {out_path}")
            written.append(str(out_path))
    summary = {
        "study_root": str(args.study_root),
        "output_root": str(args.output_root),
        "in_place": bool(args.in_place),
        "rand_prob": args.rand_prob,
        "run_ids": sorted(run_ids) if run_ids else None,
        "methods": list(methods),
        "checkpoints": sorted(checkpoints) if checkpoints else None,
        "walk_cache_keys": [_walk_cache_id(key) for key in walk_cache._cache],
        "written": written,
        "skipped_method_run_pairs": skipped,
    }
    summary_path = args.output_root / "regeneration_summary.json"
    _write_json(summary_path, summary)
    print(f"summary {summary_path}")


if __name__ == "__main__":
    main()
