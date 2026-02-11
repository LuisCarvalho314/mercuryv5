# scripts/parametric_study.py
from __future__ import annotations

import itertools
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import polars as pl

from mercury_runs.save_results import read_bundle_meta


@dataclass(frozen=True)
class StudyRunnerConfig:
    python_executable: str = "python3"
    main_path: str = "main.py"
    study_root: str = "results/studies"
    study_name: str = "level18_cd1_grid"


def _grid(parameter_space: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    parameter_names = list(parameter_space.keys())
    for values in itertools.product(*(parameter_space[name] for name in parameter_names)):
        yield dict(zip(parameter_names, values))


def _args_from_config(config: Dict[str, Any]) -> List[str]:
    arguments: List[str] = []
    for argument_name, argument_value in config.items():
        flag = f"--{argument_name}"
        if isinstance(argument_value, bool):
            if argument_value:
                arguments.append(flag)
        elif argument_value is None:
            continue
        else:
            arguments.extend([flag, str(argument_value)])
    return arguments


def _find_states_parquet(run_root: Path) -> Optional[Path]:
    bundles_root = run_root / "bundles" / "mercury"
    if not bundles_root.exists():
        return None
    candidates = sorted(bundles_root.rglob("*_states.parquet"))
    return candidates[-1] if candidates else None


def _find_metrics_parquet(run_root: Path) -> Optional[Path]:
    metrics_root = run_root / "metrics"
    if not metrics_root.exists():
        return None
    candidates = sorted(metrics_root.glob("*_precision.parquet"))
    return candidates[-1] if candidates else None


def _write_study_summary(study_root: Path, study_name: str) -> Path:
    study_dir = study_root / study_name
    if not study_dir.exists():
        raise FileNotFoundError(f"Study directory not found: {study_dir}")

    run_directories = [path for path in study_dir.iterdir() if path.is_dir()]

    summary_rows: List[dict] = []
    for run_dir in sorted(run_directories):
        states_parquet_path = _find_states_parquet(run_dir)
        metrics_parquet_path = _find_metrics_parquet(run_dir)
        if states_parquet_path is None or metrics_parquet_path is None:
            continue

        meta = read_bundle_meta(states_parquet_path)
        if meta is None:
            continue

        metrics_frame = pl.read_parquet(metrics_parquet_path)
        metric_stats = metrics_frame.select(
            pl.col("latent_precision").mean().alias("latent_precision_mean"),
            pl.col("latent_precision").min().alias("latent_precision_min"),
            pl.col("latent_precision").max().alias("latent_precision_max"),
            pl.col("sensory_precision").mean().alias("sensory_precision_mean"),
            pl.col("latent_precision").var().alias("latent_precision_var"),
        ).to_dicts()[0]

        row = {
            "run_id": meta.run_id,
            "timestamp_utc": meta.timestamp_utc,
            "level": meta.source.level,
            "sensor": meta.source.sensor,
            "sensor_range": meta.source.sensor_range,
            "memory_length": meta.memory_length,
            # Sensory params (extend as needed)
            "activation_threshold": meta.sensory_params.get("activation_threshold"),
            "sensory_weighting": meta.sensory_params.get("sensory_weighting"),
            # Latent params (extend as needed)
            "ambiguity_threshold": (meta.latent_params or {}).get("ambiguity_threshold"),
            "trace_decay": (meta.latent_params or {}).get("trace_decay"),
            "mixture_alpha": (meta.latent_params or {}).get("mixture_alpha"),
            "mixture_beta": (meta.latent_params or {}).get("mixture_beta"),
            # Action map
            "am_n_codebook": meta.action_map_params.get("n_codebook"),
            "am_lr": meta.action_map_params.get("lr"),
            "am_sigma": meta.action_map_params.get("sigma"),
            "am_key": meta.action_map_params.get("key"),
        }
        row.update(metric_stats)
        summary_rows.append(row)

    if not summary_rows:
        raise RuntimeError(
            f"No complete runs found under {study_dir}. "
            "Expected each run folder to contain bundles/**/_states.parquet and metrics/*_precision.parquet."
        )

    summary_frame = pl.DataFrame(summary_rows).sort("latent_precision_mean", descending=True)
    output_path = study_dir / "study_summary.parquet"
    summary_frame.write_parquet(output_path, compression="zstd", statistics=True)
    return output_path


def run_study() -> None:
    # ---- Edit these defaults as needed ----
    runner_config = StudyRunnerConfig()

    study_root = Path(runner_config.study_root)
    study_directory = study_root / runner_config.study_name
    study_directory.mkdir(parents=True, exist_ok=True)

    manifest_path = study_directory / "study_manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()

    # ---- Define your parameter space here ----
    parameter_space = {
        "level": [16],
        "sensor": ["cardinal distance"],
        "sensor_range": [1, 2, 3],
        "seed": [0,1,2],
        "rand_prob": [0.3],
        "memory_length": [20],
        "activation_threshold": [0.95],
        "ambiguity_threshold": [20],
        "am_lr": [0.01, 0.1],
        "am_key": [0],
        "window_length": [200],
        "mixture_alpha": [0.2],
        "mixture_beta": [0.2],
        # one-run-one-folder controls (main.py handles run_id deterministically)
        "study_root": [runner_config.study_root],
        "study_name": [runner_config.study_name],
        "reuse_existing_run": [True],
    }

    run_configurations = list(_grid(parameter_space))

    for run_index, run_configuration in enumerate(run_configurations):
        command = [runner_config.python_executable, runner_config.main_path] + _args_from_config(run_configuration)

        record = {
            "run_index": run_index,
            "command": command,
            "config": run_configuration,
        }
        with manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        subprocess.run(command, check=True)

    # ---- Automatically write summary at the end ----
    summary_path = _write_study_summary(study_root=study_root, study_name=runner_config.study_name)
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    run_study()
