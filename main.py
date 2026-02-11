# main.py
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import polars as pl

from mercury.latent.params import LatentParams
from mercury.sensory.params import SensoryParams

from src.maze_environment.generate_data import generate_data
from src.mercury_runs.pipelines import run_all_bundled
from scripts.compute_precision import compute_metrics
from src.mercury_runs.save_results import read_bundle_meta


# ------------------------
# Shared helpers
# ------------------------
def _run_subprocess_command(command: list[str], env: dict[str, str]) -> None:
    subprocess.run(command, check=True, env=env)

def _stable_hash(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def _grid(parameter_space: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    parameter_names = list(parameter_space.keys())
    for values in itertools.product(*(parameter_space[name] for name in parameter_names)):
        yield dict(zip(parameter_names, values))


def _args_from_config(config: Dict[str, Any]) -> List[str]:
    arguments: List[str] = []
    for argument_name, argument_value in config.items():
        flag = f"--{argument_name}"
        if isinstance(argument_value, bool):
            if argument_name in {"memory_replay", "memory_disambiguation"}:
                arguments.extend([flag, str(argument_value)])
            elif argument_value:
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


def _build_run_payload(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "level": args.level,
        "sensor": args.sensor,
        "sensor_range": args.sensor_range,
        "seed": args.seed,
        "rand_prob": args.rand_prob,
        "num_steps": args.num_steps,
        "memory_length": args.memory_length,
        "window_length": args.window_length,
        "sensory": {
            "allow_self_loops": args.sensory_allow_self_loops,
            "activation_threshold": args.activation_threshold,
            "topological_neighbourhood_threshold": args.topological_neighbourhood_threshold,
            "max_neurons": args.sensory_max_neurons,
            "sensory_weighting": args.sensory_weighting,
            "winning_node_lr": args.winning_node_lr,
            "topological_neighbourhood_lr": args.topological_neighbourhood_lr,
            "action_lr": args.sensory_action_lr,
            "global_context_lr": args.global_context_lr,
            "max_age": args.sensory_max_age,
            "gaussian_shape": args.sensory_gaussian_shape,
        },
        "latent": {
            "allow_self_loops": args.latent_allow_self_loops,
            "max_neurons": args.latent_max_neurons,
            "action_lr": args.latent_action_lr,
            "gaussian_shape": args.latent_gaussian_shape,
            "max_age": args.latent_max_age,
            "ambiguity_threshold": args.ambiguity_threshold,
            "trace_decay": args.trace_decay,
            "mixture_alpha": args.mixture_alpha,
            "mixture_beta": args.mixture_beta,
            "memory_replay": args.memory_replay,
            "memory_disambiguation": args.memory_disambiguation,
        },
        "action_map": {
            "n_codebook": args.am_n_codebook,
            "lr": args.am_lr,
            "sigma": args.am_sigma,
            "key": args.am_key,
        },
    }


def _run_complete(run_root: Path) -> bool:
    return _find_states_parquet(run_root) is not None and _find_metrics_parquet(run_root) is not None


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
            pl.col("latent_precision").var().alias("latent_precision_var"),
            pl.col("latent_precision").std().alias("latent_precision_std"),
            (pl.col("latent_precision").std()/pl.col(
                "latent_precision").mean()).alias(
                "latent_precision_coeff_var"),
            pl.col("sensory_precision").mean().alias("sensory_precision_mean"),
            pl.col("sensory_precision").min().alias("sensory_precision_min"),
            pl.col("sensory_precision").max().alias("sensory_precision_max"),
            pl.col("sensory_precision").var().alias("sensory_precision_var"),
            pl.col("sensory_precision").std().alias("sensory_precision_std"),
            (pl.col("sensory_precision").std() / pl.col(
                "sensory_precision").mean()).alias(
                "sensory_precision_coeff_var"),
        ).to_dicts()[0]

        row = {
            "run_id": meta.run_id,
            "timestamp_utc": meta.timestamp_utc,
            "level": meta.source.level,
            "sensor": meta.source.sensor,
            "sensor_range": meta.source.sensor_range,
            "memory_length": meta.memory_length,
            "activation_threshold": meta.sensory_params.get("activation_threshold"),
            "sensory_weighting": meta.sensory_params.get("sensory_weighting"),
            "ambiguity_threshold": (meta.latent_params or {}).get("ambiguity_threshold"),
            "trace_decay": (meta.latent_params or {}).get("trace_decay"),
            "mixture_alpha": (meta.latent_params or {}).get("mixture_alpha"),
            "mixture_beta": (meta.latent_params or {}).get("mixture_beta"),
            "memory_replay": meta.latent_params.get("memory_replay"),
            "memory_disambiguation": meta.latent_params.get("memory_disambiguation"),
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


# ------------------------
# CLI
# ------------------------

@dataclass(frozen=True)
class StudyDefaults:
    study_root: str = "results/studies"
    study_name: str = "level16_cd_grid"
    python_executable: str = "python3"
    main_path: str = "main.py"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Mode switch
    parser.add_argument("--study", action="store_true", help="Run parametric study instead of single run")

    # Parallelism for study mode
    parser.add_argument("--max_workers", type=int, default=1, help="Study mode: number of concurrent runs")
    parser.add_argument("--no_parallel", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False, help="Study mode: skip completed runs")
    parser.add_argument("--continue_on_error", action="store_true", default=False, help="Study mode: log errors and continue")
    parser.add_argument("--retry_failed", action="store_true", default=False, help="Study mode: rerun configs from study_errors.jsonl")

    # Shared / single-run args
    parser.add_argument("--level", type=int, required=False)

    parser.add_argument("--study_root", type=str, default="results/studies")
    parser.add_argument("--study_name", type=str, default="default")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--reuse_existing_run", action="store_true", default=True)
    parser.add_argument("--no_reuse_existing_run", action="store_true", default=False)

    parser.add_argument("--datasets_root", type=str, default="datasets")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rand_prob", type=float, default=0.3)
    parser.add_argument("--num_steps", type=int, default=20_000)
    parser.add_argument("--no_reuse_existing_dataset", action="store_true")

    parser.add_argument("--sensor", type=str, default="cartesian", choices=["cartesian", "cardinal distance"])
    parser.add_argument("--sensor_range", type=int, default=None)

    parser.add_argument("--window_length", type=int, default=100)
    parser.add_argument("--memory_length", type=int, default=10)

    # SensoryParams
    parser.add_argument("--sensory_allow_self_loops", action="store_true", default=False)
    parser.add_argument("--activation_threshold", type=float, default=0.95)
    parser.add_argument("--topological_neighbourhood_threshold", type=float, default=0.6)
    parser.add_argument("--sensory_max_neurons", type=int, default=50)
    parser.add_argument("--sensory_weighting", type=float, default=0.8)
    parser.add_argument("--winning_node_lr", type=float, default=0.55)
    parser.add_argument("--topological_neighbourhood_lr", type=float, default=0.9)
    parser.add_argument("--sensory_action_lr", type=float, default=0.5)
    parser.add_argument("--global_context_lr", type=float, default=0.9)
    parser.add_argument("--sensory_max_age", type=int, default=18)
    parser.add_argument("--sensory_gaussian_shape", type=int, default=2)

    # LatentParams
    parser.add_argument("--latent_allow_self_loops", action="store_true", default=True)
    parser.add_argument("--latent_max_neurons", type=int, default=300)
    parser.add_argument("--latent_action_lr", type=float, default=0.1)
    parser.add_argument("--latent_gaussian_shape", type=int, default=2)
    parser.add_argument("--latent_max_age", type=int, default=18)
    parser.add_argument("--ambiguity_threshold", type=int, default=10)
    parser.add_argument("--trace_decay", type=float, default=0.99)
    parser.add_argument("--mixture_alpha", type=float, default=0.2)
    parser.add_argument("--mixture_beta", type=float, default=0.2)
    def _strtobool(value: str) -> bool:
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off"}:
            return False
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

    parser.add_argument("--memory_replay", nargs="?", const=True, default=True, type=_strtobool)
    parser.add_argument("--memory_disambiguation", nargs="?", const=True, default=True, type=_strtobool)


    # ActionMap
    parser.add_argument("--am_n_codebook", type=int, default=4)
    parser.add_argument("--am_lr", type=float, default=0.5)
    parser.add_argument("--am_sigma", type=float, default=0.0)
    parser.add_argument("--am_key", type=int, default=0)

    return parser.parse_args()


# ------------------------
# Single run
# ------------------------

def run_single(args: argparse.Namespace) -> None:
    if args.level is None:
        raise ValueError("--level is required for single-run mode")

    reuse_existing_run = bool(args.reuse_existing_run) and (not bool(args.no_reuse_existing_run))
    reuse_existing_dataset = not bool(args.no_reuse_existing_dataset)

    datasets_root = Path(args.datasets_root)

    run_payload = _build_run_payload(args)

    run_id = args.run_id or _stable_hash(run_payload)

    run_root = Path(args.study_root) / args.study_name / run_id
    bundles_root = run_root / "bundles" / "mercury"
    metrics_root = run_root / "metrics"
    run_root.mkdir(parents=True, exist_ok=True)

    if reuse_existing_run and any(metrics_root.glob("*_precision.parquet")):
        return
    from filelock import FileLock

    lock_path = Path(args.datasets_root) / ".dataset_generation.lock"
    with FileLock(str(lock_path)):

        generate_data(
            level=args.level,
            seed=args.seed,
            rand_prob=args.rand_prob,
            num_steps=args.num_steps,
            output_root=datasets_root,
            reuse_existing=reuse_existing_dataset,
        )

    sensory_params = SensoryParams(**run_payload["sensory"])
    latent_params = LatentParams(**run_payload["latent"])
    action_map_params = dict(run_payload["action_map"])

    run_all_bundled(
        datasets_root=datasets_root,
        results_root=bundles_root,
        level=args.level,
        sensor=args.sensor,
        sensor_range=args.sensor_range,
        dataset_select="latest",
        dataset_run_id=None,
        memory_length=args.memory_length,
        embed_metadata_in_parquet=True,
        sensory_params=sensory_params,
        latent_params=latent_params,
        action_map_params=action_map_params,
        run_id=run_id,
        notes=f"study={args.study_name}",
    )

    compute_metrics(
        level=args.level,
        window_length=args.window_length,
        sensor=args.sensor,
        sensor_range=args.sensor_range,
        results_dir=bundles_root,
        out_dir=metrics_root,
        use_run_id=run_id,
    )


# ------------------------
# Study mode (grid + summary)
# ------------------------

def run_study(args: argparse.Namespace) -> None:
    defaults = StudyDefaults(study_root=args.study_root, study_name=args.study_name)

    study_root = Path(defaults.study_root)
    study_directory = study_root / defaults.study_name
    study_directory.mkdir(parents=True, exist_ok=True)

    manifest_path = study_directory / "study_manifest.jsonl"
    if manifest_path.exists() and not (args.resume or args.retry_failed):
        manifest_path.unlink()
    errors_path = study_directory / "study_errors.jsonl"
    if errors_path.exists() and not (args.resume or args.retry_failed):
        errors_path.unlink()

    # Define parameter space here (edit as needed)
    parameter_space = {
        "level": [16, 17, 18, 19, 20, 21],
        "sensor": ["cardinal distance"],
        "sensor_range": [1],
        "seed": [0, 1, 2],
        "rand_prob": [0, 0.3],
        "memory_length": [20],
        "activation_threshold": [0.95],
        "ambiguity_threshold": [0, 10],
        "am_lr": [0.1],
        "am_key": [0],
        "window_length": [100],
        "mixture_alpha": [1, 0.75, 0.5],
        "mixture_beta": [1, 0.5],
        "memory_replay": [True, False],
        "memory_disambiguation": [True, False],
        # route outputs into this study folder
        "study_root": [defaults.study_root],
        "study_name": [defaults.study_name],
        "reuse_existing_run": [False],
    }

    run_configurations = list(_grid(parameter_space))
    if args.retry_failed:
        if not errors_path.exists():
            raise FileNotFoundError(f"Missing error log: {errors_path}")
        retry_configs: List[Dict[str, Any]] = []
        for line in errors_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            config = record.get("config")
            if config:
                config = dict(config)
                config.pop("run_id", None)
                retry_configs.append(config)
        if not retry_configs:
            raise RuntimeError(f"No configs found in {errors_path}")
        run_configurations = retry_configs

    # Keep BLAS threads sane per process
    base_env = os.environ.copy()
    base_env.setdefault("OMP_NUM_THREADS", "1")
    base_env.setdefault("MKL_NUM_THREADS", "1")
    base_env.setdefault("OPENBLAS_NUM_THREADS", "1")
    base_env.setdefault("NUMEXPR_NUM_THREADS", "1")

    python_executable = defaults.python_executable
    main_path = defaults.main_path

    use_parallel = (args.max_workers > 1) and (not args.no_parallel)

    def _append_error(record: Dict[str, Any]) -> None:
        with errors_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    run_records: List[Dict[str, Any]] = []
    for run_index, run_configuration in enumerate(run_configurations):
        merged_args = argparse.Namespace(**{**vars(args), **run_configuration})
        run_payload = _build_run_payload(merged_args)
        run_id = _stable_hash(run_payload)
        run_root = study_directory / run_id
        if args.resume and _run_complete(run_root):
            continue

        run_configuration = dict(run_configuration)
        run_configuration["run_id"] = run_id
        command = [python_executable, main_path] + _args_from_config(run_configuration)

        record = {"run_index": run_index, "command": command, "config": run_configuration}
        with manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        run_records.append(
            {"run_index": run_index, "config": run_configuration, "command": command, "run_id": run_id}
        )

    print(f"Study runs scheduled: {len(run_records)}")

    if use_parallel:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # Use pool, worker is top-level so it's picklable
        with ProcessPoolExecutor(max_workers=int(args.max_workers)) as pool:
            future_map = {
                pool.submit(_run_subprocess_command, record["command"], base_env): record
                for record in run_records
            }
            for future in as_completed(future_map):
                record = future_map[future]
                try:
                    future.result()
                except Exception as exc:
                    _append_error(
                        {
                            "run_index": record["run_index"],
                            "run_id": record["run_id"],
                            "command": record["command"],
                            "config": record["config"],
                            "error": repr(exc),
                        }
                    )
                    if not args.continue_on_error:
                        raise
    else:
        for record in run_records:
            try:
                subprocess.run(record["command"], check=True, env=base_env)
            except Exception as exc:
                _append_error(
                    {
                        "run_index": record["run_index"],
                        "run_id": record["run_id"],
                        "command": record["command"],
                        "config": record["config"],
                        "error": repr(exc),
                    }
                )
                if not args.continue_on_error:
                    raise


    summary_path = _write_study_summary(study_root=study_root, study_name=defaults.study_name)
    print(f"Wrote: {summary_path}")


def main() -> None:
    args = parse_arguments()
    if args.study:
        run_study(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
