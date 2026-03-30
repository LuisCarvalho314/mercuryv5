from __future__ import annotations

import argparse
import datetime
import hashlib
import itertools
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def run_subprocess_command(command: list[str], env: dict[str, str]) -> None:
    subprocess.run(command, check=True, env=env)


def stable_hash(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def grid(parameter_space: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    parameter_names = list(parameter_space.keys())
    for values in itertools.product(*(parameter_space[name] for name in parameter_names)):
        yield dict(zip(parameter_names, values))


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp_path, path)


def utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def compact_utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")


def base_runtime_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("XDG_CACHE_HOME", "/tmp")
    env.setdefault("MERCURY_PROGRESS", "1")
    return env


def progress_enabled(default: bool = True) -> bool:
    raw = os.environ.get("MERCURY_PROGRESS")
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def args_from_config(config: Dict[str, Any]) -> List[str]:
    arguments: List[str] = []
    for argument_name, argument_value in config.items():
        flag = f"--{argument_name}"
        if isinstance(argument_value, bool):
            if argument_name in {
                "memory_replay",
                "memory_disambiguation",
                "latent_allow_self_loops",
                "sensory_allow_self_loops",
                "wandb",
                "wandb_log_artifacts",
                "baseline_pocml",
                "baseline_cscg",
                "baseline_all",
                "cscg_term_early",
                "pocml_use_ground_truth_node_ids",
                "pocml_memory_bias",
                "computational_eval",
                "structure_metrics",
                "mercury_valid_trajectories_only",
                "mercury_split_sensory_raw_latent_valid",
                "pocml_valid_trajectories_only",
                "cscg_valid_trajectories_only",
            }:
                arguments.extend([flag, str(argument_value)])
            elif argument_name == "reuse_existing_run":
                if argument_value:
                    arguments.extend([flag, "True"])
                else:
                    arguments.append("--no_reuse_existing_run")
            elif argument_value:
                arguments.append(flag)
        elif argument_value is None:
            continue
        elif isinstance(argument_value, (list, tuple)):
            if len(argument_value) == 0:
                continue
            arguments.append(flag)
            arguments.extend([str(value) for value in argument_value])
        else:
            arguments.extend([flag, str(argument_value)])
    return arguments


def strtobool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
