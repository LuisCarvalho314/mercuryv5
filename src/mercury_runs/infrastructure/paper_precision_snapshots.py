from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mercury.action_map.adapter import ActionMap
from mercury.graph.core import Graph

from .runtime import atomic_write_json, load_json


def mercury_snapshot_dir(config: Any, step: int) -> Path:
    return Path(config.results_root).parents[1] / "snapshots" / "mercury" / "paper_precision" / f"step={int(step)}"


def sensory_snapshot_dir(config: Any, step: int) -> Path:
    return Path(config.output_root).parents[1] / "snapshots" / "sensory" / "paper_precision" / f"step={int(step)}"


def cscg_snapshot_dir(config: Any, step: int) -> Path:
    return Path(config.output_root).parents[1] / "snapshots" / "cscg" / "paper_precision" / f"step={int(step)}"


def write_mercury_paper_precision_snapshot(
    *,
    config: Any,
    step: int,
    observed_samples: int,
    sensory_graph: Graph,
    latent_graph: Graph,
    action_map: ActionMap,
    obs_uniques: np.ndarray | None = None,
) -> Path:
    out_dir = mercury_snapshot_dir(config, step)
    out_dir.mkdir(parents=True, exist_ok=True)
    sensory_graph.to_npz(str(out_dir / "sensory_graph.npz"))
    latent_graph.to_npz(str(out_dir / "latent_graph.npz"))
    action_map.to_npz(str(out_dir / "action_map.npz"))
    if obs_uniques is not None:
        np.savez_compressed(out_dir / "observation_vocabulary.npz", obs_uniques=np.asarray(obs_uniques))
    atomic_write_json(
        out_dir / "snapshot.json",
        {
            "schema": "paper_precision_snapshot_v1",
            "method": "mercury",
            "step": int(step),
            "observed_samples": int(observed_samples),
            "level": int(config.level),
            "sensor": str(config.sensor),
            "sensor_range": config.sensor_range,
            "paper_precision": {
                "mode": str(getattr(config, "paper_precision_mode", "off")),
                "eval_interval": int(getattr(config, "paper_precision_eval_interval", 1)),
                "eval_checkpoints": getattr(config, "paper_precision_eval_checkpoints", None),
                "num_points": getattr(config, "paper_precision_num_points", None),
                "num_walks": int(getattr(config, "paper_precision_num_walks", 100)),
                "walk_length": int(getattr(config, "paper_precision_walk_length", 10_000)),
                "rand_prob": float(getattr(config, "rand_prob", 0.3)),
                "seed": int(getattr(config, "seed", 0)),
            },
            "input_mode": str(getattr(config, "mercury_input_mode", "pretrained_raw")),
            "has_observation_vocabulary": obs_uniques is not None,
        },
    )
    return out_dir


def load_mercury_paper_precision_snapshot(path: Path) -> dict[str, Any]:
    snapshot_dir = Path(path)
    meta = load_json(snapshot_dir / "snapshot.json") or {}
    vocab_path = snapshot_dir / "observation_vocabulary.npz"
    return {
        "meta": meta,
        "sensory_graph": Graph.from_npz(str(snapshot_dir / "sensory_graph.npz")),
        "latent_graph": Graph.from_npz(str(snapshot_dir / "latent_graph.npz")),
        "action_map": ActionMap.from_npz(str(snapshot_dir / "action_map.npz")),
        "obs_uniques": np.asarray(np.load(vocab_path, allow_pickle=False)["obs_uniques"]) if vocab_path.exists() else None,
    }


def write_sensory_paper_precision_snapshot(
    *,
    config: Any,
    step: int,
    observed_samples: int,
    sensory_graph: Graph,
    action_map: ActionMap,
    obs_uniques: np.ndarray | None = None,
) -> Path:
    out_dir = sensory_snapshot_dir(config, step)
    out_dir.mkdir(parents=True, exist_ok=True)
    sensory_graph.to_npz(str(out_dir / "sensory_graph.npz"))
    action_map.to_npz(str(out_dir / "action_map.npz"))
    if obs_uniques is not None:
        np.savez_compressed(out_dir / "observation_vocabulary.npz", obs_uniques=np.asarray(obs_uniques))
    atomic_write_json(
        out_dir / "snapshot.json",
        {
            "schema": "paper_precision_snapshot_v1",
            "method": "sensory",
            "step": int(step),
            "observed_samples": int(observed_samples),
            "level": int(config.level),
            "sensor": str(config.sensor),
            "sensor_range": config.sensor_range,
            "observation_mode": str(getattr(config, "observation_mode", "raw")),
            "paper_precision": {
                "mode": str(getattr(config, "paper_precision_mode", "off")),
                "eval_interval": int(getattr(config, "paper_precision_eval_interval", 1)),
                "eval_checkpoints": getattr(config, "paper_precision_eval_checkpoints", None),
                "num_points": getattr(config, "paper_precision_num_points", None),
                "num_walks": int(getattr(config, "paper_precision_num_walks", 100)),
                "walk_length": int(getattr(config, "paper_precision_walk_length", 10_000)),
                "rand_prob": float(getattr(config, "rand_prob", 0.3)),
                "seed": int(getattr(config, "seed", 0)),
            },
            "has_observation_vocabulary": obs_uniques is not None,
        },
    )
    return out_dir


def load_sensory_paper_precision_snapshot(path: Path) -> dict[str, Any]:
    snapshot_dir = Path(path)
    meta = load_json(snapshot_dir / "snapshot.json") or {}
    vocab_path = snapshot_dir / "observation_vocabulary.npz"
    return {
        "meta": meta,
        "sensory_graph": Graph.from_npz(str(snapshot_dir / "sensory_graph.npz")),
        "action_map": ActionMap.from_npz(str(snapshot_dir / "action_map.npz")),
        "obs_uniques": np.asarray(np.load(vocab_path, allow_pickle=False)["obs_uniques"]) if vocab_path.exists() else None,
    }


def write_cscg_paper_precision_snapshot(
    *,
    config: Any,
    stage: str,
    iteration: int,
    step: int,
    observed_samples: int,
    model_npz: dict[str, Any],
    obs_uniques: np.ndarray,
    action_uniques: np.ndarray,
) -> Path:
    out_dir = cscg_snapshot_dir(config, step)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "model.npz",
        T=np.asarray(model_npz["T"]),
        n_clones=np.asarray(model_npz["n_clones"]),
        obs_uniques=np.asarray(obs_uniques),
        action_uniques=np.asarray(action_uniques),
    )
    atomic_write_json(
        out_dir / "snapshot.json",
        {
            "schema": "paper_precision_snapshot_v1",
            "method": "cscg",
            "stage": str(stage),
            "iteration": int(iteration),
            "step": int(step),
            "observed_samples": int(observed_samples),
            "level": int(config.level),
            "sensor": str(config.sensor),
            "sensor_range": config.sensor_range,
            "training_mode": str(getattr(config, "training_mode", "full")),
            "train_algorithm": str(getattr(config, "train_algorithm", "em")),
            "paper_precision": {
                "mode": str(getattr(config, "paper_precision_mode", "off")),
                "eval_interval": int(getattr(config, "paper_precision_eval_interval", 1)),
                "eval_checkpoints": getattr(config, "paper_precision_eval_checkpoints", None),
                "num_points": getattr(config, "paper_precision_num_points", None),
                "num_walks": int(getattr(config, "paper_precision_num_walks", 100)),
                "walk_length": int(getattr(config, "paper_precision_walk_length", 10_000)),
                "rand_prob": float(getattr(config, "paper_precision_rand_prob", 0.3)),
                "seed": int(getattr(config, "paper_precision_seed", 0)),
            },
        },
    )
    return out_dir


def load_cscg_paper_precision_snapshot(path: Path) -> dict[str, Any]:
    snapshot_dir = Path(path)
    meta = load_json(snapshot_dir / "snapshot.json") or {}
    with np.load(snapshot_dir / "model.npz", allow_pickle=False) as model:
        T = np.asarray(model["T"], dtype=np.float32)
        n_clones = np.asarray(model["n_clones"], dtype=np.int64)
        if "obs_uniques" in model.files and "action_uniques" in model.files:
            obs_uniques = np.asarray(model["obs_uniques"])
            action_uniques = np.asarray(model["action_uniques"])
        else:
            vocab_path = snapshot_dir / "observation_vocabulary.npz"
            if not vocab_path.exists():
                raise KeyError(
                    "CSCG snapshot is missing obs_uniques/action_uniques in model.npz "
                    "and has no observation_vocabulary.npz sidecar"
                )
            with np.load(vocab_path, allow_pickle=False) as vocab:
                obs_uniques = np.asarray(vocab["obs_uniques"])
                action_uniques = np.asarray(vocab["action_uniques"])
    return {
        "meta": meta,
        "T": T,
        "model_npz": {"T": T, "n_clones": n_clones},
        "n_clones": n_clones,
        "obs_uniques": obs_uniques,
        "action_uniques": action_uniques,
    }


def list_paper_precision_snapshots(run_root: Path, method: str) -> list[Path]:
    root = Path(run_root) / "snapshots" / method / "paper_precision"
    if not root.exists():
        return []
    return sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith("step=")],
        key=lambda path: int(path.name.split("=", 1)[1]),
    )


def pocml_snapshot_dir(config: Any, step: int, n_states: int) -> Path:
    return Path(config.output_root).parents[1] / "snapshots" / "pocml" / "paper_precision" / f"capacity={n_states}" / f"step={int(step)}"


def write_pocml_paper_precision_snapshot(
    *,
    config: Any,
    step: int,
    observed_samples: int,
    obs_uniques: np.ndarray,
    action_uniques: np.ndarray,
    model_state_dict: Any,
    torch_module: Any,
    n_states: int | None = None,
) -> Path:
    if n_states is not None:
        resolved_n_states = int(n_states)
    elif config.n_states is not None:
        resolved_n_states = int(config.n_states)
    else:
        resolved_n_states = int(obs_uniques.shape[0])
    out_dir = pocml_snapshot_dir(config, step, resolved_n_states)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "vocabulary.npz", obs_uniques=np.asarray(obs_uniques), action_uniques=np.asarray(action_uniques))
    torch_module.save(model_state_dict, str(out_dir / "model_state_dict.pt"))
    atomic_write_json(
        out_dir / "snapshot.json",
        {
            "schema": "paper_precision_snapshot_v1",
            "method": "pocml",
            "step": int(step),
            "observed_samples": int(observed_samples),
            "level": int(config.level),
            "sensor": str(config.sensor),
            "sensor_range": config.sensor_range,
            "model_arch": {
                "n_obs": int(obs_uniques.shape[0]),
                "n_actions": int(action_uniques.shape[0]),
                "n_states": resolved_n_states,
                "state_dim": int(config.state_dim),
                "random_feature_dim": int(config.random_feature_dim),
                "alpha": float(config.alpha),
                "memory_bias": bool(config.memory_bias),
            },
            "paper_precision": {
                "mode": str(getattr(config, "paper_precision_mode", "off")),
                "eval_interval": int(getattr(config, "paper_precision_eval_interval", 1)),
                "eval_checkpoints": getattr(config, "paper_precision_eval_checkpoints", None),
                "num_points": getattr(config, "paper_precision_num_points", None),
                "num_walks": int(getattr(config, "paper_precision_num_walks", 100)),
                "walk_length": int(getattr(config, "paper_precision_walk_length", 10_000)),
                "rand_prob": float(getattr(config, "paper_precision_rand_prob", 0.3)),
                "seed": int(getattr(config, "paper_precision_seed", 0)),
            },
        },
    )
    return out_dir


def load_pocml_paper_precision_snapshot(path: Path) -> dict[str, Any]:
    snapshot_dir = Path(path)
    meta = load_json(snapshot_dir / "snapshot.json") or {}
    vocab = np.load(snapshot_dir / "vocabulary.npz", allow_pickle=False)
    return {
        "meta": meta,
        "obs_uniques": np.asarray(vocab["obs_uniques"]),
        "action_uniques": np.asarray(vocab["action_uniques"]),
        "model_state_dict_path": snapshot_dir / "model_state_dict.pt",
    }


def list_pocml_paper_precision_snapshots(run_root: Path) -> dict[int, list[Path]]:
    """Returns dict keyed by n_states (capacity), value is sorted list of step snapshot dirs."""
    root = Path(run_root) / "snapshots" / "pocml" / "paper_precision"
    if not root.exists():
        return {}
    result: dict[int, list[Path]] = {}
    for capacity_dir in sorted(root.iterdir()):
        if not capacity_dir.is_dir() or not capacity_dir.name.startswith("capacity="):
            continue
        n_states = int(capacity_dir.name.split("=", 1)[1])
        step_dirs = sorted(
            [p for p in capacity_dir.iterdir() if p.is_dir() and p.name.startswith("step=")],
            key=lambda p: int(p.name.split("=", 1)[1]),
        )
        if step_dirs:
            result[n_states] = step_dirs
    return result
