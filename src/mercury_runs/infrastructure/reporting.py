from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl
from tqdm import tqdm

from ..domain.models import PIPELINE_SCHEMA_VERSION
from .computational import derive_efficiency_metrics, read_computational_artifact, summarize_computational_raw
from .paper_precision import summarize_paper_precision_artifact
from ..save_results import read_bundle_meta
from .runtime import atomic_write_json, utc_now
from .state_columns import load_ground_truth_proxy_states
from .storage import load_run_config_model, load_run_status_model, read_native_metrics_from_bundle


def _target_method(run_payload: Dict[str, Any]) -> str:
    method = str(run_payload.get("method") or "").strip().lower()
    if method in {"mercury", "pocml", "cscg"}:
        return method
    baselines = run_payload.get("baselines") or {}
    if bool((baselines.get("pocml") or {}).get("enabled", False)) and not bool((baselines.get("cscg") or {}).get("enabled", False)):
        return "pocml"
    if bool((baselines.get("cscg") or {}).get("enabled", False)) and not bool((baselines.get("pocml") or {}).get("enabled", False)):
        return "cscg"
    return "mercury"


def _target_states_path(run_status, method: str) -> Optional[Path]:
    subsystem = run_status.subsystems.get(method)
    states_path_raw = (subsystem.artifacts if subsystem is not None else {}).get("states_parquet") if subsystem is not None else None
    if not states_path_raw:
        return None
    states_path = Path(states_path_raw)
    return states_path if states_path.exists() else None


def _target_paper_precision_path(run_status, method: str) -> Optional[Path]:
    subsystem = run_status.subsystems.get(method)
    path_raw = (subsystem.artifacts if subsystem is not None else {}).get("paper_precision_json") if subsystem is not None else None
    if not path_raw:
        return None
    path = Path(path_raw)
    return path if path.exists() else None


def _target_native_metrics(run_status, method: str, states_path: Optional[Path]) -> Dict[str, Any]:
    if method == "mercury":
        return summarize_mercury_native_metrics(states_path) if states_path is not None else {}
    subsystem = run_status.subsystems.get(method)
    return dict((subsystem.native_metrics if subsystem is not None else {}) or {})


def _target_computational_path(run_status, method: str) -> Optional[Path]:
    subsystem = run_status.subsystems.get(method)
    path_raw = (subsystem.artifacts if subsystem is not None else {}).get("computational_json") if subsystem is not None else None
    if not path_raw:
        return None
    path = Path(path_raw)
    return path if path.exists() else None


def _precision_reference(method: str, native_metrics: Dict[str, Any], paper_precision_stats: Dict[str, Any]) -> Optional[float]:
    if "latent_precision" in paper_precision_stats:
        return float(paper_precision_stats["latent_precision"])
    if "capacity_precision" in paper_precision_stats:
        return float(paper_precision_stats["capacity_precision"])
    target = _target_metric_columns(method, native_metrics).get("target_n1_accuracy")
    return float(target) if target is not None else None


def _target_metric_columns(method: str, native_metrics: Dict[str, Any]) -> Dict[str, Any]:
    if method == "mercury":
        return {
            "target_n1_accuracy": ((native_metrics.get("latent") or {}).get("n_step_observation_prediction_accuracy") or {}).get("n1"),
            "target_log_likelihood_mean": (native_metrics.get("latent") or {}).get("trajectory_log_likelihood_mean"),
            "sensory_n1_accuracy": ((native_metrics.get("sensory") or {}).get("n_step_observation_prediction_accuracy") or {}).get("n1"),
            "latent_n1_accuracy": ((native_metrics.get("latent") or {}).get("n_step_observation_prediction_accuracy") or {}).get("n1"),
            "sensory_log_likelihood_mean": (native_metrics.get("sensory") or {}).get("trajectory_log_likelihood_mean"),
            "latent_log_likelihood_mean": (native_metrics.get("latent") or {}).get("trajectory_log_likelihood_mean"),
        }
    return {
        "target_n1_accuracy": ((native_metrics.get("n_step_observation_prediction_accuracy") or {}).get("n1")),
        "target_log_likelihood_mean": native_metrics.get("trajectory_log_likelihood_mean"),
        "sensory_n1_accuracy": None,
        "latent_n1_accuracy": None,
        "sensory_log_likelihood_mean": None,
        "latent_log_likelihood_mean": None,
    }


def summarize_precision_metrics(metrics_path: Path) -> Dict[str, Any]:
    frame = pl.read_parquet(metrics_path)
    precision_cols = [col for col in frame.columns if col.endswith("_precision")]
    exprs: list[Any] = [pl.len().alias("rows")]
    for col in precision_cols:
        exprs.extend(
            [
                pl.col(col).mean().alias(f"{col}_mean"),
                pl.col(col).min().alias(f"{col}_min"),
                pl.col(col).max().alias(f"{col}_max"),
            ]
        )
    if "latent_node_count" in frame.columns:
        exprs.extend(
            [
                pl.col("latent_node_count").mean().alias("latent_node_count_mean"),
                pl.col("latent_node_count").min().alias("latent_node_count_min"),
                pl.col("latent_node_count").max().alias("latent_node_count_max"),
            ]
        )
    return frame.select(exprs).to_dicts()[0]


def resolve_dataset_parquet_from_states(states_path: Path, datasets_root: Path = Path("datasets")) -> Path:
    meta = read_bundle_meta(states_path)
    if meta is None or not meta.source.dataset_parquet_name:
        raise RuntimeError(f"Missing source dataset metadata in {states_path}")
    candidates = sorted(datasets_root.rglob(meta.source.dataset_parquet_name))
    if not candidates:
        raise FileNotFoundError(f"Dataset parquet {meta.source.dataset_parquet_name} not found under {datasets_root}")
    return candidates[-1]


def load_action_series(dataset_parquet_path: Path) -> np.ndarray:
    frame = pl.read_parquet(dataset_parquet_path)
    action_cols = [c for c in frame.columns if c.startswith("action_")]
    if not action_cols:
        raise RuntimeError(f"No action_* columns found in {dataset_parquet_path}")
    return np.argmax(frame.select(action_cols).to_numpy(), axis=1).astype(np.int32, copy=False)


def fit_transition_probs(state_series: np.ndarray, action_series: np.ndarray) -> Dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    counts: Dict[tuple[int, int], Dict[int, int]] = {}
    for t in range(min(len(state_series), len(action_series)) - 1):
        key = (int(state_series[t]), int(action_series[t]))
        counts.setdefault(key, {})
        next_state = int(state_series[t + 1])
        counts[key][next_state] = counts[key].get(next_state, 0) + 1
    probs: Dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for key, next_counts in counts.items():
        next_states = np.asarray(sorted(next_counts.keys()), dtype=np.int32)
        c = np.asarray([next_counts[int(s)] for s in next_states], dtype=np.float64)
        probs[key] = (next_states, c / float(c.sum()))
    return probs


def fit_gt_given_state_probs(state_series: np.ndarray, gt_series: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n_state = int(max(state_series.max(), 0)) + 1
    n_gt = int(max(gt_series.max(), 0)) + 1
    counts = np.full((n_state, n_gt), eps, dtype=np.float64)
    for s, g in zip(state_series.tolist(), gt_series.tolist()):
        counts[int(s), int(g)] += 1.0
    counts /= counts.sum(axis=1, keepdims=True)
    return counts


def rollout_n_step_observation_prediction_metrics(
    *,
    state_series: np.ndarray,
    gt_series: np.ndarray,
    action_series: np.ndarray,
    horizons: list[int],
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> dict[str, Any]:
    trans_probs = fit_transition_probs(state_series, action_series)
    gt_given_state = fit_gt_given_state_probs(state_series, gt_series)
    n_state = gt_given_state.shape[0]
    n_step_acc: dict[str, float] = {}
    for h in horizons:
        if h <= 0:
            continue
        correct = 0
        total = 0
        iterator = range(0, len(state_series) - h)
        if show_progress:
            iterator = tqdm(
                iterator,
                total=max(0, len(state_series) - h),
                desc=(progress_desc or f"Rollout Eval n{h}"),
                position=0,
            )
        for t in iterator:
            dist = np.zeros(n_state, dtype=np.float64)
            s0 = int(state_series[t])
            if s0 >= n_state:
                continue
            dist[s0] = 1.0
            for k in range(h):
                a = int(action_series[t + k])
                next_dist = np.zeros_like(dist)
                for s in np.nonzero(dist > 0.0)[0].tolist():
                    p_s = float(dist[s])
                    key = (int(s), a)
                    if key in trans_probs:
                        ns, ps = trans_probs[key]
                        next_dist[ns] += p_s * ps
                    else:
                        next_dist[s] += p_s
                dist = next_dist
            if int(np.argmax(dist @ gt_given_state)) == int(gt_series[t + h]):
                correct += 1
            total += 1
        n_step_acc[f"n{h}"] = (float(correct) / float(total)) if total > 0 else 0.0
    ll_sum = 0.0
    ll_total = 0
    iterator = range(0, len(state_series) - 1)
    if show_progress:
        iterator = tqdm(iterator, total=max(0, len(state_series) - 1), desc=((progress_desc or "Rollout Eval") + " LL"), position=0)
    for t in iterator:
        dist = np.zeros(n_state, dtype=np.float64)
        s0 = int(state_series[t])
        if s0 >= n_state:
            continue
        dist[s0] = 1.0
        a = int(action_series[t])
        next_dist = np.zeros_like(dist)
        for s in np.nonzero(dist > 0.0)[0].tolist():
            p_s = float(dist[s])
            key = (int(s), a)
            if key in trans_probs:
                ns, ps = trans_probs[key]
                next_dist[ns] += p_s * ps
            else:
                next_dist[s] += p_s
        gt_dist = next_dist @ gt_given_state
        p_true = float(gt_dist[int(gt_series[t + 1])]) if int(gt_series[t + 1]) < gt_dist.shape[0] else 1e-12
        ll_sum += float(np.log(max(p_true, 1e-12)))
        ll_total += 1
    return {
        "n_step_observation_prediction_accuracy": n_step_acc,
        "trajectory_log_likelihood_sum": ll_sum,
        "trajectory_log_likelihood_mean": (ll_sum / float(ll_total)) if ll_total > 0 else 0.0,
        "steps": int(len(state_series)),
    }


def summarize_mercury_native_metrics(states_path: Path) -> Dict[str, Any]:
    frame = pl.read_parquet(states_path).select("sensory_bmu", "latent_bmu")
    gt = load_ground_truth_proxy_states(states_path)
    sensory = frame.get_column("sensory_bmu").to_numpy().astype(np.int32, copy=False)
    latent = frame.get_column("latent_bmu").to_numpy().astype(np.int32, copy=False)
    dataset_path = resolve_dataset_parquet_from_states(states_path)
    actions = load_action_series(dataset_path).astype(np.int32, copy=False)
    length = min(len(gt), len(actions), len(sensory), len(latent))
    if length < 2:
        zero = {
            "n_step_observation_prediction_accuracy": {"n1": 0.0, "n3": 0.0, "n5": 0.0},
            "trajectory_log_likelihood_sum": 0.0,
            "trajectory_log_likelihood_mean": 0.0,
            "steps": int(length),
        }
        return {"sensory": zero, "latent": zero}
    gt = gt[:length]
    actions = actions[:length]
    sensory = sensory[:length]
    latent = latent[:length]
    return {
        "sensory": rollout_n_step_observation_prediction_metrics(
            state_series=sensory, gt_series=gt, action_series=actions, horizons=[1, 3, 5]
        ),
        "latent": rollout_n_step_observation_prediction_metrics(
            state_series=latent, gt_series=gt, action_series=actions, horizons=[1, 3, 5]
        ),
    }


def write_method_comparison_report(*, run_root: Path, run_id: str, run_payload: Dict[str, Any], subsystems: Dict[str, Dict[str, Any]]) -> Path:
    mercury_artifacts = (subsystems.get("mercury") or {}).get("artifacts") or {}
    mercury_states_path = Path(mercury_artifacts["states_parquet"])
    pocml_artifacts = (subsystems.get("pocml") or {}).get("artifacts") or {}
    cscg_artifacts = (subsystems.get("cscg") or {}).get("artifacts") or {}
    pocml_state_path_raw = pocml_artifacts.get("states_parquet")
    cscg_state_path_raw = cscg_artifacts.get("states_parquet")
    mercury_paper_precision_raw = mercury_artifacts.get("paper_precision_json")
    mercury_computational_raw = mercury_artifacts.get("computational_json")
    pocml_paper_precision_raw = pocml_artifacts.get("paper_precision_json")
    pocml_computational_raw = pocml_artifacts.get("computational_json")
    cscg_paper_precision_raw = cscg_artifacts.get("paper_precision_json")
    cscg_computational_raw = cscg_artifacts.get("computational_json")
    pocml_states_path = Path(pocml_state_path_raw) if pocml_state_path_raw else None
    cscg_states_path = Path(cscg_state_path_raw) if cscg_state_path_raw else None
    mercury_paper_precision_path = Path(mercury_paper_precision_raw) if mercury_paper_precision_raw else None
    mercury_computational_path = Path(mercury_computational_raw) if mercury_computational_raw else None
    pocml_paper_precision_path = Path(pocml_paper_precision_raw) if pocml_paper_precision_raw else None
    pocml_computational_path = Path(pocml_computational_raw) if pocml_computational_raw else None
    cscg_paper_precision_path = Path(cscg_paper_precision_raw) if cscg_paper_precision_raw else None
    cscg_computational_path = Path(cscg_computational_raw) if cscg_computational_raw else None
    mercury_native_metrics = summarize_mercury_native_metrics(mercury_states_path)
    mercury_paper_precision = (
        summarize_paper_precision_artifact(mercury_paper_precision_path)
        if mercury_paper_precision_path is not None and mercury_paper_precision_path.exists()
        else {}
    )
    mercury_computational = summarize_computational_raw(mercury_computational_path)
    pocml_native_metrics = read_native_metrics_from_bundle(pocml_states_path, "pocml_eval")
    pocml_paper_precision = (
        summarize_paper_precision_artifact(pocml_paper_precision_path)
        if pocml_paper_precision_path is not None and pocml_paper_precision_path.exists()
        else {}
    )
    pocml_computational = summarize_computational_raw(pocml_computational_path)
    cscg_native_metrics = read_native_metrics_from_bundle(cscg_states_path, "cscg_eval")
    cscg_paper_precision = (
        summarize_paper_precision_artifact(cscg_paper_precision_path)
        if cscg_paper_precision_path is not None and cscg_paper_precision_path.exists()
        else {}
    )
    cscg_computational = summarize_computational_raw(cscg_computational_path)
    report = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at_utc": utc_now(),
        "status": {
            "run": "completed",
            "mercury": (subsystems.get("mercury") or {}).get("status"),
            "pocml": (subsystems.get("pocml") or {}).get("status"),
            "cscg": (subsystems.get("cscg") or {}).get("status"),
        },
        "errors": {
            "pocml": (subsystems.get("pocml") or {}).get("error"),
            "cscg": (subsystems.get("cscg") or {}).get("error"),
        },
        "inputs": {
            "mercury_states_parquet": str(mercury_states_path),
            "mercury_paper_precision_json": (str(mercury_paper_precision_path) if mercury_paper_precision_path is not None else None),
            "pocml_states_parquet": (str(pocml_states_path) if pocml_states_path is not None else None),
            "pocml_paper_precision_json": (str(pocml_paper_precision_path) if pocml_paper_precision_path is not None else None),
            "cscg_states_parquet": (str(cscg_states_path) if cscg_states_path is not None else None),
            "cscg_paper_precision_json": (str(cscg_paper_precision_path) if cscg_paper_precision_path is not None else None),
        },
        "config": {
            "level": run_payload["level"],
            "sensor": run_payload["sensor"],
            "sensor_range": run_payload["sensor_range"],
            "baselines": run_payload.get("baselines") or {},
        },
        "mercury_native_metrics": mercury_native_metrics,
        "mercury_paper_precision": mercury_paper_precision,
        "mercury_computational": {
            "raw": mercury_computational,
            "derived": derive_efficiency_metrics(
                raw_metrics=mercury_computational,
                precision_value=_precision_reference("mercury", mercury_native_metrics, mercury_paper_precision),
            ),
            "notes": (read_computational_artifact(mercury_computational_path).get("notes") or []),
        },
        "pocml_native_metrics": pocml_native_metrics,
        "pocml_paper_precision": pocml_paper_precision,
        "pocml_computational": {
            "raw": pocml_computational,
            "derived": derive_efficiency_metrics(
                raw_metrics=pocml_computational,
                precision_value=_precision_reference("pocml", pocml_native_metrics, pocml_paper_precision),
            ),
            "notes": (read_computational_artifact(pocml_computational_path).get("notes") or []),
        },
        "pocml_config": read_native_metrics_from_bundle(pocml_states_path, "pocml"),
        "cscg_native_metrics": cscg_native_metrics,
        "cscg_paper_precision": cscg_paper_precision,
        "cscg_computational": {
            "raw": cscg_computational,
            "derived": derive_efficiency_metrics(
                raw_metrics=cscg_computational,
                precision_value=_precision_reference("cscg", cscg_native_metrics, cscg_paper_precision),
            ),
            "notes": (read_computational_artifact(cscg_computational_path).get("notes") or []),
        },
        "cscg_config": read_native_metrics_from_bundle(cscg_states_path, "cscg"),
    }
    report_path = run_root / "comparison_summary.json"
    atomic_write_json(report_path, report)
    return report_path


def write_study_summary(study_root: Path, study_name: str) -> Path:
    study_dir = study_root / study_name
    if not study_dir.exists():
        raise FileNotFoundError(f"Study directory not found: {study_dir}")
    summary_rows: List[dict] = []
    for run_dir in sorted([path for path in study_dir.iterdir() if path.is_dir()]):
        run_status = load_run_status_model(run_dir / "run_status.json")
        run_config = load_run_config_model(run_dir / "run_config.json")
        if run_status is None or run_config is None or int(run_status.schema_version) != PIPELINE_SCHEMA_VERSION:
            continue
        method = _target_method(run_config.run_payload)
        states_parquet_path = _target_states_path(run_status, method)
        if states_parquet_path is None:
            continue
        meta = read_bundle_meta(states_parquet_path)
        if meta is None:
            continue
        native_metrics = _target_native_metrics(run_status, method, states_parquet_path)
        paper_precision_path = _target_paper_precision_path(run_status, method)
        computational_path = _target_computational_path(run_status, method)
        paper_precision_stats = (
            summarize_paper_precision_artifact(paper_precision_path)
            if paper_precision_path is not None
            else {}
        )
        computational_stats = summarize_computational_raw(computational_path)
        derived_computational = derive_efficiency_metrics(
            raw_metrics=computational_stats,
            precision_value=_precision_reference(method, native_metrics, paper_precision_stats),
        )
        metric_columns = _target_metric_columns(method, native_metrics)
        row = {
            "run_id": run_status.run_id or meta.run_id,
            "timestamp_utc": meta.timestamp_utc,
            "method": method,
            "level": meta.source.level,
            "sensor": meta.source.sensor,
            "sensor_range": meta.source.sensor_range,
            "seed": run_config.run_payload.get("seed"),
            "rand_prob": run_config.run_payload.get("rand_prob"),
            "num_steps": run_config.run_payload.get("num_steps"),
            "window_length": run_config.run_payload.get("window_length"),
            "memory_length": run_config.run_payload.get("memory_length"),
            "activation_threshold": (run_config.run_payload.get("sensory") or {}).get("activation_threshold"),
            "sensory_weighting": (run_config.run_payload.get("sensory") or {}).get("sensory_weighting"),
            "ambiguity_threshold": (run_config.run_payload.get("latent") or {}).get("ambiguity_threshold"),
            "ambiguity_decay": (run_config.run_payload.get("latent") or {}).get("ambiguity_decay"),
            "trace_decay": (run_config.run_payload.get("latent") or {}).get("trace_decay"),
            "lambda_trace": (run_config.run_payload.get("latent") or {}).get("lambda_trace"),
            "weight_memory": (run_config.run_payload.get("latent") or {}).get("weight_memory"),
            "weight_undirected": (run_config.run_payload.get("latent") or {}).get("weight_undirected"),
            "weight_base": (run_config.run_payload.get("latent") or {}).get("weight_base"),
            "weight_action": (run_config.run_payload.get("latent") or {}).get("weight_action"),
            "memory_replay": (run_config.run_payload.get("latent") or {}).get("memory_replay"),
            "memory_disambiguation": (run_config.run_payload.get("latent") or {}).get("memory_disambiguation"),
            "am_n_codebook": (run_config.run_payload.get("action_map") or {}).get("n_codebook"),
            "am_lr": (run_config.run_payload.get("action_map") or {}).get("lr"),
            "am_sigma": (run_config.run_payload.get("action_map") or {}).get("sigma"),
            "am_key": (run_config.run_payload.get("action_map") or {}).get("key"),
            "run_status": run_status.status,
            **metric_columns,
        }
        computational_column_map = {
            "train_wall_time_seconds": "comp_train_wall_time_seconds",
            "eval_wall_time_seconds": "comp_eval_wall_time_seconds",
            "total_wall_time_seconds": "comp_total_wall_time_seconds",
            "train_steps_per_second": "comp_train_steps_per_second",
            "eval_steps_per_second": "comp_eval_steps_per_second",
            "peak_rss_mb": "comp_peak_rss_mb",
            "deployable_artifact_bytes": "comp_deployable_artifact_bytes",
            "total_artifact_bytes": "comp_total_artifact_bytes",
            "jit_warmup_wall_time_seconds": "comp_jit_warmup_wall_time_seconds",
        }
        derived_column_map = {
            "seconds_per_eval_step": "comp_eval_seconds_per_step",
            "seconds_per_train_step": "comp_train_seconds_per_step",
            "precision_per_total_second": "comp_precision_per_total_second",
            "precision_per_peak_rss_mb": "comp_precision_per_peak_rss_mb",
            "precision_per_deployable_mb": "comp_precision_per_deployable_mb",
        }
        for key, column in computational_column_map.items():
            row[column] = computational_stats.get(key)
        for key, column in derived_column_map.items():
            row[column] = derived_computational.get(key)
        if method == "pocml":
            pocml_cfg = ((run_config.run_payload.get("baselines") or {}).get("pocml") or {})
            row.update(
                {
                    "pocml_epochs": pocml_cfg.get("epochs"),
                    "pocml_trajectory_length": pocml_cfg.get("trajectory_length"),
                    "pocml_alpha": pocml_cfg.get("alpha"),
                    "pocml_batch_size": pocml_cfg.get("batch_size"),
                }
            )
        elif method == "cscg":
            cscg_cfg = ((run_config.run_payload.get("baselines") or {}).get("cscg") or {})
            row.update(
                {
                    "cscg_clones_per_obs": cscg_cfg.get("clones_per_obs"),
                    "cscg_n_iter": cscg_cfg.get("n_iter"),
                    "cscg_training_mode": cscg_cfg.get("training_mode"),
                    "cscg_train_algorithm": cscg_cfg.get("train_algorithm"),
                }
            )
        for key, value in paper_precision_stats.items():
            if isinstance(value, (int, float)):
                row[f"paper_{key}"] = value
        summary_rows.append(row)
    if not summary_rows:
        raise RuntimeError(f"No complete runs found under {study_dir}. Expected each run folder to contain valid target-method states.")
    output_path = study_dir / "study_summary.parquet"
    available_columns = set().union(*(row.keys() for row in summary_rows))
    if "paper_latent_precision" in available_columns:
        sort_column = "paper_latent_precision"
    elif "paper_capacity_precision" in available_columns:
        sort_column = "paper_capacity_precision"
    else:
        sort_column = "target_n1_accuracy"
    pl.DataFrame(summary_rows).sort(sort_column, descending=True).write_parquet(output_path, compression="zstd", statistics=True)
    return output_path
