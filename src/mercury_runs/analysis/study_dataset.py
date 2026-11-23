from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import polars as pl

from ..infrastructure.storage import load_run_config_model, load_run_status_model
from ..save_results import read_bundle_meta


ARTIFACT_KEYS: tuple[str, ...] = (
    "run_config_json",
    "run_status_json",
    "mercury_states_parquet",
    "mercury_attribution_parquet",
    "mercury_paper_precision_json",
    "mercury_computational_json",
    "mercury_latent_graph_npz",
    "mercury_internal_graph_png",
    "mercury_bmu_attribution_step_png",
    "mercury_bmu_attribution_edge_f1_png",
    "mercury_bmu_attribution_node_ratio_png",
    "internal_graphs_comparison_png",
    "comparison_summary_json",
)

HISTORY_MATRIX_KEYS: frozenset[str] = frozenset(
    {
        "sensory_cooccurrence_matrix",
        "latent_cooccurrence_matrix",
        "sensory_exact_state_cooccurrence_matrix",
        "latent_exact_state_cooccurrence_matrix",
    }
)

@dataclass(frozen=True)
class StudyDataset:
    runs: pl.DataFrame
    history: pl.DataFrame
    states: pl.DataFrame
    attribution: pl.DataFrame
    artifacts: pl.DataFrame

    def run_metadata(self, columns: Iterable[str] | None = None) -> pl.DataFrame:
        if columns is None:
            return self.runs
        selected = ["run_id", *[column for column in columns if column != "run_id"]]
        available = [column for column in selected if column in self.runs.columns]
        return self.runs.select(available)

    def history_with_run_metadata(self, columns: Iterable[str] | None = None) -> pl.DataFrame:
        metadata = self.run_metadata(columns)
        return self.history.join(metadata, on="run_id", how="left", coalesce=True)

    def states_with_run_metadata(self, columns: Iterable[str] | None = None) -> pl.DataFrame:
        metadata = self.run_metadata(columns)
        return self.states.join(metadata, on="run_id", how="left", coalesce=True)

    def attribution_with_run_metadata(self, columns: Iterable[str] | None = None) -> pl.DataFrame:
        metadata = self.run_metadata(columns)
        return self.attribution.join(metadata, on="run_id", how="left", coalesce=True)


def _resolve_existing_path(raw: str, *, study_root: Path, run_root: Path) -> Path | None:
    path = Path(raw)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend(
            [
                run_root / path,
                study_root / path,
                study_root.parent / path,
                study_root.parents[1] / path if len(study_root.parents) > 1 else study_root / path,
                study_root.parents[2] / path if len(study_root.parents) > 2 else study_root / path,
                study_root.parents[3] / path if len(study_root.parents) > 3 else study_root / path,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _artifact_path(study_root: Path, run_root: Path, status_artifacts: dict[str, Any], method: str, name: str) -> Path | None:
    direct_key = f"{method}_{name}"
    raw = status_artifacts.get(direct_key)
    if isinstance(raw, str) and raw:
        path = _resolve_existing_path(raw, study_root=study_root, run_root=run_root)
        if path is not None:
            return path

    subsystem = status_artifacts.get("subsystems")
    if isinstance(subsystem, dict):
        method_payload = subsystem.get(method)
        if isinstance(method_payload, dict):
            method_artifacts = method_payload.get("artifacts")
            if isinstance(method_artifacts, dict):
                raw = method_artifacts.get(name)
                if isinstance(raw, str) and raw:
                    path = _resolve_existing_path(raw, study_root=study_root, run_root=run_root)
                    if path is not None:
                        return path

    fallback = {
        "states_parquet": run_root / "bundles" / method,
        "attribution_parquet": run_root / "bundles" / method,
        "paper_precision_json": run_root / "metrics" / method,
        "computational_json": run_root / "metrics" / method,
        "latent_graph_npz": run_root / "bundles" / method,
        "internal_graph_png": run_root / "plots",
        "bmu_attribution_step_png": run_root / "plots",
        "bmu_attribution_edge_f1_png": run_root / "plots",
        "bmu_attribution_node_ratio_png": run_root / "plots",
    }
    base = fallback.get(name)
    if base is None or not base.exists():
        return None
    matches = sorted(base.rglob(f"*_{name}")) if base.is_dir() and name.endswith((".json", ".npz", ".png", ".parquet")) else []
    return matches[-1] if matches else None


def _resolve_mercury_artifacts(study_root: Path, run_root: Path, run_id: str, status_model: Any) -> dict[str, str | None]:
    top_level_artifacts = dict(getattr(status_model, "artifacts", {}) or {})
    subsystem_artifacts = {
        "subsystems": {
            name: {"artifacts": dict(getattr(subsystem, "artifacts", {}) or {})}
            for name, subsystem in dict(getattr(status_model, "subsystems", {}) or {}).items()
        }
    }
    artifact_payload: dict[str, Any] = {}
    artifact_payload.update(subsystem_artifacts)
    artifact_payload.update(top_level_artifacts)

    def _top(name: str) -> str | None:
        raw = top_level_artifacts.get(name)
        if isinstance(raw, str):
            path = _resolve_existing_path(raw, study_root=study_root, run_root=run_root)
            if path is not None:
                return str(path)
        return None

    resolved: dict[str, str | None] = {
        "run_id": run_id,
        "run_config_json": _top("run_config_json") or str(run_root / "run_config.json"),
        "run_status_json": _top("run_status_json") or str(run_root / "run_status.json"),
        "comparison_summary_json": _top("comparison_summary_json"),
        "internal_graphs_comparison_png": _top("internal_graphs_comparison_png"),
    }
    mercury_names = {
        "mercury_states_parquet": "states_parquet",
        "mercury_attribution_parquet": "attribution_parquet",
        "mercury_paper_precision_json": "paper_precision_json",
        "mercury_computational_json": "computational_json",
        "mercury_latent_graph_npz": "latent_graph_npz",
        "mercury_internal_graph_png": "internal_graph_png",
        "mercury_bmu_attribution_step_png": "bmu_attribution_step_png",
        "mercury_bmu_attribution_edge_f1_png": "bmu_attribution_edge_f1_png",
        "mercury_bmu_attribution_node_ratio_png": "bmu_attribution_node_ratio_png",
    }
    for column, name in mercury_names.items():
        path = _artifact_path(study_root, run_root, artifact_payload, "mercury", name)
        resolved[column] = str(path) if path is not None else None
    return resolved


def _payload_value(run_payload: dict[str, Any], cli_args: dict[str, Any], *keys: str) -> Any:
    current: Any = run_payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            current = None
            break
        current = current[key]
    if current is not None:
        return current
    current = cli_args
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _normalize_scalar_metadata(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    return None


def _run_record_from_models(
    *,
    run_root: Path,
    run_id: str,
    status_model: Any,
    config_model: Any,
    artifacts: dict[str, str | None],
) -> dict[str, Any]:
    run_payload = dict(getattr(config_model, "run_payload", {}) or {})
    cli_args = dict(getattr(config_model, "cli_args", {}) or {})
    states_path_raw = artifacts.get("mercury_states_parquet")
    meta = read_bundle_meta(Path(states_path_raw)) if states_path_raw else None
    level = _payload_value(run_payload, cli_args, "level")
    sensor = _payload_value(run_payload, cli_args, "sensor")
    sensor_range = _payload_value(run_payload, cli_args, "sensor_range")
    if meta is not None:
        level = meta.source.level
        sensor = meta.source.sensor
        sensor_range = meta.source.sensor_range
    row = {
        "run_id": run_id,
        "timestamp_utc": getattr(status_model, "finished_at_utc", None) or getattr(config_model, "saved_at_utc", None),
        "method": _payload_value(run_payload, cli_args, "method") or "mercury",
        "level": level,
        "sensor": sensor,
        "sensor_range": sensor_range,
        "seed": _payload_value(run_payload, cli_args, "seed"),
        "rand_prob": _payload_value(run_payload, cli_args, "rand_prob"),
        "num_steps": _payload_value(run_payload, cli_args, "num_steps"),
        "window_length": _payload_value(run_payload, cli_args, "window_length"),
        "memory_length": _payload_value(run_payload, cli_args, "memory_length"),
        "activation_threshold": _first_not_none(
            _payload_value(run_payload, cli_args, "sensory", "activation_threshold"),
            _payload_value(run_payload, cli_args, "activation_threshold"),
        ),
        "sensory_weighting": _first_not_none(
            _payload_value(run_payload, cli_args, "sensory", "sensory_weighting"),
            _payload_value(run_payload, cli_args, "sensory_weighting"),
        ),
        "ambiguity_threshold": _first_not_none(
            _payload_value(run_payload, cli_args, "latent", "ambiguity_threshold"),
            _payload_value(run_payload, cli_args, "ambiguity_threshold"),
        ),
        "ambiguity_decay": _first_not_none(
            _payload_value(run_payload, cli_args, "latent", "ambiguity_decay"),
            _payload_value(run_payload, cli_args, "ambiguity_decay"),
        ),
        "ambiguity_counter_mode": _first_not_none(
            _payload_value(run_payload, cli_args, "latent", "ambiguity_counter_mode"),
            _payload_value(run_payload, cli_args, "ambiguity_counter_mode"),
        ),
        "ambiguity_response_mode": _first_not_none(
            _payload_value(run_payload, cli_args, "ambiguity_response_mode"),
        ),
        "trace_decay": _first_not_none(
            _payload_value(run_payload, cli_args, "latent", "trace_decay"),
            _payload_value(run_payload, cli_args, "trace_decay"),
        ),
        "lambda_trace": _first_not_none(
            _payload_value(run_payload, cli_args, "latent", "lambda_trace"),
            _payload_value(run_payload, cli_args, "lambda_trace"),
        ),
        "weight_memory": _first_not_none(
            _payload_value(run_payload, cli_args, "latent", "weight_memory"),
            _payload_value(run_payload, cli_args, "weight_memory"),
        ),
        "weight_undirected": _first_not_none(
            _payload_value(run_payload, cli_args, "latent", "weight_undirected"),
            _payload_value(run_payload, cli_args, "weight_undirected"),
        ),
        "weight_base": _first_not_none(
            _payload_value(run_payload, cli_args, "latent", "weight_base"),
            _payload_value(run_payload, cli_args, "weight_base"),
        ),
        "weight_action": _first_not_none(
            _payload_value(run_payload, cli_args, "latent", "weight_action"),
            _payload_value(run_payload, cli_args, "weight_action"),
        ),
        "memory_replay": _payload_value(run_payload, cli_args, "latent", "memory_replay")
        if _payload_value(run_payload, cli_args, "latent", "memory_replay") is not None
        else _payload_value(run_payload, cli_args, "memory_replay"),
        "memory_disambiguation": _payload_value(run_payload, cli_args, "latent", "memory_disambiguation")
        if _payload_value(run_payload, cli_args, "latent", "memory_disambiguation") is not None
        else _payload_value(run_payload, cli_args, "memory_disambiguation"),
        "ambiguity_response_enabled": _payload_value(run_payload, cli_args, "latent", "ambiguity_response_enabled")
        if _payload_value(run_payload, cli_args, "latent", "ambiguity_response_enabled") is not None
        else _payload_value(run_payload, cli_args, "ambiguity_response_enabled"),
        "ambiguity_refinement_enabled": _payload_value(run_payload, cli_args, "latent", "ambiguity_refinement_enabled")
        if _payload_value(run_payload, cli_args, "latent", "ambiguity_refinement_enabled") is not None
        else _payload_value(run_payload, cli_args, "ambiguity_refinement_enabled"),
        "transition_reconstruction_enabled": _payload_value(run_payload, cli_args, "latent", "transition_reconstruction_enabled")
        if _payload_value(run_payload, cli_args, "latent", "transition_reconstruction_enabled") is not None
        else _payload_value(run_payload, cli_args, "transition_reconstruction_enabled"),
        "am_n_codebook": _first_not_none(
            _payload_value(run_payload, cli_args, "action_map", "n_codebook"),
            _payload_value(run_payload, cli_args, "am_n_codebook"),
        ),
        "am_lr": _first_not_none(
            _payload_value(run_payload, cli_args, "action_map", "lr"),
            _payload_value(run_payload, cli_args, "am_lr"),
        ),
        "am_sigma": _first_not_none(
            _payload_value(run_payload, cli_args, "action_map", "sigma"),
            _payload_value(run_payload, cli_args, "am_sigma"),
        ),
        "am_key": _first_not_none(
            _payload_value(run_payload, cli_args, "action_map", "key"),
            _payload_value(run_payload, cli_args, "am_key"),
        ),
        "run_status": getattr(status_model, "status", None),
        "run_root": str(run_root),
    }
    for key, raw_value in cli_args.items():
        if key in row:
            continue
        value = _normalize_scalar_metadata(raw_value)
        if value is not None:
            row[key] = value
    return row


def _merge_run_frames(discovered: pl.DataFrame, summary: pl.DataFrame) -> pl.DataFrame:
    if summary.is_empty():
        return discovered
    joined = discovered.join(summary, on="run_id", how="full", suffix="_summary", coalesce=True)
    for column in list(joined.columns):
        if not column.endswith("_summary"):
            continue
        base = column[: -len("_summary")]
        if base in joined.columns:
            joined = joined.with_columns(pl.coalesce(pl.col(base), pl.col(column)).alias(base)).drop(column)
        else:
            joined = joined.rename({column: base})
    return joined


def _scan_run_roots(study_root: Path) -> list[Path]:
    return sorted(path for path in study_root.iterdir() if path.is_dir() and path.name != "plots")


def _load_runs_and_artifacts(study_root: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    summary_path = study_root / "study_summary.parquet"
    summary = pl.read_parquet(summary_path) if summary_path.exists() else pl.DataFrame({"run_id": []}, schema={"run_id": pl.Utf8})

    artifact_rows: list[dict[str, Any]] = []
    discovered_rows: list[dict[str, Any]] = []
    for run_root in _scan_run_roots(study_root):
        run_status = load_run_status_model(run_root / "run_status.json")
        run_config = load_run_config_model(run_root / "run_config.json")
        if run_status is None or run_config is None:
            continue
        run_id = str(getattr(run_status, "run_id", None) or getattr(run_config, "run_id", None) or run_root.name)
        artifacts = _resolve_mercury_artifacts(study_root, run_root, run_id, run_status)
        artifact_rows.append(artifacts)
        discovered_rows.append(
            _run_record_from_models(
                run_root=run_root,
                run_id=run_id,
                status_model=run_status,
                config_model=run_config,
                artifacts=artifacts,
            )
        )

    artifacts_frame = pl.DataFrame(artifact_rows) if artifact_rows else pl.DataFrame({"run_id": []}, schema={"run_id": pl.Utf8})
    for key in ARTIFACT_KEYS:
        if key in artifacts_frame.columns:
            artifacts_frame = artifacts_frame.with_columns(pl.col(key).is_not_null().alias(f"has_{key}"))
    discovered = pl.DataFrame(discovered_rows) if discovered_rows else pl.DataFrame({"run_id": []}, schema={"run_id": pl.Utf8})
    if summary.is_empty():
        runs = discovered
    else:
        runs = _merge_run_frames(discovered, summary)
    if not artifacts_frame.is_empty():
        runs = runs.join(artifacts_frame, on="run_id", how="left", coalesce=True)
    return runs, artifacts_frame


def _history_rows_from_payload(run_id: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    history = payload.get("history")
    if not isinstance(history, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        row = {"run_id": run_id}
        for key, value in item.items():
            if key not in HISTORY_MATRIX_KEYS and isinstance(value, (dict, list)):
                continue
            row[key] = value
        rows.append(row)
    return rows


def _load_json_payload(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


def _concat_frames(frames: Iterable[pl.DataFrame]) -> pl.DataFrame:
    materialized = [frame for frame in frames if not frame.is_empty()]
    if not materialized:
        return pl.DataFrame()
    return pl.concat(materialized, how="diagonal_relaxed")


def _load_history_frame(runs: pl.DataFrame, artifacts: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    if artifacts.is_empty() or "mercury_paper_precision_json" not in artifacts.columns:
        return pl.DataFrame()
    for record in artifacts.select("run_id", "mercury_paper_precision_json").iter_rows(named=True):
        payload = _load_json_payload(record["mercury_paper_precision_json"])
        if payload is None:
            continue
        rows.extend(_history_rows_from_payload(str(record["run_id"]), payload))
    history = pl.DataFrame(rows) if rows else pl.DataFrame()
    if history.is_empty():
        return history
    return history


def _load_states_frame(artifacts: pl.DataFrame) -> pl.DataFrame:
    if artifacts.is_empty() or "mercury_states_parquet" not in artifacts.columns:
        return pl.DataFrame()
    frames: list[pl.DataFrame] = []
    for record in artifacts.select("run_id", "mercury_states_parquet").iter_rows(named=True):
        path_raw = record["mercury_states_parquet"]
        if not path_raw:
            continue
        frame = pl.read_parquet(path_raw).with_row_index("step", offset=1).with_columns(pl.lit(str(record["run_id"])).alias("run_id"))
        frames.append(frame)
    return _concat_frames(frames)


def _load_attribution_frame(artifacts: pl.DataFrame) -> pl.DataFrame:
    if artifacts.is_empty() or "mercury_attribution_parquet" not in artifacts.columns:
        return pl.DataFrame()
    frames: list[pl.DataFrame] = []
    for record in artifacts.select("run_id", "mercury_attribution_parquet").iter_rows(named=True):
        path_raw = record["mercury_attribution_parquet"]
        if not path_raw:
            continue
        frame = pl.read_parquet(path_raw).with_columns(pl.lit(str(record["run_id"])).alias("run_id"))
        frames.append(frame)
    return _concat_frames(frames)


def load_mercury_study_dataset(study_root: Path) -> StudyDataset:
    if not study_root.exists():
        raise FileNotFoundError(f"Study directory not found: {study_root}")
    runs, artifacts = _load_runs_and_artifacts(study_root)
    if runs.is_empty():
        raise RuntimeError(f"No readable runs found under {study_root}")
    return StudyDataset(
        runs=runs,
        history=_load_history_frame(runs, artifacts),
        states=_load_states_frame(artifacts),
        attribution=_load_attribution_frame(artifacts),
        artifacts=artifacts,
    )
