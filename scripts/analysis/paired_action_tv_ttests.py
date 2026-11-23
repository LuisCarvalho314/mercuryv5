from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")


DEFAULT_METRIC = "latent_action_conditioned_mean_total_variation"
DEFAULT_CANONICAL_METRIC = "action_conditioned.tv_occupancy"
DEFAULT_X_KEYS = ("cumulative_step", "observed_samples", "step", "iteration")
METHOD_ALIASES = {
    "hybrid": "mercury_cscg_hybrid",
    "mercury/hybrid": "mercury_cscg_hybrid",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _method_name(raw: str) -> str:
    value = raw.strip()
    return METHOD_ALIASES.get(value, value)


def _run_dirs(study_root: Path) -> list[Path]:
    return sorted(
        path
        for path in study_root.iterdir()
        if path.is_dir() and path.name != "plots" and (path / "run_config.json").exists()
    )


def _paper_precision_path(run_root: Path, method: str) -> Path | None:
    metrics_dir = run_root / "metrics" / method
    if metrics_dir.exists():
        matches = sorted(metrics_dir.glob("*_paper_precision.json"))
        if matches:
            return matches[-1]
    fallback = sorted(run_root.glob(f"**/metrics/{method}/*_paper_precision.json"))
    return fallback[-1] if fallback else None


def _step_from_snapshot_path(path: Path) -> int:
    stem = path.stem
    if stem.startswith("step="):
        try:
            return int(stem.split("=", 1)[1])
        except ValueError:
            return -1
    return -1


def _canonical_snapshot_path(run_root: Path, method: str, protocol_id: str, snapshot_step: int | None) -> Path | None:
    snapshot_root = run_root / "evaluations" / protocol_id / "snapshots"
    method_root = snapshot_root / method
    roots = [method_root, snapshot_root]
    for root in roots:
        if not root.exists():
            continue
        if snapshot_step is not None:
            candidate = root / f"step={int(snapshot_step):08d}.json"
            if candidate.exists():
                return candidate
            continue
        matches = sorted(root.glob("step=*.json"), key=_step_from_snapshot_path)
        if matches:
            return matches[-1]
    return None


def _nested_value(payload: dict[str, Any], dotted_key: str) -> Any:
    current: Any = payload
    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def canonical_metric_observations(
    run_root: Path,
    *,
    method: str,
    protocol_id: str,
    metric: str,
    snapshot_step: int | None,
) -> list[dict[str, Any]]:
    path = _canonical_snapshot_path(run_root, method, protocol_id, snapshot_step)
    if path is None:
        return []
    payload = _load_json(path)
    value = _numeric(_nested_value(payload, metric))
    if value is None:
        return []
    protocol = payload.get("protocol") if isinstance(payload.get("protocol"), dict) else {}
    return [
        {
            "value": value,
            "canonical_snapshot": str(path),
            "protocol_id": payload.get("protocol_id", protocol_id),
            "protocol_number_of_walks": protocol.get("number_of_walks"),
            "protocol_walk_length": protocol.get("walk_length"),
            "walk_cache_id": payload.get("walk_cache_id"),
            "snapshot_step": payload.get("snapshot_step", _step_from_snapshot_path(path)),
            "observed_samples": payload.get("observed_samples"),
            "ground_truth_checksum": (payload.get("ground_truth") or {}).get("transition_count_checksum")
            if isinstance(payload.get("ground_truth"), dict)
            else None,
        }
    ]


def _flatten(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten(next_prefix, child, output)
        return
    if isinstance(value, (str, int, float, bool)) or value is None:
        output[prefix] = value


def _run_metadata(run_root: Path) -> dict[str, Any]:
    payload = _load_json(run_root / "run_config.json")
    metadata: dict[str, Any] = {"run_id": payload.get("run_id", run_root.name), "run_root": str(run_root)}
    for section in ("run_payload", "cli_args"):
        section_payload = payload.get(section)
        if isinstance(section_payload, dict):
            flattened: dict[str, Any] = {}
            _flatten("", section_payload, flattened)
            metadata.update(flattened)
    return metadata


def _numeric(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _parse_protocol_ids_by_rand(raw: str | None) -> dict[float, str]:
    if not raw:
        return {}
    mapping: dict[float, str] = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"Expected RAND=PROTOCOL_ID item, got {item!r}")
        rand_raw, protocol_id = item.split("=", 1)
        mapping[float(rand_raw)] = protocol_id.strip()
    return mapping


def _protocol_id_for_row(row: dict[str, Any], protocol_id_by_rand: dict[float, str]) -> str | None:
    if len(protocol_id_by_rand) == 1:
        return next(iter(protocol_id_by_rand.values()))
    rand_prob = _numeric(row.get("rand_prob"))
    if rand_prob is None:
        return None
    for key, protocol_id in protocol_id_by_rand.items():
        if math.isclose(float(key), rand_prob, rel_tol=0.0, abs_tol=1e-12):
            return protocol_id
    return None


def _history_metric(payload: dict[str, Any], metric: str, phase: str | None) -> float | None:
    history = payload.get("history")
    if not isinstance(history, list):
        return None
    candidates: list[dict[str, Any]] = []
    for row in history:
        if not isinstance(row, dict) or metric not in row:
            continue
        if phase is not None and row.get("phase") != phase:
            continue
        if _numeric(row.get(metric)) is not None:
            candidates.append(row)
    if not candidates:
        return None

    def sort_key(row: dict[str, Any]) -> tuple[float, ...]:
        values = tuple(_numeric(row.get(key)) or -math.inf for key in DEFAULT_X_KEYS)
        return values

    return _numeric(max(candidates, key=sort_key).get(metric))


def _history_metric_rows(
    payload: dict[str, Any],
    metric: str,
    phase: str | None,
    checkpoint_key: str,
) -> list[dict[str, Any]]:
    history = payload.get("history")
    if not isinstance(history, list):
        return []
    rows: list[dict[str, Any]] = []
    for row in history:
        if not isinstance(row, dict) or metric not in row:
            continue
        if phase is not None and row.get("phase") != phase:
            continue
        value = _numeric(row.get(metric))
        if value is None or checkpoint_key not in row:
            continue
        checkpoint = row.get(checkpoint_key)
        output = {
            "value": value,
            "checkpoint": checkpoint,
            "checkpoint_key": checkpoint_key,
        }
        for key, raw in row.items():
            if isinstance(raw, (str, int, float, bool)) or raw is None:
                output[f"history_{key}"] = raw
        rows.append(output)
    return rows


def metric_value(path: Path, metric: str, source: str, phase: str | None) -> float | None:
    payload = _load_json(path)
    if source == "history":
        return _history_metric(payload, metric, phase)
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        value = _numeric(metrics.get(metric))
        if value is not None:
            return value
    return _history_metric(payload, metric, phase)


def metric_observations(
    path: Path,
    metric: str,
    source: str,
    phase: str | None,
    checkpoint_key: str | None,
) -> list[dict[str, Any]]:
    payload = _load_json(path)
    protocol = payload.get("protocol") if isinstance(payload.get("protocol"), dict) else {}
    protocol_context = {
        "protocol_num_walks": protocol.get("num_walks"),
        "protocol_walk_length": protocol.get("walk_length"),
    }
    if checkpoint_key is not None:
        rows = _history_metric_rows(payload, metric, phase, checkpoint_key)
        return [{**protocol_context, **row} for row in rows]
    value = metric_value(path, metric, source, phase)
    if value is None:
        return []
    return [{**protocol_context, "value": value}]


def collect_pairs(
    study_root: Path,
    *,
    method_a: str,
    method_b: str,
    metric: str,
    source: str,
    phase_a: str | None,
    phase_b: str | None,
    checkpoint_key: str | None,
    protocol_id_by_rand: dict[float, str],
    snapshot_step: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_root in _run_dirs(study_root):
        metadata = _run_metadata(run_root)
        if source == "canonical":
            protocol_id = _protocol_id_for_row(metadata, protocol_id_by_rand)
            if protocol_id is None:
                continue
            observations_a = canonical_metric_observations(
                run_root, method=method_a, protocol_id=protocol_id, metric=metric, snapshot_step=snapshot_step
            )
            observations_b = canonical_metric_observations(
                run_root, method=method_b, protocol_id=protocol_id, metric=metric, snapshot_step=snapshot_step
            )
        else:
            path_a = _paper_precision_path(run_root, method_a)
            path_b = _paper_precision_path(run_root, method_b)
            if path_a is None or path_b is None:
                continue
            observations_a = metric_observations(path_a, metric, source, phase_a, checkpoint_key)
            observations_b = metric_observations(path_b, metric, source, phase_b, checkpoint_key)
        if checkpoint_key is not None:
            by_checkpoint_b = {row["checkpoint"]: row for row in observations_b}
            observation_pairs = [(row_a, by_checkpoint_b[row_a["checkpoint"]]) for row_a in observations_a if row_a["checkpoint"] in by_checkpoint_b]
        else:
            observation_pairs = [(observations_a[0], observations_b[0])] if observations_a and observations_b else []
        for obs_a, obs_b in observation_pairs:
            rows.append(
                {
                    **metadata,
                    **{f"a_{key}": value for key, value in obs_a.items() if key != "value"},
                    **{f"b_{key}": value for key, value in obs_b.items() if key != "value"},
                    "checkpoint": obs_a.get("checkpoint"),
                    "method_a": method_a,
                    "method_b": method_b,
                    "metric": metric,
                    "value_a": obs_a["value"],
                    "value_b": obs_b["value"],
                    "diff": obs_a["value"] - obs_b["value"],
                    "paper_precision_a": str(path_a),
                    "paper_precision_b": str(path_b),
                }
            )
    return rows


def collect_method_observations(
    study_root: Path,
    *,
    methods: set[str],
    metric: str,
    source: str,
    phases: dict[str, str | None],
    checkpoint_key: str | None,
    protocol_id_by_rand: dict[float, str],
    snapshot_step: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_root in _run_dirs(study_root):
        metadata = _run_metadata(run_root)
        method = _method_name(str(metadata.get("method") or ""))
        if method not in methods:
            continue
        if source == "canonical":
            protocol_id = _protocol_id_for_row(metadata, protocol_id_by_rand)
            if protocol_id is None:
                continue
            observations = canonical_metric_observations(
                run_root, method=method, protocol_id=protocol_id, metric=metric, snapshot_step=snapshot_step
            )
            artifact_path = None
        else:
            path = _paper_precision_path(run_root, method)
            if path is None:
                continue
            observations = metric_observations(path, metric, source, phases.get(method), checkpoint_key)
            artifact_path = str(path)
        for observation in observations:
            rows.append(
                {
                    **metadata,
                    **{key: value for key, value in observation.items() if key != "value"},
                    "method": method,
                    "metric": metric,
                    "value": observation["value"],
                    "paper_precision": artifact_path,
                }
            )
    return rows


def collect_pairs_by_metadata(
    study_root: Path,
    *,
    method_a: str,
    method_b: str,
    metric: str,
    source: str,
    phase_a: str | None,
    phase_b: str | None,
    pair_by: list[str],
    allow_many_pairs: bool,
    checkpoint_key: str | None,
    protocol_id_by_rand: dict[float, str],
    snapshot_step: int | None,
) -> list[dict[str, Any]]:
    observations = collect_method_observations(
        study_root,
        methods={method_a, method_b},
        metric=metric,
        source=source,
        phases={method_a: phase_a, method_b: phase_b},
        checkpoint_key=checkpoint_key,
        protocol_id_by_rand=protocol_id_by_rand,
        snapshot_step=snapshot_step,
    )
    effective_pair_by = list(pair_by)
    if checkpoint_key is not None and "checkpoint" not in effective_pair_by:
        effective_pair_by.append("checkpoint")
    by_method_and_key: dict[tuple[str, tuple[Any, ...]], list[dict[str, Any]]] = {}
    for row in observations:
        by_method_and_key.setdefault((str(row["method"]), _group_key(row, effective_pair_by)), []).append(row)

    duplicate_keys = [(method, key, len(rows)) for (method, key), rows in by_method_and_key.items() if len(rows) > 1]
    if duplicate_keys and not allow_many_pairs:
        examples = ", ".join(
            f"{method} {dict(zip(pair_by, key, strict=True))} n={count}"
            for method, key, count in duplicate_keys[:5]
        )
        raise RuntimeError(
            "Pairing columns are not unique within a method. "
            f"Add more --pair-by columns. Examples: {examples}"
        )

    keys_a = {key for method, key in by_method_and_key if method == method_a}
    keys_b = {key for method, key in by_method_and_key if method == method_b}
    rows: list[dict[str, Any]] = []
    for key in sorted(keys_a & keys_b, key=lambda item: tuple(str(value) for value in item)):
        for row_a in by_method_and_key[(method_a, key)]:
            for row_b in by_method_and_key[(method_b, key)]:
                metadata = {column: value for column, value in zip(effective_pair_by, key, strict=True)}
                method_metadata = {
                    **{f"a_{column}": value for column, value in row_a.items() if isinstance(value, (str, int, float, bool)) or value is None},
                    **{f"b_{column}": value for column, value in row_b.items() if isinstance(value, (str, int, float, bool)) or value is None},
                }
                rows.append(
                    {
                        **method_metadata,
                        **metadata,
                        "run_id_a": row_a.get("run_id"),
                        "run_id_b": row_b.get("run_id"),
                        "run_root_a": row_a.get("run_root"),
                        "run_root_b": row_b.get("run_root"),
                        "method_a": method_a,
                        "method_b": method_b,
                        "metric": metric,
                        "value_a": row_a["value"],
                        "value_b": row_b["value"],
                        "diff": row_a["value"] - row_b["value"],
                        "paper_precision_a": row_a.get("paper_precision"),
                        "paper_precision_b": row_b.get("paper_precision"),
                        "canonical_snapshot_a": row_a.get("canonical_snapshot"),
                        "canonical_snapshot_b": row_b.get("canonical_snapshot"),
                    }
                )
    return rows


def _group_key(row: dict[str, Any], columns: Iterable[str]) -> tuple[Any, ...]:
    return tuple(row.get(column) for column in columns)


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else math.nan


def _std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if values.size > 1 else math.nan


def _paired_ttest(values_a: np.ndarray, values_b: np.ndarray, *, alternative: str, confidence: float) -> dict[str, Any]:
    diff = values_a - values_b
    n = int(diff.size)
    mean_diff = _mean(diff)
    sd_diff = _std(diff)
    sem = float(sd_diff / math.sqrt(n)) if n > 1 and math.isfinite(sd_diff) else math.nan
    if n > 1 and sem > 0:
        test = stats.ttest_rel(values_a, values_b, alternative=alternative)
        t_stat = float(test.statistic)
        p_value = float(test.pvalue)
        alpha = 1.0 - confidence
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
        ci_low = mean_diff - t_crit * sem
        ci_high = mean_diff + t_crit * sem
        cohen_dz = mean_diff / sd_diff
    elif n > 1 and np.allclose(diff, diff[0]):
        t_stat = math.inf if mean_diff > 0 else (-math.inf if mean_diff < 0 else math.nan)
        p_value = 0.0 if mean_diff != 0 else 1.0
        ci_low = mean_diff
        ci_high = mean_diff
        cohen_dz = math.inf if mean_diff != 0 else math.nan
    else:
        t_stat = math.nan
        p_value = math.nan
        ci_low = math.nan
        ci_high = math.nan
        cohen_dz = math.nan
    return {
        "n": n,
        "mean_a": _mean(values_a),
        "sd_a": _std(values_a),
        "mean_b": _mean(values_b),
        "sd_b": _std(values_b),
        "mean_diff": mean_diff,
        "sd_diff": sd_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "t": t_stat,
        "df": n - 1 if n > 0 else math.nan,
        "p": p_value,
        "cohen_dz": cohen_dz,
    }


def summarize_pairs(
    pairs: list[dict[str, Any]],
    *,
    group_by: list[str],
    alternative: str,
    confidence: float,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in pairs:
        groups.setdefault(_group_key(row, group_by), []).append(row)

    summaries: list[dict[str, Any]] = []
    for key, group_rows in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        values_a = np.asarray([row["value_a"] for row in group_rows], dtype=np.float64)
        values_b = np.asarray([row["value_b"] for row in group_rows], dtype=np.float64)
        summary = _paired_ttest(values_a, values_b, alternative=alternative, confidence=confidence)
        summary.update({column: value for column, value in zip(group_by, key, strict=True)})
        summaries.append(summary)
    return summaries


def _format(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.6g}"
    return str(value)


def print_markdown(
    summaries: list[dict[str, Any]],
    *,
    group_by: list[str],
    method_a: str,
    method_b: str,
    confidence: float,
) -> None:
    ci_label = f"{confidence * 100:g}%_ci"
    headers = [
        *group_by,
        "n",
        f"{method_a}_mean",
        f"{method_b}_mean",
        "mean_diff",
        ci_label,
        "t",
        "df",
        "p",
        "cohen_dz",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in summaries:
        values = [row.get(column) for column in group_by]
        values.extend(
            [
                row["n"],
                row["mean_a"],
                row["mean_b"],
                row["mean_diff"],
                f"[{_format(row['ci_low'])}, {_format(row['ci_high'])}]",
                row["t"],
                row["df"],
                row["p"],
                row["cohen_dz"],
            ]
        )
        print("| " + " | ".join(_format(value) for value in values) + " |")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run paired t-tests on action-conditioned total variation between two methods in a study."
    )
    parser.add_argument("--study-root", required=True, type=Path, help="Study directory containing run subdirectories.")
    parser.add_argument("--method-a", default="mercury_cscg_hybrid", help="First method. Aliases: hybrid, mercury/hybrid.")
    parser.add_argument("--method-b", default="cscg", help="Second method.")
    parser.add_argument("--metric", default=DEFAULT_METRIC, help="Metric key to compare.")
    parser.add_argument(
        "--source",
        choices=("metrics", "history", "canonical"),
        default="metrics",
        help="Use top-level final paper-precision metrics, history rows, or canonical evaluation snapshots.",
    )
    parser.add_argument(
        "--protocol-id",
        default=None,
        help="Canonical evaluation protocol id to use with --source canonical when all rows share one protocol.",
    )
    parser.add_argument(
        "--protocol-ids-by-rand",
        default=None,
        help="Comma-separated RAND=PROTOCOL_ID mapping for --source canonical, e.g. 0.0=canonical-v4-...,0.3=canonical-v4-...",
    )
    parser.add_argument(
        "--snapshot-step",
        type=int,
        default=None,
        help="Canonical snapshot step to read. Defaults to latest step available for each run.",
    )
    parser.add_argument("--phase-a", default=None, help="Optional history phase filter for method A.")
    parser.add_argument("--phase-b", default=None, help="Optional history phase filter for method B.")
    parser.add_argument(
        "--group-by",
        nargs="*",
        default=[],
        help="Run one paired t-test per metadata group, e.g. --group-by level rand_prob.",
    )
    parser.add_argument(
        "--pair-by",
        nargs="*",
        default=None,
        help=(
            "Pair separate method runs by metadata columns. "
            "Omit this when both method outputs live in each run directory."
        ),
    )
    parser.add_argument(
        "--allow-many-pairs",
        action="store_true",
        help=(
            "Allow repeated runs per --pair-by key and emit all method-A/method-B combinations. "
            "Use prefixed group columns such as b_cscg_clone_allocation to split variants."
        ),
    )
    parser.add_argument(
        "--alternative",
        choices=("two-sided", "less", "greater"),
        default="two-sided",
        help="Alternative hypothesis for scipy.stats.ttest_rel on method_a - method_b.",
    )
    parser.add_argument("--confidence", type=float, default=0.95, help="Two-sided confidence level for mean difference CI.")
    parser.add_argument("--pairs-output", type=Path, default=None, help="Optional CSV path for paired observations.")
    parser.add_argument("--summary-output", type=Path, default=None, help="Optional CSV path for t-test summaries.")
    args = parser.parse_args()

    method_a = _method_name(args.method_a)
    method_b = _method_name(args.method_b)
    metric = str(args.metric)
    protocol_id_by_rand = _parse_protocol_ids_by_rand(args.protocol_ids_by_rand)
    if args.protocol_id:
        protocol_id_by_rand = {0.0: str(args.protocol_id)}
    if args.source == "canonical":
        if metric == DEFAULT_METRIC:
            metric = DEFAULT_CANONICAL_METRIC
        if not protocol_id_by_rand:
            raise RuntimeError("--source canonical requires --protocol-id or --protocol-ids-by-rand")
    if args.pair_by is None:
        pairs = collect_pairs(
            args.study_root,
            method_a=method_a,
            method_b=method_b,
            metric=metric,
            source=args.source,
            phase_a=args.phase_a,
            phase_b=args.phase_b,
            checkpoint_key=None,
            protocol_id_by_rand=protocol_id_by_rand,
            snapshot_step=args.snapshot_step,
        )
    else:
        pairs = collect_pairs_by_metadata(
            args.study_root,
            method_a=method_a,
            method_b=method_b,
            metric=metric,
            source=args.source,
            phase_a=args.phase_a,
            phase_b=args.phase_b,
            pair_by=list(args.pair_by),
            allow_many_pairs=bool(args.allow_many_pairs),
            checkpoint_key=None,
            protocol_id_by_rand=protocol_id_by_rand,
            snapshot_step=args.snapshot_step,
        )
    if not pairs:
        raise RuntimeError(f"No paired {method_a}/{method_b} observations found under {args.study_root}")

    missing_group_columns = [column for column in args.group_by if all(column not in row for row in pairs)]
    if missing_group_columns:
        raise RuntimeError(f"Group columns not found in run metadata: {', '.join(missing_group_columns)}")

    summaries = summarize_pairs(
        pairs,
        group_by=list(args.group_by),
        alternative=args.alternative,
        confidence=float(args.confidence),
    )
    print_markdown(
        summaries,
        group_by=list(args.group_by),
        method_a=method_a,
        method_b=method_b,
        confidence=float(args.confidence),
    )
    print(f"\npaired observations: {len(pairs)}")
    print(f"difference: {method_a} - {method_b}; metric: {metric}")

    if args.pairs_output is not None:
        write_csv(args.pairs_output, pairs)
        print(f"wrote {args.pairs_output}")
    if args.summary_output is not None:
        write_csv(args.summary_output, summaries)
        print(f"wrote {args.summary_output}")


if __name__ == "__main__":
    main()
