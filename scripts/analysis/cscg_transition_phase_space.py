#!/usr/bin/env python3
"""Project saved CSCG evaluation-v5 transitions into an environment-wise PCA space.

This reads frozen evaluation records and saved model snapshots. It does not run
inference or training, and it does not redefine the paper's transition metric.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from scripts.analysis.evaluate_v5 import EvaluationTask, _initial_snapshot, _trained_snapshot
from mercury_runs.infrastructure.paper_precision import (
    _action_tv_by_pair,
    complete_and_normalize_projected_rows,
    compute_occupancy_weighted_action_tv,
    compute_action_projection_weights,
    condition_non_selfloop_action_distributions,
    normalize_action_rows,
    project_transition_tensor_canonical,
)


PLOT_ROOT = ROOT / "results/current/action_alignment_rerun/notebook_plots/main_study_eval_v5"
MANIFEST = PLOT_ROOT / "evaluation_v5_manifest.json"
SOURCE_ROOT = ROOT / "results/current/action_alignment_rerun/studies/main_study"
STEP_BY_PASS = {0: 0, 1: 2_000, 2: 4_000, 1000: 2_000_000}
ENVIRONMENTS = {13: "3x3", 21: "1x4", 20: "1x5", 22: "3x4"}
CONDITIONS = {
    ("cscg", "oracle_capacity"): ("Oracle-C", "#176f8a", "-"),
    ("cscg", "oracle_capacity_first_order"): ("Oracle-C+1st", "#d18a13", "--"),
    ("cscg", "oracle_capacity_structural"): ("Oracle-C+Struct", "#b44856", ":"),
    ("cscg", "vanilla_cscg"): ("Standard CSCG", "#637441", "-."),
    ("mercury_cscg_hybrid", "source_graph_clone_state"): ("MERCURY-initialised CSCG", "#8354a2", (0, (5, 1, 1, 1))),
    ("tmgwr_cscg_hybrid", "source_graph_first_order"): ("TMGWR-initialised CSCG", "#735d52", (0, (3, 1, 1, 1, 1, 1))),
}
CONDITION_ORDER = [value[0] for value in CONDITIONS.values()]
ORACLE_ORDER = CONDITION_ORDER[:3]
MARKERS = {0: "o", 1: "s", 2: "^", 1000: "D"}


def _checksum(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()[:16]


def _source_run(run_id: str) -> Path:
    return SOURCE_ROOT / run_id


def _raw_transition(run_id: str, config: dict[str, Any], step: int) -> np.ndarray:
    task = EvaluationTask(
        study_name="main_study",
        source_run=_source_run(run_id),
        output_run=PLOT_ROOT,
        config=config,
        checkpoints=(step,),
        reference_step=4_000,
    )
    snapshot, _ = _initial_snapshot(task) if step == 0 else _trained_snapshot(task, step)
    return np.asarray(snapshot["T"])


def _project_with(
    transition: np.ndarray,
    alignment: np.ndarray,
    source_cooccurrence: np.ndarray,
    valid_rows: np.ndarray,
    n_true: int,
) -> np.ndarray:
    weights = compute_action_projection_weights(source_cooccurrence, alignment, n_true)
    projected = project_transition_tensor_canonical(
        T_hat_norm=normalize_action_rows(transition),
        alignment=alignment,
        omega=weights,
        n_true=n_true,
    )
    return complete_and_normalize_projected_rows(projected, valid_rows)


def _read_records(manifest_path: Path, level: int) -> tuple[pd.DataFrame, list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    study_root = ROOT / manifest["source_shadow_study"]
    protocol_id = manifest["protocol_id"]
    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    expected_gt: np.ndarray | None = None
    expected_gt_nonself: np.ndarray | None = None
    expected_move_rows: np.ndarray | None = None
    expected_valid_rows: np.ndarray | None = None
    expected_cache: str | None = None
    max_stored_error = 0.0
    max_tv_error = 0.0
    max_nonself_error = 0.0
    missing_movement_mass_rows = 0
    source_phase_records_checked = 0
    for config_path in sorted(study_root.glob("*/run_config.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))["cli_args"]
        if int(config.get("level", -1)) != level:
            continue
        key = (str(config["method"]), str(config["cscg_structural_preset"]))
        if key not in CONDITIONS:
            continue
        run_id = config_path.parent.name
        condition = CONDITIONS[key][0]
        hybrid = key[0] in {"mercury_cscg_hybrid", "tmgwr_cscg_hybrid"}
        matched_hybrid_budget = (
            hybrid
            and int(config["cscg_n_iter"]) < 2_000_000
            and int(config["num_steps"]) + int(config["cscg_n_iter"]) == 2_000_000
        )
        source_record_path: Path | None = None
        if hybrid:
            if int(config["num_steps"]) != 2_000 or not (
                int(config["cscg_n_iter"]) == 2_000_000 or matched_hybrid_budget
            ):
                raise ValueError(f"Unexpected hybrid stage lengths: {run_id}")
            source_namespace = key[0] + ("_mercury" if key[0] == "mercury_cscg_hybrid" else "_tmgwr")
            source_record_path = config_path.parent / "evaluations" / protocol_id / "snapshots" / source_namespace / "step=00002000.json"
            if not source_record_path.is_file():
                raise FileNotFoundError(f"Missing hybrid source-phase record: {source_record_path}")
            source_record = json.loads(source_record_path.read_text(encoding="utf-8"))
            if (int(source_record["snapshot_step"]) != 2_000 or int(source_record["observed_samples"]) != 2_000
                    or source_record["protocol_id"] != protocol_id):
                raise ValueError(f"Incompatible hybrid source-phase record: {source_record_path}")
            source_phase_records_checked += 1
        for pass_number, step in STEP_BY_PASS.items():
            record_path = config_path.parent / "evaluations" / protocol_id / "snapshots" / key[0] / f"step={step:08d}.json"
            if not record_path.is_file():
                raise FileNotFoundError(f"Missing {ENVIRONMENTS[level]} evaluation record: {record_path}")
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record["protocol_id"] != protocol_id or int(record["snapshot_step"]) != step or int(record["level"]) != level:
                raise ValueError(f"Incompatible evaluation record: {record_path}")
            expected_phase_samples = (
                int(config["cscg_n_iter"])
                if matched_hybrid_budget and step == 2_000_000
                else step
            )
            if int(record["observed_samples"]) != expected_phase_samples:
                raise ValueError(f"CSCG stage sample count differs from checkpoint: {record_path}")
            if matched_hybrid_budget and step == 2_000_000:
                budget = record.get("training_budget") or {}
                if (
                    int(budget.get("source_samples", -1)) != int(config["num_steps"])
                    or int(budget.get("cscg_phase_samples", -1)) != expected_phase_samples
                    or int(budget.get("cumulative_samples", -1)) != step
                ):
                    raise ValueError(f"Hybrid final budget provenance differs: {record_path}")
            if hybrid and step == 0:
                initialization = record["initialization"]
                expected_source = "mercury" if key[0] == "mercury_cscg_hybrid" else "sensory"
                graph_path = str(initialization.get("source_graph_path", ""))
                if (f"/{expected_source}/paper_precision/step=2000/" not in graph_path
                        or int(initialization.get("reference_snapshot_step", -1)) != 4_000
                        or initialization.get("pseudocount_policy") != "applied_once_to_reconstructed_counts"):
                    raise ValueError(f"Hybrid CSCG handoff provenance differs: {record_path}")
            if expected_cache is None:
                expected_cache = record["walk_cache_id"]
            elif record["walk_cache_id"] != expected_cache:
                raise ValueError(f"Walk cache differs: {record_path}")
            action = record["action_conditioned"]
            q = np.asarray(action["P_hat_full"], dtype=np.float64)
            gt = np.asarray(action["P_true_full"], dtype=np.float64)
            n_true = int(record["state_representation"]["n_true_states"])
            n_actions = int(record["state_representation"]["n_actions"])
            if q.shape != (n_actions, n_true, n_true + 1) or gt.shape != q.shape:
                raise ValueError(f"Unexpected {ENVIRONMENTS[level]} tensor shape {q.shape}: {record_path}")
            if expected_gt is None:
                expected_gt = gt
            elif not np.array_equal(gt, expected_gt):
                raise ValueError(f"GT tensor differs: {record_path}")
            weights = np.asarray(action["evaluation_occupancy_weights"], dtype=np.float64)
            valid_rows = np.asarray(action["valid_rows"], dtype=bool)
            counts = np.asarray(action["source_action_counts"], dtype=np.float64)
            if expected_valid_rows is None:
                expected_valid_rows = valid_rows
            elif not np.array_equal(valid_rows, expected_valid_rows):
                raise ValueError(f"GT valid-row mask differs: {record_path}")
            q_nonself, gt_nonself, move_rows = condition_non_selfloop_action_distributions(q, gt, valid_rows)
            if expected_move_rows is None:
                expected_move_rows = move_rows
                expected_gt_nonself = gt_nonself
            elif not np.array_equal(move_rows, expected_move_rows) or not np.array_equal(gt_nonself, expected_gt_nonself):
                raise ValueError(f"GT movement representation differs: {record_path}")
            q_without_self = q.copy()
            q_without_self[:, np.arange(n_true), np.arange(n_true)] = 0.0
            no_predicted_mass = move_rows & (q_without_self.sum(axis=-1) <= 0.0)
            if np.any(q_nonself[no_predicted_mass, -1] != 1.0):
                raise ValueError(f"Missing predicted movement mass was not sent to sink: {record_path}")
            if np.any(q_nonself[:, np.arange(n_true), np.arange(n_true)] != 0.0):
                raise ValueError(f"No-self representation retained diagonal mass: {record_path}")
            missing_movement_mass_rows += int(no_predicted_mass.sum())
            pair_tv = _action_tv_by_pair(q, gt, valid_rows)
            distance = float(action["tv_occupancy"])
            similarity = float(action["tv_similarity_occupancy"])
            recomputed_distance, _, recomputed_weights, _, _ = compute_occupancy_weighted_action_tv(pair_tv, valid_rows, counts)
            pair_nonself = _action_tv_by_pair(q_nonself, gt_nonself, move_rows)
            nonself_distance, _, _, _, _ = compute_occupancy_weighted_action_tv(pair_nonself, move_rows, counts)
            nonself_similarity = float(action["tv_non_self_loop_similarity_occupancy"])
            max_stored_error = max(max_stored_error, abs((1.0 - distance) - similarity))
            max_tv_error = max(max_tv_error, abs(recomputed_distance - distance),
                               float(np.max(np.abs(recomputed_weights - weights))),
                               float(np.max(np.abs(pair_tv - np.asarray(action["tv_by_pair_full"])))) )
            max_nonself_error = max(max_nonself_error, abs((1.0 - nonself_distance) - nonself_similarity),
                                    abs(nonself_distance - float(action["tv_non_self_loop_occupancy"])),
                                    float(np.max(np.abs(pair_nonself - np.asarray(action["tv_by_pair_non_self_loop"])))) )
            if (not np.isfinite(q).all() or not np.isfinite(q_nonself).all() or not np.isfinite(distance)
                    or not np.isfinite(float((record.get("model_fit") or {}).get("bits_per_symbol", np.nan)))):
                raise ValueError(f"Non-finite projected transition or TV: {record_path}")
            row = {
                "environment": ENVIRONMENTS[level], "level": level, "condition": condition,
                "method": key[0], "preset": key[1], "seed": int(config["seed"]),
                "run_id": run_id, "checkpoint": pass_number, "snapshot_step": step,
                "observed_samples": int(record["observed_samples"]),
                "training_phase": "cscg", "source_stage_observed_samples": 2_000 if hybrid else 0,
                "cscg_stage_observed_samples": step,
                "source_phase_record_path": str(source_record_path.relative_to(ROOT)) if source_record_path else None,
                "s_a_tv": similarity, "d_gt": distance,
                "s_a_tv_nonself": nonself_similarity, "d_gt_nonself": nonself_distance,
                "nll_bits_per_symbol": (record.get("model_fit") or {}).get("bits_per_symbol"),
                "record_path": str(record_path.relative_to(ROOT)),
            }
            rows.append(row)
            samples.append({"row": row, "q": q, "q_nonself": q_nonself,
                            "alignment": np.asarray(record["alignment"]["learned_to_true"], dtype=np.int64),
                            "H": np.asarray(record["alignment"]["action_source_cooccurrence"], dtype=np.int64),
                            "valid_rows": valid_rows,
                            "config": config, "transition_checksum": record["inference"]["transition_checksum"]})
    frame = pd.DataFrame(rows)
    if len(frame) != 6 * 30 * 4 or frame.duplicated(["condition", "seed", "checkpoint"]).any():
        raise ValueError(f"Expected 720 unique seed/checkpoint records; found {len(frame)}")
    seed_sets = [set(frame.loc[frame.condition == condition, "seed"]) for condition in CONDITION_ORDER]
    if any(len(seeds) != 30 or seeds != seed_sets[0] for seeds in seed_sets):
        raise ValueError("Conditions do not share the same 30 seeds")
    if any(set(group.checkpoint) != set(STEP_BY_PASS) for _, group in frame.groupby(["condition", "seed"])):
        raise ValueError("Incomplete seed trajectory")
    if int(expected_move_rows.sum()) != {13: 16, 21: 6, 20: 8, 22: 34}[level]:
        raise ValueError("Unexpected number of GT movement rows")
    if source_phase_records_checked != 60:
        raise ValueError(f"Expected 60 hybrid source-phase records; found {source_phase_records_checked}")
    if max_stored_error > 1e-14 or max_tv_error > 1e-14 or max_nonself_error > 1e-14:
        raise ValueError(f"Stored TV parity failed: full={max_tv_error}, no-self={max_nonself_error}")
    assert expected_gt is not None
    return frame, samples, expected_gt, {"protocol_id": protocol_id, "walk_cache_id": expected_cache,
                                         "hybrid_source_phase_records_checked": source_phase_records_checked,
                                         "max_similarity_identity_error": max_stored_error,
                                         "max_weighted_tv_error": max_tv_error,
                                         "max_nonself_tv_error": max_nonself_error,
                                         "movement_rows": int(expected_move_rows.sum()),
                                         "predicted_sink_only_movement_rows": missing_movement_mass_rows,
                                         "move_mask": expected_move_rows,
                                         "gt_nonself": expected_gt_nonself}


def _check_summary_values(frame: pd.DataFrame) -> dict[str, Any]:
    path = PLOT_ROOT / "pass_2_to_1000_statistical_analysis/pass_2_to_1000_paired_seed_values.csv"
    table = pd.read_csv(path)
    metrics = {"action_tv_similarity": "s_a_tv",
               "action_no_self_loop_tv_similarity": "s_a_tv_nonself",
               "eval_nll_bps": "nll_bits_per_symbol"}
    selected = table[(table["level"] == int(frame["level"].iloc[0])) & (table["metric"].isin(metrics))]
    lookup = frame.set_index(["run_id", "checkpoint"])
    differences = []
    for record in selected.itertuples(index=False):
        for checkpoint, column in [(2, "value_pass_2"), (1000, "value_pass_1000")]:
            differences.append(abs(float(lookup.loc[(record.run_id, checkpoint), metrics[record.metric]]) - float(getattr(record, column))))
    if len(differences) != 1080 or max(differences) > 1e-12:
        raise ValueError(f"Published summary parity failed: {len(differences)} values, max error={max(differences, default=float('nan'))}")
    return {"summary_values_checked": len(differences), "max_summary_error": max(differences)}


def _alignment_diagnostics(
    frame: pd.DataFrame, samples: list[dict[str, Any]], pca: PCA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    environment = str(frame["environment"].iloc[0])
    by_key = {(item["row"]["run_id"], item["row"]["checkpoint"]): item for item in samples}
    summaries: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    for (run_id, condition, seed), group in frame.groupby(["run_id", "condition", "seed"], sort=False):
        checkpoints = sorted(group["checkpoint"].tolist(), key=lambda value: list(STEP_BY_PASS).index(value))
        for prior_checkpoint, current_checkpoint in zip(checkpoints[:-1], checkpoints[1:]):
            prior = by_key[(run_id, prior_checkpoint)]
            current = by_key[(run_id, current_checkpoint)]
            old_map, new_map = prior["alignment"], current["alignment"]
            if old_map.shape != new_map.shape or not np.array_equal(prior["valid_rows"], current["valid_rows"]):
                raise ValueError(f"Incomparable learned states or valid GT rows: {run_id}")
            changed = old_map != new_map
            for state_id in range(len(old_map)):
                states.append({"environment": environment, "condition": condition, "seed": seed, "run_id": run_id,
                               "from_checkpoint": prior_checkpoint, "to_checkpoint": current_checkpoint,
                               "learned_state": state_id, "from_gt_state": int(old_map[state_id]),
                               "to_gt_state": int(new_map[state_id]), "changed": bool(changed[state_id])})
            old_t = _raw_transition(run_id, prior["config"], STEP_BY_PASS[prior_checkpoint])
            new_t = _raw_transition(run_id, current["config"], STEP_BY_PASS[current_checkpoint])
            if _checksum(old_t) != prior["transition_checksum"] or _checksum(new_t) != current["transition_checksum"]:
                raise ValueError(f"Raw transition checksum differs from evaluated model: {run_id}")
            n_true = prior["q"].shape[1]
            fixed = _project_with(new_t, old_map, prior["H"], prior["valid_rows"], n_true)
            new_map_old_h = _project_with(new_t, new_map, prior["H"], prior["valid_rows"], n_true)
            reprojected_old = _project_with(old_t, old_map, prior["H"], prior["valid_rows"], n_true)
            reprojected_new = _project_with(new_t, new_map, current["H"], current["valid_rows"], n_true)
            if not np.allclose(reprojected_old, prior["q"], atol=1e-12) or not np.allclose(reprojected_new, current["q"], atol=1e-12):
                raise ValueError(f"Canonical projection does not reproduce stored tensors: {run_id}")
            components = np.stack([
                (fixed - prior["q"]).ravel(),
                (new_map_old_h - fixed).ravel(),
                (current["q"] - new_map_old_h).ravel(),
            ])
            total = (current["q"] - prior["q"]).ravel()
            if not np.allclose(components.sum(axis=0), total, atol=1e-12):
                raise ValueError(f"Projection diagnostic fails to reconstruct displacement: {run_id}")
            delta_pc = components @ pca.components_[:2].T
            total_pc = total @ pca.components_[:2].T
            summaries.append({
                "environment": environment, "condition": condition, "seed": seed, "run_id": run_id,
                "from_checkpoint": prior_checkpoint, "to_checkpoint": current_checkpoint,
                "n_learned_states": len(old_map), "n_changed": int(changed.sum()),
                "fraction_changed": float(changed.mean()),
                "n_became_unaligned": int(((old_map >= 0) & (new_map < 0)).sum()),
                "n_became_aligned": int(((old_map < 0) & (new_map >= 0)).sum()),
                "n_changed_gt_assignment": int(((old_map >= 0) & (new_map >= 0) & changed).sum()),
                "changed_state_ids": json.dumps(np.flatnonzero(changed).tolist()),
                "pca_displacement": float(np.linalg.norm(total_pc)),
                "transition_only_pc_displacement": float(np.linalg.norm(delta_pc[0])),
                "alignment_pc_displacement": float(np.linalg.norm(delta_pc[1])),
                "occupancy_weight_pc_displacement": float(np.linalg.norm(delta_pc[2])),
                "transition_only_pc1": float(delta_pc[0, 0]), "transition_only_pc2": float(delta_pc[0, 1]),
                "alignment_pc1": float(delta_pc[1, 0]), "alignment_pc2": float(delta_pc[1, 1]),
                "occupancy_weight_pc1": float(delta_pc[2, 0]), "occupancy_weight_pc2": float(delta_pc[2, 1]),
            })
    return pd.DataFrame(summaries), pd.DataFrame(states)


def _interval_measurements(frame: pd.DataFrame, samples: list[dict[str, Any]],
                           move_rows: np.ndarray, diagnostics: pd.DataFrame) -> pd.DataFrame:
    by_key = {(item["row"]["run_id"], item["row"]["checkpoint"]): item for item in samples}
    rows = []
    for (run_id, condition, seed), group in frame.groupby(["run_id", "condition", "seed"], sort=False):
        group = group.set_index("checkpoint").loc[list(STEP_BY_PASS)]
        for start, end in zip(list(STEP_BY_PASS)[:-1], list(STEP_BY_PASS)[1:]):
            old, new = by_key[(run_id, start)], by_key[(run_id, end)]
            full_change = new["q"] - old["q"]
            move_change = (new["q_nonself"] - old["q_nonself"])[move_rows]
            full_step = float(np.linalg.norm(full_change))
            move_step = float(np.linalg.norm(move_change))
            n_true = full_change.shape[1]
            diagonal = full_change[:, np.arange(n_true), np.arange(n_true)]
            diagonal_step = float(np.linalg.norm(diagonal))
            remaining_step = float(np.sqrt(max(0.0, full_step**2 - diagonal_step**2)))
            rows.append({"environment": old["row"]["environment"], "level": old["row"]["level"],
                         "condition": condition, "seed": seed, "run_id": run_id,
                         "from_checkpoint": start, "to_checkpoint": end,
                         "v_full": full_step, "v_nonself": move_step,
                         "r_move": move_step / full_step if full_step > 1e-12 else np.nan,
                         "r_move_missing_reason": "v_full_at_or_below_1e-12" if full_step <= 1e-12 else None,
                         "v_full_diagonal": diagonal_step, "v_full_remaining": remaining_step,
                         "delta_d_gt": float(group.loc[end, "d_gt"] - group.loc[start, "d_gt"]),
                         "delta_d_gt_nonself": float(group.loc[end, "d_gt_nonself"] - group.loc[start, "d_gt_nonself"]),
                         "delta_nll_bits_per_symbol": float(group.loc[end, "nll_bits_per_symbol"] - group.loc[start, "nll_bits_per_symbol"])})
    intervals = pd.DataFrame(rows).merge(diagnostics, on=["environment", "condition", "seed", "run_id",
                                                   "from_checkpoint", "to_checkpoint"], validate="one_to_one")
    if len(intervals) != 540 or not np.isfinite(intervals[["v_full", "v_nonself", "delta_d_gt",
                                                          "delta_d_gt_nonself", "delta_nll_bits_per_symbol"]]).all().all():
        raise ValueError("Invalid interval measurements")
    return intervals


def _checkpoint_interval_figure(frame: pd.DataFrame, intervals: pd.DataFrame, output: Path) -> None:
    styles = {value[0]: value[1:] for value in CONDITIONS.values()}
    panels = [(frame, "d_gt", "Full structural distance", "checkpoint"),
              (frame, "d_gt_nonself", "No-self structural distance", "checkpoint"),
              (frame, "nll_bits_per_symbol", "Held-out NLL (bits/symbol)", "checkpoint"),
              (intervals, "v_full", "Full-vector step magnitude", "to_checkpoint"),
              (intervals, "v_nonself", "No-self-vector step magnitude", "to_checkpoint")]
    fig, axes = plt.subplots(3, 2, figsize=(13, 13), constrained_layout=True)
    for ax, (data, metric, title, clock) in zip(axes.flat, panels):
        ticks = list(STEP_BY_PASS) if clock == "checkpoint" else list(STEP_BY_PASS)[1:]
        x = np.arange(len(ticks))
        for condition in CONDITION_ORDER:
            subset = data[data.condition == condition]
            color, linestyle = styles[condition]
            for _, seed in subset.groupby("seed"):
                ax.plot(x, seed.set_index(clock).loc[ticks, metric], color=color, linestyle=linestyle,
                        alpha=0.1, linewidth=0.7)
            mean = subset.groupby(clock)[metric].mean().loc[ticks]
            ax.plot(x, mean, color=color, linestyle=linestyle, marker="o", linewidth=2.2,
                    label=condition)
        ax.set_xticks(x, [str(tick) for tick in ticks])
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes.flat[-1].axis("off")
    axes.flat[0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"CSCG checkpoint metrics and interval motion — {frame.environment.iloc[0]}")
    _save_figure(fig, output)


def _change_scatter_figure(intervals: pd.DataFrame, output: Path) -> None:
    styles = {value[0]: value[1:] for value in CONDITIONS.values()}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for ax, metric, title in zip(axes, ["delta_d_gt", "delta_d_gt_nonself"],
                                  ["Full structural-distance change", "No-self structural-distance change"]):
        for condition in CONDITION_ORDER:
            subset = intervals[intervals.condition == condition]
            color = styles[condition][0]
            ax.scatter(subset[metric], subset.delta_nll_bits_per_symbol, color=color, s=10, alpha=0.16)
            means = subset.groupby(["from_checkpoint", "to_checkpoint"])[[metric, "delta_nll_bits_per_symbol"]].mean()
            for (start, end), row in means.iterrows():
                ax.scatter(row[metric], row.delta_nll_bits_per_symbol, color=color, s=85,
                           edgecolor="black", linewidth=0.5, label=condition if (start, end) == (0, 1) else None)
                ax.annotate(f"{start}→{end}", (row[metric], row.delta_nll_bits_per_symbol),
                            xytext=(4, 4), textcoords="offset points", fontsize=6)
        ax.axhline(0, color="0.4", lw=0.7)
        ax.axvline(0, color="0.4", lw=0.7)
        ax.set_xlabel(title)
        ax.set_ylabel("Δ held-out NLL (bits/symbol)")
        ax.grid(alpha=0.15)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Structural and likelihood changes — {intervals.environment.iloc[0]}")
    _save_figure(fig, output)


def _save_figure(fig: plt.Figure, output: Path) -> None:
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _condition_handles(conditions: list[str]) -> list[Line2D]:
    styles = {value[0]: value[1:] for value in CONDITIONS.values()}
    return [Line2D([0], [0], color=styles[name][0], linestyle=styles[name][1], linewidth=2.4, label=name)
            for name in conditions]


def _phase_figure(frame: pd.DataFrame, conditions: list[str], variance: np.ndarray, output: Path,
                  *, representation: str = "full", colour: str = "d_gt",
                  plane: tuple[int, int] = (1, 2)) -> None:
    selected = frame[frame["condition"].isin(conditions)]
    environment = str(frame["environment"].iloc[0]).replace("x", "×")
    fig, ax = plt.subplots(figsize=(9.5, 7.5), constrained_layout=True)
    styles = {value[0]: value[1:] for value in CONDITIONS.values()}
    suffix = "" if representation == "full" else "_nonself"
    xcol, ycol = f"pc{plane[0]}{suffix}", f"pc{plane[1]}{suffix}"
    norm = Normalize(vmin=float(frame[colour].min()), vmax=float(frame[colour].max()))
    cmap = plt.get_cmap("viridis_r")
    order = list(STEP_BY_PASS)
    coincident: list[str] = []
    for condition in conditions:
        subset = selected[selected["condition"] == condition]
        line_color, linestyle = styles[condition]
        for _, seed in subset.groupby("seed"):
            seed = seed.set_index("checkpoint").loc[order]
            coordinates = seed[[xcol, ycol]].to_numpy()
            ax.plot(coordinates[:, 0], coordinates[:, 1], color=line_color, linestyle=linestyle, alpha=0.14, linewidth=0.8, zorder=1)
            for start, end in zip(coordinates[:-1], coordinates[1:]):
                ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": line_color,
                            "alpha": 0.10, "lw": 0.6, "mutation_scale": 7}, zorder=1)
            for checkpoint, row in seed.iterrows():
                ax.scatter(row[xcol], row[ycol], c=[cmap(norm(row[colour]))], marker=MARKERS[checkpoint],
                           s=22, alpha=0.38, linewidths=0, zorder=2)
        means = subset.groupby("checkpoint", as_index=True)[[xcol, ycol, colour]].mean().loc[order]
        coordinates = means[[xcol, ycol]].to_numpy()
        all_near = float(np.max(np.linalg.norm(coordinates - coordinates[0], axis=1))) < 0.03
        if all_near:
            coincident.append(condition)
        ax.plot(coordinates[:, 0], coordinates[:, 1], color=line_color, linestyle=linestyle, linewidth=2.5, zorder=3)
        for start, end in zip(coordinates[:-1], coordinates[1:]):
            ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "-|>", "color": line_color,
                        "lw": 1.8, "mutation_scale": 14}, zorder=3)
        for checkpoint, row in means.iterrows():
            ax.scatter(row[xcol], row[ycol], c=[cmap(norm(row[colour]))], marker=MARKERS[checkpoint],
                       s=115, edgecolors=line_color, linewidths=1.8, zorder=4)
            if not all_near:
                label = ax.annotate(str(checkpoint), (row[xcol], row[ycol]), xytext=(5, 5), textcoords="offset points",
                                    fontsize=8, color=line_color, weight="bold", zorder=5)
                label.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground="white")])
    gt = frame.attrs["gt_coordinate" if representation == "full" else "gt_coordinate_nonself"]
    gt_xy = (gt[plane[0] - 1], gt[plane[1] - 1])
    ax.scatter(*gt_xy, marker="*", s=300, color="black", edgecolors="white", linewidths=1.2, zorder=7)
    ax.annotate("GT", gt_xy, xytext=(-8, 9), textcoords="offset points", ha="right", fontsize=11, weight="bold")
    if coincident:
        ax.text(0.99, 0.025,
                "Within each listed condition, its four checkpoint means cluster:\n" + ", ".join(coincident),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9, "boxstyle": "round,pad=0.35"})
    ax.set_xlabel(f"PC{plane[0]} ({variance[plane[0] - 1] * 100:.1f}%)")
    ax.set_ylabel(f"PC{plane[1]} ({variance[plane[1] - 1] * 100:.1f}%)")
    title = (f"Projected CSCG training trajectories — {environment}" if representation == "full"
             else f"Projected CSCG training trajectories ({representation}) — {environment}")
    fig.suptitle(title, fontsize=15)
    ax.set_title("Hybrid pass 0 follows 2,000 source-stage samples; passes 1/2/1000 count CSCG training only.",
                 fontsize=9, pad=5)
    ax.grid(alpha=0.18)
    condition_legend = ax.legend(handles=_condition_handles(conditions), title="Initialisation", loc="upper left", fontsize=8)
    ax.add_artist(condition_legend)
    marker_handles = [Line2D([0], [0], marker=MARKERS[k], color="none", markerfacecolor="gray", markersize=7,
                             label=str(k)) for k in order]
    ax.legend(handles=marker_handles, title="Checkpoint (pass)", loc="lower left", ncol=4, fontsize=8)
    colour_labels = {"d_gt": r"$d_{GT}=1-S^A_{TV}$", "d_gt_nonself": r"$1-S^A_{TV,\neg self}$",
                     "nll_bits_per_symbol": "Held-out NLL (bits/symbol)"}
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=colour_labels[colour], shrink=0.77)
    _save_figure(fig, output)


def _distance_figure(frame: pd.DataFrame, output: Path) -> None:
    environment = str(frame["environment"].iloc[0]).replace("x", "×")
    fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    styles = {value[0]: value[1:] for value in CONDITIONS.values()}
    order = list(STEP_BY_PASS)
    x = np.arange(len(order))
    for condition in CONDITION_ORDER:
        subset = frame[frame["condition"] == condition]
        color, linestyle = styles[condition]
        for _, seed in subset.groupby("seed"):
            values = seed.set_index("checkpoint").loc[order, "d_gt"].to_numpy()
            ax.plot(x, values, color=color, linestyle=linestyle, alpha=0.13, linewidth=0.8)
        means = subset.groupby("checkpoint")["d_gt"].mean().loc[order].to_numpy()
        ax.plot(x, means, color=color, linestyle=linestyle, marker="o", linewidth=2.5, markersize=6, label=condition)
    ax.set_xticks(x, [str(value) for value in order])
    ax.set_xlabel("CSCG pass since initialisation / hybrid handoff")
    ax.set_ylabel(r"Structural distance $1-S^A_{TV}$")
    fig.suptitle(f"Structural distance through CSCG training — {environment}", fontsize=15)
    ax.set_title("Hybrid pass 0 follows the MERCURY/TMGWR source stage (2,000 samples).", fontsize=9, pad=5)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    _save_figure(fig, output)


def _start_finish_figure(frame: pd.DataFrame, output: Path) -> None:
    environment = str(frame["environment"].iloc[0]).replace("x", "×")
    fig, ax = plt.subplots(figsize=(7, 6.5), constrained_layout=True)
    styles = {value[0]: value[1:] for value in CONDITIONS.values()}
    paired = frame[frame["checkpoint"].isin([0, 1000])].pivot(index=["condition", "seed"], columns="checkpoint", values="d_gt")
    lower = float(paired.min().min())
    upper = float(paired.max().max())
    padding = max(0.025, (upper - lower) * 0.06)
    ax.plot([lower - padding, upper + padding], [lower - padding, upper + padding], color="black", linestyle="--", linewidth=1, label="No change")
    marker_by_condition = dict(zip(CONDITION_ORDER, ["o", "s", "^", "D", "P", "X"]))
    for condition in CONDITION_ORDER:
        subset = paired.loc[condition]
        ax.scatter(subset[0], subset[1000], color=styles[condition][0], marker=marker_by_condition[condition],
                   s=43, alpha=0.70, label=condition)
    ax.set_xlim(lower - padding, upper + padding)
    ax.set_ylim(lower - padding, upper + padding)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Initial structural distance $d_{GT}^{(0)}$")
    ax.set_ylabel(r"Final structural distance $d_{GT}^{(1000)}$")
    ax.set_title(f"CSCG structural change after initialisation / handoff — {environment}")
    ax.text(0.04, 0.96, "Below line: improvement", transform=ax.transAxes, va="top", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.2)
    _save_figure(fig, output)


def _write_report(
    frame: pd.DataFrame, diagnostics: pd.DataFrame, variance: np.ndarray,
    gt_coordinate: np.ndarray, validation: dict[str, Any], output: Path,
) -> None:
    environment = str(frame["environment"].iloc[0]).replace("x", "×")
    distances = frame.groupby(["condition", "checkpoint"])["d_gt"].mean().unstack()
    changes = diagnostics.groupby(["condition", "from_checkpoint", "to_checkpoint"])["n_changed"].mean()
    lines = [
        f"# {environment} CSCG projected training trajectories", "",
        "## Checkpoint clock", "",
        "MERCURY and TMGWR each have a separate source-model stage with 2,000 observed samples. Hybrid CSCG pass 0 is the graph-initialised CSCG model **after that source stage and before CSCG optimisation**; it is reconstructed by evaluation-v5 from the saved source graph. CSCG passes 1, 2, and 1000 then use 2,000, 4,000, and 2,000,000 CSCG-stage samples, respectively. The source-model evaluation at source step 2,000 is a distinct record and is not included in this CSCG PCA.",
        "", "The source and CSCG stages may consume the same underlying training sequence. Their sample counts describe stage exposure, not distinct samples; the CSV keeps them in separate columns.", "",
        f"Single PCA fitted to 720 learned projected tensors ({frame.attrs['n_features']} features each); PC1–PC4 explain {', '.join(f'{part:.1%}' for part in variance)} respectively. The PC3/PC4 figures show the other two saved dimensions.",
        f"Projected GT coordinate (PC1–PC4): ({', '.join(f'{part:.4f}' for part in gt_coordinate)}). GT was excluded from the PCA fit.",
        f"The 720 stored similarities reproduce occupancy weighted TV exactly; {validation['summary_values_checked']} published pass 2/pass 1000 values agree within {validation['max_summary_error']:.2g}.",
        "", "## Mean structural distance", "",
        "| Condition | Pass 0 | Pass 1 | Pass 2 | Pass 1000 | Change 0→1000 |", 
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITION_ORDER:
        values = distances.loc[condition]
        lines.append(f"| {condition} | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} | {values[1000]:.3f} | {values[1000] - values[0]:+.3f} |")
    lines.extend(["", "## Learned-state alignment changes", "",
                  "Mean number of learned states whose hard GT assignment changed per seed:", "",
                  "| Condition | 0→1 | 1→2 | 2→1000 |", "|---|---:|---:|---:|"])
    for condition in CONDITION_ORDER:
        values = [changes.loc[(condition, before, after)] for before, after in [(0, 1), (1, 2), (2, 1000)]]
        lines.append(f"| {condition} | {values[0]:.2f} | {values[1]:.2f} | {values[2]:.2f} |")
    lines.extend(["", "The diagnostic CSV decomposes each projected transition change in order: saved transition tensor change with the old alignment and source weights; alignment change with old source co-occurrence; then action-conditioned source-weight change. The three vectors sum to the observed displacement. Their norms do not sum, and this is a projection sensitivity diagnostic rather than a replacement for $1-S^A_{TV}$.", ""])
    (output / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_oracle_findings(frame: pd.DataFrame, intervals: pd.DataFrame, output: Path) -> None:
    oracle = intervals[intervals.condition == "Oracle-C"].sort_values(["from_checkpoint", "seed"])
    columns = ["environment", "seed", "from_checkpoint", "to_checkpoint", "v_full", "v_nonself",
               "r_move", "r_move_missing_reason", "v_full_diagonal", "v_full_remaining",
               "delta_d_gt", "delta_d_gt_nonself", "delta_nll_bits_per_symbol"]
    oracle[columns].to_csv(output / "oracle_c_seed_distributions.csv", index=False)
    metrics = [("v_full", "Full-vector motion"), ("v_nonself", "No-self motion"),
               ("r_move", "Motion ratio"), ("v_full_diagonal", "Full diagonal motion"),
               ("v_full_remaining", "Full remaining-entry motion"),
               ("delta_d_gt", "Δ full structural distance"),
               ("delta_d_gt_nonself", "Δ no-self structural distance"),
               ("delta_nll_bits_per_symbol", "Δ held-out NLL")]
    lines = [f"# Oracle-C checkpoint findings: {frame.environment.iloc[0]}", "",
             "Thirty paired seeds per interval. Changes equal destination minus source; negative distance or NLL changes indicate improvement. NLL is the saved held-out bits per symbol, not a distance to GT.",
             "", "The full and no-self PCA bases are separate within this environment. Motion is measured in the original vectors. The ratio compares differently conditioned vectors and is descriptive; it does not partition full-transition change.",
             "", "| Interval | Measure | Mean | Median | Q1–Q3 | IQR | Min–max | Positive / zero / negative |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for (start, end), group in oracle.groupby(["from_checkpoint", "to_checkpoint"], sort=True):
        for metric, label in metrics:
            values = group[metric].dropna().to_numpy()
            if len(values) == 0:
                continue
            q1, median, q3 = np.percentile(values, [25, 50, 75])
            signs = f"{np.sum(values > 1e-12)} / {np.sum(np.abs(values) <= 1e-12)} / {np.sum(values < -1e-12)}"
            lines.append(f"| {start}→{end} | {label} | {values.mean():.4g} | {median:.4g} | {q1:.4g}–{q3:.4g} | {q3-q1:.4g} | {values.min():.4g}–{values.max():.4g} | {signs} |")
    missing = int(oracle.r_move.isna().sum())
    lines.extend(["", f"The motion ratio is missing for {missing} of 90 Oracle-C intervals because full-vector motion is at or below 10⁻¹². The per-seed values and reasons are in `oracle_c_seed_distributions.csv`.",
                  "", "## Descriptive assessment", ""])
    for (start, end), group in oracle.groupby(["from_checkpoint", "to_checkpoint"], sort=True):
        n_full = int((group.delta_d_gt < 0).sum())
        n_move = int((group.delta_d_gt_nonself < 0).sum())
        n_nll = int((group.delta_nll_bits_per_symbol < 0).sum())
        both = int(((group.delta_d_gt < 0) & (group.delta_nll_bits_per_symbol < 0)).sum())
        lines.append(f"- **{start}→{end}:** full structural distance improves in {n_full}/30 seeds, no-self distance in {n_move}/30, and NLL in {n_nll}/30. Full structural distance and NLL both improve in {both}/30. Median full/no-self motion is {group.v_full.median():.4g}/{group.v_nonself.median():.4g}; median diagonal/remaining full motion is {group.v_full_diagonal.median():.4g}/{group.v_full_remaining.median():.4g}.")
    lines.extend(["", "H1–H4 are assessed descriptively through (H1) full-vector motion, (H2) no-self motion and the descriptive ratio, (H3) changes in both structural distances, and (H4) the paired NLL and structural changes. The tables show seed spread and signs without a dynamical-systems label or a cutoff for a bottleneck. Diagonal and remaining-entry norms describe full-vector self-transition changes; their squared norms sum to the squared full step.",
                  "", "Hybrid pass 0 follows a separate 2,000-sample MERCURY or TMGWR source stage. Passes 1, 2, and 1000 count CSCG optimisation after that handoff.", ""])
    (output / "oracle_c_findings.md").write_text("\n".join(lines), encoding="utf-8")


def _run_environment(manifest: Path, level: int, output: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frame, samples, gt, validation = _read_records(manifest, level)
    move_rows = validation.pop("move_mask")
    gt_nonself = validation.pop("gt_nonself")
    validation.update(_check_summary_values(frame))
    matrix = np.stack([item["q"].ravel(order="C") for item in samples])
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite full-transition features")
    pca = PCA(n_components=4, svd_solver="full")
    coordinates = pca.fit_transform(matrix)
    gt_coordinate = pca.transform(gt.ravel(order="C").reshape(1, -1))[0]
    frame[["pc1", "pc2", "pc3", "pc4"]] = coordinates
    previous_path = output / "trajectory_coordinates.csv"
    if previous_path.is_file():
        previous = pd.read_csv(previous_path).set_index(["run_id", "checkpoint"])[["pc1", "pc2"]]
        current = frame.set_index(["run_id", "checkpoint"])[["pc1", "pc2"]]
        if set(previous.index) != set(current.index) or not np.allclose(
                previous.loc[current.index], current, rtol=0, atol=1e-9):
            raise ValueError(f"Full PCA coordinates differ from prior analysis: {output}")
    basis_path = output / "pca_basis_and_transitions.npz"
    if basis_path.is_file():
        with np.load(basis_path) as prior:
            recovered = (matrix - prior["feature_mean"]) @ prior["components"].T
            if not np.allclose(recovered, coordinates[:, :recovered.shape[1]], rtol=0, atol=1e-9):
                raise ValueError(f"Saved full PCA basis does not reproduce current coordinates: {output}")
    nonself_matrix = np.stack([item["q_nonself"][move_rows].ravel(order="C") for item in samples])
    if not np.isfinite(nonself_matrix).all():
        raise ValueError("Non-finite no-self features")
    pca_nonself = PCA(n_components=4, svd_solver="full")
    frame[["pc1_nonself", "pc2_nonself", "pc3_nonself", "pc4_nonself"]] = pca_nonself.fit_transform(nonself_matrix)
    gt_coordinate_nonself = pca_nonself.transform(gt_nonself[move_rows].ravel(order="C").reshape(1, -1))[0]
    frame.attrs["gt_coordinate"] = gt_coordinate
    frame.attrs["gt_coordinate_nonself"] = gt_coordinate_nonself
    frame.attrs["n_features"] = int(matrix.shape[1])
    alignment_summary, alignment_states = _alignment_diagnostics(frame, samples, pca)
    intervals = _interval_measurements(frame, samples, move_rows, alignment_summary)
    interval_columns = [column for column in intervals if column not in {"environment", "level", "condition", "seed", "run_id", "to_checkpoint"}]
    frame = frame.merge(intervals.drop(columns=["environment", "level", "condition", "seed"]),
                        left_on=["run_id", "checkpoint"], right_on=["run_id", "to_checkpoint"],
                        how="left", validate="one_to_one").drop(columns="to_checkpoint")
    if frame.loc[frame.checkpoint == 0, interval_columns].notna().any().any():
        raise ValueError("Pass 0 has interval measurements")
    frame.attrs["gt_coordinate"] = gt_coordinate
    frame.attrs["gt_coordinate_nonself"] = gt_coordinate_nonself
    frame.attrs["n_features"] = int(matrix.shape[1])
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "trajectory_coordinates.csv", index=False)
    intervals.to_csv(output / "interval_measurements.csv", index=False)
    alignment_summary.to_csv(output / "alignment_change_summary.csv", index=False)
    alignment_states.to_csv(output / "alignment_changes_by_state.csv", index=False)
    np.savez_compressed(output / "pca_basis_and_transitions.npz", learned_vectors=matrix,
                        gt_vector=gt.ravel(order="C"), feature_mean=pca.mean_, components=pca.components_,
                        explained_variance_ratio=pca.explained_variance_ratio_, gt_coordinate=gt_coordinate)
    np.savez_compressed(output / "pca_basis_nonself.npz", learned_vectors=nonself_matrix,
                        gt_vector=gt_nonself[move_rows].ravel(order="C"), movement_mask=move_rows,
                        feature_mean=pca_nonself.mean_, components=pca_nonself.components_,
                        explained_variance_ratio=pca_nonself.explained_variance_ratio_,
                        gt_coordinate=gt_coordinate_nonself)
    metadata = {**validation, "environment": ENVIRONMENTS[level], "level": level, "n_records": len(frame),
                "n_seeds_per_condition": 30, "checkpoint_to_sample_step": STEP_BY_PASS,
                "hybrid_source_stage_observed_samples": 2_000,
                "checkpoint_clock": "CSCG-stage samples since initialisation/handoff; hybrid source-stage records are separate and excluded",
                "feature_order": f"C-order (action, GT source, GT target 0..{gt.shape[1]-1} then bottom)",
                "feature_shape": list(gt.shape), "pca_fit": "centred learned projected tensors only; no feature scaling",
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "nonself_explained_variance_ratio": pca_nonself.explained_variance_ratio_.tolist(),
                "nonself_feature_shape": [int(move_rows.sum()), int(gt.shape[-1])],
                "nonself_feature_order": "C-order GT-defined movement rows in action/source order, each with all targets including zero self slot and sink",
                "nonself_pca_fit": "centred learned conditioned tensors only; no feature scaling",
                "projected_gt_coordinate": gt_coordinate.tolist(),
                "projected_gt_coordinate_nonself": gt_coordinate_nonself.tolist(),
                "alignment_diagnostic": "Sequential exact vector decomposition: transition change with old alignment/H; alignment change with old H; action-source occupancy H change. Component norms are not additive."}
    (output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    _write_report(frame, alignment_summary, pca.explained_variance_ratio_, gt_coordinate, validation, output)
    _phase_figure(frame, ORACLE_ORDER, pca.explained_variance_ratio_, output / "phase_space_oracle_conditions")
    _phase_figure(frame, CONDITION_ORDER, pca.explained_variance_ratio_, output / "phase_space_all_conditions")
    _phase_figure(frame, ORACLE_ORDER, pca.explained_variance_ratio_, output / "phase_space_oracle_conditions_nll",
                  colour="nll_bits_per_symbol")
    _phase_figure(frame, CONDITION_ORDER, pca.explained_variance_ratio_, output / "phase_space_all_conditions_nll",
                  colour="nll_bits_per_symbol")
    _phase_figure(frame, ORACLE_ORDER, pca_nonself.explained_variance_ratio_, output / "phase_space_nonself_oracle_conditions",
                  representation="no-self", colour="d_gt_nonself")
    _phase_figure(frame, CONDITION_ORDER, pca_nonself.explained_variance_ratio_, output / "phase_space_nonself_all_conditions",
                  representation="no-self", colour="d_gt_nonself")
    for conditions, scope in [(ORACLE_ORDER, "oracle_conditions"), (CONDITION_ORDER, "all_conditions")]:
        _phase_figure(frame, conditions, pca.explained_variance_ratio_, output / f"phase_space_{scope}_pc3_pc4",
                      plane=(3, 4))
        _phase_figure(frame, conditions, pca.explained_variance_ratio_, output / f"phase_space_{scope}_nll_pc3_pc4",
                      colour="nll_bits_per_symbol", plane=(3, 4))
        _phase_figure(frame, conditions, pca_nonself.explained_variance_ratio_,
                      output / f"phase_space_nonself_{scope}_pc3_pc4",
                      representation="no-self", colour="d_gt_nonself", plane=(3, 4))
    _distance_figure(frame, output / "structural_distance_by_checkpoint")
    _start_finish_figure(frame, output / "initial_vs_final_distance")
    _checkpoint_interval_figure(frame, intervals, output / "checkpoint_metrics_and_motion")
    _change_scatter_figure(intervals, output / "nll_vs_structural_changes")
    _write_oracle_findings(frame, intervals, output)
    print(f"{ENVIRONMENTS[level]}: {len(frame)} observations; PC1/PC2={pca.explained_variance_ratio_[0]:.4f}/{pca.explained_variance_ratio_[1]:.4f}; summary parity max={validation['max_summary_error']:.3g}")
    return frame, intervals, alignment_summary, alignment_states, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=PLOT_ROOT / "cscg_transition_phase_space_all_environments")
    parser.add_argument("--levels", nargs="+", type=int, default=list(ENVIRONMENTS))
    args = parser.parse_args()
    if not args.levels or any(level not in ENVIRONMENTS for level in args.levels) or len(set(args.levels)) != len(args.levels):
        parser.error(f"Select unique levels from {list(ENVIRONMENTS)}")
    all_frames: list[pd.DataFrame] = []
    all_diagnostics: list[pd.DataFrame] = []
    all_intervals: list[pd.DataFrame] = []
    all_state_changes: list[pd.DataFrame] = []
    index: list[dict[str, Any]] = []
    for level in args.levels:
        frame, intervals, diagnostics, state_changes, metadata = _run_environment(
            args.manifest, level, args.output_dir / ENVIRONMENTS[level]
        )
        combined_frame = frame.copy()
        combined_frame.attrs = {}
        all_frames.append(combined_frame)
        all_diagnostics.append(diagnostics)
        all_intervals.append(intervals)
        all_state_changes.append(state_changes)
        index.append({"environment": ENVIRONMENTS[level], "level": level,
                      "n_records": metadata["n_records"], "n_features": int(np.prod(metadata["feature_shape"])),
                      "pc1_explained_variance_ratio": metadata["explained_variance_ratio"][0],
                      "pc2_explained_variance_ratio": metadata["explained_variance_ratio"][1],
                      "pc3_explained_variance_ratio": metadata["explained_variance_ratio"][2],
                      "pc4_explained_variance_ratio": metadata["explained_variance_ratio"][3],
                      "gt_pc1": metadata["projected_gt_coordinate"][0],
                      "gt_pc2": metadata["projected_gt_coordinate"][1],
                      "gt_pc3": metadata["projected_gt_coordinate"][2],
                      "gt_pc4": metadata["projected_gt_coordinate"][3],
                      "nonself_pc3_explained_variance_ratio": metadata["nonself_explained_variance_ratio"][2],
                      "nonself_pc4_explained_variance_ratio": metadata["nonself_explained_variance_ratio"][3],
                      "nonself_gt_pc3": metadata["projected_gt_coordinate_nonself"][2],
                      "nonself_gt_pc4": metadata["projected_gt_coordinate_nonself"][3],
                      "max_summary_error": metadata["max_summary_error"]})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(all_frames, ignore_index=True).to_csv(args.output_dir / "trajectory_coordinates_all_environments.csv", index=False)
    pd.concat(all_diagnostics, ignore_index=True).to_csv(args.output_dir / "alignment_change_summary_all_environments.csv", index=False)
    pd.concat(all_intervals, ignore_index=True).to_csv(args.output_dir / "interval_measurements_all_environments.csv", index=False)
    pd.concat(all_state_changes, ignore_index=True).to_csv(args.output_dir / "alignment_changes_by_state_all_environments.csv", index=False)
    pd.DataFrame(index).to_csv(args.output_dir / "environment_pca_index.csv", index=False)
    print(f"Saved {sum(len(frame) for frame in all_frames)} observations to {args.output_dir}")


if __name__ == "__main__":
    main()
