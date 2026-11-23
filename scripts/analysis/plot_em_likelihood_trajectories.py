from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mercury_runs.analysis.plots import publication_style as pub


@dataclass(frozen=True)
class TrajectoryRecord:
    run_id: str
    method: str
    level: int
    rand_prob: float
    seed: int
    batch_size: int
    clone_allocation: str
    clones_per_obs: int
    clones_per_state: int
    x: np.ndarray
    y: np.ndarray
    original_points: int
    original_final_x: int
    initial_y: float
    final_y: float
    min_y: float
    mean_y: float


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sensor_path_parts(sensor: str, sensor_range: int | None) -> list[str]:
    parts = [f"sensor={sensor}"]
    if sensor == "cardinal distance":
        parts.append(f"range={sensor_range}")
    return parts


def _model_path(run_dir: Path, run_id: str, args: dict) -> Path | None:
    method = str(args.get("method", ""))
    level = int(args["level"])
    sensor = str(args["sensor"])
    sensor_range = args.get("sensor_range")
    suffix_parts = [f"level={level}", *_sensor_path_parts(sensor, sensor_range)]
    if method == "cscg":
        return run_dir.joinpath("bundles", "cscg", *suffix_parts, f"{run_id}_cscg_model.npz")
    if method == "mercury_cscg_hybrid":
        return run_dir.joinpath(
            "bundles",
            "mercury_cscg_hybrid",
            *suffix_parts,
            f"{run_id}_mercury_cscg_hybrid_model.npz",
        )
    return None


def _downsample(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if x.size <= max_points:
        return x, y
    indices = np.unique(np.linspace(0, x.size - 1, num=max_points, dtype=np.int64))
    return x[indices], y[indices]


def _load_trajectory(run_config_path: Path, *, max_points: int) -> TrajectoryRecord | None:
    config = _read_json(run_config_path)
    args = config.get("cli_args", {})
    method = str(args.get("method", ""))
    if method not in {"cscg", "mercury_cscg_hybrid"}:
        return None
    run_id = str(config.get("run_id") or config.get("requested_run_id") or run_config_path.parent.name)
    model_path = _model_path(run_config_path.parent, run_id, args)
    if model_path is None or not model_path.exists():
        return None
    with np.load(model_path) as model:
        if "convergence_em" not in model.files:
            return None
        y = np.asarray(model["convergence_em"], dtype=np.float64)
    if y.size == 0:
        return None
    batch_size = int(args.get("cscg_batch_size") or 1)
    x = np.arange(1, y.size + 1, dtype=np.float64) * float(batch_size)
    original_points = int(y.size)
    original_final_x = int(x[-1])
    initial_y = float(y[0])
    final_y = float(y[-1])
    min_y = float(np.min(y))
    mean_y = float(np.mean(y))
    x, y = _downsample(x, y, max_points=max_points)
    return TrajectoryRecord(
        run_id=run_id,
        method=method,
        level=int(args["level"]),
        rand_prob=float(args.get("rand_prob", 0.0)),
        seed=int(args.get("seed", -1)),
        batch_size=batch_size,
        clone_allocation=str(args.get("cscg_clone_allocation") or "uniform"),
        clones_per_obs=int(args.get("cscg_clones_per_obs") or 0),
        clones_per_state=int(args.get("cscg_clones_per_state") or 0),
        x=x,
        y=y,
        original_points=original_points,
        original_final_x=original_final_x,
        initial_y=initial_y,
        final_y=final_y,
        min_y=min_y,
        mean_y=mean_y,
    )


def load_records(study_root: Path, *, max_points: int) -> list[TrajectoryRecord]:
    records = []
    for path in sorted(study_root.glob("*/run_config.json")):
        record = _load_trajectory(path, max_points=max_points)
        if record is not None:
            records.append(record)
    return records


def _summary_frame(records: Iterable[TrajectoryRecord]) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append(
            {
                "run_id": record.run_id,
                "method": record.method,
                "level": record.level,
                "rand_prob": record.rand_prob,
                "seed": record.seed,
                "batch_size": record.batch_size,
                "clone_allocation": record.clone_allocation,
                "clones_per_obs": record.clones_per_obs,
                "clones_per_state": record.clones_per_state,
                "points": record.original_points,
                "observed_samples_final": record.original_final_x,
                "em_nll_bps_initial": record.initial_y,
                "em_nll_bps_final": record.final_y,
                "em_nll_bps_min": record.min_y,
                "em_nll_bps_mean": record.mean_y,
            }
        )
    return pd.DataFrame(rows)


def _clone_config_key(record: TrajectoryRecord) -> tuple[str, int]:
    allocation = str(record.clone_allocation).strip().lower()
    if allocation in {"ground_truth", "oracle_ground_truth"}:
        return ("gt", int(record.clones_per_state))
    return ("uniform", int(record.clones_per_obs))


def _clone_config_label(key: tuple[str, int]) -> str:
    kind, value = key
    if kind == "gt":
        return f"GT c/state={value}"
    return f"uniform c/obs={value}"


def _aggregate(records: list[TrajectoryRecord]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    min_final_x = min(float(record.x[-1]) for record in records)
    max_len = min(max(record.y.size for record in records), 700)
    grid = np.unique(np.linspace(1.0, min_final_x, num=max_len))
    interpolated = np.vstack([np.interp(grid, record.x, record.y) for record in records])
    count = int(interpolated.shape[0])
    mean = np.mean(interpolated, axis=0)
    if count <= 1:
        ci95 = np.zeros_like(mean)
    else:
        ci95 = 1.96 * np.std(interpolated, axis=0, ddof=1) / np.sqrt(float(count))
    return grid, mean, ci95, count


def _method_label(method: str) -> str:
    return pub.method_label(method) if method == "mercury_cscg_hybrid" else "CSCG"


def _rand_label(rand_prob: float) -> str:
    return f"rand={rand_prob:g}"


def _configure_axis(ax) -> None:
    ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Observed samples")
    ax.set_ylabel("EM negative log2 likelihood / symbol")


def plot_overview(records: list[TrajectoryRecord], out_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    levels = sorted({record.level for record in records})
    rands = sorted({record.rand_prob for record in records})
    methods = ["cscg", "mercury_cscg_hybrid"]
    colors = {
        ("gt", 1): "#4c78a8",
        ("gt", 2): "#72b7b2",
        ("uniform", 4): "#f58518",
        ("uniform", 8): "#54a24b",
        ("uniform", 10): "#b279a2",
    }
    linestyles = {1: "-", 10: "--", 40: "-.", 100: ":"}

    for method in methods:
        method_records = [record for record in records if record.method == method]
        if not method_records:
            continue
        fig, axes = plt.subplots(len(levels), len(rands), figsize=(12, 9), sharex=False, sharey=False)
        axes = np.asarray(axes).reshape(len(levels), len(rands))
        for row, level in enumerate(levels):
            for col, rand_prob in enumerate(rands):
                ax = axes[row, col]
                cell = [record for record in method_records if record.level == level and record.rand_prob == rand_prob]
                for (clone_key, batch), group in sorted(_group_by(cell, lambda r: (_clone_config_key(r), r.batch_size)).items()):
                    grid, mean, ci95, count = _aggregate(group)
                    label = f"{_clone_config_label(clone_key)}, batch={batch}, n={count}"
                    color = colors.get(clone_key, "#666666")
                    ax.plot(grid, mean, color=color, linestyle=linestyles.get(batch, "-"), linewidth=1.6, label=label)
                    ax.fill_between(grid, mean - ci95, mean + ci95, color=color, alpha=0.14, linewidth=0)
                ax.set_title(f"level {level}, {_rand_label(rand_prob)}", fontsize=10)
                _configure_axis(ax)
                if cell:
                    ax.set_xscale("log")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8)
        fig.suptitle(f"{_method_label(method)} EM objective trajectories (mean with 95% CI)", fontsize=14)
        fig.tight_layout(rect=(0, 0.08, 1, 0.96))
        output = out_dir / f"em_nll_trajectory_overview_{method}.png"
        fig.savefig(output, dpi=180)
        outputs.append(output)
        plt.close(fig)
    return outputs


def plot_batch_profiles(records: list[TrajectoryRecord], out_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    methods = ["cscg", "mercury_cscg_hybrid"]
    colors = {1: "#4c78a8", 10: "#f58518", 40: "#54a24b", 100: "#b279a2"}
    for method in methods:
        method_records = [record for record in records if record.method == method]
        if not method_records:
            continue
        for level in sorted({record.level for record in method_records}):
            for rand_prob in sorted({record.rand_prob for record in method_records if record.level == level}):
                cell = [record for record in method_records if record.level == level and record.rand_prob == rand_prob]
                clone_keys = sorted({_clone_config_key(record) for record in cell})
                fig, axes = plt.subplots(1, len(clone_keys), figsize=(4.8 * len(clone_keys), 4), sharey=True)
                axes = np.asarray(axes).reshape(-1)
                for ax, clone_key in zip(axes, clone_keys):
                    clone_cell = [record for record in cell if _clone_config_key(record) == clone_key]
                    for batch, group in sorted(_group_by(clone_cell, lambda r: r.batch_size).items()):
                        grid, mean, ci95, count = _aggregate(group)
                        ax.plot(grid, mean, color=colors.get(batch, "#666666"), linewidth=1.8, label=f"batch={batch}, n={count}")
                        ax.fill_between(grid, mean - ci95, mean + ci95, color=colors.get(batch, "#666666"), alpha=0.16, linewidth=0)
                    ax.set_title(_clone_config_label(clone_key), fontsize=10)
                    ax.set_xscale("log")
                    _configure_axis(ax)
                axes[-1].legend(loc="best", frameon=False, fontsize=8)
                fig.suptitle(f"{_method_label(method)} level {level}, {_rand_label(rand_prob)} (mean with 95% CI)", fontsize=13)
                fig.tight_layout(rect=(0, 0, 1, 0.94))
                rand_key = str(rand_prob).replace(".", "p")
                output = out_dir / f"em_nll_trajectory_{method}_level_{level}_rand_{rand_key}_by_batch.png"
                fig.savefig(output, dpi=180)
                outputs.append(output)
                plt.close(fig)
    return outputs


def plot_rand03_batch40_method_comparison(records: list[TrajectoryRecord], out_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    cell = [record for record in records if np.isclose(record.rand_prob, 0.3) and record.batch_size == 40]
    if not cell:
        return outputs
    levels = sorted({record.level for record in cell})
    fig, axes = plt.subplots(1, len(levels), figsize=(5.2 * len(levels), 4.4), sharey=False)
    axes = np.asarray(axes).reshape(-1)
    styles = {
        ("cscg", ("gt", 1)): (pub.method_color("ground_truth_state_1"), "-"),
        ("cscg", ("gt", 2)): (pub.method_color("ground_truth_state_2"), "-"),
        ("cscg", ("uniform", 4)): (pub.method_color("uniform_obs_4"), "-"),
        ("cscg", ("uniform", 8)): (pub.method_color("uniform_obs_8"), "-"),
        ("mercury_cscg_hybrid", ("uniform", 10)): (pub.method_color("hybrid"), "--"),
    }
    fallback_colors = {"cscg": pub.method_color("vanilla_cscg"), "mercury_cscg_hybrid": pub.method_color("hybrid")}
    for ax, level in zip(axes, levels):
        level_records = [record for record in cell if record.level == level]
        for (method, clone_key), group in sorted(_group_by(level_records, lambda r: (r.method, _clone_config_key(r))).items()):
            grid, mean, ci95, count = _aggregate(group)
            color, linestyle = styles.get((method, clone_key), (fallback_colors.get(method, "#666666"), "-"))
            label = f"{_method_label(method)}, {_clone_config_label(clone_key)}, n={count}"
            ax.plot(grid, mean, color=color, linestyle=linestyle, linewidth=1.9, label=label)
            ax.fill_between(grid, mean - ci95, mean + ci95, color=color, alpha=0.15, linewidth=0)
        ax.set_title(f"level {level}", fontsize=10)
        ax.set_xscale("log")
        _configure_axis(ax)
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    fig.suptitle("rand=0.3, batch=40 EM objective comparison (mean with 95% CI)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 0.9, 0.93))
    output = out_dir / "em_nll_trajectory_rand_0p3_batch_40_cscg_hybrid_all_levels.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    outputs.append(output)
    plt.close(fig)
    return outputs


def _group_by(records: Iterable[TrajectoryRecord], key_fn) -> dict:
    groups = defaultdict(list)
    for record in records:
        groups[key_fn(record)].append(record)
    return dict(groups)


def plot_selected_raw(records: list[TrajectoryRecord], out_dir: Path, max_points: int) -> list[Path]:
    outputs: list[Path] = []
    for method in sorted({record.method for record in records}):
        selected = []
        for key, group in sorted(_group_by([r for r in records if r.method == method], lambda r: (r.level, r.rand_prob, r.batch_size, _clone_config_key(r))).items()):
            selected.extend(sorted(group, key=lambda r: (r.seed, r.run_id))[:3])
        if not selected:
            continue
        fig, ax = plt.subplots(figsize=(11, 5.5))
        for record in selected:
            x, y = _downsample(record.x, record.y, max_points=max_points)
            label = f"L{record.level} r={record.rand_prob:g} b={record.batch_size} {_clone_config_label(_clone_config_key(record))} s={record.seed}"
            ax.plot(x, y, linewidth=0.9, alpha=0.65, label=label)
        ax.set_xscale("log")
        _configure_axis(ax)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=7)
        fig.suptitle(f"{_method_label(method)} sampled raw EM trajectories", fontsize=13)
        fig.tight_layout()
        output = out_dir / f"em_nll_trajectory_sampled_raw_{method}.png"
        fig.savefig(output, dpi=180, bbox_inches="tight")
        outputs.append(output)
        plt.close(fig)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CSCG EM negative-log-likelihood convergence trajectories.")
    parser.add_argument(
        "--study-root",
        type=Path,
        default=Path("results/current/studies/mercury_cscg_hybrid_vs_cscg_batch_size_30_seed_common_eval_walks2"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/current/notebook_plots/mercury_cscg_hybrid_vs_cscg_batch_size_30_seed_common_eval_walks2/em_likelihood_trajectories"),
    )
    parser.add_argument("--raw-max-points", type=int, default=1200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.study_root, max_points=int(args.raw_max_points))
    if not records:
        raise SystemExit(f"No convergence_em trajectories found under {args.study_root}")
    summary = _summary_frame(records)
    summary_path = args.out_dir / "em_nll_trajectory_summary.csv"
    summary.to_csv(summary_path, index=False)
    outputs = [summary_path]
    outputs.extend(plot_overview(records, args.out_dir))
    outputs.extend(plot_batch_profiles(records, args.out_dir))
    outputs.extend(plot_rand03_batch40_method_comparison(records, args.out_dir))
    outputs.extend(plot_selected_raw(records, args.out_dir, max_points=int(args.raw_max_points)))
    manifest_path = args.out_dir / "em_nll_trajectory_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "study_root": str(args.study_root),
                "out_dir": str(args.out_dir),
                "num_runs": len(records),
                "outputs": [str(path) for path in outputs],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Loaded {len(records)} trajectories")
    print(f"Wrote {len(outputs)} outputs plus manifest to {args.out_dir}")


if __name__ == "__main__":
    main()
