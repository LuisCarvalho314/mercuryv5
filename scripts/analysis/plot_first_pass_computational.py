"""Create the defensive first-pass computational appendix outputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
import numpy as np
import polars as pl
from scipy.stats import t

from mercury.graph.core import Graph

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = ("mercury", "sensory", "cscg")
LABELS = {"mercury": "MERCURY", "sensory": "TMGWR", "cscg": "CSCG (B=40)"}
COLORS = {"mercury": "#D55E00", "sensory": "#E69F00", "cscg": "#264653"}
OFFSETS = {"mercury": -0.23, "sensory": 0.0, "cscg": 0.23}
LEVEL_ORDER = (13, 21, 20, 22)
LEVEL_LABELS = {13: "3x3", 21: "1x4", 20: "1x5", 22: "3x4"}


def _plot_method(config: dict) -> str | None:
    """Map full-study hybrids onto the three first-pass comparison series."""
    method = str(config["method"])
    if method == "mercury_cscg_hybrid":
        return "mercury"
    if method == "tmgwr_cscg_hybrid":
        return "sensory"
    if method == "cscg" and str(config.get("cscg_structural_preset") or "vanilla_cscg") != "vanilla_cscg":
        return None
    return method if method in METHODS else None


def _method_by_run(study_root: Path) -> dict[str, str | None]:
    return {
        run_root.name: _plot_method(_run_config(run_root))
        for run_root in study_root.iterdir()
        if (run_root / "run_config.json").exists()
    }


def _ci95(values: np.ndarray) -> float:
    return float(t.ppf(0.975, values.size - 1) * values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0


def _configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#374151",
            "axes.labelcolor": "#111827",
            "xtick.color": "#111827",
            "ytick.color": "#111827",
        }
    )


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    for path in paths:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return paths


def _training_summary(frame: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for method in METHODS:
        for level in LEVEL_ORDER:
            values = np.asarray(
                frame.filter((pl.col("method") == method) & (pl.col("level") == level))["comp_train_wall_time_seconds"],
                dtype=float,
            )
            rows.append(
                {
                    "method": method,
                    "method_label": LABELS[method],
                    "level": int(level),
                    "environment": LEVEL_LABELS[level],
                    "n": int(values.size),
                    "mean_seconds": float(values.mean()),
                    "sd_seconds": float(values.std(ddof=1)),
                    "ci95_seconds": _ci95(values),
                }
            )
    return pl.DataFrame(rows)


def _plot_training_time(
    frame: pl.DataFrame,
    summary: pl.DataFrame,
    output_dir: Path,
    *,
    compact: bool = False,
    full_study: bool = False,
) -> list[Path]:
    _configure_style()
    levels = [level for level in LEVEL_ORDER if level in set(frame["level"].unique().to_list())]
    x_base = np.arange(len(levels), dtype=float)
    fig, axis = plt.subplots(figsize=(1.78, 1.82) if compact else (7.2, 4.15), constrained_layout=True)
    for method in METHODS:
        color = COLORS[method]
        for index, level in enumerate(levels):
            values = np.asarray(
                frame.filter((pl.col("method") == method) & (pl.col("level") == level)).sort("seed")[
                    "comp_train_wall_time_seconds"
                ],
                dtype=float,
            )
            center = x_base[index] + OFFSETS[method]
            jitter = np.linspace(-0.055, 0.055, values.size)
            axis.scatter(
                center + jitter,
                values,
                s=7 if compact else 11,
                color=color,
                alpha=0.18,
                linewidths=0,
                zorder=1,
            )
            row = summary.filter((pl.col("method") == method) & (pl.col("level") == level)).row(0, named=True)
            axis.errorbar(
                center,
                row["mean_seconds"],
                yerr=row["ci95_seconds"],
                fmt="D",
                color=color,
                markeredgecolor="white",
                markeredgewidth=0.7,
                markersize=4.5 if compact else 6,
                elinewidth=1.6,
                capsize=3,
                zorder=3,
            )
    handles = [
        plt.Line2D(
            [],
            [],
            marker="D",
            linestyle="",
            color=COLORS[m],
            label="CSCG" if compact and m == "cscg" else LABELS[m],
            markersize=4.5 if compact else 6,
        )
        for m in METHODS
    ]
    axis.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.23 if compact else 1.13),
        ncol=2 if compact else 3,
        frameon=False,
        fontsize=5.0 if compact else None,
        handletextpad=0.2 if compact else 0.3,
        columnspacing=0.45 if compact else 0.8,
        labelspacing=0.2 if compact else 0.5,
    )
    axis.set_xticks(x_base, [LEVEL_LABELS[level] for level in levels])
    axis.set_ylabel(
        "Time (s)"
        if compact
        else ("Measured training wall time (s)" if full_study else "Training time for 2,000 observations (s)")
    )
    axis.set_xlabel("" if compact else "Environment")
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    if compact:
        axis.tick_params(labelsize=5.5, length=2.2, width=0.6)
        axis.yaxis.label.set_size(6.0)
    else:
        axis.text(
            0.0,
            -0.22,
            "Points are seeds; diamonds show means and bars show 95% confidence intervals.",
            transform=axis.transAxes,
            color="#4B5563",
            fontsize=8.5,
        )
    stem = "first_pass_training_time_compact" if compact else "first_pass_training_time"
    return _save(fig, output_dir, stem)


def _run_config(run_root: Path) -> dict:
    return json.loads((run_root / "run_config.json").read_text(encoding="utf-8"))["cli_args"]


def _mercury_diagnostics(study_root: Path) -> pl.DataFrame:
    rows = []
    for run_root in sorted(path for path in study_root.iterdir() if (path / "run_config.json").exists()):
        config = _run_config(run_root)
        if _plot_method(config) != "mercury":
            continue
        source_root = (
            run_root / "bundles" / "mercury"
            if (run_root / "bundles" / "mercury").exists()
            else run_root / "bundles" / "mercury_cscg_hybrid" / "mercury"
        )
        states_path = next(source_root.rglob("*_states.parquet"))
        final = pl.read_parquet(states_path).tail(1).to_dicts()[0]
        rows.append(
            {
                "run_id": run_root.name,
                "level": int(config["level"]),
                "seed": int(config["seed"]),
                "live_timesteps": int(config["num_steps"]),
                "refinement_events": int(final["ambiguity_response_cumulative"]),
                "replayed_timesteps": int(final["replay_inference_cumulative"]),
                "replay_percent": 100.0 * float(final["replay_inference_cumulative"]) / float(config["num_steps"]),
            }
        )
    return pl.DataFrame(rows)


def _plot_replay(
    diagnostics: pl.DataFrame,
    output_dir: Path,
    *,
    compact: bool = False,
) -> tuple[list[Path], pl.DataFrame]:
    _configure_style()
    levels = [level for level in LEVEL_ORDER if level in set(diagnostics["level"].unique().to_list())]
    rows = []
    fig, axis = plt.subplots(figsize=(1.78, 1.68) if compact else (6.6, 3.9), constrained_layout=True)
    for index, level in enumerate(levels):
        group = diagnostics.filter(pl.col("level") == level).sort("seed")
        values = np.asarray(group["replay_percent"], dtype=float)
        x = float(index)
        jitter = np.linspace(-0.09, 0.09, values.size)
        axis.scatter(
            x + jitter,
            values,
            s=9 if compact else 18,
            color=COLORS["mercury"],
            alpha=0.22,
            linewidths=0,
        )
        mean = float(values.mean())
        ci95 = _ci95(values)
        axis.errorbar(
            x,
            mean,
            yerr=ci95,
            fmt="D",
            color=COLORS["mercury"],
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=4.8 if compact else 6.5,
            elinewidth=1.7,
            capsize=3,
            zorder=3,
        )
        rows.append(
            {
                "level": int(level),
                "environment": LEVEL_LABELS[level],
                "n": int(values.size),
                "mean_replayed_timesteps": float(group["replayed_timesteps"].mean()),
                "sd_replayed_timesteps": float(group["replayed_timesteps"].std()),
                "mean_replay_percent": mean,
                "sd_replay_percent": float(values.std(ddof=1)),
                "ci95_replay_percent": ci95,
                "mean_refinement_events": float(group["refinement_events"].mean()),
                "max_refinement_events": int(group["refinement_events"].max()),
            }
        )
    axis.set_xticks(np.arange(len(levels)), [LEVEL_LABELS[level] for level in levels])
    axis.set_ylabel("Replay (%)" if compact else "Replayed timesteps / live timesteps (%)")
    axis.set_xlabel("" if compact else "Environment")
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    if compact:
        axis.tick_params(labelsize=5.5, length=2.2, width=0.6)
        axis.yaxis.label.set_size(6.0)
    else:
        axis.text(
            0.0,
            -0.23,
            "Each run contains 2,000 live observations. Points are seeds; diamonds show means and 95% CIs.",
            transform=axis.transAxes,
            color="#4B5563",
            fontsize=8.5,
        )
    stem = "mercury_replay_burden_compact" if compact else "mercury_replay_burden"
    return _save(fig, output_dir, stem), pl.DataFrame(rows)


def _graph_size(graph_path: Path) -> tuple[int, int]:
    graph = Graph.from_npz(str(graph_path))
    return int(graph.n), int(np.count_nonzero(graph.adj))


def _representation_rows(study_root: Path) -> pl.DataFrame:
    rows = []
    for run_root in sorted(path for path in study_root.iterdir() if (path / "run_config.json").exists()):
        config = _run_config(run_root)
        method = _plot_method(config)
        if method is None:
            continue
        if method == "mercury":
            source_root = (
                run_root / "bundles" / "mercury"
                if (run_root / "bundles" / "mercury").exists()
                else run_root / "bundles" / "mercury_cscg_hybrid" / "mercury"
            )
            states_path = next(source_root.rglob("*_states.parquet"))
            graph_path = next(source_root.rglob("*_latent_graph.npz"))
            final_nodes = int(pl.read_parquet(states_path, columns=["latent_node_count"])["latent_node_count"][-1])
            _, directed_edges = _graph_size(graph_path)
        elif method == "sensory":
            source_root = (
                run_root / "bundles" / "sensory"
                if (run_root / "bundles" / "sensory").exists()
                else run_root / "bundles" / "tmgwr_cscg_hybrid" / "sensory"
            )
            states_path = next(source_root.rglob("*_states.parquet"))
            graph_path = next(source_root.rglob("*_sensory_graph.npz"))
            final_nodes = int(pl.read_parquet(states_path, columns=["sensory_node_count"])["sensory_node_count"][-1])
            _, directed_edges = _graph_size(graph_path)
        elif method == "cscg":
            states_path = next((run_root / "bundles" / "cscg").rglob("*_states.parquet"))
            final_nodes = int(pl.read_parquet(states_path, columns=["latent_node_count"])["latent_node_count"][-1])
            directed_edges = None
        else:
            continue
        rows.append(
            {
                "method": method,
                "method_label": LABELS[method],
                "level": int(config["level"]),
                "environment": LEVEL_LABELS[int(config["level"])],
                "seed": int(config["seed"]),
                "final_nodes_or_allocated_clones": final_nodes,
                "directed_edges": directed_edges,
            }
        )
    return pl.DataFrame(rows)


def _representation_summary(rows: pl.DataFrame) -> pl.DataFrame:
    return (
        rows.group_by(["method", "method_label", "level", "environment"])
        .agg(
            pl.len().alias("n"),
            pl.col("final_nodes_or_allocated_clones").mean().alias("nodes_or_clones_mean"),
            pl.col("final_nodes_or_allocated_clones").std().alias("nodes_or_clones_sd"),
            pl.col("directed_edges").mean().alias("directed_edges_mean"),
            pl.col("directed_edges").std().alias("directed_edges_sd"),
        )
        .with_columns(
            pl.col("method").replace_strict({m: i for i, m in enumerate(METHODS)}).alias("method_order"),
            pl.col("level").replace_strict({level: i for i, level in enumerate(LEVEL_ORDER)}).alias("level_order"),
        )
        .sort(["level_order", "method_order"])
        .drop(["level_order", "method_order"])
    )


def _write_representation_markdown(summary: pl.DataFrame, path: Path) -> None:
    lines = [
        "| Environment | Method | Final nodes / allocated clones | Directed graph edges |",
        "|---|---|---:|---:|",
    ]
    for row in summary.iter_rows(named=True):
        nodes = f"{row['nodes_or_clones_mean']:.1f} ± {row['nodes_or_clones_sd']:.1f}"
        edges = "—" if row["directed_edges_mean"] is None else f"{row['directed_edges_mean']:.1f} ± {row['directed_edges_sd']:.1f}"
        lines.append(f"| {row['environment']} | {row['method_label']} | {nodes} | {edges} |")
    lines.extend(
        [
            "",
            "Values are mean ± SD across 30 seeds after 2,000 observations.",
            "For CSCG, the reported representation size is allocated clone states; graph-edge counts are not directly comparable to MERCURY/TMGWR graph edges.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_measurement_notes(study_root: Path, path: Path, *, full_study: bool = False) -> None:
    artifact_path = next(study_root.glob("*/metrics/*/*_computational.json"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    runtime = artifact["runtime_metadata"]
    thread_env = runtime.get("thread_env") or {}
    threads = ", ".join(f"{key}={value}" for key, value in thread_env.items())
    lines = [
        "# Main-study computational profiling notes" if full_study else "# First-pass computational profiling notes",
        "",
        (
            "- Scope: full main-study training; 30 seeds per method and environment. The plot compares the two "
            "initialised CSCG hybrids with the standard B=40 CSCG and excludes the three oracle controls."
            if full_study
            else "- Scope: one 2,000-observation pass; 30 seeds per method and environment."
        ),
        f"- Platform: {runtime.get('platform')}; {runtime.get('cpu_count')} logical CPUs; Python {runtime.get('python_version')}.",
        f"- Thread settings: {threads}.",
        (
            "- Measured training workload: 2,002,000 steps for MERCURY→CSCG and TMGWR→CSCG "
            "(2,000 source observations plus 2,000,000 CSCG steps); 2,000,000 steps for standard CSCG."
            if full_study
            else "- Runs were scheduled sequentially (`max_workers=1`) to avoid inter-run CPU contention."
        ),
        (
            "- Computational timing used one repetition. Paper-precision and structural outputs were enabled in the main study."
            if full_study
            else "- Paper-precision and structural-evaluation passes were disabled during profiling."
        ),
        "- The implementations and workloads differ and are not asserted to be equally optimised. Wall-clock comparisons are descriptive implementation measurements, not formal complexity estimates.",
        "- CSCG JIT warm-up and native evaluation are recorded separately from its training time. MERCURY and TMGWR timings include their method-specific setup and artifact persistence before return from training.",
        "- Replay burden is `replay_inference_cumulative / 2000`; it counts replay inference steps, not live observations.",
        "- Directed edge counts are reported only for MERCURY/TMGWR graph representations. CSCG allocated clone count is shown instead because its transition tensor is not directly comparable to graph-edge counts.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-root",
        type=Path,
        default=Path("results/current/studies/main_study_first_pass_computational"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/current/notebook_plots/main_study_first_pass_computational"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pl.read_parquet(args.study_root / "study_summary.parquet")
    method_by_run = _method_by_run(args.study_root)
    full_study = any(
        _run_config(run_root).get("method") in {"mercury_cscg_hybrid", "tmgwr_cscg_hybrid"}
        for run_root in args.study_root.iterdir()
        if (run_root / "run_config.json").exists()
    )
    if full_study:
        LABELS.update({"mercury": "MERCURY→CSCG", "sensory": "TMGWR→CSCG"})
    frame = frame.with_columns(
        pl.col("run_id").replace_strict(method_by_run, default=None).alias("method")
    ).filter(pl.col("method").is_in(METHODS))
    training = _training_summary(frame)
    training_csv = args.output_dir / "first_pass_training_time_summary.csv"
    training.write_csv(training_csv)
    outputs = [*_plot_training_time(frame, training, args.output_dir, full_study=full_study), training_csv]
    outputs.extend(_plot_training_time(frame, training, args.output_dir, compact=True, full_study=full_study))

    diagnostics = _mercury_diagnostics(args.study_root)
    replay_paths, replay_summary = _plot_replay(diagnostics, args.output_dir)
    compact_replay_paths, _ = _plot_replay(diagnostics, args.output_dir, compact=True)
    replay_runs_csv = args.output_dir / "mercury_replay_burden_runs.csv"
    replay_summary_csv = args.output_dir / "mercury_replay_burden_summary.csv"
    diagnostics.write_csv(replay_runs_csv)
    replay_summary.write_csv(replay_summary_csv)
    outputs.extend([*replay_paths, *compact_replay_paths, replay_runs_csv, replay_summary_csv])

    representation = _representation_summary(_representation_rows(args.study_root))
    representation_csv = args.output_dir / "representation_size_summary.csv"
    representation_md = args.output_dir / "representation_size_table.md"
    representation.write_csv(representation_csv)
    _write_representation_markdown(representation, representation_md)
    outputs.extend([representation_csv, representation_md])

    notes_path = args.output_dir / "measurement_notes.md"
    _write_measurement_notes(args.study_root, notes_path, full_study=full_study)
    outputs.append(notes_path)

    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
