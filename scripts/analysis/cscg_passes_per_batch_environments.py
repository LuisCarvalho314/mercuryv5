"""Compare CSCG passes-per-batch pilots across maze levels 13, 20, 21, and 22."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from cscg_passes_per_batch_pilot import METRICS


STUDIES = {
    13: Path("results/current/studies/cscg_passes_per_batch_other_levels"),
    20: Path("results/current/studies/cscg_passes_per_batch_other_levels"),
    21: Path("results/current/studies/cscg_passes_per_batch_pilot"),
    22: Path("results/current/studies/cscg_passes_per_batch_other_levels"),
}
OUTPUT = Path("results/current/studies/cscg_passes_per_batch_other_levels")
PASSES = (1, 2, 5)


def load_level(level: int) -> list[dict]:
    study = STUDIES[level]
    rows = []
    for line in (study / "study_manifest.jsonl").read_text().splitlines():
        config = json.loads(line)["config"]
        if int(config["level"]) != level:
            continue
        run_id = config["requested_run_id"]
        root = study / run_id
        records = [
            path for path in root.glob("evaluations/*/snapshots/cscg/step=00002000.json")
            if json.loads(path.read_text())["protocol"]["evaluation_seed"] == 0
        ]
        if len(records) != 1:
            raise ValueError(f"Expected one final evaluation for {run_id}; found {len(records)}")
        record = json.loads(records[0].read_text())
        timing = json.loads((root / "metrics" / "cscg" / f"{run_id}_computational.json").read_text())
        if record["observed_samples"] != 2000 or record["snapshot_step"] != 2000:
            raise ValueError(f"Unexpected sample budget for {run_id}")
        if (record["protocol"]["protocol_name"] != "canonical-v5"
                or record["protocol"]["number_of_walks"] != 20
                or record["protocol"]["walk_length"] != 1000):
            raise ValueError(f"Unexpected evaluation protocol for {run_id}")
        rows.append({
            "level": level,
            "seed": int(config["seed"]),
            "passes": int(config["cscg_iterations_per_batch"]),
            "run_id": run_id,
            "walk_cache_id": record["walk_cache_id"],
            "latent_precision": record["state_representation"]["latent_precision"],
            "latent_recall": record["state_representation"]["node_recall"],
            "tv_similarity": record["action_conditioned"]["tv_similarity_occupancy"],
            "tv_non_self_loop_similarity": record["action_conditioned"]["tv_non_self_loop_similarity_occupancy"],
            "eval_nll": record["model_fit"]["bits_per_symbol"],
            "train_seconds": timing["metrics"]["train_wall_time_seconds"],
        })
    if {(row["seed"], row["passes"]) for row in rows} != {(seed, passes) for seed in range(3) for passes in PASSES}:
        raise ValueError(f"Level {level} does not contain the expected nine conditions")
    if len({row["walk_cache_id"] for row in rows}) != 1:
        raise ValueError(f"Level {level} evaluation walks differ between conditions")
    return sorted(rows, key=lambda row: (row["seed"], row["passes"]))


def fmt(name: str, value: float, *, signed: bool = False) -> str:
    digits = 6 if name == "eval_nll" else 3
    return f"{value:{'+' if signed else ''}.{digits}f}"


def mean(rows: list[dict], passes: int, name: str) -> float:
    return sum(row[name] for row in rows if row["passes"] == passes) / 3


def write_report(data: dict[int, list[dict]]) -> None:
    lines = [
        "# CSCG passes per batch across four maze levels",
        "",
        "Vanilla CSCG; seeds 0–2 paired within each level; batch size 40; 2,000 observed training samples (50 batches); 1, 2, or 5 EM passes per batch. Each level used a common set of 20 evaluation walks of length 1,000. Evaluation used the canonical v5 protocol.",
        "",
        "Higher is better for purity, concentration, and transition similarities. Lower is better for evaluation NLL. Training seconds are single-run training phase wall times; first-use compilation and system noise may affect them. EM update counts are exactly 50, 100, and 250 for 1, 2, and 5 passes.",
        "",
        "## Level means",
        "",
        "| Level | Passes | Purity | Concentration | Action TV sim. | No self loop sim. | Eval NLL | Train s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for level, rows in data.items():
        for passes in PASSES:
            cells = [str(level), str(passes)] + [fmt(name, mean(rows, passes, name)) for name, _, _ in METRICS]
            lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Mean paired changes from one pass",
        "",
        "| Level | Passes | Δ purity | Δ concentration | Δ action TV sim. | Δ no self loop sim. | Δ eval NLL | Δ train s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for level, rows in data.items():
        for passes in (2, 5):
            cells = [str(level), str(passes)] + [
                fmt(name, mean(rows, passes, name) - mean(rows, 1, name), signed=True)
                for name, _, _ in METRICS
            ]
            lines.append("| " + " | ".join(cells) + " |")
    for level, rows in data.items():
        by_key = {(row["seed"], row["passes"]): row for row in rows}
        lines += [
            "",
            f"## Level {level}: paired seeds",
            "",
            f"Evaluation walk cache: `{rows[0]['walk_cache_id']}`.",
            "",
            "| Seed | Passes | Purity | Δ | Concentration | Δ | Action TV sim. | Δ | No self loop sim. | Δ | Eval NLL | Δ | Train s | Δ |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            baseline = by_key[row["seed"], 1]
            cells = [str(row["seed"]), str(row["passes"])]
            for name, _, _ in METRICS:
                cells += [fmt(name, row[name]), fmt(name, row[name] - baseline[name], signed=True)]
            lines.append("| " + " | ".join(cells) + " |")
    (OUTPUT / "pilot_environment_results.md").write_text("\n".join(lines) + "\n")


def plot(data: dict[int, list[dict]]) -> None:
    for level, rows in data.items():
        if level == 21:
            continue  # Existing level 21 seed plots remain in its original study.
        by_key = {(row["seed"], row["passes"]): row for row in rows}
        for paired, suffix in ((False, "metrics"), (True, "paired_changes")):
            fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
            for ax, (name, label, direction) in zip(axes.flat, METRICS):
                for seed in range(3):
                    baseline = by_key[seed, 1][name] if paired else 0
                    ax.plot(PASSES, [by_key[seed, p][name] - baseline for p in PASSES], marker="o", alpha=0.55, label=f"Seed {seed}")
                if paired:
                    ax.axhline(0, color="black", linewidth=0.8)
                else:
                    ax.plot(PASSES, [mean(rows, p, name) for p in PASSES], color="black", marker="s", linewidth=2, label="Mean")
                ax.set(title=f"{label} {direction}", xlabel="EM passes per batch", xticks=PASSES)
                ax.grid(alpha=0.2)
            axes[0, 0].legend(frameon=False, fontsize=8)
            fig.suptitle(f"Maze level {level}")
            fig.savefig(OUTPUT / f"pilot_level{level}_{suffix}.png", dpi=180)
            plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, (name, label, direction) in zip(axes.flat, METRICS):
        for level, rows in data.items():
            baseline = mean(rows, 1, name)
            ax.plot(PASSES, [mean(rows, p, name) - baseline for p in PASSES], marker="o", label=f"Level {level}")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(title=f"Mean change in {label} {direction}", xlabel="EM passes per batch", xticks=PASSES)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(OUTPUT / "pilot_environment_paired_changes.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    level_data = {level: load_level(level) for level in STUDIES}
    write_report(level_data)
    plot(level_data)
    print(OUTPUT / "pilot_environment_results.md")
    print(OUTPUT / "pilot_environment_paired_changes.png")
