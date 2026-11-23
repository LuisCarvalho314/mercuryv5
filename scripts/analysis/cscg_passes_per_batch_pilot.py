"""Summarize the matched CSCG passes-per-batch screening study."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


STUDY = Path("results/current/studies/cscg_passes_per_batch_pilot")
METRICS = (
    ("latent_precision", "Latent purity", "↑"),
    ("latent_recall", "Concentration", "↑"),
    ("tv_similarity", "Action TV similarity", "↑"),
    ("tv_non_self_loop_similarity", "Action TV, no self loops", "↑"),
    ("eval_nll", "Eval NLL (bits/symbol)", "↓"),
    ("train_seconds", "Training time (s)", "↓"),
)


def load_rows() -> list[dict]:
    rows = []
    for line in (STUDY / "study_manifest.jsonl").read_text().splitlines():
        config = json.loads(line)["config"]
        run_id = config["requested_run_id"]
        root = STUDY / run_id
        records = [
            path for path in root.glob("evaluations/*/snapshots/cscg/step=00002000.json")
            if json.loads(path.read_text())["protocol"]["evaluation_seed"] == 0
        ]
        if len(records) != 1:
            raise ValueError(f"Expected one final evaluation for {run_id}; found {len(records)}")
        record = json.loads(records[0].read_text())
        computational = json.loads((root / "metrics" / "cscg" / f"{run_id}_computational.json").read_text())
        if record["observed_samples"] != 2000 or record["snapshot_step"] != 2000:
            raise ValueError(f"Unexpected sample budget for {run_id}")
        rows.append({
            "seed": int(config["seed"]),
            "passes": int(config["cscg_iterations_per_batch"]),
            "run_id": run_id,
            "walk_cache_id": record["walk_cache_id"],
            "latent_precision": record["state_representation"]["latent_precision"],
            "latent_recall": record["state_representation"]["node_recall"],
            "tv_similarity": record["action_conditioned"]["tv_similarity_occupancy"],
            "tv_non_self_loop_similarity": record["action_conditioned"]["tv_non_self_loop_similarity_occupancy"],
            "eval_nll": record["model_fit"]["bits_per_symbol"],
            "train_seconds": computational["metrics"]["train_wall_time_seconds"],
        })
    if {(row["seed"], row["passes"]) for row in rows} != {(s, p) for s in range(3) for p in (1, 2, 5)}:
        raise ValueError("Study does not contain the expected nine conditions")
    if len({row["walk_cache_id"] for row in rows}) != 1:
        raise ValueError("Evaluation walks differ between conditions")
    return sorted(rows, key=lambda row: (row["seed"], row["passes"]))


def write_report(rows: list[dict]) -> None:
    by_key = {(row["seed"], row["passes"]): row for row in rows}
    lines = [
        "# CSCG passes per batch pilot",
        "",
        "Level 21; seeds 0–2; batch size 40; 2,000 observed training samples; vanilla CSCG; 20 shared evaluation walks of length 1,000.",
        "",
        f"Walk cache: `{rows[0]['walk_cache_id']}`. Training times are the recorded training phase wall times; first-use compilation may be included. Each run used 50 batches, so 1/2/5 passes made 50/100/250 batch EM updates.",
        "",
        "| Seed | Passes | Purity | Δ | Concentration | Δ | Action TV sim. | Δ | No self loop sim. | Δ | Eval NLL | Δ | Train s | Δ |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        baseline = by_key[row["seed"], 1]
        cells = [str(row["seed"]), str(row["passes"])]
        for name, _, _ in METRICS:
            digits = 6 if name == "eval_nll" else 3
            cells += [f"{row[name]:.{digits}f}", f"{row[name] - baseline[name]:+.{digits}f}"]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Mean across three paired seeds", "", "| Passes | Purity | Concentration | Action TV sim. | No self loop sim. | Eval NLL | Train s |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for passes in (1, 2, 5):
        values = [f"{sum(by_key[seed, passes][name] for seed in range(3)) / 3:.{6 if name == 'eval_nll' else 3}f}" for name, _, _ in METRICS]
        lines.append("| " + " | ".join([str(passes), *values]) + " |")
    lines += [
        "",
        "## Reading",
        "",
        "Five passes improved concentration in two of three seeds, but reduced both action conditioned transition similarities in all three. Evaluation NLL changed by less than 0.00002 bits per symbol. Two passes gave no consistent representation gain. The recorded wall times vary more than the pass count predicts, so these single-run timings do not resolve runtime scaling; the EM update counts show the additional work exactly.",
    ]
    (STUDY / "pilot_results.md").write_text("\n".join(lines) + "\n")


def plot(rows: list[dict]) -> None:
    by_key = {(row["seed"], row["passes"]): row for row in rows}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, (name, label, direction) in zip(axes.flat, METRICS):
        for seed in range(3):
            ax.plot((1, 2, 5), [by_key[seed, p][name] for p in (1, 2, 5)], marker="o", alpha=0.55, label=f"Seed {seed}")
        ax.plot((1, 2, 5), [sum(by_key[s, p][name] for s in range(3)) / 3 for p in (1, 2, 5)], color="black", marker="s", linewidth=2, label="Mean")
        ax.set(title=f"{label} {direction}", xlabel="EM passes per batch", xticks=(1, 2, 5))
        ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(STUDY / "pilot_metrics.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, (name, label, direction) in zip(axes.flat, METRICS):
        for seed in range(3):
            baseline = by_key[seed, 1][name]
            ax.plot((1, 2, 5), [by_key[seed, p][name] - baseline for p in (1, 2, 5)], marker="o", alpha=0.6, label=f"Seed {seed}")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(title=f"Change in {label} {direction}", xlabel="EM passes per batch", xticks=(1, 2, 5))
        ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(STUDY / "pilot_paired_changes.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    data = load_rows()
    write_report(data)
    plot(data)
    print(STUDY / "pilot_results.md")
    print(STUDY / "pilot_metrics.png")
    print(STUDY / "pilot_paired_changes.png")
