"""Offline JEPA-CSCG: full-sequence EMA teacher, patch-smoothed student (research pilot).

PYTHONPATH=.:src .venv/bin/python scripts/run_cscg_offline_jepa_experiment.py

Arms: offline (full-sequence EM), patch (EM on random-offset patch tilings), ema_control
(patch + EMA teacher, lambda = 0), jepa (patch + predictive clone-marginal KL to the
full-sequence EMA teacher). Data loading, held-out walks and canonical evaluation are
imported unchanged from the online distillation runner. Only the student is the
primary outcome; EMA teachers are evaluated at the final pass as a diagnostic.
Units: E data passes (one random-offset tiling each) x J coupled iterations per pass;
every arm receives E*J student EM updates and E*J*transitions transition presentations.
Evaluation records store the data-pass index in their step/observed_samples field.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from mercury_runs.algorithms.cscg.distillation import row_normalize
from mercury_runs.algorithms.cscg.offline_jepa import METHODS, OfflineJepaConfig, train_offline
from mercury_runs.application.evaluation_protocol import default_canonical_v4_protocol
from mercury_runs.application.evaluation_service import build_empirical_ground_truth, generate_and_cache_walks
from mercury_runs.infrastructure.paper_precision import (
    exact_cartesian_reference_positions, exact_valid_sensorimotor_transitions, infer_num_actions_from_valid_transitions,
)
from scripts.analysis.cscg_distillation_canonical_report import canonical_metrics
from scripts.run_cscg_distillation_experiment import (
    ROOT, evaluate_snapshot, indexed_evaluation, load_training, write_json,
)

LABELS = dict(offline="Offline CSCG (full-sequence EM)", patch="Patch CSCG", ema_control="EMA control",
              jepa="Offline JEPA-CSCG")
METRICS = dict(similarity="Action transition similarity (1 - TV)", nonself_similarity="No-self-loop similarity",
               edge_f1_auc="Action edge F1 (threshold AUC)", purity="Clone purity",
               concentration="Latent concentration (canonical recall)", nll_bits="Held-out NLL (bits/symbol)")


BUDGET = ("tilings", "student_em_iterations", "msteps", "teacher_updates", "teacher_refreshes",
          "teacher_inference_passes", "transition_presentations", "student_patch_calls")


def checkpoint_schedule(passes):
    grid = [0, 1, 2, 3, 5, 7, 10, 15, 20, 25, *range(30, passes + 1, 10), passes]
    return sorted({e for e in grid if e <= passes})


def check_budget(method, cost, config, transitions, n_trajectories):
    E, J = config.passes, config.coupled_iterations_per_pass
    expected = dict(passes=E, student_em_iterations=E * J, msteps=E * J,
                    transition_presentations=E * J * transitions,
                    tilings=0 if method == "offline" else E,
                    teacher_updates=E * J if method in ("ema_control", "jepa") else 0,
                    teacher_refreshes=E * J if method == "jepa" else 0,
                    teacher_inference_passes=E * J * n_trajectories if method == "jepa" else 0)
    wrong = {k: (cost[k], v) for k, v in expected.items() if cost[k] != v}
    if wrong:
        raise AssertionError(f"{method} budget mismatch (actual, expected): {wrong}")


def evaluate(T, n_clones, context, method, pass_index, destination):
    metrics = evaluate_snapshot(T, n_clones, context["indexed"], context["gt"], context["protocol"],
                                context["level"], method, pass_index, destination)
    canonical = canonical_metrics(json.loads(destination.read_text()))
    return dict(similarity=1 - metrics["action_tv"], nonself_similarity=1 - metrics["nonself_tv"],
                edge_f1_auc=canonical["action_edge_f1_auc"],
                nonself_edge_f1_auc=canonical["action_no_self_loop_edge_f1_auc"],
                purity=metrics["purity"], concentration=metrics["node_recall"], nll_bits=metrics["nll_bits"],
                unmatched_mass=metrics["unmatched_mass"], state_coverage=metrics["state_coverage"],
                clone_utilisation=metrics["clone_utilisation"])


def run_arm(output, method, config, context, checkpoints):
    folder = output / f"patch={config.patch_length}" / method if method != "offline" else output / "offline"
    complete = folder / "rows.json"
    if complete.exists():
        print(f"skip {folder.relative_to(output)}", flush=True)
        return json.loads(complete.read_text())
    print(f"train {folder.relative_to(output)}", flush=True)
    folder.mkdir(parents=True, exist_ok=True)
    metadata = {}
    result = train_offline(context["initial"], context["n_clones"], context["trajectories"], method, config,
                           context["seed"], set(checkpoints), callback=lambda e, T, meta: metadata.update({e: meta}))
    check_budget(method, result["cost"], config, sum(len(a) for _, a in context["trajectories"]),
                 len(context["trajectories"]))
    np.savez_compressed(folder / "snapshots.npz", passes=np.array(checkpoints),
                        T=np.stack([result["snapshots"][e] for e in checkpoints]),
                        teacher=np.stack([result["teachers"][e] for e in checkpoints]), n_clones=context["n_clones"])
    write_json(folder / "training.json", dict(diagnostics=result["diagnostics"], cost=result["cost"],
                                              config=asdict(config),
                                              checkpoint_cost={str(k): v for k, v in metadata.items()}))
    rows = []
    patch_length = None if method == "offline" else config.patch_length
    for pass_index in checkpoints:
        scores = evaluate(result["snapshots"][pass_index], context["n_clones"], context, method, pass_index,
                          folder / "evaluations" / f"pass={pass_index:04d}.json")
        rows.append(dict(method=method, role="student", patch_length=patch_length, pass_index=pass_index,
                         **metadata[pass_index], **scores))
    if method in ("ema_control", "jepa"):
        pass_index = checkpoints[-1]
        scores = evaluate(result["teachers"][pass_index], context["n_clones"], context, method + "_teacher", pass_index,
                          folder / "evaluations" / f"teacher_pass={pass_index:04d}.json")
        rows.append(dict(method=method, role="teacher", patch_length=patch_length, pass_index=pass_index,
                         **metadata[pass_index], **scores))
    write_json(complete, rows)
    final = rows[-2] if rows[-1]["role"] == "teacher" else rows[-1]
    print(f"done {method}: similarity={final['similarity']:.4f} F1auc={final['edge_f1_auc']:.4f} "
          f"student={final['student_seconds']:.1f}s teacher={final['teacher_seconds']:.1f}s", flush=True)
    return rows


def report(output, rows, manifest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(rows)
    df.to_csv(output / "learning_curves.csv", index=False)
    students = df[df.role == "student"]
    lengths = sorted(students.patch_length.dropna().unique())
    offline = students[students.method == "offline"].sort_values("pass_index")
    fig, axes = plt.subplots(len(lengths), len(METRICS), figsize=(4 * len(METRICS), 3.4 * len(lengths)), squeeze=False)
    for i, length in enumerate(lengths):
        for j, (metric, label) in enumerate(METRICS.items()):
            ax = axes[i, j]
            ax.plot(offline.pass_index, offline[metric], color="black", lw=1.6, label=LABELS["offline"])
            for method, style in [("patch", "-"), ("ema_control", ":"), ("jepa", "-")]:
                part = students[(students.method == method) & (students.patch_length == length)].sort_values("pass_index")
                ax.plot(part.pass_index, part[metric], style, lw=1.4, label=LABELS[method])
            ax.set(title=f"L={int(length)}: {label}", xlabel=f"Data pass (x J={manifest['config']['coupled_iterations_per_pass']} EM updates)")
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=150)
    fig.savefig(output / "learning_curves.pdf")
    plt.close(fig)

    final_pass = int(df.pass_index.max())
    keys = ["method", "role", "patch_length"]
    area = students.sort_values("pass_index").groupby(keys, dropna=False).apply(
        lambda part: np.trapezoid(part.similarity, part.pass_index) / final_pass, include_groups=False)
    final = df[df.pass_index == final_pass].merge(area.rename("time_mean_similarity").reset_index(), on=keys, how="left")
    final = final.sort_values(["patch_length", "method"], na_position="first")
    final.to_csv(output / "final_summary.csv", index=False)
    config = manifest["config"]
    E, J = config["passes"], config["coupled_iterations_per_pass"]
    lines = ["# Offline JEPA-CSCG pilot", "",
             f"Level {manifest['level']}, seed {manifest['seed']}, {manifest['observations']:,} training observations "
             f"({manifest['transitions']:,} transitions), {manifest['clones_per_observation']} clones per symbol, "
             f"pseudocount {config['pseudocount']}.",
             f"E = {E} data passes (one random-offset tiling each) x J = {J} coupled iterations per pass: every arm "
             f"receives {E * J} student EM updates (one global M-step each) and {E * J * manifest['transitions']:,} "
             "transition presentations. Patch arms share identical tilings.",
             f"JEPA weight lambda = {config['jepa_weight']} (per-transition KL weight lambda/(L-1)); EMA beta = "
             f"{config['ema']} per coupled iteration; JEPA targets are refreshed from the current teacher every iteration.",
             "Single seed: implementation pilot, not evidence of a general effect.", "",
             f"## Student endpoints after pass {final_pass} (higher is better except NLL)", "",
             "| Patch L | Model | Similarity | No-self sim. | Edge F1 AUC | Purity | Concentration | NLL bits | Mean similarity over passes |",
             "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in final[final.role == "student"].itertuples():
        length = "full" if pd.isna(r.patch_length) else int(r.patch_length)
        lines.append(f"| {length} | {LABELS[r.method]} | {r.similarity:.4f} | {r.nonself_similarity:.4f} | "
                     f"{r.edge_f1_auc:.4f} | {r.purity:.4f} | {r.concentration:.4f} | {r.nll_bits:.4f} | "
                     f"{r.time_mean_similarity:.4f} |")
    lines += ["", "## EMA teacher diagnostic (not the primary outcome)", "",
              "| Patch L | Arm | Similarity | Edge F1 AUC | NLL bits |", "| ---: | --- | ---: | ---: | ---: |"]
    for r in final[final.role == "teacher"].itertuples():
        lines.append(f"| {int(r.patch_length)} | {LABELS[r.method]} | {r.similarity:.4f} | {r.edge_f1_auc:.4f} | {r.nll_bits:.4f} |")
    lines += ["", "## Budgets and cost (measured counters; the runner asserts the required ones)", "",
              "Teacher time is full-sequence smoothing plus target construction, reported separately from student time.", "",
              "| Patch L | Model | Tilings | Student EM iterations | M-steps | EMA updates | Teacher inference passes | "
              "Transition presentations | Student patch calls | Student s | Teacher s |",
              "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in final[final.role == "student"].itertuples():
        length = "full" if pd.isna(r.patch_length) else int(r.patch_length)
        lines.append(f"| {length} | {LABELS[r.method]} | {r.tilings} | {r.student_em_iterations} | {r.msteps} | "
                     f"{r.teacher_updates} | {r.teacher_inference_passes} | {r.transition_presentations:,} | "
                     f"{r.student_patch_calls:,} | {r.student_seconds:.1f} | {r.teacher_seconds:.1f} |")
    lines += ["", "The EMA control student is identical to patch CSCG by construction (lambda = 0 leaves the student "
              "independent of its teacher); the runner asserts this. Canonical metrics use causal carried-filter "
              "decoding on shared held-out walks, without teacher information."]
    (output / "results.md").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--level", type=int, default=22)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--rand-prob", type=float, default=.3)
    p.add_argument("--clones", type=int, default=10)
    p.add_argument("--pseudocount", type=float, default=.002)
    p.add_argument("--passes", type=int, default=100, help="E: data passes, one tiling each")
    p.add_argument("--coupled-iterations", type=int, default=1, help="J: coupled iterations per pass")
    p.add_argument("--patch-lengths", nargs="+", type=int, default=[40])
    p.add_argument("--jepa-weight", type=float, default=.5, help="lambda in [0, 1]")
    p.add_argument("--ema", type=float, default=.9, help="teacher EMA beta per coupled iteration")
    p.add_argument("--walks", type=int, default=10)
    p.add_argument("--walk-length", type=int, default=500)
    p.add_argument("--output", type=Path)
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args()
    base = OfflineJepaConfig(patch_length=args.patch_lengths[0], jepa_weight=args.jepa_weight, ema=args.ema,
                             pseudocount=args.pseudocount, passes=args.passes,
                             coupled_iterations_per_pass=args.coupled_iterations)
    output = (args.output or ROOT / f"results/diagnostics/cscg_offline_jepa_level{args.level}_seed{args.seed}_J{args.coupled_iterations}_v2").resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = checkpoint_schedule(args.passes)
    shared_config = {k: v for k, v in asdict(base).items() if k != "patch_length"}  # L varies within a directory
    manifest = dict(config=shared_config, patch_lengths=args.patch_lengths, level=args.level, seed=args.seed,
                    observations=args.steps, transitions=args.steps - 1, rand_prob=args.rand_prob, clones_per_observation=args.clones,
                    checkpoint_passes=checkpoints, evaluation_walks=args.walks, walk_length=args.walk_length,
                    evaluation_seed=10000, methods=list(METHODS), schema_version=2)
    if args.report_only:
        rows = [r for path in sorted(output.glob("**/rows.json")) for r in json.loads(path.read_text())]
        report(output, rows, json.loads((output / "experiment.json").read_text()))
        return
    implementation = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in [
        "src/mercury_runs/algorithms/cscg/offline_jepa.py", "src/mercury_runs/algorithms/cscg/distillation.py",
        "scripts/run_cscg_offline_jepa_experiment.py"]}
    for name, value in [("implementation.json", implementation)]:
        path = output / name
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError("Implementation differs from the saved experiment; use a fresh output directory")
        write_json(path, value)
    manifest_path = output / "experiment.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if {k: v for k, v in saved.items() if k != "patch_lengths"} != {k: v for k, v in manifest.items() if k != "patch_lengths"}:
            raise ValueError("Output directory contains a different experiment")
        manifest["patch_lengths"] = sorted(set(saved["patch_lengths"]) | set(args.patch_lengths))
    write_json(manifest_path, manifest)

    protocol = default_canonical_v4_protocol(
        level=args.level, sensor="cardinal distance", sensor_range=1, rand_prob=args.rand_prob,
        evaluation_seed=10000, number_of_walks=args.walks, walk_length=args.walk_length,
        valid_trajectories_only=False, edge_thresholds=[1e-6])
    protocol.decoder = "causal_clone_filter_argmax"
    protocol.decoder_configuration_source = "shared_experiment_filter"
    protocol.inference_state_initialisation = "uniform_clone_prior"
    walks = generate_and_cache_walks(study_root=output, protocol=protocol, level=args.level)
    gt = build_empirical_ground_truth(
        walk_data=walks, n_true=len(exact_cartesian_reference_positions(args.level)),
        n_actions=infer_num_actions_from_valid_transitions(exact_valid_sensorimotor_transitions(level_index=args.level)),
        level=args.level)
    data = load_training(args.level, args.seed, args.rand_prob, args.steps)
    n_clones = np.full(len(data["vocabulary"]), args.clones, dtype=int)
    n_states = int(n_clones.sum())
    # Same initialisation rule as the online distillation runner, shared by every arm.
    initial = row_normalize(np.random.RandomState(args.seed).rand(len(data["action_vocabulary"]), n_states, n_states)
                            + args.pseudocount)
    context = dict(level=args.level, seed=args.seed, gt=gt, protocol=protocol, n_clones=n_clones, initial=initial,
                   trajectories=[(data["x"], data["actions"])],
                   indexed=indexed_evaluation(walks, data["vocabulary"], data["action_vocabulary"]))
    write_json(output / "provenance.json", dict({k: data[k] for k in ["path", "sha256", "collisions"]},
                                                initial_T_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
                                                evaluation_protocol=protocol.protocol_id(), ground_truth_checksum=gt["checksum"]))
    rows = run_arm(output, "offline", base, context, checkpoints)
    for length in args.patch_lengths:
        config = replace(base, patch_length=length)
        arms = {method: run_arm(output, method, config, context, checkpoints) for method in ("patch", "ema_control", "jepa")}
        for a, b in zip(arms["patch"], arms["ema_control"]):
            if b["role"] == "student" and any(a[k] != b[k] for k in METRICS):
                raise AssertionError("EMA control student must equal patch CSCG")
        rows += [r for part in arms.values() for r in part]
    all_rows = [r for path in sorted(output.glob("**/rows.json")) for r in json.loads(path.read_text())]
    report(output, all_rows, manifest)
    print(f"Results: {output / 'results.md'}", flush=True)


if __name__ == "__main__":
    main()
