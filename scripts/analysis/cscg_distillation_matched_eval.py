"""Canonical online evaluation of online JEPA-CSCG snapshots with a training-matched decoder.

Evaluation-v5 (scripts/analysis/evaluate_v5.py) decodes each frozen CSCG with the
inference its training used. This script applies the same principle to the
online distillation runs: walks, empirical ground truth, metric computation
(`_evaluate_metrics`) and record format are canonical evaluation-v5; only the
decoder is the arm's own training inference.

Training inference is a carried causal filter f_t = P(z_t | x_<=t) plus fixed-lag
smoothing. The decoded state is argmax P(z_t | x_<=t+L) with
  L = 1      for short EM, the replay-NLL control and JEPA (their data/source
             posterior conditions z_u on x_{u+1}: pair counts / smoothed source),
  L = W - 1  for long EM (window W; a node's last estimate in training is at the
             start of the window, smoothed from its filtered boundary belief).
Lag 0 (pure filter) is also written as a decoder-sensitivity check. NLL is the
carried filter's causal predictive loss -log2 p(x_t | x_<t), as in training.

  PYTHONPATH=.:src .venv/bin/python scripts/analysis/cscg_distillation_matched_eval.py evaluate \
      --run <experiment dir> [--methods ...] [--steps final|all] [--workers N]
  PYTHONPATH=.:src .venv/bin/python scripts/analysis/cscg_distillation_matched_eval.py report \
      --condition "W40=<dir>" --condition "W100=<dir>" --output DIR --title T
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numba
import numpy as np
import pandas as pd

from mercury_runs.application.evaluation_service import (
    _load_walk_cache, _walk_cache_filename, build_empirical_ground_truth, generate_and_cache_walks,
)
from mercury_runs.algorithms.cscg.prepare import map_to_indices
from mercury_runs.infrastructure.paper_precision import (
    exact_cartesian_reference_positions, exact_valid_sensorimotor_transitions, infer_num_actions_from_valid_transitions,
)
from scripts.analysis.cscg_distillation_canonical_report import LABELS, METRICS, canonical_metrics
from scripts.analysis.evaluate_v5 import EvaluationV5Protocol, _evaluate_metrics

ROOT = Path(__file__).resolve().parents[2]
SHORT_LAG = ("short", "nll_control", "predictive", "posterior")
OUTPUT = "evaluation_v5_training_matched"
CANONICAL_BUDGET = (100, 10_000, 0.1)   # evaluation-v5 walks, steps per walk, random-action probability


def output_name(walks=CANONICAL_BUDGET[0], walk_length=CANONICAL_BUDGET[1], rand_prob=CANONICAL_BUDGET[2]):
    """Canonical setting -> OUTPUT; any other setting gets its own directory, never mixed."""
    name = OUTPUT if (walks, walk_length) == CANONICAL_BUDGET[:2] else f"{OUTPUT}_walks{walks}x{walk_length}"
    return name if rand_prob == CANONICAL_BUDGET[2] else f"{name}_p{rand_prob:g}"


@dataclass
class TrainingMatchedOnlineProtocol(EvaluationV5Protocol):
    """Evaluation-v5 walks and metrics; decoder = the online run's own training inference."""

    schema_version: str = "evaluation_v5_training_matched_online_v1"
    protocol_name: str = "evaluation-v5-online-matched-v1"
    decoder: str = "carried_filter_fixed_lag_marginal_argmax"
    decoder_configuration_source: str = "online_training_inference"
    online_inference_mode: str = "frozen_carried_filter_fixed_lag"
    inference_state_initialisation: str = "uniform_clone_prior"
    inference_state_reset_policy: str = "reset_per_walk"
    batch_size_source: str = "not_applicable"
    batch_overlap_samples: int = 0
    batch_construction: str = "none_carried_belief"
    posterior_inference: str = "carried_forward_filter_times_fixed_lag_backward"
    overlap_metric_policy: str = "not_applicable"
    decoder_lag: int = 1


def protocol_for(level, lag, walks=CANONICAL_BUDGET[0], walk_length=CANONICAL_BUDGET[1],
                 rand_prob=CANONICAL_BUDGET[2]):
    # Canonical study walk set (main_study / cscg_batch_sweep evaluation-v5); a non-canonical
    # setting keeps the seed, metrics and decoder and changes only walk count/length/rand_prob.
    return TrainingMatchedOnlineProtocol(
        sensor="cardinal distance", sensor_range=1, rand_prob=float(rand_prob), evaluation_seed=0,
        number_of_walks=int(walks), walk_length=int(walk_length), edge_thresholds=[1e-6], decoder_lag=int(lag))


def decoder_lag(method, window):
    """Training-matched lag: W - 1 for windowed smoothers (long EM, and the EMA teacher with its
    context W_T), 1 for lag-one inference (short EM, controls, and the JEPA student at any
    student context: under a frozen tensor the carried filter makes the context exact)."""
    return window - 1 if method in ("long", "teacher") else 1


@numba.njit(cache=True)
def fixed_lag_decode(T, starts, sizes, x, incoming, lag):
    """Carried filter + fixed-lag smoothing over one walk.

    incoming[t] is the action into x_t (ignored at t = 0). Returns decoded global
    clone ids (argmax, smallest id on ties) and causal -log2 p(x_t | x_<t).
    """
    n = x.shape[0]
    n_states = T.shape[1]
    width = sizes.max()
    filt = np.zeros((n, width))
    nll = np.zeros(n)
    s0, k0 = starts[x[0]], sizes[x[0]]
    for i in range(k0):
        filt[0, i] = 1.0 / k0
    nll[0] = -np.log2(k0 / n_states)
    for t in range(1, n):
        sp, kp = starts[x[t - 1]], sizes[x[t - 1]]
        sc, kc = starts[x[t]], sizes[x[t]]
        a = incoming[t]
        mass = 0.0
        for j in range(kc):
            v = 0.0
            for i in range(kp):
                v += filt[t - 1, i] * T[a, sp + i, sc + j]
            filt[t, j] = v
            mass += v
        if not mass > 0.0:
            raise ValueError("zero predictive probability")
        for j in range(kc):
            filt[t, j] /= mass
        nll[t] = -np.log2(mass)
    states = np.empty(n, dtype=np.int64)
    beta = np.zeros(width)
    nxt = np.zeros(width)
    for t in range(n):
        end = min(t + lag, n - 1)
        ke = sizes[x[end]]
        for j in range(ke):
            beta[j] = 1.0
        for u in range(end - 1, t - 1, -1):
            su, ku = starts[x[u]], sizes[x[u]]
            sv, kv = starts[x[u + 1]], sizes[x[u + 1]]
            a = incoming[u + 1]
            total = 0.0
            for i in range(ku):
                v = 0.0
                for j in range(kv):
                    v += T[a, su + i, sv + j] * beta[j]
                nxt[i] = v
                total += v
            for i in range(ku):
                beta[i] = nxt[i] / total
        s, k = starts[x[t]], sizes[x[t]]
        best, arg = -1.0, 0
        for i in range(k):
            v = filt[t, i] * beta[i]
            if v > best:
                best, arg = v, i
        states[t] = s + arg
    return states, nll


def find_walk_cache(protocol, level):
    """Reuse any existing canonical evaluation-v5 walk cache with the same walk key."""
    filename = _walk_cache_filename(protocol, level)
    key = protocol.walk_cache_key()
    for path in sorted((ROOT / "results/current").rglob(f"evaluation_caches/*/walks/{filename}")):
        try:
            meta = json.loads((path.parent.parent / "protocol.json").read_text())
        except (OSError, ValueError):
            continue
        candidate = EvaluationV5Protocol(**{k: v for k, v in meta.items() if k in EvaluationV5Protocol.__dataclass_fields__})
        if candidate.walk_cache_key() == key:
            return _load_walk_cache(path.parent, path.name), str(path.relative_to(ROOT))
    return None, None


_CONTEXT = {}


def level_context(level, cache_root, path=None, budget=CANONICAL_BUDGET):
    """Walks + ground truth for one level. `path` (repo-relative cache file) skips the search."""
    if (level, budget) not in _CONTEXT:
        protocol = protocol_for(level, 1, *budget)
        if path is not None:
            walks, source = _load_walk_cache((ROOT / path).parent, (ROOT / path).name), path
        else:
            walks, source = find_walk_cache(protocol, level)
        if walks is None:
            walks = generate_and_cache_walks(study_root=cache_root, protocol=protocol, level=level)
            source = str((cache_root / "evaluation_caches" / protocol.protocol_id() / "walks" /
                          _walk_cache_filename(protocol, level)).relative_to(ROOT))
        gt = build_empirical_ground_truth(
            walk_data=walks, n_true=len(exact_cartesian_reference_positions(level)),
            n_actions=infer_num_actions_from_valid_transitions(exact_valid_sensorimotor_transitions(level_index=level)),
            level=level)
        _CONTEXT[level, budget] = (walks, gt, source)
    return _CONTEXT[level, budget]


def viterbi_name(n_iter, pseudocount):
    return f"viterbi{int(n_iter)}_pc{pseudocount:g}"


def viterbi_decode(T, n_clones, step, walks, viterbi):
    """Existing CSCG offline Viterbi: refine the snapshot with learn_viterbi_T (n_iter, the given
    pseudocount) on the training observations up to this checkpoint, then MAP-decode each walk
    with CHMM.decode. Returns the refined tensor, decoded states and causal NLL (CHMM.bps)."""
    from mercury_runs.algorithms.cscg.prepare import align_post_step_actions_to_cscg, import_cscg_module
    from mercury_runs.algorithms.cscg.train import viterbi_refined_eval_model_npz
    n_iter, pseudocount, train_x, train_actions = viterbi
    chmm = import_cscg_module(ROOT)
    clones = np.asarray(n_clones, dtype=np.int64)
    def model_for(x, actions, tensor):
        model = chmm.CHMM(n_clones=clones, x=x, a=align_post_step_actions_to_cscg(actions),
                          pseudocount=pseudocount, dtype=np.float32, seed=0)
        model.C = np.zeros(tensor.shape, dtype=np.float32)   # every action, even if absent from x
        model.T = np.asarray(tensor, dtype=np.float32).copy()
        return model
    if step >= 2:
        refined = viterbi_refined_eval_model_npz(
            model_for(train_x[:step], train_actions[:step], T), obs_idx=train_x[:step],
            act_idx=train_actions[:step], n_clones=clones, n_iter=n_iter, pseudocount=pseudocount)["T"]
    else:   # no training transitions yet: the pseudocount alone, as CHMM.update_T applies it
        refined = (T + pseudocount) / (T + pseudocount).sum(-1, keepdims=True)
    decoded, nlls = [], []
    for x, a, _, _ in walks:
        model = model_for(x, a, refined)
        decoded.append(np.asarray(model.decode(x, align_post_step_actions_to_cscg(a))[1], dtype=np.int64))
        nlls.append(np.asarray(model.bps(x, align_post_step_actions_to_cscg(a)), dtype=np.float64))
    return np.asarray(refined, dtype=np.float64), decoded, nlls


def evaluate_run(folder, window, steps_mode, cache_root, cache_paths, budget=CANONICAL_BUDGET, teacher_window=None,
                 viterbi=None):
    """Student snapshots, plus EMA-teacher snapshots when the run saved them (method "teacher",
    records under teacher_<decoder>/; training-matched lag W_T - 1, as for long EM).
    viterbi = (n_iter, pseudocount, train_x, train_actions) adds the offline-Viterbi decoder."""
    rows = json.loads((folder / "rows.json").read_text())
    level, seed, student = rows[0]["level"], rows[0]["seed"], rows[0]["method"]
    walks, gt, source = level_context(level, cache_root, cache_paths[level], budget)
    with np.load(folder / "snapshots.npz") as saved:
        steps, n_clones = saved["steps"], saved["n_clones"]
        models = [(student, saved["T"].astype(np.float64), decoder_lag(student, window), "")]
        if "teacher_T" in saved:
            models.append(("teacher", saved["teacher_T"].astype(np.float64),
                           decoder_lag("teacher", teacher_window or window), "teacher_"))
        obs_uniques, action_uniques = saved["obs_uniques"], saved["action_uniques"]
    starts = np.r_[0, np.cumsum(n_clones)[:-1]].astype(np.int64); sizes = n_clones.astype(np.int64)
    indexed = [(map_to_indices(obs_uniques, np.asarray(x)).astype(np.int64),
                map_to_indices(action_uniques, np.asarray(a)).astype(np.int64),
                np.asarray(truth, dtype=np.int64), np.asarray(ga, dtype=np.int64))
               for x, a, truth, ga in zip(walks["filtered_obs_per_walk"], walks["filtered_actions_per_walk"],
                                          walks["gt_exact_per_walk"], walks["gt_action_per_walk"])]
    selected = range(len(steps)) if steps_mode == "all" else [len(steps) - 1]
    out = []
    for method, Ts, matched, prefix in models:
      for k in selected:
        T = np.ascontiguousarray(Ts[k])
        # common_lag1: one decoder configuration for every arm (carried filter + lag-one
        # smoothing), separating the learned tensor from each arm's inference context.
        decoders = [("training_matched", matched), ("filter", 0), ("common_lag1", 1)]
        if viterbi is not None:
            decoders.append((viterbi_name(*viterbi[:2]), -1))   # lag -1: MAP path, not a fixed lag
        for decoder, lag in decoders:
            target = folder / output_name(*budget) / f"{prefix}{decoder}" / f"step={int(steps[k]):05d}.json"
            if target.exists():
                record = json.loads(target.read_text())
            else:
                started = time.perf_counter()
                evaluated = T
                if lag < 0:
                    evaluated, decoded, nlls = viterbi_decode(T, n_clones, int(steps[k]), indexed, viterbi)
                else:
                    decoded, nlls = [], []
                    for x, a, _, _ in indexed:
                        states, nll = fixed_lag_decode(T, starts, sizes, x, a, lag)
                        decoded.append(states); nlls.append(nll)
                total = float(np.concatenate(nlls).sum()); count = int(sum(len(v) for v in nlls))
                protocol = protocol_for(level, max(lag, 0), *budget)
                inference = dict(decoder=protocol.decoder, lag=int(lag), decoder_role=decoder,
                                 carried_belief=True, parameter_updates=False, walk_cache=source,
                                 transition_checksum=hashlib.sha256(T.tobytes()).hexdigest()[:16])
                if lag < 0:
                    # Metrics, NLL and checksum refer to the Viterbi-refined tensor actually decoded.
                    inference.update(decoder="cscg_offline_viterbi_map", carried_belief=False,
                                     parameter_updates=True, offline_viterbi_iterations=int(viterbi[0]),
                                     viterbi_pseudocount=float(viterbi[1]),
                                     refinement_data="training stream x_{0:step}",
                                     transition_checksum=hashlib.sha256(evaluated.tobytes()).hexdigest()[:16])
                record = _evaluate_metrics(
                    protocol=protocol, level=level, method=method, step=int(steps[k]), observed_samples=int(steps[k]),
                    walk_cache_id=protocol.walk_cache_key(), gt=gt,
                    decoded=dict(decoded_walks=decoded, gt_exact_walks=[w[2] for w in indexed],
                                 gt_action_walks=[w[3] for w in indexed]),
                    transition=evaluated, n_learned=int(n_clones.sum()), inference=inference,
                    model_fit=dict(bits_per_symbol=total / count, scored_symbols=count, initial_observation_included=True,
                                   negative_log2_likelihood_sum=total, parameter_updates=lag < 0))
                record["protocol_id"] = protocol.protocol_id()
                record["evaluation_seconds"] = time.perf_counter() - started
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(record))
            scores = canonical_metrics(record)
            out.append(dict(level=level, seed=seed, method=method, step=int(steps[k]), decoder=decoder, lag=int(lag),
                            evaluation_walks=budget[0], walk_length=budget[1], evaluation_rand_prob=budget[2],
                            evaluation_seconds=record.get("evaluation_seconds"), **{m: scores[m] for m in METRICS}))
    return out


def evaluate(args):
    run = args.run.resolve()
    manifest = json.loads((run / "experiment.json").read_text())
    window = int(manifest["experiment"]["window"])
    teacher_window = int(manifest["experiment"].get("teacher_window") or window)
    folders = [p.parent for p in sorted(run.glob("level=*/seed=*/*/rows.json"))
               if p.parent.name in (args.methods or manifest["methods"])]
    cache_root = args.walk_cache_root.resolve()
    budget = (args.walks, args.walk_length, args.rand_prob)
    output = run / output_name(*budget); output.mkdir(parents=True, exist_ok=True)
    for level in sorted({json.loads((f / "rows.json").read_text())[0]["level"] for f in folders}):
        level_context(level, cache_root, budget=budget)   # locate or generate caches once, before the workers start
    cache_paths = {level: context[2] for (level, b), context in _CONTEXT.items() if b == budget}
    viterbi = {}
    if args.offline_viterbi_iterations > 0:
        # The run's own training stream (same loader and vocabulary as training), for the
        # offline Viterbi refinement; actions are post-step, as stored in the dataset.
        from scripts.run_cscg_distillation_experiment import load_training
        for f in folders:
            first = json.loads((f / "rows.json").read_text())[0]
            data = load_training(first["level"], first["seed"], manifest["rand_prob"], manifest["observations"])
            viterbi[f] = (args.offline_viterbi_iterations, args.viterbi_pseudocount,
                          np.asarray(data["x"], dtype=np.int64), np.r_[0, data["actions"]].astype(np.int64))
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate_run, f, window, args.steps, cache_root, cache_paths, budget, teacher_window,
                               viterbi.get(f)): f
                   for f in folders}
        for done, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result())
            print(f"[{done}/{len(folders)}] {futures[future].relative_to(run)}", flush=True)
    frame = pd.DataFrame(rows).sort_values(["level", "seed", "method", "decoder", "step"])
    frame.to_csv(output / f"canonical_metrics_{args.steps}.csv", index=False)
    (output / "protocol.json").write_text(json.dumps(dict(
        protocol=asdict(protocol_for(13, 1, *budget)), window=window,
        evaluation_budget=dict(walks=budget[0], walk_length=budget[1], total_steps=budget[0] * budget[1],
                               rand_prob=budget[2], canonical=budget == CANONICAL_BUDGET),
        lags={**{m: decoder_lag(m, window) for m in manifest["methods"]},
              "teacher": decoder_lag("teacher", teacher_window)},
        offline_viterbi=(None if args.offline_viterbi_iterations <= 0 else dict(
            decoder=viterbi_name(args.offline_viterbi_iterations, args.viterbi_pseudocount),
            iterations=args.offline_viterbi_iterations, pseudocount=args.viterbi_pseudocount,
            refinement="cscg.train.viterbi_refined_eval_model_npz on the training stream x_{0:step}",
            decode="CHMM.decode (max-product + backtrace)", nll="CHMM.bps under the refined tensor")),
        walk_caches={str(level): path for level, path in sorted(cache_paths.items())}), indent=2) + "\n")
    print(f"Wrote {output / f'canonical_metrics_{args.steps}.csv'}")


def report(args):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.lines import Line2D
    from scripts.analysis.cscg_distillation_teacher_window_report import (
        BASELINE_LABELS, COLORS, INK, LOWER_IS_BETTER, METRIC_LABELS, MUTED, dot_figure, finish,
    )
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    frames = []
    for item in args.condition:
        label, path = item.split("=", 1)
        frames.append(pd.read_csv(Path(path) / OUTPUT / f"canonical_metrics_{args.steps}.csv").assign(condition=label))
    data = pd.concat(frames, ignore_index=True)
    final = data[data.step == data.step.max()]
    labels = list(dict.fromkeys(data.condition))
    rows = []
    table = final.set_index(["decoder", "condition", "level", "seed", "method"])
    for (decoder, condition, level, seed), _ in final.groupby(["decoder", "condition", "level", "seed"]):
        get = lambda m, c=condition: table.loc[(decoder, c, level, seed, m)]
        for baseline in ("short", "long", "nll_control"):
            for metric in METRICS:
                rows.append(dict(decoder=decoder, contrast=f"predictive_minus_{baseline}:{condition}", level=level, seed=seed,
                                 metric=f"final_{metric}", delta=get("predictive")[metric] - get(baseline)[metric]))
        if len(labels) == 2 and condition == labels[1]:
            for method in ("predictive", "long", "nll_control"):
                for metric in METRICS:
                    rows.append(dict(decoder=decoder, contrast=f"{method}:{labels[1]}_minus_{labels[0]}", level=level, seed=seed,
                                     metric=f"final_{metric}",
                                     delta=get(method)[metric] - get(method, labels[0])[metric]))
    diffs = pd.DataFrame(rows)
    lower = diffs.metric.str.endswith(LOWER_IS_BETTER)
    diffs["benefit"] = np.where(lower, -diffs.delta, diffs.delta)
    summary = diffs.groupby(["decoder", "contrast", "level", "metric"]).agg(
        mean=("delta", "mean"), std=("delta", "std"), count=("delta", "count"),
        wins=("benefit", lambda x: int((x > 1e-12).sum()))).reset_index()
    data.to_csv(output / "learning_curves.csv", index=False)
    final.to_csv(output / "final_per_seed.csv", index=False)
    diffs.to_csv(output / "per_seed_paired_differences.csv", index=False)
    summary.to_csv(output / "paired_summary.csv", index=False)
    headline = ("action_tv_similarity", "action_no_self_loop_tv_similarity", "action_edge_f1_auc",
                "latent_precision", "latent_recall", "eval_nll_bps")
    lines = [f"# {args.title}", "",
             "Canonical online evaluation (evaluation-v5 walks: 100 × 10,000 steps, rand_prob 0.1, seed 0; evaluation-v5",
             "metric computation and extractor) with each arm's training inference as decoder: carried causal filter plus",
             "fixed-lag smoothing, argmax P(z_t | x_<=t+L). L = 1 for short EM, replay-NLL control and JEPA; L = W − 1 for",
             "long EM. NLL is the carried filter's causal predictive loss. `filter` rows (L = 0 for every arm) are a",
             "decoder-sensitivity check. Only the student is evaluated.", ""]
    order = sorted(set(summary.contrast), key=lambda c: (":" not in c.split("_minus_")[0] and "predictive_minus" not in c, c))
    for decoder in ("training_matched", "filter"):
        part = summary[summary.decoder == decoder].set_index(["contrast", "level", "metric"])
        lines += [f"## {'Primary: training-matched decoder' if decoder == 'training_matched' else 'Check: causal filter decoder (L = 0)'}", "",
                  "Final checkpoint: mean ± SD (wins/n; wins = left side better).", "",
                  "| Contrast | Level | " + " | ".join(METRIC_LABELS.get(m, m) for m in headline) + " |",
                  "| --- | ---: |" + " ---: |" * len(headline)]
        for contrast in order:
            for level in sorted({k[1] for k in part.index if k[0] == contrast}):
                cells = [part.loc[(contrast, level, f"final_{m}")] for m in headline]
                lines.append(f"| {contrast} | {level} | " + " | ".join(
                    f"{c['mean']:+.4f} ± {c['std']:.4f} ({int(c.wins)}/{int(c['count'])})" for c in cells) + " |")
        lines.append("")
    means = final.groupby(["decoder", "condition", "level", "method"]).mean(numeric_only=True)
    lines += ["## Final checkpoint means", "", "| Decoder | Condition | Level | Model | " +
              " | ".join(METRIC_LABELS.get(m, m) for m in headline) + " |", "| --- | --- | ---: | --- |" + " ---: |" * len(headline)]
    lines += [f"| {d} | {c} | {l} | {LABELS[m]} | " + " | ".join(f"{r[x]:.4f}" for x in headline) + " |"
              for (d, c, l, m), r in means.iterrows()]
    (output / "results.md").write_text("\n".join(lines) + "\n")
    # Figure: JEPA − baselines for each condition (hollow = first, filled = second), plus window rows.
    for decoder in ("training_matched", "filter"):
        part = diffs[diffs.decoder == decoder].assign(setting="")
        groups = [("", l) for l in sorted(part.level.unique())]
        plot_rows = [(f"JEPA − {BASELINE_LABELS[b]}", [(f"predictive_minus_{b}:{c}", i > 0) for i, c in enumerate(labels)], COLORS[b])
                     for b in ("short", "long", "nll_control")]
        if len(labels) == 2:
            plot_rows += [(f"{LABELS[m]}: {labels[1]} − {labels[0]}", [(f"{m}:{labels[1]}_minus_{labels[0]}", True)], COLORS[m])
                          for m in ("predictive", "long")]
        fig, _ = dot_figure(part, groups, plot_rows, output, "", "")
        handles = [Line2D([], [], marker="o", linestyle="", markersize=7, markeredgewidth=1.5, markeredgecolor=MUTED,
                          markerfacecolor=MUTED if i else "white", label=c) for i, c in enumerate(labels)]
        handles.append(Line2D([], [], color=INK, linewidth=2, label="Mean over seeds"))
        finish(fig, handles, output, f"paired_jepa_effect_{decoder}",
               f"{args.title} — {decoder.replace('_', '-')} decoder, final checkpoint, per seed (dots) and mean (tick)")
    print(f"Report: {output / 'results.md'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--run", type=Path, required=True)
    e.add_argument("--methods", nargs="+")
    e.add_argument("--steps", choices=["final", "all"], default="final")
    e.add_argument("--workers", type=int, default=8)
    e.add_argument("--walks", type=int, default=CANONICAL_BUDGET[0],
                   help="Held-out walks (canonical 100); a non-canonical budget writes to its own directory")
    e.add_argument("--walk-length", type=int, default=CANONICAL_BUDGET[1], help="Steps per walk (canonical 10,000)")
    e.add_argument("--rand-prob", type=float, default=CANONICAL_BUDGET[2],
                   help="Random-action probability of the held-out walks (canonical 0.1)")
    e.add_argument("--offline-viterbi-iterations", type=int, default=0,
                   help="Add the offline-Viterbi decoder: learn_viterbi_T for this many iterations "
                        "(early stop as in CSCG) on x_{0:step}, then CHMM.decode; 0 = off")
    e.add_argument("--viterbi-pseudocount", type=float, default=0.0,
                   help="Pseudocount of the offline-Viterbi refinement (CSCG eval default 0)")
    e.add_argument("--walk-cache-root", type=Path, default=ROOT / "results/diagnostics/evaluation_v5_online_matched_walks",
                   help="Where canonical walks are generated when no existing evaluation-v5 cache matches")
    r = sub.add_parser("report")
    r.add_argument("--condition", action="append", required=True, help="Label=<experiment dir>")
    r.add_argument("--steps", choices=["final", "all"], default="final")
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--title", default="Online JEPA-CSCG: canonical online evaluation, training-matched decoder")
    args = p.parse_args()
    evaluate(args) if args.command == "evaluate" else report(args)


if __name__ == "__main__":
    main()
