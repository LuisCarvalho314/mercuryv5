"""Isolated online-count scaling diagnostic; never edits canonical runners or studies.

Run: PYTHONPATH=src uv run python scripts/analysis/cscg_online_count_scale_diagnostic.py
Results: results/diagnostics/cscg_online_count_scale_v1/
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from mercury.graph.core import Graph
from mercury_runs.algorithms.cscg.api import _initialize_cscg_structure, _resolve_cscg_capacity
from mercury_runs.algorithms.cscg.config import CSCGConfig
from mercury_runs.algorithms.cscg.prepare import (
    action_to_indices,
    align_post_step_actions_to_cscg,
    import_cscg_module,
    iter_cscg_training_batches,
    load_dataset_arrays,
    load_ground_truth_states,
    map_to_indices,
    to_indices,
)
from mercury_runs.algorithms.mercury_cscg_hybrid.prepare import (
    build_hybrid_assignment, build_source_graph_observation_assignment,
)
from mercury_runs.algorithms.mercury_cscg_hybrid.train import initialize_cscg_from_mercury_graph
from mercury_runs.application.evaluation_protocol import default_canonical_v4_protocol
from mercury_runs.application.evaluation_service import (
    build_empirical_ground_truth,
    generate_and_cache_walks,
)
from mercury_runs.infrastructure.paper_precision import (
    exact_cartesian_reference_positions,
    exact_valid_sensorimotor_transitions,
    infer_num_actions_from_valid_transitions,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/current/studies/default_matrix_smoke_30_seed_onehot_clean_more_mazes"
OUTPUT = ROOT / "results/diagnostics/cscg_online_count_scale_v1"
LAMBDA = 0.9999
KAPPA = 0.002
CHECKPOINTS = (0, 1, 2, 5, 10, 20, 50)
CONDITIONS = ("CURRENT", "NO_KAPPA", "SCALED_KAPPA", "CONSISTENT_SCALE")
METHODS = ("vanilla", "mercury", "tmgwr", "oracle")


def source_runs() -> dict[tuple[str, int, int], tuple[dict, Path]]:
    rows = [json.loads(line)["config"] for line in (SOURCE / "study_manifest.jsonl").read_text().splitlines()]
    return {
        (str(row["method"]), int(row["level"]), int(row["seed"])): (row, SOURCE / row["requested_run_id"])
        for row in rows if row["method"] in {"mercury_cscg_hybrid", "tmgwr_cscg_hybrid"} and int(row["seed"]) < 3
    }


def training_data(level: int, seed: int, source_root: Path) -> dict:
    directory = ROOT / f"datasets/level={level}/sensor=cardinal distance/range=1"
    matches = sorted(directory.glob(f"*_seed{seed}_p0.3_steps2000_allsteps_randomstart_*.parquet"))
    if len(matches) != 1:
        raise ValueError(f"Expected one paired training dataset for level={level}, seed={seed}: {matches}")
    parquet = matches[0]
    observations, actions, collisions, _, _ = load_dataset_arrays(
        datasets_root=ROOT / "datasets", level=level, sensor="cardinal distance", sensor_range=1,
        dataset_select="run_id", dataset_run_id=parquet.stem,
    )
    if len(observations) != 2000:
        raise ValueError(f"Expected 2000 observations in {parquet}")
    obs_idx, obs_values = to_indices(observations)
    act_idx, action_values = action_to_indices(actions)
    ground_truth_path = next((source_root / "ground_truth").glob("*_ground_truth_states.parquet"))
    ground_truth = load_ground_truth_states(ground_truth_path)
    if len(ground_truth) != 2000:
        raise ValueError(f"Source ground truth length mismatch: {ground_truth_path}")
    batches = []
    for batch in iter_cscg_training_batches(
        parquet_path=parquet, batch_size=40, include_prev=True, valid_trajectories_only=False,
    ):
        batches.append((
            map_to_indices(obs_values, np.asarray(batch["observations"])),
            align_post_step_actions_to_cscg(map_to_indices(action_values, np.asarray(batch["actions"]))),
            int(batch["filtered_steps"]),
        ))
    if len(batches) != 50 or sum(x[2] for x in batches) != 2000:
        raise ValueError(f"Unexpected batch partition for {parquet}")
    return dict(parquet=parquet, obs_idx=obs_idx, act_idx=act_idx, obs_values=obs_values,
                action_values=action_values, ground_truth=ground_truth, batches=batches,
                collision_count=int(np.asarray(collisions).sum()))


def initial_model(method: str, level: int, seed: int, source_row: dict, source_root: Path, data: dict, chmm):
    n_obs = len(data["obs_values"])
    n_actions = len(data["action_values"])
    if method in {"vanilla", "oracle"}:
        oracle = method == "oracle"
        config = CSCGConfig(
            datasets_root=ROOT / "datasets", output_root=OUTPUT, run_id="diagnostic",
            level=level, sensor="cardinal distance", sensor_range=1,
            ground_truth_states_parquet=next((source_root / "ground_truth").glob("*_ground_truth_states.parquet")),
            seed=seed, clones_per_obs=(1 if oracle else 10), clone_allocation=("ground_truth" if oracle else "uniform"),
            clone_capacity_mode=("ground_truth" if oracle else "fixed"),
            structural_transfer_level=("clone_state" if oracle else "first_order_observation"),
            structural_init_mode=("binary" if oracle else "none"),
            structural_source=("oracle_maze_topology" if oracle else "none"),
            pseudocount=KAPPA,
        )
        n_clones = _resolve_cscg_capacity(
            config=config, obs_idx=data["obs_idx"], ground_truth_states=data["ground_truth"], n_obs=n_obs,
        )
        model = chmm.CHMM(
            n_clones=n_clones, x=np.array([0, min(1, n_obs - 1), 0]),
            a=np.array([0, n_actions - 1, 0]), pseudocount=KAPPA, dtype=np.float32, seed=seed,
        )
        initial_counts, _ = _initialize_cscg_structure(
            config=config, model=model, n_clones=n_clones, obs_idx=data["obs_idx"],
            act_idx=data["act_idx"], ground_truth_states=data["ground_truth"],
            n_obs=n_obs, n_actions=n_actions,
        )
        return model, initial_counts, None

    graph_path = next((source_root / ("bundles/tmgwr_cscg_hybrid/sensory" if method == "tmgwr"
                                      else "bundles/mercury_cscg_hybrid/mercury")).glob(
        "**/*_sensory_graph.npz" if method == "tmgwr" else "**/*_latent_graph.npz"))
    graph = Graph.from_npz(str(graph_path))
    if method == "tmgwr":
        assignment = build_source_graph_observation_assignment(
            graph=graph, obs_values=data["obs_values"], n_obs=n_obs,
            clone_capacity_mode="source", clones_per_obs=1,
        )
    else:
        assignment = build_hybrid_assignment(
            latent_graph=graph, obs_idx=data["obs_idx"], n_obs=n_obs,
            memory_length=int(source_row["memory_length"]),
        )
    model = initialize_cscg_from_mercury_graph(
        chmm_mod=chmm, latent_graph=graph, assignment=assignment,
        obs_idx=data["obs_idx"], act_idx=data["act_idx"], pseudocount=KAPPA, seed=seed,
        transfer_level=("first_order_observation" if method == "tmgwr" else "clone_state"),
        init_mode="binary", structural_source="source_graph", level=level,
    )
    return model, np.asarray(model.C, dtype=np.float32).copy(), graph_path


def checkpoint(model, counts, prior, observed_total, batch, kappa, initial_t, initial_mass):
    prior_mass = float(prior.sum(dtype=np.float64))
    total_mass = float(counts.sum(dtype=np.float64))
    return dict(batch=batch, running_mass=total_mass, prior_mass=prior_mass,
                prior_survival_fraction=(prior_mass / initial_mass if initial_mass else 0.0),
                prior_fraction=(prior_mass / (prior_mass + observed_total) if prior_mass + observed_total else 0.0),
                observed_mass=observed_total, pseudocount_per_cell=kappa,
                prior_recurrence_max_abs_error=float(np.max(np.abs(prior - initial_t * (LAMBDA ** batch))))
                if initial_mass else 0.0,
                T=np.asarray(model.T).copy(), counts=np.asarray(counts).copy(), prior=np.asarray(prior).copy())


def train_one(model, initial_counts, data, condition, chmm, passes):
    # The constructor and structural T are left untouched before batch 1.
    constructor_counts = np.asarray(model.C, dtype=np.float32).copy()
    initial = np.zeros_like(model.C, dtype=np.float32) if initial_counts is None else initial_counts.copy()
    if condition == "CONSISTENT_SCALE":
        initial *= 1.0 - LAMBDA
    counts = initial.copy()
    prior = initial.astype(np.float64)
    initial_prior = prior.copy()
    initial_mass = float(prior.sum())
    observed_total = 0.0
    after_kappa = KAPPA if condition == "CURRENT" else (0.0 if condition == "NO_KAPPA" else (1.0 - LAMBDA) * KAPPA)
    checkpoints = {0: checkpoint(model, counts, prior, observed_total, 0, after_kappa, initial_prior, initial_mass)}
    t0 = time.perf_counter()
    failure = None
    for batch_num, (obs_idx, act_idx, _) in enumerate(data["batches"], start=1):
        try:
            batch_model = model if passes == 1 else copy.deepcopy(model)
            for local_pass in range(passes):
                _, mess_fwd = chmm.forward(batch_model.T.transpose(0, 2, 1), batch_model.Pi_x,
                                           batch_model.n_clones, obs_idx, act_idx, store_messages=True)
                mess_bwd = chmm.backward(batch_model.T, batch_model.n_clones, obs_idx, act_idx)
                batch_counts = np.zeros_like(model.C, dtype=np.float32)
                chmm.updateC(batch_counts, batch_model.T, batch_model.n_clones,
                             mess_fwd, mess_bwd, obs_idx, act_idx)
                if local_pass + 1 < passes:
                    batch_model.C = batch_counts.astype(batch_model.dtype, copy=True)
                    batch_model.pseudocount = after_kappa
                    batch_model.update_T()
        except (AssertionError, FloatingPointError) as exc:
            failure = {"batch": batch_num, "error": type(exc).__name__,
                       "reason": "forward probability collapsed with the specified transition matrix"}
            break
        counts *= LAMBDA
        counts += (1.0 - LAMBDA) * batch_counts
        prior *= LAMBDA
        observed_total = LAMBDA * observed_total + (1.0 - LAMBDA) * float(batch_counts.sum(dtype=np.float64))
        model.C = counts.astype(model.dtype, copy=True)
        model.pseudocount = after_kappa
        model.update_T()
        if batch_num in CHECKPOINTS:
            checkpoints[batch_num] = checkpoint(model, counts, prior, observed_total,
                                                batch_num, after_kappa, initial_prior, initial_mass)
    train_seconds = time.perf_counter() - t0
    first_counts = checkpoints.get(1, checkpoints[0])["counts"]
    row = np.unravel_index(np.argmax(first_counts.sum(axis=2)), first_counts.shape[:2])
    records = []
    n_states = int(model.n_clones.sum())
    for batch in CHECKPOINTS:
        if batch not in checkpoints:
            continue
        item = checkpoints[batch]
        a, i = row
        learned = float(item["counts"][a, i].sum(dtype=np.float64))
        prior_row = float(item["prior"][a, i].sum(dtype=np.float64))
        transition_row_mass = (float(constructor_counts[a, i].sum(dtype=np.float64))
                               if batch == 0 else learned)
        pseudomass = n_states * (KAPPA if batch == 0 else after_kappa)
        t_row = item["T"][a, i]
        records.append({k: v for k, v in item.items() if k not in {"T", "counts", "prior"}} |
                       dict(row_action=int(a), row_clone=int(i), learned_row_mass=learned,
                            prior_row_mass=prior_row, pseudocount_row_mass=pseudomass,
                            transition_count_row_mass=transition_row_mass,
                            pseudocount_fraction=(pseudomass / (transition_row_mass + pseudomass)
                                                  if transition_row_mass + pseudomass else 0.0),
                            row_T_sum=float(t_row.sum()),
                            row_T_tv_from_uniform=float(0.5 * np.abs(t_row - 1.0 / n_states).sum())))
    if any(item["prior_recurrence_max_abs_error"] > 1e-9 for item in records):
        raise AssertionError("Seeded-prior recurrence was violated")
    return records, train_seconds, failure


def evaluate(model, data, level, method, condition, seed, protocol, walk_data, gt, output):
    from mercury_runs.algorithms.cscg.snapshot_adapter import CSCGSnapshotAdapter
    from mercury_runs.application.evaluation_service import (
        compute_transition_statistics, serialise_evaluation_result,
    )
    from mercury_runs.infrastructure.paper_precision import (
        concatenate_walk_series, compute_fixed_map_precision_details, evaluate_transition_structure_canonical,
    )
    adapter = CSCGSnapshotAdapter()
    snapshot = dict(T=np.asarray(model.T).copy(), model_npz={"T": np.asarray(model.T).copy()},
                    n_clones=np.asarray(model.n_clones), obs_uniques=data["obs_values"],
                    action_uniques=data["action_values"], meta={"step": 2000, "observed_samples": 2000, "batch_size": 40})
    decoded = adapter.decode_walks(snapshot, walk_data, batch_size=40)
    if not decoded["decoded_walks"]:
        raise RuntimeError("Canonical decoder produced no evaluation walks")
    C, H, N_ai = compute_transition_statistics(
        decoded_walks=decoded["decoded_walks"], true_walks=decoded["gt_exact_walks"],
        action_walks=decoded["gt_action_walks"], n_learned=int(model.n_clones.sum()),
        n_true=gt["n_true"], n_actions=gt["n_actions"],
    )
    details = compute_fixed_map_precision_details(
        inferred_states=concatenate_walk_series(decoded["decoded_walks"]),
        ground_truth_states=concatenate_walk_series(decoded["gt_exact_walks"]),
        metric_name="latent_precision", inferred_state_count=int(model.n_clones.sum()),
        ground_truth_state_count=gt["n_true"],
    )
    result = evaluate_transition_structure_canonical(
        T_hat=adapter.transition_tensor(snapshot), T_true=gt["T_GT"].astype(np.float64),
        cooccurrence=C, action_source_cooccurrence=H, source_action_counts=N_ai,
        edge_thresholds=tuple(protocol.edge_thresholds),
    )
    fit = adapter.negative_log2_likelihood(snapshot, walk_data)
    record = serialise_evaluation_result(
        protocol=protocol, level=level, method=method, step=2000, observed_samples=2000,
        walk_cache_id=protocol.walk_cache_key(), gt=gt, result=result,
        inference_meta=adapter.inference_metadata(snapshot, batch_size=40),
        latent_precision=details["latent_precision"], node_recall=details["latent_recall"],
        n_learned=int(model.n_clones.sum()), n_true=gt["n_true"], n_actions=gt["n_actions"], model_fit=fit,
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "canonical_v5_record.json").write_text(json.dumps(record))
    return dict(P_Z=record["state_representation"]["latent_precision"],
                C_Z=record["state_representation"]["node_recall"],
                S_A_TV=record["action_conditioned"]["tv_similarity_occupancy"],
                S_A_TV_nonself=record["action_conditioned"]["tv_non_self_loop_similarity_occupancy"],
                eval_NLL=record["model_fit"]["bits_per_symbol"],
                evaluated_walks=len(decoded["decoded_walks"]), walk_cache_id=record["walk_cache_id"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", type=int, nargs="+", default=[13, 20, 21, 22])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--walks", type=int, default=20)
    parser.add_argument("--walk-length", type=int, default=1000)
    parser.add_argument("--passes", type=int, default=1)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    base_config = {
        "purpose": "isolated CSCG online-count scale sensitivity diagnostic",
        "source_study": str(SOURCE.relative_to(ROOT)),
        "lambda": LAMBDA, "kappa": KAPPA, "batch_size": 40,
        "training_observations": 2000, "rand_prob": 0.3,
        "evaluation_seed": 0, "evaluation_walks": args.walks,
        "evaluation_walk_length": args.walk_length,
        "checkpoints": list(CHECKPOINTS), "conditions": list(CONDITIONS),
        "methods": list(METHODS),
    }
    config_path = OUTPUT / "diagnostic_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != base_config:
            raise ValueError("Existing diagnostic output uses a different base configuration")
    else:
        config_path.write_text(json.dumps(base_config, indent=2))
    source = source_runs()
    chmm = import_cscg_module(ROOT)
    for level in args.levels:
        protocol = default_canonical_v4_protocol(
            level=level, sensor="cardinal distance", sensor_range=1, rand_prob=0.3,
            evaluation_seed=0, number_of_walks=args.walks, walk_length=args.walk_length,
            valid_trajectories_only=False, edge_thresholds=[1e-6],
        )
        walks = generate_and_cache_walks(study_root=OUTPUT, protocol=protocol, level=level)
        n_true = int(exact_cartesian_reference_positions(level).shape[0])
        n_actions = infer_num_actions_from_valid_transitions(exact_valid_sensorimotor_transitions(level_index=level))
        gt = build_empirical_ground_truth(walk_data=walks, n_true=n_true, n_actions=n_actions, level=level)
        for seed in args.seeds:
            mercury_row, mercury_root = source["mercury_cscg_hybrid", level, seed]
            data = training_data(level, seed, mercury_root)
            for method in args.methods:
                source_row, source_root = (source["tmgwr_cscg_hybrid", level, seed]
                                           if method == "tmgwr" else (mercury_row, mercury_root))
                for condition in args.conditions:
                    out = OUTPUT / f"level={level}" / f"seed={seed}" / method / condition
                    if args.passes != 1:
                        out = out / f"passes={args.passes}"
                    result_path = out / "result.json"
                    if result_path.exists():
                        print(f"skip {level} {seed} {method} {condition}", flush=True)
                        continue
                    print(f"run {level} {seed} {method} {condition}", flush=True)
                    model, initial_counts, graph_path = initial_model(method, level, seed, source_row, source_root, data, chmm)
                    if method in {"mercury", "tmgwr"} and initial_counts is not None:
                        initial_counts *= float(source_row.get("mercury_cscg_prior_weight", 1.0))
                    initial_T = np.asarray(model.T).copy()
                    count_records, train_seconds, failure = train_one(model, initial_counts, data,
                                                                       condition, chmm, args.passes)
                    if np.allclose(initial_T, model.T):
                        print("warning: final T equals initial T", flush=True)
                    metrics = (evaluate(model, data, level, method, condition, seed, protocol, walks, gt, out)
                               if failure is None else {})
                    result = dict(level=level, seed=seed, method=method, condition=condition,
                                  passes=args.passes,
                                  dataset=str(data["parquet"].relative_to(ROOT)),
                                  source_run=str(source_root.relative_to(ROOT)),
                                  source_graph=(str(graph_path.relative_to(ROOT)) if graph_path else None),
                                  initial_structural_count_mass=(float(initial_counts.sum()) if initial_counts is not None else 0.0),
                                  train_seconds=train_seconds, clone_count=int(model.n_clones.sum()),
                                  source_node_count=(int(Graph.from_npz(str(graph_path)).n) if graph_path else None),
                                  completed_batches=count_records[-1]["batch"], failure=failure,
                                  count_diagnostics=count_records, **metrics)
                    out.mkdir(parents=True, exist_ok=True)
                    result_path.write_text(json.dumps(result, indent=2))
                    print(f"done {level} {seed} {method} {condition}: {metrics}", flush=True)


if __name__ == "__main__":
    main()
