from __future__ import annotations

import argparse
import logging
import sys

from ..application import evaluate_study, evaluate_study_snapshots, run_single, run_study
from ..infrastructure import strtobool


_POCML_ARGS = {
    "pocml_valid_trajectories_only",
    "pocml_epochs",
    "pocml_trajectory_length",
    "pocml_max_trajectories",
    "pocml_state_dim",
    "pocml_random_feature_dim",
    "pocml_n_states",
    "pocml_precision_capacities",
    "pocml_alpha",
    "pocml_lr_q",
    "pocml_lr_v",
    "pocml_lr_all",
    "pocml_lr_m",
    "pocml_reg_m",
    "pocml_max_iter_m",
    "pocml_eps_m",
    "pocml_batch_size",
    "pocml_memory_bias",
    "pocml_use_ground_truth_node_ids",
}
_CSCG_ARGS = {
    "cscg_valid_trajectories_only",
    "cscg_clones_per_obs",
    "cscg_clone_preset",
    "cscg_structural_preset",
    "cscg_clone_allocation",
    "cscg_clones_per_state",
    "cscg_structural_transfer_level",
    "cscg_structural_init_mode",
    "cscg_structural_source",
    "cscg_clone_capacity_mode",
    "cscg_n_iter",
    "cscg_pseudocount",
    "cscg_term_early",
    "cscg_train_algorithm",
    "cscg_training_mode",
    "cscg_batch_size",
    "cscg_iterations_per_batch",
    "cscg_online_lambda",
    "cscg_eval_viterbi",
    "cscg_seed",
    "mercury_cscg_prior_weight",
}
_SENSORY_ARGS = {
    "sensory_observation_mode",
    "sensory_allow_self_loops",
    "activation_threshold",
    "topological_neighbourhood_threshold",
    "sensory_max_neurons",
    "sensory_weighting",
    "winning_node_lr",
    "topological_neighbourhood_lr",
    "sensory_action_lr",
    "global_context_lr",
    "sensory_max_age",
    "sensory_gaussian_shape",
}
_ACTION_MAP_ARGS = {
    "am_n_codebook",
    "am_lr",
    "am_sigma",
    "am_key",
    "am_identity_for_one_hot",
}
_MERCURY_DATASET_ARGS = {
    "mercury_valid_trajectories_only",
    "mercury_split_sensory_raw_latent_valid",
    "mercury_input_mode",
}
_MERCURY_LATENT_ARGS = {
    "memory_length",
    "latent_allow_self_loops",
    "latent_max_neurons",
    "latent_action_lr",
    "latent_gaussian_shape",
    "latent_max_age",
    "ambiguity_threshold",
    "ambiguity_decay",
    "ambiguity_counter_mode",
    "trace_decay",
    "lambda_trace",
    "weight_memory",
    "weight_undirected",
    "weight_base",
    "weight_action",
    "memory_replay",
    "memory_disambiguation",
    "ambiguity_response_enabled",
    "ambiguity_refinement_enabled",
    "transition_reconstruction_enabled",
}
_BASELINE_ARGS = {"baseline_pocml", "baseline_cscg", "baseline_all"}
_METHOD_SPECIFIC_ARGS = (
    _BASELINE_ARGS
    | _POCML_ARGS
    | _CSCG_ARGS
    | _MERCURY_DATASET_ARGS
    | _MERCURY_LATENT_ARGS
    | _SENSORY_ARGS
    | _ACTION_MAP_ARGS
)
_APPLICABLE_ARGS_BY_METHOD = {
    "mercury": _BASELINE_ARGS | _MERCURY_DATASET_ARGS | _MERCURY_LATENT_ARGS | _SENSORY_ARGS | _ACTION_MAP_ARGS | _POCML_ARGS | _CSCG_ARGS,
    "mercury_cscg_hybrid": _MERCURY_DATASET_ARGS | _MERCURY_LATENT_ARGS | _SENSORY_ARGS | _ACTION_MAP_ARGS | _CSCG_ARGS,
    "mercury_cscg_hybrid_mercury": _MERCURY_DATASET_ARGS | _MERCURY_LATENT_ARGS | _SENSORY_ARGS | _ACTION_MAP_ARGS,
    "tmgwr_cscg_hybrid": _SENSORY_ARGS | _ACTION_MAP_ARGS | _CSCG_ARGS,
    "tmgwr_cscg_hybrid_tmgwr": _SENSORY_ARGS | _ACTION_MAP_ARGS,
    "sensory": _SENSORY_ARGS | _ACTION_MAP_ARGS,
    "pocml": _POCML_ARGS,
    "cscg": _CSCG_ARGS,
}


def _raw_cli_method(argv: list[str]) -> str | None:
    for index, token in enumerate(argv[1:], start=1):
        if token == "--method" and index + 1 < len(argv):
            return argv[index + 1]
        if token.startswith("--method="):
            return token.split("=", 1)[1]
    return None


def _explicit_cli_dests(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    explicit: set[str] = set()
    for token in argv[1:]:
        if not token.startswith("--"):
            continue
        option = token.split("=", 1)[0]
        action = parser._option_string_actions.get(option)  # noqa: SLF001 - argparse exposes no public lookup.
        if action is not None:
            explicit.add(action.dest)
    return explicit


def _validate_method_specific_cli(parser: argparse.ArgumentParser, args: argparse.Namespace, argv: list[str]) -> None:
    if args.study or args.evaluate_study or args.evaluate_study_snapshots:
        return
    method = str(args.method or "mercury")
    explicit = _explicit_cli_dests(parser, argv)
    invalid = sorted(explicit & (_METHOD_SPECIFIC_ARGS - _APPLICABLE_ARGS_BY_METHOD.get(method, set())))
    if invalid:
        flag_lookup = {action.dest: option for option, action in parser._option_string_actions.items() if option.startswith("--")}  # noqa: SLF001
        flags = ", ".join(flag_lookup.get(dest, f"--{dest}") for dest in invalid)
        parser.error(f"arguments not applicable to --method {method}: {flags}")


def _suppress_inapplicable_help(parser: argparse.ArgumentParser, method: str) -> None:
    invalid = _METHOD_SPECIFIC_ARGS - _APPLICABLE_ARGS_BY_METHOD.get(method, set())
    for action in parser._actions:  # noqa: SLF001 - argparse has no public action iterator.
        if action.dest in invalid:
            action.help = argparse.SUPPRESS


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", action="store_true", help="Run parametric study instead of single run")
    parser.add_argument("--_worker_run", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=["mercury", "sensory", "pocml", "cscg", "mercury_cscg_hybrid", "tmgwr_cscg_hybrid", "mercury_cscg_hybrid_mercury", "tmgwr_cscg_hybrid_tmgwr"],
    )
    parser.add_argument("--max_workers", type=int, default=1, help="Study mode: number of concurrent runs")
    parser.add_argument("--no_parallel", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--continue_on_error", action="store_true", default=False)
    parser.add_argument("--retry_failed", action="store_true", default=False)
    parser.add_argument("--retry_incomplete", action="store_true", default=False)
    parser.add_argument("--wandb", nargs="?", const=True, default=False, type=strtobool)
    parser.add_argument("--wandb_project", type=str, default="mercuryv5")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_job_type", type=str, default="run")
    parser.add_argument("--wandb_log_artifacts", nargs="?", const=True, default=True, type=strtobool)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--baseline_pocml", nargs="?", const=True, default=False, type=strtobool)
    parser.add_argument("--baseline_all", nargs="?", const=True, default=False, type=strtobool)
    parser.add_argument("--pocml_epochs", type=int, default=30)
    parser.add_argument("--pocml_trajectory_length", type=int, default=15)
    parser.add_argument("--pocml_max_trajectories", type=int, default=None)
    parser.add_argument("--pocml_state_dim", type=int, default=50)
    parser.add_argument("--pocml_random_feature_dim", type=int, default=500)
    parser.add_argument("--pocml_n_states", type=int, default=None)
    parser.add_argument("--pocml_precision_capacities", type=int, nargs="+", default=None)
    parser.add_argument("--pocml_alpha", type=float, default=2.0)
    parser.add_argument("--pocml_lr_q", type=float, default=0.1)
    parser.add_argument("--pocml_lr_v", type=float, default=0.08)
    parser.add_argument("--pocml_lr_all", type=float, default=0.05)
    parser.add_argument("--pocml_lr_m", type=float, default=1.0)
    parser.add_argument("--pocml_reg_m", type=float, default=0.0)
    parser.add_argument("--pocml_max_iter_m", type=int, default=1)
    parser.add_argument("--pocml_eps_m", type=float, default=1e-3)
    parser.add_argument("--pocml_batch_size", type=int, default=64)
    parser.add_argument("--pocml_memory_bias", nargs="?", const=True, default=True, type=strtobool)
    parser.add_argument("--pocml_use_ground_truth_node_ids", nargs="?", const=True, default=False, type=strtobool)
    parser.add_argument("--baseline_cscg", nargs="?", const=True, default=False, type=strtobool)
    parser.add_argument("--cscg_clones_per_obs", type=int, default=10)
    parser.add_argument("--cscg_clone_preset", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cscg_structural_preset", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cscg_clone_allocation", type=str, default="uniform", choices=["uniform", "ground_truth", "oracle_ground_truth"])
    parser.add_argument("--cscg_clones_per_state", type=int, default=1)
    parser.add_argument("--cscg_structural_transfer_level", type=str, default="first_order_observation", choices=["clone_state", "first_order_observation"])
    parser.add_argument("--cscg_structural_init_mode", type=str, default="binary", choices=["none", "binary", "count"])
    parser.add_argument("--cscg_structural_source", type=str, default="oracle_maze_topology", choices=["none", "oracle_maze_topology", "oracle_empirical_ground_truth", "source_graph"])
    parser.add_argument("--cscg_clone_capacity_mode", type=str, default="source", choices=["source", "ground_truth", "fixed"])
    parser.add_argument("--cscg_n_iter", type=int, default=1000)
    parser.add_argument("--cscg_pseudocount", type=float, default=2e-3)
    parser.add_argument("--cscg_term_early", nargs="?", const=True, default=True, type=strtobool)
    parser.add_argument("--cscg_train_algorithm", type=str, default="em", choices=["em", "viterbi"])
    parser.add_argument("--cscg_training_mode", type=str, default="online_em", choices=["full", "online_em"])
    parser.add_argument("--cscg_batch_size", type=int, default=40)
    parser.add_argument("--cscg_iterations_per_batch", type=int, default=1)
    parser.add_argument("--cscg_online_lambda", type=float, default=0.9999)
    parser.add_argument("--cscg_eval_viterbi", nargs="?", const=True, default=False, type=strtobool)
    parser.add_argument("--cscg_seed", type=int, default=None)
    parser.add_argument("--mercury_cscg_prior_weight", type=float, default=1.0)
    parser.add_argument("--level", type=int, required=False)
    parser.add_argument("--study_root", type=str, default="results/current/studies")
    parser.add_argument("--study_name", type=str, default="default")
    parser.add_argument("--study_config", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--reuse_existing_run", nargs="?", const=True, default=True, type=strtobool)
    parser.add_argument("--no_reuse_existing_run", action="store_true", default=False)
    parser.add_argument("--datasets_root", type=str, default="datasets")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rand_prob", type=float, default=0.3)
    parser.add_argument("--num_steps", type=int, default=20_000)
    parser.add_argument("--valid_trajectories_only", nargs="?", const=True, default=False, type=strtobool)
    parser.add_argument("--mercury_valid_trajectories_only", nargs="?", const=True, default=False, type=strtobool)
    parser.add_argument("--mercury_split_sensory_raw_latent_valid", nargs="?", const=True, default=False, type=strtobool)
    parser.add_argument("--mercury_input_mode", type=str, default="direct_cscg_one_hot", choices=["pretrained_raw", "pretrained_cscg_one_hot", "direct_cscg_one_hot"])
    parser.add_argument("--sensory_observation_mode", type=str, default="cscg_one_hot", choices=["raw", "cscg_one_hot"])
    parser.add_argument("--pocml_valid_trajectories_only", nargs="?", const=True, default=False, type=strtobool)
    parser.add_argument("--cscg_valid_trajectories_only", nargs="?", const=True, default=False, type=strtobool)
    parser.add_argument("--no_reuse_existing_dataset", action="store_true")
    parser.add_argument("--sensor", type=str, default="cardinal distance", choices=["cartesian", "cardinal distance"])
    parser.add_argument("--sensor_range", type=int, default=1)
    parser.add_argument("--window_length", type=int, default=100)
    parser.add_argument("--paper_precision", nargs="?", const=True, default=False, type=strtobool)
    parser.add_argument("--paper_precision_mode", type=str, default="per_iteration", choices=["off", "final", "per_iteration"])
    parser.add_argument("--paper_precision_schedule", type=str, default="log", choices=["log", "linear"])
    parser.add_argument("--paper_precision_eval_interval", type=int, default=1)
    parser.add_argument("--paper_precision_eval_checkpoints", type=int, nargs="+", default=None)
    parser.add_argument("--paper_precision_num_points", type=int, default=None)
    parser.add_argument("--paper_precision_num_walks", type=int, default=20)
    parser.add_argument("--paper_precision_walk_length", type=int,
                        default=1_000)

    parser.add_argument("--structure_metrics", nargs="?", const=True, default=True, type=strtobool)
    parser.add_argument("--structure_metrics_epsilon", type=float, default=1e-6)
    parser.add_argument("--structure_metrics_epsilons", type=float, nargs="+", default=None)
    parser.add_argument("--structure_metrics_ignore_self_loops", nargs="?", const=True, default=True, type=strtobool)
    parser.add_argument("--structure_metrics_scope", type=str, default="exact_level", choices=["exact_level", "walk_local"])
    parser.add_argument("--structure_metrics_ground_truth_source", type=str, default="empirical_walks", choices=["empirical_walks", "maze_topology"])
    parser.add_argument("--reevaluate_paper_precision_snapshots", action="store_true", default=False)
    parser.add_argument("--evaluate_study_snapshots", action="store_true", default=False)
    parser.add_argument("--evaluate_study", action="store_true", default=False,
                        help="Run versioned canonical snapshot evaluation over a completed study")
    parser.add_argument("--eval_num_walks", type=int, default=100,
                        help="Number of evaluation walks for evaluate-study")
    parser.add_argument("--eval_walk_length", type=int, default=10_000,
                        help="Length of each evaluation walk for evaluate-study")
    parser.add_argument("--eval_seed", type=int, default=0,
                        help="Evaluation seed for evaluate-study")
    parser.add_argument("--eval_edge_thresholds", type=float, nargs="+", default=[1e-6],
                        help="Edge probability thresholds for evaluate-study")
    parser.add_argument("--eval_force", action="store_true", default=False,
                        help="Force re-evaluation even if results exist")
    parser.add_argument("--eval_snapshot_steps", type=int, nargs="+", default=None,
                        help="Restrict evaluate-study to these snapshot steps")
    parser.add_argument("--eval_cscg_batch_size", type=int, default=None,
                        help="Restrict evaluate-study to runs with this CSCG batch size")
    parser.add_argument("--eval_cscg_clone_preset", type=str, default=None,
                        help="Restrict evaluate-study to runs with this CSCG clone preset")
    parser.add_argument("--snapshot_source_run_id", type=str, default=None)
    parser.add_argument("--computational_eval", nargs="?", const=True, default=False, type=strtobool)
    parser.add_argument("--computational_eval_profile", type=str, default="full", choices=["full"])
    parser.add_argument("--memory_length", type=int, default=10)
    parser.add_argument("--sensory_allow_self_loops", nargs="?", const=True,
                        default=True, type=strtobool)
    parser.add_argument("--activation_threshold", type=float, default=0.95)
    parser.add_argument("--topological_neighbourhood_threshold", type=float, default=0.6)
    parser.add_argument("--sensory_max_neurons", type=int, default=50)
    parser.add_argument("--sensory_weighting", type=float, default=1.0)
    parser.add_argument("--winning_node_lr", type=float, default=0.55)
    parser.add_argument("--topological_neighbourhood_lr", type=float, default=0.9)
    parser.add_argument("--sensory_action_lr", type=float, default=0.5)
    parser.add_argument("--global_context_lr", type=float, default=0.9)
    parser.add_argument("--sensory_max_age", type=int, default=18)
    parser.add_argument("--sensory_gaussian_shape", type=int, default=2)
    parser.add_argument("--latent_allow_self_loops", nargs="?", const=True, default=True, type=strtobool)
    parser.add_argument("--latent_max_neurons", type=int, default=300)
    parser.add_argument("--latent_action_lr", type=float, default=0.1)
    parser.add_argument("--latent_gaussian_shape", type=int, default=2)
    parser.add_argument("--latent_max_age", type=int, default=20)
    parser.add_argument("--ambiguity_threshold", type=int, default=40)
    parser.add_argument("--ambiguity_decay", type=float, default=0.99)
    parser.add_argument("--ambiguity_counter_mode", type=str, default="action", choices=["action", "node"])
    parser.add_argument(
        "--ambiguity_response_mode",
        type=str,
        default=None,
        choices=[
            "no_ambiguity_response",
            "local_repair_only",
            "refinement_without_reconstruction",
            "full_mercury",
        ],
    )
    parser.add_argument("--trace_decay", type=float, default=1.0)
    parser.add_argument("--lambda_trace", type=float, default=0.00)
    parser.add_argument("--weight_memory", type=float, default=1.0)
    parser.add_argument("--weight_undirected", type=float, default=0.0)
    parser.add_argument("--weight_base", type=float, default=0.0)
    parser.add_argument("--weight_action", type=float, default=0.0)
    parser.add_argument("--memory_replay", nargs="?", const=True, default=True, type=strtobool)
    parser.add_argument("--memory_disambiguation", nargs="?", const=True, default=True, type=strtobool)
    parser.add_argument("--ambiguity_response_enabled", nargs="?", const=True, default=True, type=strtobool)
    parser.add_argument("--ambiguity_refinement_enabled", nargs="?", const=True, default=True, type=strtobool)
    parser.add_argument("--transition_reconstruction_enabled", nargs="?", const=True, default=True, type=strtobool)
    parser.add_argument("--am_n_codebook", type=int, default=4)
    parser.add_argument("--am_lr", type=float, default=0.5)
    parser.add_argument("--am_sigma", type=float, default=0.0)
    parser.add_argument("--am_key", type=int, default=0)
    parser.add_argument("--am_identity_for_one_hot", nargs="?", const=True, default=False, type=strtobool)
    parser.add_argument(
        "--random_start",
        nargs="?",
        const=True,
        default=True,
        type=strtobool,
    )
    if any(token in {"-h", "--help"} for token in sys.argv[1:]):
        method_for_help = _raw_cli_method(sys.argv)
        if method_for_help is not None:
            _suppress_inapplicable_help(parser, method_for_help)
    args = parser.parse_args()
    _validate_method_specific_cli(parser, args, sys.argv)
    return args



def _run_evaluate_study(args: argparse.Namespace) -> None:
    from pathlib import Path
    from ..application.evaluation_protocol import EvaluationProtocol

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    study_root = Path(args.study_root) / args.study_name
    protocol = EvaluationProtocol(
        sensor=str(args.sensor),
        sensor_range=args.sensor_range,
        rand_prob=float(args.rand_prob),
        evaluation_seed=int(args.eval_seed),
        number_of_walks=int(args.eval_num_walks),
        walk_length=int(args.eval_walk_length),
        valid_trajectories_only=bool(args.valid_trajectories_only),
        edge_thresholds=list(args.eval_edge_thresholds),
    )
    method = str(args.method) if args.method else "cscg"
    snapshot_steps = None if args.eval_snapshot_steps is None else {int(step) for step in args.eval_snapshot_steps}
    evaluate_study(
        study_root=study_root,
        level=int(args.level),
        method=method,
        protocol=protocol,
        resume=not bool(args.eval_force),
        force=bool(args.eval_force),
        max_workers=int(args.max_workers),
        no_parallel=bool(args.no_parallel),
        run_rand_prob=float(args.rand_prob),
        snapshot_steps=snapshot_steps,
        cscg_batch_size=args.eval_cscg_batch_size,
        cscg_clone_preset=args.eval_cscg_clone_preset,
    )


def main() -> None:
    args = parse_arguments()
    if bool(getattr(args, "evaluate_study", False)):
        _run_evaluate_study(args)
    elif bool(getattr(args, "evaluate_study_snapshots", False)):
        evaluate_study_snapshots(args)
    elif args.study:
        run_study(args)
    else:
        run_single(args)
