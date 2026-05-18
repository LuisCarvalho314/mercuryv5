__all__ = [
    "PhaseTimer",
    "append_jsonl",
    "args_from_config",
    "artifact_layout",
    "atomic_write_json",
    "base_runtime_env",
    "baseline_enabled",
    "build_computational_payload",
    "build_run_config_payload",
    "build_run_payload",
    "collect_run_artifacts",
    "computational_eval_enabled",
    "compact_utc_now",
    "computational_notes",
    "derive_efficiency_metrics",
    "grid",
    "load_json",
    "load_run_config_model",
    "load_run_status_model",
    "maybe_log_study_wandb",
    "maybe_log_wandb",
    "peak_rss_mb",
    "prepare_study_logs",
    "read_computational_artifact",
    "read_native_metrics_from_bundle",
    "resolve_run_identity",
    "runtime_metadata",
    "run_complete",
    "run_subprocess_command",
    "safe_artifact_bytes",
    "stable_hash",
    "strtobool",
    "subsystem_status",
    "summarize_computational_raw",
    "utc_now",
    "validate_reusable_run",
    "write_computational_artifact",
]


def __getattr__(name: str):
    if name in {
        "PhaseTimer",
        "build_computational_payload",
        "computational_notes",
        "derive_efficiency_metrics",
        "peak_rss_mb",
        "read_computational_artifact",
        "runtime_metadata",
        "safe_artifact_bytes",
        "summarize_computational_raw",
        "write_computational_artifact",
    }:
        from . import computational

        return getattr(computational, name)
    if name in {
        "append_jsonl",
        "args_from_config",
        "atomic_write_json",
        "base_runtime_env",
        "compact_utc_now",
        "grid",
        "load_json",
        "run_subprocess_command",
        "stable_hash",
        "strtobool",
        "utc_now",
    }:
        from . import runtime

        return getattr(runtime, name)
    if name in {
        "artifact_layout",
        "baseline_enabled",
        "build_run_config_payload",
        "build_run_payload",
        "collect_run_artifacts",
        "computational_eval_enabled",
        "load_run_config_model",
        "load_run_status_model",
        "prepare_study_logs",
        "read_native_metrics_from_bundle",
        "resolve_run_identity",
        "run_complete",
        "subsystem_status",
        "validate_reusable_run",
    }:
        from . import storage

        return getattr(storage, name)
    if name in {"maybe_log_study_wandb", "maybe_log_wandb"}:
        from . import wandb_logger

        return getattr(wandb_logger, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
