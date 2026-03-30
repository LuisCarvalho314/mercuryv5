from __future__ import annotations

from copy import deepcopy
import inspect
from pathlib import Path
from typing import Any, Callable, Optional

from .config import MercuryConfig
from .evaluate import compute_mercury_paper_precision_metrics, write_mercury_paper_precision_payload
from .evaluate import compute_mercury_precision as _compute_mercury_precision
from .evaluate import summarize_mercury_run
from .prepare import generate_mercury_datasets
from .train import run_mercury_bundle
from ...infrastructure.paper_precision import resolve_eval_checkpoints
from ...infrastructure.runtime import progress_enabled


def _supports_keyword_argument(func: Callable[..., Any], argument_name: str) -> bool:
    parameters = inspect.signature(func).parameters.values()
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD or parameter.name == argument_name
        for parameter in parameters
    )


def run_mercury(
    config: MercuryConfig,
    *,
    progress_callback: Optional[Callable[..., None]] = None,
) -> Path:
    show_progress = progress_enabled(default=True)
    dataset_run_ids = generate_mercury_datasets(config=config)
    if progress_callback is not None:
        progress_callback(stage="dataset", current=1, total=1, message="Datasets ready")
    paper_precision_history: list[dict[str, object]] = []
    final_snapshot: dict[str, Any] = {}
    paper_precision_mode = str(config.paper_precision_mode).strip().lower()
    paper_precision_checkpoints = set(
        resolve_eval_checkpoints(
            total_units=int(config.num_steps),
            num_points=config.paper_precision_num_points,
            eval_interval=int(config.paper_precision_eval_interval),
        )
    )

    def _paper_precision_callback(*, step: int, sensory_graph, latent_graph, action_map) -> None:
        if step == int(config.num_steps):
            final_snapshot.update(
                sensory_graph=deepcopy(sensory_graph),
                latent_graph=deepcopy(latent_graph),
                action_map=deepcopy(action_map),
            )
        if paper_precision_mode != "per_iteration":
            return
        if int(step) not in paper_precision_checkpoints:
            return
        metrics_kwargs = {
            "config": config,
            "sensory_graph": deepcopy(sensory_graph),
            "latent_graph": deepcopy(latent_graph),
            "action_map": deepcopy(action_map),
            "show_progress": show_progress,
            "progress_desc": f"Mercury Paper Precision @ step {step}",
        }
        if progress_callback is not None and _supports_keyword_argument(compute_mercury_paper_precision_metrics, "progress_callback"):
            metrics_kwargs["progress_callback"] = lambda **kwargs: progress_callback(stage=f"paper_precision@{step}", **kwargs)
        metrics = compute_mercury_paper_precision_metrics(
            **metrics_kwargs,
        )
        paper_precision_history.append({"step": int(step), "observed_samples": int(step), **metrics})

    step_callback = None
    if bool(config.paper_precision_enabled) and paper_precision_mode in {"final", "per_iteration"}:
        step_callback = _paper_precision_callback

    bundle_kwargs = {
        "config": config,
        "dataset_run_ids": dataset_run_ids,
        "latent_step_callback": step_callback,
        "show_progress": show_progress,
    }
    if progress_callback is not None and _supports_keyword_argument(run_mercury_bundle, "progress_callback"):
        bundle_kwargs["progress_callback"] = progress_callback
    states_path = run_mercury_bundle(**bundle_kwargs)
    if progress_callback is not None:
        progress_callback(stage="native_eval", current=0, total=1, message="Starting native eval")
    if bool(config.paper_precision_enabled):
        if paper_precision_mode == "per_iteration":
            final_metrics = (
                paper_precision_history[-1]
                if paper_precision_history
                else compute_mercury_paper_precision_metrics(
                    **{
                        key: value
                        for key, value in {
                            "config": config,
                            "sensory_graph": final_snapshot.get("sensory_graph"),
                            "latent_graph": final_snapshot.get("latent_graph"),
                            "action_map": final_snapshot.get("action_map"),
                            "show_progress": show_progress,
                            "progress_desc": "Mercury Paper Precision",
                            "progress_callback": (
                                (lambda **kwargs: progress_callback(stage="paper_precision", **kwargs))
                                if (
                                    progress_callback is not None
                                    and _supports_keyword_argument(compute_mercury_paper_precision_metrics, "progress_callback")
                                )
                                else None
                            ),
                        }.items()
                        if value is not None or key != "progress_callback"
                    }
                )
            )
            metrics = {key: value for key, value in final_metrics.items() if key not in {"step", "observed_samples"}}
            write_mercury_paper_precision_payload(
                config=config,
                metrics=metrics,
                history=paper_precision_history,
                schedule_unit="training_step",
            )
        else:
            if not final_snapshot:
                raise RuntimeError("Mercury final paper precision requires a captured final training snapshot.")
            metrics_kwargs = {
                "config": config,
                "sensory_graph": final_snapshot["sensory_graph"],
                "latent_graph": final_snapshot["latent_graph"],
                "action_map": final_snapshot["action_map"],
                "show_progress": show_progress,
                "progress_desc": "Mercury Paper Precision",
            }
            if progress_callback is not None and _supports_keyword_argument(compute_mercury_paper_precision_metrics, "progress_callback"):
                metrics_kwargs["progress_callback"] = lambda **kwargs: progress_callback(stage="paper_precision", **kwargs)
            metrics = compute_mercury_paper_precision_metrics(**metrics_kwargs)
            write_mercury_paper_precision_payload(
                config=config,
                metrics=metrics,
                history=[],
                schedule_unit="final_only",
            )
    _compute_mercury_precision(config=config)
    if progress_callback is not None:
        progress_callback(stage="native_eval", current=1, total=1, message="Native eval complete")
    return states_path


def compute_mercury_precision(config: MercuryConfig) -> None:
    _compute_mercury_precision(config=config)


def compute_mercury_paper_precision(config: MercuryConfig) -> None:
    raise RuntimeError("Mercury paper precision must be computed from a captured training snapshot via run_mercury(...).")


__all__ = [
    "MercuryConfig",
    "compute_mercury_precision",
    "compute_mercury_paper_precision",
    "generate_mercury_datasets",
    "run_mercury",
    "summarize_mercury_run",
]
