from __future__ import annotations

import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .runtime import atomic_write_json

try:  # pragma: no cover - platform-dependent import
    import resource
except Exception:  # pragma: no cover
    resource = None


COMPUTATIONAL_SCHEMA_VERSION = 1
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def runtime_metadata() -> Dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "thread_env": {name: os.environ.get(name) for name in THREAD_ENV_VARS},
    }


def peak_rss_mb() -> Optional[float]:
    if resource is None:  # pragma: no cover - platform-dependent
        return None
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:  # pragma: no cover
        return None
    if sys.platform == "darwin":
        return float(rss) / (1024.0 * 1024.0)
    return float(rss) / 1024.0


def safe_artifact_bytes(paths: Iterable[Path]) -> int:
    total = 0
    for path in paths:
        if path.exists() and path.is_file():
            total += int(path.stat().st_size)
    return total


def computational_notes(*, peak_rss: Optional[float], include_jit_note: bool = False) -> list[str]:
    notes: list[str] = ["artifact scope: deployable=model/state artifacts, total=all method-specific persisted outputs"]
    if peak_rss is None:
        notes.append("RSS unavailable")
    if include_jit_note:
        notes.append("JIT warmup excluded from steady-state timings")
    return notes


class PhaseTimer:
    def __init__(self) -> None:
        self.started_at = time.perf_counter()
        self.phase_started_at = self.started_at
        self.current_phase = "train"
        self.durations: Dict[str, float] = {"train": 0.0, "eval": 0.0, "paper_precision": 0.0, "jit_warmup": 0.0}

    def switch(self, phase: str) -> None:
        now = time.perf_counter()
        self.durations[self.current_phase] = self.durations.get(self.current_phase, 0.0) + (now - self.phase_started_at)
        self.current_phase = phase
        self.phase_started_at = now

    def finish(self) -> Dict[str, float]:
        now = time.perf_counter()
        self.durations[self.current_phase] = self.durations.get(self.current_phase, 0.0) + (now - self.phase_started_at)
        self.durations["measured_total"] = self.durations.get("train", 0.0) + self.durations.get("eval", 0.0)
        self.durations["wall_total"] = now - self.started_at
        return dict(self.durations)


def build_computational_payload(
    *,
    method: str,
    measurement_profile: str,
    raw_metrics: Dict[str, Any],
    n_train_steps_measured: Optional[int],
    n_eval_steps_measured: Optional[int],
    timing_repetitions: int = 1,
    notes: Optional[list[str]] = None,
) -> Dict[str, Any]:
    return {
        "schema_version": COMPUTATIONAL_SCHEMA_VERSION,
        "method": method,
        "measurement_profile": measurement_profile,
        "measurement_scope": "cpu_edge_proxy",
        "runtime_metadata": runtime_metadata(),
        "measurement_context": {
            "n_train_steps_measured": n_train_steps_measured,
            "n_eval_steps_measured": n_eval_steps_measured,
            "timing_repetitions": int(timing_repetitions),
        },
        "metrics": raw_metrics,
        "notes": notes or [],
    }


def write_computational_artifact(path: Path, payload: Dict[str, Any]) -> Path:
    atomic_write_json(path, payload)
    return path


def read_computational_artifact(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def summarize_computational_raw(path: Optional[Path]) -> Dict[str, Any]:
    payload = read_computational_artifact(path)
    metrics = payload.get("metrics") or {}
    return dict(metrics) if isinstance(metrics, dict) else {}


def _metric_div(numerator: Any, denominator: Any) -> Optional[float]:
    if numerator is None or denominator in {None, 0, 0.0}:
        return None
    try:
        return float(numerator) / float(denominator)
    except Exception:
        return None


def derive_efficiency_metrics(*, raw_metrics: Dict[str, Any], precision_value: Optional[float], success_rate: Optional[float] = None) -> Dict[str, Any]:
    deployable_mb = _metric_div(raw_metrics.get("deployable_artifact_bytes"), 1024.0 * 1024.0)
    total_mb = _metric_div(raw_metrics.get("total_artifact_bytes"), 1024.0 * 1024.0)
    return {
        "seconds_per_train_step": _metric_div(raw_metrics.get("train_wall_time_seconds"), raw_metrics.get("n_train_steps_measured")),
        "seconds_per_eval_step": _metric_div(raw_metrics.get("eval_wall_time_seconds"), raw_metrics.get("n_eval_steps_measured")),
        "precision_per_total_second": _metric_div(precision_value, raw_metrics.get("total_wall_time_seconds")),
        "success_rate_per_total_second": _metric_div(success_rate, raw_metrics.get("total_wall_time_seconds")),
        "precision_per_train_second": _metric_div(precision_value, raw_metrics.get("train_wall_time_seconds")),
        "precision_per_peak_rss_mb": _metric_div(precision_value, raw_metrics.get("peak_rss_mb")),
        "precision_per_deployable_mb": _metric_div(precision_value, deployable_mb),
        "precision_per_total_mb": _metric_div(precision_value, total_mb),
    }
