# scripts/study_summary.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import polars as pl

from src.mercury_runs.save_results import read_bundle_meta


def _find_states_parquet(run_root: Path) -> Optional[Path]:
    bundles_root = run_root / "bundles" / "mercury"
    if not bundles_root.exists():
        return None
    candidates = sorted(bundles_root.rglob("*_states.parquet"))
    return candidates[-1] if candidates else None


def summarize(study_root: Path, study_name: str) -> Path:
    study_dir = study_root / study_name
    run_dirs = [path for path in study_dir.iterdir() if path.is_dir()]

    rows: List[dict] = []
    for run_dir in sorted(run_dirs):
        states_path = _find_states_parquet(run_dir)
        if states_path is None:
            continue

        meta = read_bundle_meta(states_path)
        if meta is None:
            continue

        metrics_files = sorted((run_dir / "metrics").glob("*_precision.parquet"))
        if not metrics_files:
            continue

        metrics_frame = pl.read_parquet(metrics_files[-1])
        stats = metrics_frame.select(
            pl.col("latent_precision").mean().alias("latent_precision_mean"),
            pl.col("latent_precision").min().alias("latent_precision_min"),
            pl.col("latent_precision").max().alias("latent_precision_max"),
            pl.col("sensory_precision").mean().alias("sensory_precision_mean"),
            pl.col("latent_precision").var().alias("latent_precision_var"),
        ).to_dicts()[0]

        row = {
            "run_id": meta.run_id,
            "timestamp_utc": meta.timestamp_utc,
            "level": meta.source.level,
            "sensor": meta.source.sensor,
            "sensor_range": meta.source.sensor_range,
            "memory_length": meta.memory_length,
            "activation_threshold": meta.sensory_params.get("activation_threshold"),
            "sensory_weighting": meta.sensory_params.get("sensory_weighting"),
            "ambiguity_threshold": (meta.latent_params or {}).get("ambiguity_threshold"),
            "trace_decay": (meta.latent_params or {}).get("trace_decay"),
            "am_lr": meta.action_map_params.get("lr"),
            "am_sigma": meta.action_map_params.get("sigma"),
            "am_key": meta.action_map_params.get("key"),
            "am_n_codebook": meta.action_map_params.get("n_codebook"),
        }
        row.update(stats)
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No runnable runs found under: {study_dir}")

    summary_frame = pl.DataFrame(rows).sort("latent_precision_mean", descending=True)
    out_path = study_dir / "study_summary.parquet"
    summary_frame.write_parquet(out_path, compression="zstd", statistics=True)
    return out_path


if __name__ == "__main__":
    output = summarize(Path("results/studies"), "level18_cd1_grid")
    print(f"Wrote: {output}")
