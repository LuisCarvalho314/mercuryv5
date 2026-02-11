# scripts/study_summarize_from_parquet_meta.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import polars as pl

from mercury_runs.save_results import read_bundle_meta


def _find_states_parquet(run_root: Path) -> Optional[Path]:
    bundles_root = run_root / "bundles" / "mercury"
    if not bundles_root.exists():
        return None
    candidates = list(bundles_root.rglob("*_states.parquet"))
    if not candidates:
        return None
    # Prefer the single one per run; if multiple exist, pick latest by name
    return sorted(candidates)[-1]


def summarize_study(study_root: Path, study_name: str) -> Path:
    study_dir = study_root / study_name
    if not study_dir.exists():
        raise FileNotFoundError(study_dir)

    run_dirs = [path for path in study_dir.iterdir() if path.is_dir()]

    rows: List[dict] = []
    for run_dir in sorted(run_dirs):
        states_path = _find_states_parquet(run_dir)
        if states_path is None:
            continue

        meta = read_bundle_meta(states_path)
        if meta is None:
            continue

        metrics_dir = run_dir / "metrics"
        metrics_candidates = sorted(metrics_dir.glob("*_precision.parquet"))
        if not metrics_candidates:
            continue

        metrics_path = metrics_candidates[-1]
        metrics_frame = pl.read_parquet(metrics_path)

        summary = metrics_frame.select(
            pl.col("latent_precision").mean().alias("latent_precision_mean"),
            pl.col("latent_precision").min().alias("latent_precision_min"),
            pl.col("latent_precision").max().alias("latent_precision_max"),
            pl.col("sensory_precision").mean().alias("sensory_precision_mean"),
            pl.col("latent_precision").var().alias("latent_precision_var"),

        ).to_dicts()[0]

        # Pull key hparams from embedded meta
        row = {
            "run_id": meta.run_id,
            "timestamp_utc": meta.timestamp_utc,
            "level": meta.source.level,
            "sensor": meta.source.sensor,
            "sensor_range": meta.source.sensor_range,
            "memory_length": meta.memory_length,

            # ActionMap
            "am_lr": meta.action_map_params.get("lr"),
            "am_sigma": meta.action_map_params.get("sigma"),
            "am_key": meta.action_map_params.get("key"),
            "am_n_codebook": meta.action_map_params.get("n_codebook"),

            # Sensory / latent knobs (add more as needed)
            "activation_threshold": meta.sensory_params.get("activation_threshold"),
            "sensory_weighting": meta.sensory_params.get("sensory_weighting"),
            "ambiguity_threshold": (meta.latent_params or {}).get("ambiguity_threshold"),
            "trace_decay": (meta.latent_params or {}).get("trace_decay"),
        }

        row.update(summary)
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No runs with readable states+meta+metrics found in: {study_dir}")

    out_frame = pl.DataFrame(rows).sort("latent_precision_mean", descending=True)

    out_path = study_dir / "study_summary.parquet"
    out_frame.write_parquet(out_path, compression="zstd", statistics=True)
    return out_path


if __name__ == "__main__":
    output_path = summarize_study(Path("results/studies/"),
                                  "memory_replay_memory_disambiguation")
    print(f"Wrote: {output_path}")
