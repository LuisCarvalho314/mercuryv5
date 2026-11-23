"""Rerender selected transition residual maps from saved JSON sidecars.

Reads payload JSON files written alongside PNGs by
`mercury_runs.analysis.plots.selected_residual_maps` and re-renders the figures
(useful for tweaking layout without re-running evaluation).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

from mercury_runs.analysis.residual_map_rendering import save_selected_figure


DEFAULT_INPUT_DIR = Path("results/current/notebook_plots")


def _is_selected_payload(payload: object) -> bool:
    panels = payload.get("panels") if isinstance(payload, dict) else None
    if not isinstance(panels, dict) or not panels:
        return False
    return all(str(key).startswith("L") and "/" in str(key) for key in panels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--vis-threshold", type=float, default=0.0)
    args = parser.parse_args()

    matplotlib.use("Agg", force=True)
    for sidecar in sorted(args.input_dir.rglob("transition_residual_*.json")):
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        if not _is_selected_payload(payload):
            continue
        output_path = Path(payload.get("figure") or sidecar.with_suffix(".png"))
        save_selected_figure(
            payload=payload,
            output_path=output_path,
            vis_threshold=float(args.vis_threshold),
            font_scale=float(payload.get("font_scale", 1.60)),
            header_font_scale=float(
                payload.get("header_font_scale", payload.get("font_scale", 1.60))
            ),
            omit_leading_zero=bool(payload.get("omit_leading_zero", False)),
        )
        print(output_path)


if __name__ == "__main__":
    main()
