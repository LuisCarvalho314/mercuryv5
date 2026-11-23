"""CLI entry point for per-run transition residual map figures.

Usage:
    python -m mercury_runs.analysis.plots.transition_residual_maps_cli \\
        --run_root results/current/studies/<study>/<run_id>
"""
from __future__ import annotations

import argparse
from pathlib import Path

from mercury_runs.analysis.transition_residual_maps import SUPPORTED_METHODS, generate_transition_residual_maps


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ground-truth-aligned transition residual maps.")
    parser.add_argument("--run_root", type=str, required=True, help="Run directory under results/current/studies/<study>/<run_id>")
    parser.add_argument("--methods", nargs="*", default=None, choices=SUPPORTED_METHODS)
    parser.add_argument("--output", type=str, default=None, help="Output PNG path. Defaults to <run_root>/plots/transition_residual_maps.png")
    parser.add_argument("--vis_threshold", type=float, default=1e-3, help="Plot-only residual threshold for edge visibility")
    parser.add_argument("--show_edge_labels", action="store_true", help="Annotate displayed residual edges")
    parser.add_argument("--separate_files", action="store_true", help="Also save one PNG per method")
    args = parser.parse_args()

    outputs = generate_transition_residual_maps(
        run_root=Path(args.run_root),
        methods=args.methods,
        output_path=Path(args.output) if args.output else None,
        vis_threshold=float(args.vis_threshold),
        show_edge_labels=bool(args.show_edge_labels),
        separate_files=bool(args.separate_files),
    )
    for name, path in outputs.items():
        print(f"Saved {name}: {path}")


if __name__ == "__main__":
    main()
